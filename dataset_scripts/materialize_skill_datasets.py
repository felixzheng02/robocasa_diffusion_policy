"""
Step 3 of the pick/place split: write the two skill datasets.

Reads the ranges chosen by select_split_points.py and emits, for every source dataset, a
<Task>_pick and a <Task>_place LeRobot dataset laid out exactly like the originals, so the
registry and every downstream loader address them unchanged.

Frames come from decoding the source mp4s rather than re-rendering the scene (which is what
convert_hdf5_lerobot.py does), so the pixels are the ones the demos actually shipped with.
No simulator is needed here.

Example:
    python -m robocasa.scripts.dataset_scripts.materialize_skill_datasets \\
        --split pretrain target --num_procs 8
"""

import argparse
import json
import multiprocessing as mp
import shutil
import time
import traceback
from pathlib import Path

import av
import numpy as np
import pandas as pd

import robocasa.utils.lerobot_utils as LU
import diffusion_policy.skills.skill_utils as SU
from robocasa.scripts.dataset_scripts.convert_hdf5_lerobot import FPS, VIDEO_INFO, add_task_name
from robocasa.utils.dataset_registry_utils import get_ds_meta

CAMERAS = ["robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right"]

# Per-dataset PNG staging pool. Each job opens two datasets, so the live process count is
# 2 x IMAGE_WRITER_PROCESSES x --num_procs. Keep this small: the pools are the dominant
# memory consumer here, well ahead of the one episode of decoded frames each worker holds.
IMAGE_WRITER_PROCESSES = 2
IMAGE_WRITER_THREADS = 4


def build_features(img_shape):
    """
    Describe the columns a skill dataset holds, matching the originals exactly.

    Kept byte-identical to `convert_hdf5_lerobot.py`'s feature dict on purpose: it is what
    makes the generated datasets interchangeable with the source ones, so playback, the
    dataset soups and the GR00T loader all work on them unmodified.

    Inputs
    ------
    img_shape : tuple[int, int, int]
        The `(height, width, channel)` of one decoded camera frame, probed from the source
        mp4 rather than assumed.

    Outputs
    -------
    features : dict[str, dict]
        One entry per column. Three video columns, one per camera in `CAMERAS`, plus six
        array columns:
        - `annotation.human.task_description` -- `(1,)` int64 instruction id
        - `annotation.human.task_name` -- `(1,)` int64 task-name id
        - `observation.state` -- `(16,)` float64
        - `action` -- `(12,)` float64
        - `next.reward` -- `(1,)` float32
        - `next.done` -- `(1,)` bool

    Procedure
    ---------
    1. Build one video feature per camera, all sharing the probed image shape and the
       standard video encoding settings.
    2. Add the two annotation columns, the state and action columns, and the reward and done
       flags.
    3. Return the combined dict.
    """
    features = {
        f"observation.images.{cam}": {
            "dtype": "video",
            "shape": img_shape,
            "names": ["height", "width", "channel"],
            "video_info": VIDEO_INFO,
        }
        for cam in CAMERAS
    }
    features.update(
        {
            "annotation.human.task_description": {"dtype": "int64", "shape": (1,)},
            "annotation.human.task_name": {"dtype": "int64", "shape": (1,)},
            "observation.state": {"dtype": "float64", "shape": (16,)},
            "action": {"dtype": "float64", "shape": (12,)},
            "next.reward": {"dtype": "float32", "shape": (1,)},
            "next.done": {"dtype": "bool", "shape": (1,)},
        }
    )
    return features


def decode_video(path):
    """
    Decode a whole camera video into memory as an array of frames.

    Decoding the shipped mp4 is deliberate, rather than re-rendering the scene. Re-rendering
    would re-derive the pixels through scene setup, texture generation and camera configs,
    and could silently diverge from the video the demo actually shipped with.

    Inputs
    ------
    path : pathlib.Path or str
        Path to one episode's mp4 for one camera.

    Outputs
    -------
    frames : np.ndarray, `(T, H, W, 3)`, uint8
        Every frame in the file, in order, as RGB. Roughly 50 MB for a typical episode,
        which is why callers decode one episode at a time and drop it before the next.

    Procedure
    ---------
    1. Open the container.
    2. Decode the first video stream sequentially, converting each frame to RGB.
    3. Stack the frames into one array along a new leading time axis.
    """
    with av.open(str(path)) as container:
        return np.stack(
            [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        )


def source_paths(src_dir, ep_idx):
    """
    Locate one episode's parquet file and its three camera videos.

    Both live under a chunk sub-directory whose name is not known ahead of time, so each is
    found by glob and then required to be unique.

    Inputs
    ------
    src_dir : pathlib.Path
        The source LeRobot dataset directory.
    ep_idx : int
        Zero-based episode index.

    Outputs
    -------
    parquet : pathlib.Path
        That episode's parquet file, holding its actions, states and rewards.
    videos : dict[str, pathlib.Path]
        One mp4 path per camera in `CAMERAS`, keyed by camera name. Always three entries.

    Raises
    ------
    AssertionError
        If the parquet or any camera video is missing, or matches more than once. Loud
        because a silently missing camera would produce a dataset with a blank view.

    Procedure
    ---------
    1. Glob for the episode's parquet under any data chunk and require exactly one hit.
    2. For each camera, glob for its mp4 under any video chunk and require exactly one hit.
    3. Return the parquet path and the camera-to-path mapping.
    """
    parquet = sorted(src_dir.glob(f"data/*/episode_{ep_idx:06d}.parquet"))
    assert len(parquet) == 1, f"episode {ep_idx}: {parquet}"
    videos = {}
    for cam in CAMERAS:
        hits = sorted(
            src_dir.glob(f"videos/*/observation.images.{cam}/episode_{ep_idx:06d}.mp4")
        )
        assert len(hits) == 1, f"episode {ep_idx} cam {cam}: {hits}"
        videos[cam] = hits[0]
    return parquet[0], videos


def skill_reward(sig_len, lo, hi, skill, t_moved, orig_reward):
    """
    Build the reward column for one skill segment.

    Place inherits the task's real reward. Pick cannot: the original task only rewards
    success after the object has been released, so a pick segment's slice of it would be all
    zeros and the policy would train against a reward that never fires. Pick's reward is
    therefore synthesised from the pick criterion itself -- the object is up.

    Inputs
    ------
    sig_len : int
        Length of the source episode. Accepted for signature symmetry with the caller's other
        arguments and not read.
    lo : int
        First frame of the segment, inclusive.
    hi : int
        Last frame of the segment, inclusive.
    skill : str
        Either `"pick"` or `"place"`.
    t_moved : int
        Frame at which the object had visibly moved, in source-episode coordinates. Only used
        for pick.
    orig_reward : np.ndarray, `(T,)`, float32
        The source episode's recorded reward. Only used for place.

    Outputs
    -------
    reward : np.ndarray, `(hi - lo + 1,)`, float32
        For place, the matching slice of the original reward. For pick, zeros up to
        `t_moved` and 1.0 from there to the end of the segment. Clamped so a `t_moved`
        before the segment start yields an all-ones array rather than an error.

    Procedure
    ---------
    1. For place, slice the original reward from `lo` to `hi` inclusive and return it.
    2. For pick, allocate a zero array as long as the segment.
    3. Set every frame from `t_moved` onwards to 1.0, converting `t_moved` into
       segment-relative coordinates and clamping at 0.
    """
    if skill == "place":
        return orig_reward[lo : hi + 1].astype(np.float32)
    r = np.zeros(hi - lo + 1, dtype=np.float32)
    r[max(t_moved - lo, 0) :] = 1.0
    return r


def write_episode(dataset, ep, skill, frames, df, task_id, task_name_idx):
    """
    Append one skill segment of one source episode to a skill dataset.

    Actions and states are copied verbatim out of the source parquet, one row at a time, so
    the generated dataset carries byte-identical control data. Only the four columns listed
    below are recomputed.

    Inputs
    ------
    dataset : LU.LerobotDatasetWrapper
        The open output dataset for this skill. Mutated: one episode is appended.
    ep : dict
        This episode's entry from the splits file. Reads the `pick`/`place` range, the
        matching `<skill>_task` string, and `t_moved`.
    skill : str
        Either `"pick"` or `"place"`, selecting which range of the episode to write.
    frames : dict[str, np.ndarray]
        Decoded video per camera, each `(T, H, W, 3)` uint8, covering the whole source
        episode -- this function indexes into it rather than expecting a pre-sliced view.
    df : pandas.DataFrame
        The source episode's parquet, with `observation.state`, `action` and `next.reward`.
    task_id : int
        Instruction id for this segment's task string, matching `meta/tasks.jsonl`.
    task_name_idx : int
        Id standing for the original task name.

    Outputs
    -------
    None
        The episode is written into `dataset` as a side effect.

    Procedure
    ---------
    1. Read the segment's inclusive frame range and its task string.
    2. Build the reward column for the segment.
    3. For each frame in the range, gather the three camera images at that source index.
    4. Copy the state and action rows verbatim from the parquet.
    5. Attach the instruction and task-name ids, the reward for this step, and a done flag
       true only on the segment's final frame.
    6. Add the frame to the dataset under this segment's task string.
    7. Close the episode once every frame has been added.
    """
    lo, hi = ep[skill]
    lang = ep[f"{skill}_task"]
    reward = skill_reward(
        len(df), lo, hi, skill, ep["t_moved"],
        np.asarray(df["next.reward"].tolist(), dtype=np.float32).reshape(-1),
    )
    for i, t in enumerate(range(lo, hi + 1)):
        frame = {f"observation.images.{cam}": frames[cam][t] for cam in CAMERAS}
        frame["observation.state"] = np.asarray(df["observation.state"].iloc[t])
        frame["action"] = np.asarray(df["action"].iloc[t])
        frame["annotation.human.task_description"] = np.array([task_id], dtype=np.int64)
        frame["annotation.human.task_name"] = np.array([task_name_idx], dtype=np.int64)
        frame["next.reward"] = np.array([reward[i]], dtype=np.float32)
        frame["next.done"] = np.array([t == hi], dtype=bool)
        dataset.add_frame(frame, task=lang)
    dataset.save_episode()


def process_dataset(src_dir, splits_path):
    """
    Write both the pick and the place dataset for one source dataset.

    One source dataset is one unit of work for one worker process. Episodes are decoded one
    at a time and written to both output datasets before being dropped: holding every
    episode's frames would cost about 50 MB each, i.e. tens of GB on the 500-episode target
    datasets.

    Inputs
    ------
    src_dir : str or pathlib.Path
        The source LeRobot dataset directory.
    splits_path : str or pathlib.Path
        The `*_splits.json` written by step 2 for this dataset.

    Outputs
    -------
    summary : str
        A one-line report, or `"<task>: nothing to write"` when the splits file contains no
        usable episodes.

    Side effects
    ------------
    Creates two dataset directories beside the source, `<Task>_pick/<date>/lerobot` and
    `<Task>_place/<date>/lerobot`, **deleting any existing copy first**. Each gets the same
    `data/`, `videos/`, `meta/` and `extras/` layout as the original.

    Procedure
    ---------
    1. Read the splits file and sort its usable episodes numerically; bail out early if empty.
    2. Derive the two output directories from the source path, keeping the date folder.
    3. Assign each distinct task string an id in first-seen order, so the parquet column
       agrees with the `meta/tasks.jsonl` lerobot writes as episodes are added.
    4. Probe one frame of one video to learn the image shape.
    5. Delete any existing output directory and create both datasets with that feature set.
    6. For each episode: locate its files, read the parquet and sim states, decode all three
       videos, and read its metadata.
    7. For each skill, append the segment, then save the sliced sim states, the scene XML and
       an `ep_meta` extended with a `skill` block recording the source task, source episode,
       source range and the three split frame indices.
    8. Drop the decoded frames before moving to the next episode.
    9. Copy the source `dataset_meta.json` into both outputs with only `total` rewritten,
       register the task name, add the GR00T metadata, and remove the PNG staging directory.
    10. Shut down both image-writer pools in a `finally`, then return the summary.

    Notes
    -----
    Two things here are deliberate and should not be "fixed":

    - `extras/dataset_meta.json` still names the **original** task, not `<Task>_pick`, because
      playback and the chained evaluator must rebuild the real scene.
    - The `finally` that stops the image writers is load-bearing. `LeRobotDataset` never shuts
      its pool down on its own, and a worker handling many datasets in sequence leaked four
      processes per job: 45 live writers were observed on the target split and 67 on pretrain,
      which exhausted RAM and all 7 GB of swap. Because it scales with job turnover rather
      than concurrency, lowering `--num_procs` alone never fixes it.

    This function is **not resumable** -- it removes and rewrites each output in full. Resume
    at dataset granularity with `--tasks` instead.
    """
    src_dir = Path(src_dir)
    splits = json.loads(Path(splits_path).read_text())
    episodes = splits["episodes"]
    ep_ids = sorted(episodes, key=int)
    if not ep_ids:
        return f"{splits['env_name']}: nothing to write"

    t0 = time.time()
    task_dir = src_dir.parent.parent  # .../atomic/<Task>
    date = src_dir.parent.name
    out_dirs = {
        skill: task_dir.parent / f"{task_dir.name}_{skill}" / date / "lerobot"
        for skill in SU.SKILLS
    }

    # Instruction ids in first-seen order, matching how lerobot assigns task indices as
    # episodes are added, so the parquet column agrees with meta/tasks.jsonl.
    task_to_id = {skill: {} for skill in SU.SKILLS}
    for ep_id in ep_ids:
        for skill in SU.SKILLS:
            lang = episodes[ep_id][f"{skill}_task"]
            task_to_id[skill].setdefault(lang, len(task_to_id[skill]))
    task_name_idx = {skill: len(task_to_id[skill]) for skill in SU.SKILLS}

    # peek at one frame to learn the image shape
    _, probe_videos = source_paths(src_dir, int(ep_ids[0]))
    with av.open(str(probe_videos[CAMERAS[0]])) as container:
        img_shape = next(container.decode(video=0)).to_ndarray(format="rgb24").shape

    datasets = {}
    try:
        for skill, out_dir in out_dirs.items():
            if out_dir.exists():
                shutil.rmtree(out_dir)
            datasets[skill] = LU.LerobotDatasetWrapper.create(
                repo_id=out_dir,
                robot_type="PandaOmron",
                fps=FPS,
                features=build_features(img_shape),
                image_writer_threads=IMAGE_WRITER_THREADS,
                image_writer_processes=IMAGE_WRITER_PROCESSES,
            )

        for out_idx, ep_id in enumerate(ep_ids):
            ep = episodes[ep_id]
            ep_idx = int(ep_id)
            parquet_path, videos = source_paths(src_dir, ep_idx)
            df = pd.read_parquet(parquet_path)
            states = LU.get_episode_states(src_dir, ep_idx)
            frames = {cam: decode_video(p) for cam, p in videos.items()}
            ep_meta_src = LU.get_episode_meta(src_dir, ep_idx)
            model_gz = src_dir / "extras" / f"episode_{ep_idx:06d}" / "model.xml.gz"

            for skill in SU.SKILLS:
                lang = ep[f"{skill}_task"]
                write_episode(
                    datasets[skill], ep, skill, frames, df,
                    task_to_id[skill][lang], task_name_idx[skill],
                )
                lo, hi = ep[skill]
                ep_meta = dict(ep_meta_src)
                ep_meta["lang"] = lang
                ep_meta["skill"] = {
                    "skill": skill,
                    "obj": ep["obj"],
                    "receptacle": ep["target"] if skill == "place" else None,
                    "src_task": splits["env_name"],
                    "src_episode": ep_idx,
                    "src_range": [lo, hi],
                    "t_grasp": ep["t_grasp"],
                    "t_moved": ep["t_moved"],
                    "t_success": ep["t_success"],
                }
                LU.save_extra_demo_info_raw(
                    out_dirs[skill], states[lo : hi + 1], ep_meta, model_gz, out_idx
                )
            del frames

        # env metadata deliberately still names the ORIGINAL task, so playback and the
        # chained evaluator rebuild the real scene.
        src_meta = json.loads((src_dir / "extras" / "dataset_meta.json").read_text())
        for skill, out_dir in out_dirs.items():
            meta = dict(src_meta)
            meta["total"] = int(datasets[skill].meta.total_frames)
            (out_dir / "extras" / "dataset_meta.json").write_text(json.dumps(meta, indent=4))
            add_task_name(out_dir, splits["env_name"], task_name_idx[skill])
            LU.add_groot_specific_metadata(out_dir.parent)
            shutil.rmtree(out_dir / "images", ignore_errors=True)
    finally:
        # Release the image-writer pools. A worker handles many datasets in sequence and
        # LeRobotDataset never shuts its pool down on its own, so without this every job
        # leaves its processes behind: observed 45 live writers on the slow target split and
        # 67 on the faster pretrain split, which exhausted RAM and all 7 GB of swap. In a
        # finally block because a failed job must not leak either.
        for ds in datasets.values():
            ds.stop_image_writer()

    return (
        f"{splits['env_name']}: {len(ep_ids)} episodes -> pick + place "
        f"in {time.time() - t0:.0f}s"
    )


def worker(job_queue, result_queue):
    """
    Worker-process loop: take source datasets off the queue and materialise them until none
    are left.

    Runs in a spawned child process, so it must be importable at module level and cannot rely
    on state inherited from the parent.

    Inputs
    ------
    job_queue : multiprocessing.Queue
        Holds `(src_dir, splits_path)` string pairs. An empty queue is how the worker learns
        to exit.
    result_queue : multiprocessing.Queue
        Where one summary string per finished dataset is put. The parent reads exactly one
        result per job, so this must receive a message even when the dataset fails.

    Outputs
    -------
    None
        Returns once the job queue is empty. Results travel through `result_queue`.

    Procedure
    ---------
    1. Try to take a job without blocking; return from the function if that fails, which
       means the queue is drained.
    2. Materialise that dataset and put its summary on the result queue.
    3. If it raises, print the traceback and put a `"FAILED ..."` line on the result queue
       instead, so the parent's count still balances.
    4. Loop.
    """
    while True:
        try:
            src_dir, splits_path = job_queue.get_nowait()
        except Exception:
            return
        try:
            result_queue.put(process_dataset(src_dir, splits_path))
        except Exception as exc:
            traceback.print_exc()
            result_queue.put(f"FAILED {src_dir}: {type(exc).__name__}: {exc}")


def main():
    """
    Command-line entry point: write the pick and place datasets for every split dataset.

    The expensive final step -- hours of video decoding and re-encoding. Launch it under tmux
    or `setsid nohup`.

    Inputs
    ------
    Read from the command line, not from arguments:
    --tasks : list[str]
        Task names to materialise. Defaults to all 18 PickPlace tasks.
    --split : list[str]
        Splits to materialise. Defaults to both `pretrain` and `target`.
    --num_procs : int
        Worker processes. Defaults to 8. Live image-writer processes are
        `2 x IMAGE_WRITER_PROCESSES x --num_procs`.
    --cache_dir : str or None
        Where the splits files live. Defaults to a `skill_cache` folder beside each dataset.

    Outputs
    -------
    None
        Prints the job count first, then one summary line per dataset as it finishes.
        The real output is the skill datasets on disk.

    Procedure
    ---------
    1. Parse the arguments.
    2. For every split and task, resolve the source dataset and its splits file.
    3. Skip pairs with no registry entry or no splits file.
    4. Print how many datasets will run and with how many workers.
    5. Create spawn-context queues and load every job.
    6. Start up to `--num_procs` workers, never more than there are jobs.
    7. Read exactly one result per job and print it as it arrives.
    8. Join every worker before returning.

    Notes
    -----
    The first printed line is the sanity check: under zsh, `--tasks $TASK_LIST` arrives as a
    single argument, so the count must read the number of datasets you expected, not 1.

    Checking a run for completeness means counting `episodes.jsonl` lines and confirming
    `stats.json`, `tasks.jsonl` and `extras/dataset_meta.json` -- **not** `info.json` or the
    directory existing, both of which are written at dataset creation, so a killed job leaves
    them behind on an empty dataset.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=SU.PICK_PLACE_TASKS)
    p.add_argument("--split", nargs="+", default=["pretrain", "target"])
    p.add_argument("--num_procs", type=int, default=8)
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    jobs = []
    for split in args.split:
        for task in args.tasks:
            meta = get_ds_meta(task, split, "human")
            if meta is None:
                continue
            src = Path(meta["path"])
            cache = Path(args.cache_dir) if args.cache_dir else src.parent / "skill_cache"
            splits_path = cache / f"{task}_{split}_splits.json"
            if splits_path.exists():
                jobs.append((str(src), str(splits_path)))

    print(f"{len(jobs)} datasets to materialize with {args.num_procs} workers")

    ctx = mp.get_context("spawn")
    job_queue, result_queue = ctx.Queue(), ctx.Queue()
    for j in jobs:
        job_queue.put(j)
    procs = [
        ctx.Process(target=worker, args=(job_queue, result_queue))
        for _ in range(min(args.num_procs, len(jobs)))
    ]
    for pr in procs:
        pr.start()
    for _ in jobs:
        print(result_queue.get(), flush=True)
    for pr in procs:
        pr.join()


if __name__ == "__main__":
    main()
