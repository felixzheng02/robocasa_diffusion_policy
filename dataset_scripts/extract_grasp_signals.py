"""
Step 1 of the pick/place split: replay every demo in the simulator and record the
per-frame signals that determine where the pick ends and the place begins.

This is the only expensive step (it rebuilds the MuJoCo scene once per episode, ~7 s), so
it is kept separate from the actual choice of split points. Re-tuning a threshold means
re-running select_split_points.py over the cache this writes, which takes seconds.

No rendering happens here, so no GL context is needed.

Example:
    python -m robocasa.scripts.dataset_scripts.extract_grasp_signals \\
        --split pretrain --num_procs 6
"""

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import robosuite

import robocasa.utils.lerobot_utils as LU
import diffusion_policy.skills.skill_utils as SU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils.dataset_registry_utils import get_ds_meta

def make_env(dataset_dir):
    """
    Rebuild the simulator env a dataset was recorded in, configured for physics only.

    Rendering is switched off in all three places it can be enabled, because this step reads
    contacts and positions out of the physics state and never looks at pixels. That is what
    lets the whole step run headless with no GL context.

    Inputs
    ------
    dataset_dir : pathlib.Path
        A LeRobot dataset directory, the one holding `meta/` and `data/`.

    Outputs
    -------
    env : robosuite env
        A fresh env matching the dataset's recorded kwargs, with no renderer, no offscreen
        renderer and no camera observations. Not yet reset to any episode.
    env_name : str
        The task class name recorded in the dataset metadata, e.g.
        `"PickPlaceCounterToCabinet"`.

    Procedure
    ---------
    1. Read the dataset's recorded env metadata.
    2. Copy its `env_kwargs` so the original metadata is not mutated.
    3. Set the task name on the copy.
    4. Turn off the on-screen renderer, the offscreen renderer, and camera observations.
    5. Construct the robosuite env from those kwargs and return it with the task name.
    """
    env_meta = LU.get_env_metadata(dataset_dir)
    kwargs = dict(env_meta["env_kwargs"])
    kwargs["env_name"] = env_meta["env_name"]
    kwargs["has_renderer"] = False
    kwargs["has_offscreen_renderer"] = False
    kwargs["use_camera_obs"] = False
    return robosuite.make(**kwargs), env_meta["env_name"]


def episode_signals(env, dataset_dir, ep_idx):
    """
    Replay one recorded episode frame by frame and measure the signals the splitter needs.

    Every frame is visited. Binary-searching for the grasp would save very little, because a
    fixed scene rebuild dominates the cost of an episode rather than the per-frame pass, and
    it would buy off-by-one bugs around the very boundary this whole pipeline exists to find.

    Inputs
    ------
    env : robosuite env
        From `make_env`, matching this dataset. Reused across episodes and left reset to this
        episode's final frame on return.
    dataset_dir : pathlib.Path
        The LeRobot dataset directory.
    ep_idx : int
        Zero-based episode index within that dataset.

    Outputs
    -------
    out : dict[str, np.ndarray]
        One entry per `SU.SIGNAL_KEYS`, each `(T,)` float32 where `T` is the episode length:
        - `held` -- strict fingerpad grasp, 0.0 or 1.0
        - `held_loose` -- any gripper contact, 0.0 or 1.0 (diagnostic and fallback tier)
        - `obj_z` -- object height in metres
        - `gdist` -- end effector to object distance in metres
        - `success` -- replayed `_check_success`, recorded as a tripwire and deliberately
          **not** used to pick split points
        - `reward` -- the reward as recorded at collection time, read from the parquet
        - `grip_cmd` -- the commanded gripper value, read from the parquet
    slots : dict[str, str]
        This episode's `{"obj": ..., "target": ...}` noun phrases.

    Raises
    ------
    AssertionError
        If the episode does not have exactly one parquet file, or if its parquet row count
        disagrees with its sim state count -- either means the dataset is malformed.

    Procedure
    ---------
    1. Load the episode's sim states, metadata and scene XML.
    2. Locate its parquet file, require exactly one, and read it.
    3. Require the parquet row count to match the number of sim states.
    4. Do one full scene reset from the XML, metadata and first state -- the expensive part.
    5. Allocate a `(T,)` float32 array for every signal key.
    6. For each frame, restore that sim state and record the strict grasp, the loose contact,
       the object height, the gripper distance and the replayed success flag.
    7. Replace the `reward` array wholesale with the parquet's `next.reward` column.
    8. Replace `grip_cmd` wholesale with column 11 of each recorded action.
    9. Return the signals together with this episode's slots.

    Notes
    -----
    Two ordering traps live in steps 7 and 8. `next.reward` is stored as a scalar per row,
    not a one-element array, hence the `.tolist()` and `.reshape(-1)`. And `action[11]` is the
    gripper command in **parquet** ordering, which is not the ordering the policy emits.
    """
    states = LU.get_episode_states(dataset_dir, ep_idx)
    ep_meta = LU.get_episode_meta(dataset_dir, ep_idx)
    model_xml = LU.get_episode_model_xml(dataset_dir, ep_idx)

    parquet = sorted(dataset_dir.glob(f"data/*/episode_{ep_idx:06d}.parquet"))
    assert len(parquet) == 1, f"expected one parquet for episode {ep_idx}, got {parquet}"
    df = pd.read_parquet(parquet[0])
    assert len(df) == len(states), (
        f"episode {ep_idx}: {len(df)} parquet rows but {len(states)} sim states"
    )

    # the expensive part: reload the scene for this episode
    reset_to(env, {"model": model_xml, "ep_meta": json.dumps(ep_meta), "states": states[0]})

    T = len(states)
    out = {k: np.zeros(T, dtype=np.float32) for k in SU.SIGNAL_KEYS}
    for t in range(T):
        reset_to(env, {"states": states[t]})
        out["held"][t] = SU.is_holding_obj(env)
        # diagnostic only: any gripper geom touching the object, fingerpads or not
        out["held_loose"][t] = env.check_contact(env.robots[0].gripper["right"], env.objects["obj"])
        out["obj_z"][t] = SU.obj_pos(env)[2]
        out["gdist"][t] = SU.gripper_obj_dist(env)
        out["success"][t] = env._check_success()

    # sim-free cross-checks straight from the parquet.
    # next.reward is stored as a scalar per row; reshape keeps us safe either way.
    out["reward"] = np.asarray(df["next.reward"].tolist(), dtype=np.float32).reshape(-1)
    # action column is in parquet order, where index 11 is gripper_close
    out["grip_cmd"] = np.asarray([a[11] for a in df["action"]], dtype=np.float32)

    return out, SU.make_skill_slots(env.__class__.__name__, ep_meta)


def process_dataset(dataset_dir, out_path):
    """
    Replay every episode of one dataset and write its signals cache to disk.

    One dataset is one unit of work for one worker process. A single episode that raises is
    recorded and skipped rather than losing the whole dataset, since this step runs for hours.

    Inputs
    ------
    dataset_dir : str or pathlib.Path
        The LeRobot dataset directory to replay.
    out_path : pathlib.Path
        Where to write the `.npz` cache. A `.json` sidecar is written next to it with the
        same stem. Parent directories are created if missing.

    Outputs
    -------
    summary : str
        A one-line report, e.g. `"PickPlaceCounterToCabinet: 240/240 episodes in 2570s"`.
        Printed by the parent process as each dataset finishes.

    Side effects
    ------------
    Writes two files:
    - `<out_path>` -- compressed npz holding every signal concatenated across episodes, plus
      `ep_len` `(n_ok,)` int64 giving each episode's length and `ep_index` `(n_ok,)` int64
      giving its original index. The signals are one flat run, so a consumer must use
      `ep_len` to cut them back apart.
    - `<out_path>.json` -- the task name, dataset path, episode count, per-episode slots, a
      map of failed episodes to their error text, and the elapsed seconds.

    Procedure
    ---------
    1. Build the env for this dataset and count its episodes.
    2. For each episode, extract its signals and slots.
    3. On an exception, record the episode index and error, print the traceback, and continue.
    4. Otherwise append each signal, the episode length, and the slots.
    5. Close the env once every episode has been attempted.
    6. Concatenate each signal across episodes into one flat array, or use an empty dict if
       nothing succeeded.
    7. Write the compressed npz with the lengths and original indices alongside the signals.
    8. Write the JSON sidecar.
    9. Return the summary line.
    """
    dataset_dir = Path(dataset_dir)
    env, env_name = make_env(dataset_dir)
    n_eps = len(LU.get_episodes(dataset_dir))

    per_signal = {k: [] for k in SU.SIGNAL_KEYS}
    ep_len, slots, failed = [], [], {}

    t0 = time.time()
    for ep_idx in range(n_eps):
        try:
            sig, sl = episode_signals(env, dataset_dir, ep_idx)
        except Exception as exc:
            failed[str(ep_idx)] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            continue
        for k in SU.SIGNAL_KEYS:
            per_signal[k].append(sig[k])
        ep_len.append(len(sig["held"]))
        slots.append({"episode": ep_idx, **sl})

    env.close()

    arrays = {k: np.concatenate(per_signal[k]) for k in SU.SIGNAL_KEYS} if ep_len else {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ep_len=np.asarray(ep_len, dtype=np.int64),
        ep_index=np.asarray([s["episode"] for s in slots], dtype=np.int64),
        **arrays,
    )
    out_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "env_name": env_name,
                "dataset": str(dataset_dir),
                "n_episodes": n_eps,
                "slots": slots,
                "failed": failed,
                "seconds": round(time.time() - t0, 1),
            },
            indent=2,
        )
    )
    return f"{env_name}: {len(ep_len)}/{n_eps} episodes in {time.time() - t0:.0f}s"


def worker(job_queue, result_queue):
    """
    Worker-process loop: take datasets off the queue and replay them until none are left.

    Runs in a spawned child process, so it must be importable at module level and must not
    rely on anything inherited from the parent's memory.

    Inputs
    ------
    job_queue : multiprocessing.Queue
        Holds `(dataset_dir, out_path)` string pairs. An empty queue is how the worker learns
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
    2. Replay that dataset and put its summary on the result queue.
    3. If replaying raises, print the traceback and put a `"FAILED ..."` line on the result
       queue instead, so the parent's count still balances.
    4. Loop.
    """
    while True:
        try:
            dataset_dir, out_path = job_queue.get_nowait()
        except Exception:
            return
        try:
            result_queue.put(process_dataset(dataset_dir, Path(out_path)))
        except Exception as exc:
            traceback.print_exc()
            result_queue.put(f"FAILED {dataset_dir}: {type(exc).__name__}: {exc}")


def main():
    """
    Command-line entry point: replay the requested datasets in parallel and cache their signals.

    Inputs
    ------
    Read from the command line, not from arguments:
    --tasks : list[str]
        Task names to process. Defaults to all 18 PickPlace tasks.
    --split : list[str]
        Splits to process. Defaults to both `pretrain` and `target`.
    --num_procs : int
        Worker processes. Defaults to 6. Capped at the number of jobs.
    --cache_dir : str or None
        Where to write caches. Defaults to a `skill_cache` folder beside each dataset.

    Outputs
    -------
    None
        Prints the job count first, then one summary line per dataset as it finishes.
        The real output is the signal caches on disk.

    Procedure
    ---------
    1. Parse the arguments.
    2. For every split and task, look up the dataset metadata and skip pairs with no entry.
    3. Skip any dataset whose directory is not present, printing a note.
    4. Build the output cache path and add a job for it.
    5. Print how many datasets will run and with how many workers.
    6. Create spawn-context queues and load every job onto the job queue.
    7. Start up to `--num_procs` workers, never more than there are jobs.
    8. Read exactly one result per job and print it as it arrives.
    9. Join every worker before returning.

    Notes
    -----
    zsh does not word-split unquoted expansions, so `--tasks $TASK_LIST` arrives as a single
    argument and the registry lookup fails immediately. Use a zsh array or `${=VAR}`. The
    first printed line is the check: it must name the number of datasets you expected, not 1.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=SU.PICK_PLACE_TASKS)
    p.add_argument("--split", nargs="+", default=["pretrain", "target"])
    p.add_argument("--num_procs", type=int, default=6)
    p.add_argument(
        "--cache_dir",
        default=None,
        help="where to write signal caches (default: <dataset>/../skill_cache)",
    )
    args = p.parse_args()

    jobs = []
    for split in args.split:
        for task in args.tasks:
            meta = get_ds_meta(task, split, "human")
            if meta is None:
                continue
            ds = Path(meta["path"])
            if not ds.exists():
                print(f"skip {task}/{split}: not downloaded")
                continue
            cache = Path(args.cache_dir) if args.cache_dir else ds.parent / "skill_cache"
            jobs.append((str(ds), str(cache / f"{task}_{split}_signals.npz")))

    print(f"{len(jobs)} datasets to process with {args.num_procs} workers")

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
