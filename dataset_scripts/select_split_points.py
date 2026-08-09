"""
Step 2 of the pick/place split: turn the signal caches from extract_grasp_signals.py into
concrete pick / place frame ranges.

Pure numpy, no simulator, so it runs in seconds and can be re-run freely while tuning the
thresholds in diffusion_policy.skills.skill_utils.

Example:
    python -m robocasa.scripts.dataset_scripts.select_split_points --split pretrain target
"""

import argparse
import json
from pathlib import Path

import numpy as np

import diffusion_policy.skills.skill_utils as SU
from robocasa.utils.dataset_registry_utils import get_ds_meta


def split_episode(sig, T):
    """
    Decide where the pick ends and the place begins for one recorded episode.

    This is the offline half of the split criterion; `SU.GraspMoveDetector` is its online
    twin. The rule finds the grasp that actually transported the object, then cuts the
    episode into two overlapping ranges around it.

    Inputs
    ------
    sig : dict[str, np.ndarray]
        This episode's signals, each `(T,)`. Reads `reward`, `obj_z`, `held` and `held_loose`.
    T : int
        The episode length in frames. Used to clamp both ranges to the last valid index.

    Outputs
    -------
    result : dict
        On success, a dict of frame indices:
        - `t_grasp` -- first frame of the transport grasp
        - `t_moved` -- frame at which the object had visibly moved
        - `t_success` -- first frame the task was recorded as solved
        - `pick` -- `[start, end]`, inclusive both ends, always starting at 0
        - `place` -- `[start, end]`, inclusive both ends
        - `n_grasp_runs` -- how many grasp runs the winning tier found, so multi-grasp
          episodes can be counted
        - `overlap` -- frames shared by the two ranges
        On failure, `{"reject": reason}` where reason is one of `"never_succeeds"`,
        `"no_grasp_before_success"`, `"no_move"`, `"pick_too_short"`, `"place_too_short"`.

    Procedure
    ---------
    1. Take success from the recorded reward and reject the episode if it never succeeds.
    2. Take `t_success` as the first successful frame.
    3. For the strict contact tier and then the loose one: close short gaps, find runs of at
       least `MIN_GRASP_RUN` frames, and keep those starting before success.
    4. Walking those candidates from last to first, accept the first one after which the
       object's height changes by `MOVE_DZ` within `MOVE_WINDOW` frames.
    5. Stop as soon as a tier produces a grasp; only fall through to the loose tier when the
       strict one produced nothing.
    6. Reject with `"no_move"` if grasps were seen but none moved the object, or
       `"no_grasp_before_success"` if none were seen at all.
    7. Set pick to run from frame 0 to `t_moved + PICK_TAIL_PAD`, clamped to the episode end.
    8. Set place to run from `t_grasp` to `t_success + PLACE_TAIL_PAD`, clamped likewise.
    9. Reject either range shorter than `MIN_SEG_LEN` frames.
    10. Return the indices, the ranges, the run count and the overlap.

    Notes
    -----
    Several choices here are load-bearing and were each driven by a measured failure:

    - Success comes from the recorded reward, **not** from re-running `_check_success` on the
      replayed state. They agree on most tasks but not all: `PickPlaceCounterToBlender` tests
      `obj_inside_of(th=0.01)`, a tolerance that does not survive state restoration, and 85 of
      its 106 demos have a recorded success the replayed predicate misses.
    - The transport grasp is the **last** run starting before success, not the first. A human
      who re-grasps to nudge the object after placing it would otherwise hand us the
      adjustment. This is well defined because every `PickPlace._check_success` conjoins
      `gripper_obj_far`, so no run can straddle `t_success`.
    - The movement test is unsigned, because half these tasks carry the object downwards.
    - Ending pick at `t_moved` rather than a fixed offset from `t_grasp` keeps the pick
      policy's terminal states matched to the runtime handoff even when a lift is slow.
    - The two ranges deliberately **overlap** on the lift phase, which is exactly where the
      runtime handoff fires, so the handoff state is in-distribution for both policies.
    """
    # next.reward as recorded at collection time, not env._check_success() re-evaluated on
    # replay. They agree on most tasks, but not all: PickPlaceCounterToBlender checks
    # obj_inside_of(..., th=0.01), and that tolerance does not survive state restoration —
    # 85 of its 106 demos have a recorded success that the replayed predicate misses.
    success = sig["reward"] > 0
    if not success.any():
        return {"reject": "never_succeeds"}
    t_success = int(np.argmax(success))

    # Two tiers of contact. Fingerpad contact is the precise signal, but handled objects
    # (ladles, measuring cups) are grasped by the handle and never register on both pads, so
    # fall back to any gripper contact when the strict tier yields nothing usable.
    def moved_after(t_grasp):
        """
        Find when the object first moves far enough after a candidate grasp.

        Inputs
        ------
        t_grasp : int
            Frame the candidate grasp starts on. The object's height there becomes the
            reference the movement is measured against.

        Outputs
        -------
        t_moved : int or None
            Absolute frame index of the first sample within `MOVE_WINDOW` frames whose height
            differs from the reference by at least `MOVE_DZ`. `None` if the object never
            moves that far in the window, which disqualifies this candidate grasp.

        Procedure
        ---------
        1. Read the object's height at the grasp frame as the reference.
        2. Slice the next `MOVE_WINDOW` frames of height, truncating at the episode end.
        3. Find every offset in that window whose unsigned deviation reaches `MOVE_DZ`.
        4. Return the first such offset as an absolute frame index, or None if there is none.
        """
        z0 = sig["obj_z"][t_grasp]
        window = sig["obj_z"][t_grasp : t_grasp + SU.MOVE_WINDOW]
        hits = np.flatnonzero(np.abs(window - z0) >= SU.MOVE_DZ)
        return t_grasp + int(hits[0]) if len(hits) else None

    t_grasp = t_moved = None
    n_runs = 0
    saw_grasp = False
    for key in ("held", "held_loose"):
        held = SU.close_gaps(sig[key] > 0, SU.GAP_CLOSE)
        runs = SU.true_runs(held, SU.MIN_GRASP_RUN)
        # The transport grasp is the last run that *starts* before success. A human who
        # re-grasps to nudge the object after placing it would otherwise hand us the
        # adjustment. Well defined because every PickPlace._check_success conjoins
        # gripper_obj_far, so no run can straddle t_success.
        candidates = [r for r in runs if r[0] < t_success]
        saw_grasp |= bool(candidates)
        # Require the object to actually move once grasped: this is what rejects closing on
        # a distractor, a grasp that never took, and (for the loose tier) a mere brush.
        # Unsigned, because half these tasks carry the object downwards.
        for run in reversed(candidates):
            hit = moved_after(int(run[0]))
            if hit is not None:
                t_grasp, t_moved, n_runs = int(run[0]), hit, len(runs)
                break
        if t_grasp is not None:
            break

    if t_grasp is None:
        return {"reject": "no_move" if saw_grasp else "no_grasp_before_success"}

    # Ending pick at t_moved rather than a fixed offset from t_grasp keeps the pick policy's
    # terminal states matched to the runtime handoff predicate even when the lift is slow.
    # Truncating place at t_success drops the idle tail every human demo has, which would
    # otherwise teach the place policy to stall.
    pick = (0, min(t_moved + SU.PICK_TAIL_PAD, T - 1))
    place = (t_grasp, min(t_success + SU.PLACE_TAIL_PAD, T - 1))

    for name, (a, b) in (("pick", pick), ("place", place)):
        if b - a + 1 < SU.MIN_SEG_LEN:
            return {"reject": f"{name}_too_short"}

    return {
        "t_grasp": t_grasp,
        "t_moved": t_moved,
        "t_success": t_success,
        "pick": [pick[0], pick[1]],
        "place": [place[0], place[1]],
        "n_grasp_runs": n_runs,
        "overlap": pick[1] - place[0] + 1,
    }


def process_cache(npz_path):
    """
    Split every episode in one signal cache and summarise how well the criterion did.

    The cache stores all episodes' signals concatenated into flat arrays, so the first job is
    cutting them back apart using the recorded per-episode lengths.

    Inputs
    ------
    npz_path : str or pathlib.Path
        Path to a `*_signals.npz` written by step 1. Its `.json` sidecar must sit beside it
        with the same stem, since the slots come from there.

    Outputs
    -------
    result : dict
        Ready to serialise as the `*_splits.json` file:
        - `env_name`, `dataset` -- carried over from the sidecar
        - `params` -- the seven threshold values in force, recorded so a results file can be
          matched to the criterion that produced it
        - `summary` -- episode counts, reject rate, and (only when at least one episode
          succeeded) the mean pick/place/overlap fractions of episode length, the mean
          overlap in frames, and how many episodes had more than one grasp run
        - `episodes` -- keyed by episode index as a string; each value is `split_episode`'s
          output plus `obj`, `target`, `pick_task` and `place_task`
        - `rejected` -- episode index string to reject reason, kept so a disagreement with
          the criterion is recoverable without re-running step 1

    Procedure
    ---------
    1. Load the npz and its JSON sidecar, and index the sidecar's slots by episode.
    2. Turn the per-episode lengths into cumulative start offsets.
    3. For each episode, slice its own signals out of the flat arrays.
    4. Run `split_episode` on that slice.
    5. Record the reason and move on if it was rejected.
    6. Otherwise attach the episode's object and receptacle, and the encoded pick and place
       task strings.
    7. Count successes and rejections and compute the reject rate.
    8. When anything succeeded, compute the mean pick, place and overlap lengths as
       fractions of each episode's own length, plus the multi-grasp count.
    9. Return the metadata, thresholds, summary, per-episode splits and rejections.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    slots_by_ep = {s["episode"]: s for s in meta["slots"]}

    ep_len = data["ep_len"]
    ep_index = data["ep_index"]
    starts = np.concatenate(([0], np.cumsum(ep_len)))

    episodes, rejected = {}, {}
    for i, ep in enumerate(ep_index):
        a, b = int(starts[i]), int(starts[i + 1])
        sig = {k: data[k][a:b] for k in SU.SIGNAL_KEYS if k in data}
        out = split_episode(sig, int(ep_len[i]))
        if "reject" in out:
            rejected[str(int(ep))] = out["reject"]
            continue
        slots = slots_by_ep[int(ep)]
        out["obj"] = slots["obj"]
        out["target"] = slots["target"]
        out["pick_task"] = SU.slots_to_task_string("pick", slots)
        out["place_task"] = SU.slots_to_task_string("place", slots)
        episodes[str(int(ep))] = out

    n_ok = len(episodes)
    summary = {
        "n_episodes": int(len(ep_index)),
        "n_ok": n_ok,
        "n_rejected": len(rejected),
        "reject_rate": round(len(rejected) / max(len(ep_index), 1), 4),
    }
    if n_ok:
        lens = np.array([[e["pick"][1] - e["pick"][0] + 1,
                          e["place"][1] - e["place"][0] + 1,
                          e["overlap"]] for e in episodes.values()], dtype=float)
        totals = np.array([int(ep_len[list(ep_index).index(int(k))]) for k in episodes])
        summary.update(
            mean_pick_frac=round(float((lens[:, 0] / totals).mean()), 3),
            mean_place_frac=round(float((lens[:, 1] / totals).mean()), 3),
            mean_overlap_frac=round(float((lens[:, 2] / totals).mean()), 3),
            mean_overlap_frames=round(float(lens[:, 2].mean()), 1),
            n_multi_grasp=int(sum(e["n_grasp_runs"] > 1 for e in episodes.values())),
        )
    return {
        "env_name": meta["env_name"],
        "dataset": meta["dataset"],
        "params": {
            k: getattr(SU, k)
            for k in ("MOVE_DZ", "MOVE_WINDOW", "MIN_GRASP_RUN", "GAP_CLOSE",
                      "PICK_TAIL_PAD", "PLACE_TAIL_PAD", "MIN_SEG_LEN")
        },
        "summary": summary,
        "episodes": episodes,
        "rejected": rejected,
    }


def main():
    """
    Command-line entry point: choose split points for every cached dataset and report quality.

    Fast and idempotent, because it touches no simulator. Re-run it freely after changing a
    threshold in `skill_utils`; the printed table doubles as the split-quality report.

    Inputs
    ------
    Read from the command line, not from arguments:
    --tasks : list[str]
        Task names to process. Defaults to all 18 PickPlace tasks.
    --split : list[str]
        Splits to process. Defaults to both `pretrain` and `target`.
    --cache_dir : str or None
        Where the caches live. Defaults to a `skill_cache` folder beside each dataset.

    Outputs
    -------
    None
        Writes one `*_splits.json` per dataset and prints a per-dataset table, the overall
        usable and rejected counts, and a tally of reject reasons ordered by frequency.

    Procedure
    ---------
    1. Parse the arguments.
    2. For every split and task, resolve the dataset and its cache directory.
    3. Skip silently where there is no registry entry or no signals cache -- an unprocessed
       dataset is not an error here.
    4. Split every episode in that cache and write the result as `*_splits.json`.
    5. Collect each dataset's summary row.
    6. Print the header and one row per dataset.
    7. Print the totals and the overall reject rate.
    8. Re-read each written file, tally the reject reasons, and print them most common first.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=SU.PICK_PLACE_TASKS)
    p.add_argument("--split", nargs="+", default=["pretrain", "target"])
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    rows = []
    for split in args.split:
        for task in args.tasks:
            meta = get_ds_meta(task, split, "human")
            if meta is None:
                continue
            ds = Path(meta["path"])
            cache = Path(args.cache_dir) if args.cache_dir else ds.parent / "skill_cache"
            npz = cache / f"{task}_{split}_signals.npz"
            if not npz.exists():
                continue

            result = process_cache(npz)
            out = cache / f"{task}_{split}_splits.json"
            out.write_text(json.dumps(result, indent=2))
            rows.append((task, split, result["summary"]))

    print(f"{'task':32s} {'split':9s} {'ok':>5s} {'rej':>4s} {'rate':>6s} "
          f"{'pick':>6s} {'place':>6s} {'ovlp':>6s} {'multi':>6s}")
    for task, split, s in rows:
        print(f"{task:32s} {split:9s} {s['n_ok']:5d} {s['n_rejected']:4d} "
              f"{s['reject_rate']:6.1%} {s.get('mean_pick_frac', 0):6.2f} "
              f"{s.get('mean_place_frac', 0):6.2f} {s.get('mean_overlap_frac', 0):6.2f} "
              f"{s.get('n_multi_grasp', 0):6d}")

    tot_ok = sum(s["n_ok"] for _, _, s in rows)
    tot_rej = sum(s["n_rejected"] for _, _, s in rows)
    print(f"\ntotal: {tot_ok} usable episodes, {tot_rej} rejected "
          f"({tot_rej / max(tot_ok + tot_rej, 1):.1%})")

    reasons = {}
    for task, split, _ in rows:
        cache = Path(get_ds_meta(task, split, "human")["path"]).parent / "skill_cache"
        rej = json.loads((cache / f"{task}_{split}_splits.json").read_text())["rejected"]
        for reason in rej.values():
            reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        print("reject reasons:", dict(sorted(reasons.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
