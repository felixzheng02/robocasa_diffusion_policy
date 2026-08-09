"""
Harness sanity check for the pick/place evaluator.

Replays *recorded demo actions* through the same env + wrapper + convert_action + detector
path that eval_chained_pick_place.py uses, and asserts the grasp detector fires. If ground
truth does not trigger it, the harness is broken and any policy score from the evaluator is
meaningless — so run this before reading anything into an eval number.

It exercises, in one shot:
  * base_env()'s unwrapping of the gym wrapper stack
  * the three-way action ordering (parquet vs policy vs convert_action)
  * GraspMoveDetector working online, frame by frame

Note the env is freshly sampled, *not* reset to the demo's initial state, so the recorded
actions are replayed open-loop into a different scene. That still exercises every code path;
what it cannot do is guarantee the grasp succeeds. Use --from_demo_state to additionally
restore the source episode's scene, which makes the grasp itself reproducible.

Example:
    MUJOCO_GL=egl python check_eval_harness.py --task PickPlaceCounterToCabinet --n 3
"""

import argparse
import json
import pathlib
from pathlib import Path

import numpy as np

import robocasa  # noqa: F401  (registers the gym envs)
import robocasa.utils.lerobot_utils as LU
import diffusion_policy.skills.skill_utils as SU
from robocasa.utils.dataset_registry_utils import get_ds_meta

from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from robocasa.utils.env_helpers import create_env
from robocasa.utils.env_helpers import base_env


def load_shape_meta(skill="pick"):
    """
    Read a skill's observation and action schema straight from its task config.

    Loading the YAML rather than a checkpoint means this check needs no trained weights, so
    the harness can be verified before or independently of any training run.

    Inputs
    ------
    skill : str
        Either `"pick"` or `"place"`, selecting `pretrain_<skill>_skill.yaml`.

    Outputs
    -------
    shape_meta : dict
        The config's `shape_meta` as plain Python, with an `obs` sub-dict mapping each key to
        its `shape` and `type`, plus an `action` entry. Interpolations are left unresolved.

    Procedure
    ---------
    1. Build the config path relative to **this file**, not the working directory.
    2. Load the YAML.
    3. Convert its `shape_meta` block to a plain dict and return it.

    Notes
    -----
    The path is `__file__`-anchored on purpose: this script is run from several directories
    and, after the workspace split, no longer necessarily from the repo root.
    """
    from omegaconf import OmegaConf

    # __file__-anchored, not cwd-relative: this script is run from several directories
    # and, after the workspace split, no longer necessarily from the repo root.
    cfg = OmegaConf.load(
        pathlib.Path(__file__).parent
        / "diffusion_policy/config/task/robocasa"
        / f"pretrain_{skill}_skill.yaml"
    )
    return OmegaConf.to_container(cfg.shape_meta, resolve=False)


def replay_episode(task, split, shape_meta, ep_idx, actions, seed, demo=None):
    """
    Replay one episode's recorded actions open-loop and see whether the detector fires.

    This is ground truth going through the real evaluation path. If a human demo's own
    actions do not trigger the grasp criterion, the harness is broken and no policy score
    from the evaluator means anything.

    Inputs
    ------
    task : str
        Task class name, e.g. `"PickPlaceCounterToCabinet"`.
    split : str
        Which split to sample the scene from.
    shape_meta : dict
        The pick skill's schema, from `load_shape_meta`.
    ep_idx : int
        Episode index, recorded in the output so results can be traced back.
    actions : np.ndarray, `(T, 12)`
        The episode's recorded actions, in the ordering `env.step` expects.
    seed : int
        Scene seed for the freshly created env.
    demo : tuple or None
        When given, `(states, ep_meta, model_xml)` for the source episode, used to restore
        the original scene. `None` replays into a freshly sampled scene instead.

    Outputs
    -------
    out : dict
        - `episode` (int), `obj` (str) -- which episode and object this was
        - `n_actions` (int) -- how many actions the demo had
        - `steps_run` (int) -- how many were executed before the detector fired or they ran out
        - `detector_fired` (bool), `fired_at` (int or None) -- the outcome and its step
        - `held_frames` (int) -- frames with a strict grasp, a softer signal than the detector
        - `final_dz` (float) -- net object height change in metres, 4 dp

    Procedure
    ---------
    1. Create the env and wrap it exactly as the evaluator does.
    2. Install a zero-filled placeholder slot embedding and reset, then unwrap to the sim.
    3. If a demo was supplied, restore its scene with a full model reload.
    4. Read the episode's slots and start a fresh detector, latching the object's height.
    5. Step every recorded action in turn, counting frames with a strict grasp.
    6. Feed each step to the detector and stop as soon as it fires.
    7. Measure the net height change, close the env, and return the outcome.

    Notes
    -----
    Restoring a demo needs the **full model reload**, not just `set_state_from_flattened`. A
    freshly sampled scene has a different layout and object set, so the state vectors are not
    even the same length -- 79 versus 107 qvel observed -- and a bare state restore raises.

    Without `demo`, the recorded actions are replayed open-loop into a *different* scene.
    Every code path still runs, which is the point of the check, but the grasp succeeding is
    then luck rather than evidence.
    """
    wrapper = RobomimicImageWrapper(
        env=create_env(split=split, env_name=task, seed=seed),
        shape_meta=shape_meta,
        init_state=None,
        render_obs_key="robot0_agentview_right_image",
    )
    dim = shape_meta["obs"]["obj_emb"]["shape"][0]
    wrapper.slot_embs = {"obj_emb": np.zeros(dim, dtype=np.float32)}
    wrapper.reset()
    sim = base_env(wrapper)

    if demo is not None:
        states, ep_meta, model_xml = demo
        reset_to(sim, {"states": states[0], "model": model_xml,
                       "ep_meta": json.dumps(ep_meta)})

    slots = SU.make_skill_slots(task, sim.get_ep_meta())
    detector = SU.GraspMoveDetector()
    z0 = SU.obj_pos(sim)[2]
    held_frames = 0

    for t, a in enumerate(actions):
        wrapper.step(a)
        if SU.is_holding_obj(sim):
            held_frames += 1
        detector.update(sim)
        if detector.fired:
            break

    dz = SU.obj_pos(sim)[2] - z0
    out = {
        "episode": ep_idx,
        "obj": slots["obj"],
        "n_actions": len(actions),
        "steps_run": t + 1,
        "detector_fired": bool(detector.fired),
        "fired_at": detector.t,
        "held_frames": held_frames,
        "final_dz": round(float(dz), 4),
    }
    wrapper.env.close()
    return out


def main():
    """
    Command-line entry point: replay demos through the eval path and judge the harness.

    Run this before reading anything into an eval number. A low policy score could otherwise
    equally mean a broken harness, and this is what separates the two.

    Inputs
    ------
    Read from the command line, not from arguments:
    --task : str
        Task to replay. Defaults to `PickPlaceCounterToCabinet`.
    --split : str
        Split to draw scenes from. Defaults to `pretrain`.
    --n : int
        Episodes to replay. Defaults to 3.
    --seed : int
        Base scene seed; episode `i` uses `seed + i`. Defaults to 100000.
    --from_demo_state : flag
        Restore each demo's original scene before replaying. Recommended -- without it the
        result is not meaningful (see notes).
    --output : str or None
        Where to write the result JSON. Defaults to `harness_check.json` beside this script.

    Outputs
    -------
    None
        Prints a line per episode, the fired count, and one of three verdicts. Writes the
        per-episode records as JSON.

    Procedure
    ---------
    1. Parse the arguments.
    2. Resolve the **pick skill** dataset, which holds the trimmed segments and their actions.
    3. Load the pick schema from the task config.
    4. Print what is about to run, including whether demo states will be restored.
    5. For each episode, read its recorded actions and, if asked, its states, metadata and
       scene XML.
    6. Replay it and print the per-episode outcome line.
    7. Count how many replays fired the detector.
    8. Print `HARNESS OK` if all fired, `HARNESS SUSPECT` with an ordered checklist if none
       did, or `PARTIAL` if some did.
    9. Write the results to the output path.

    Notes
    -----
    The three verdicts are deliberately distinct. `PARTIAL` means the plumbing works and the
    misses are open-loop drift; `HARNESS SUSPECT` means ground truth never fires, and the
    checklist is ordered by how likely each cause is -- action ordering first, then
    `base_env()`, then the scene.

    `--from_demo_state` is effectively required for a meaningful result. Without it the
    actions are replayed into a different scene, so every code path runs but a successful
    grasp is luck.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PickPlaceCounterToCabinet")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--n", type=int, default=3, help="episodes to replay")
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument(
        "--from_demo_state",
        action="store_true",
        help="restore each demo's initial sim state before replaying (recommended)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="where to write the result JSON; defaults beside this script so a run from "
             "another directory does not scatter results",
    )
    args = p.parse_args()

    # the *pick* skill dataset holds the trimmed segments and their recorded actions
    meta = get_ds_meta(f"{args.task}_pick", args.split, "human")
    src = Path(meta["path"])
    shape_meta = load_shape_meta("pick")

    print(f"replaying {args.n} episodes of {args.task}_pick ({args.split})")
    print(f"  from {src}")
    print(f"  restore demo state: {args.from_demo_state}\n")

    results = []
    for ep in range(args.n):
        actions = LU.get_episode_actions(src, ep)
        demo = None
        if args.from_demo_state:
            demo = (
                LU.get_episode_states(src, ep),
                LU.get_episode_meta(src, ep),
                LU.get_episode_model_xml(src, ep),
            )
        r = replay_episode(
            args.task, args.split, shape_meta, ep, actions, args.seed + ep, demo=demo,
        )
        results.append(r)
        print(f"  ep{ep}: obj={r['obj']:16s} steps={r['steps_run']:4d}/{r['n_actions']:4d} "
              f"held={r['held_frames']:4d} dz={r['final_dz']:+.3f} "
              f"fired={'YES @' + str(r['fired_at']) if r['detector_fired'] else 'no'}")

    n_fired = sum(r["detector_fired"] for r in results)
    print(f"\ndetector fired on {n_fired}/{len(results)} replayed demos")
    if n_fired == len(results):
        print("HARNESS OK - ground-truth actions trigger the grasp criterion")
    elif n_fired == 0:
        print("HARNESS SUSPECT - ground truth never fires. Check, in order:")
        print("  1. action ordering (convert_action vs the parquet column order)")
        print("  2. base_env() returning the right object")
        print("  3. whether the replayed scene matches the demo (use --from_demo_state)")
    else:
        print("PARTIAL - plumbing works; misses are likely open-loop drift, not a bug")

    out = pathlib.Path(args.output) if args.output else (
        pathlib.Path(__file__).parent / "harness_check.json")
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
