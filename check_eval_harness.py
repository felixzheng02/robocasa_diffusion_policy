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
from pathlib import Path

import numpy as np

import robocasa  # noqa: F401  (registers the gym envs)
import robocasa.utils.lerobot_utils as LU
import robocasa.utils.skill_utils as SU
from robocasa.utils.dataset_registry_utils import get_ds_meta

from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from robocasa.utils.env_helpers import create_env
from robocasa.utils.env_helpers import base_env


def load_shape_meta(skill="pick"):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        f"diffusion_policy/config/task/robocasa/pretrain_{skill}_skill.yaml"
    )
    return OmegaConf.to_container(cfg.shape_meta, resolve=False)


def replay_episode(task, split, shape_meta, ep_idx, actions, seed, demo=None):
    """
    Open-loop replay of one episode's recorded actions. Returns outcome dict.

    `demo`, when given, is (states, ep_meta, model_xml) for the source episode. Restoring it
    needs the full model reload, not just set_state_from_flattened: a freshly sampled scene
    has a different layout and object set, so the state vectors are not even the same length
    (79 vs 107 qvel observed).
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

    Path("harness_check.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
