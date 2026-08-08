"""
Chained pick -> place evaluation.

Runs the pick policy until it has the object in hand and moving, hands off to the place
policy, and scores the rollout with the original task's own _check_success. The handoff uses
robocasa.utils.skill_utils.GraspMoveDetector, which is the same criterion that ended the
pick segments in training, so the state at the handoff is in-distribution for both policies.

Three numbers are reported per task:
    pick_success       the detector fired at all
    place_given_pick   the task succeeded, among rollouts where it fired
    task_success       end-to-end

Example:
    MUJOCO_GL=egl python eval_chained_pick_place.py \\
        --pick_checkpoint outputs/pick_skill/checkpoints/latest.ckpt \\
        --place_checkpoint outputs/place_skill/checkpoints/latest.ckpt \\
        --split pretrain --num_rollouts 50
"""

import argparse
import collections
import copy
import json
import os
import pathlib

import imageio
import numpy as np
import torch
from termcolor import colored

import robocasa  # noqa: F401  (registers the gym envs)
import robocasa.utils.skill_utils as SU
from robomimic.utils.lang_utils import LangEncoder

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.lerobot_dataset import SLOT_KEYS
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from robocasa.utils.env_helpers import base_env, create_env
from diffusion_policy.skills.eval_helpers import (
    load_policy, obs_to_frame, stack_obs)
from robocasa.utils.dataset_registry_utils import get_task_horizon


def run_episode(env_name, split, seed, shape_meta, policies, encoder,
                horizon, n_obs_steps, device, video_path=None):
    """
    One rollout. Returns a dict of outcome flags.

    With a place policy this is the chained pick -> place rollout. Without one it evaluates
    the pick skill alone: run until the grasp criterion fires, then stop. The criterion is
    GraspMoveDetector either way, i.e. exactly what ended the pick segments in training.
    """
    pick_only = "place" not in policies

    # The place obs keys are a superset of pick's, so use whichever schema is in play.
    wrapper_meta = shape_meta["pick" if pick_only else "place"]
    wrapper = RobomimicImageWrapper(
        env=create_env(split=split, env_name=env_name, seed=seed),
        shape_meta=wrapper_meta,
        init_state=None,
        render_obs_key="robot0_agentview_right_image",
    )
    # The slots are only knowable once the scene exists, but get_observation needs them to
    # be present, so reset with placeholders and re-derive the first observation after.
    dim = wrapper_meta["obs"]["obj_emb"]["shape"][0]
    wrapper.slot_embs = {
        k: np.zeros(dim, dtype=np.float32) for k in SLOT_KEYS if k in wrapper_meta["obs"]
    }
    wrapper.reset()
    sim = base_env(wrapper)

    # slots come from the env itself, not the recorded combined instruction
    slot_text = SU.make_skill_slots(env_name, sim.get_ep_meta())
    obj_emb = encoder.get_lang_emb(slot_text["obj"]).numpy()
    embs = {
        "pick": {"obj_emb": obj_emb},
        "place": {
            "obj_emb": obj_emb,
            "recep_emb": encoder.get_lang_emb(slot_text["target"]).numpy(),
        },
    }

    phase = "pick"
    detector = SU.GraspMoveDetector()
    # Pick-only rollouts get the same budget the pick phase would have had in the chain, so
    # the two numbers stay comparable.
    pick_budget = int(0.5 * horizon)
    budget = pick_budget if pick_only else horizon
    history = collections.deque(maxlen=n_obs_steps)
    success = False
    handoff_step = None

    for policy in policies.values():
        policy.reset()

    # The wrapper always emits every slot its own schema declares -- that schema is the place
    # one, so dropping recep_emb during the pick phase would leave get_observation with a key
    # it must fill. Phase selection happens on the policy input instead, below.
    wrapper.slot_embs = {k: v for k, v in embs["place"].items() if k in wrapper_meta["obs"]}
    first_obs = dict(wrapper.get_observation(wrapper.last_raw_obs))
    history.append(first_obs)

    frames = []
    if video_path is not None:
        f = obs_to_frame(first_obs)
        if f is not None:
            frames.append(f)

    z0 = SU.obj_pos(sim)[2]
    step = 0
    while step < budget:
        # Feed each policy only the keys it was trained on: the pick policy's normalizer has
        # no entry for recep_emb and raises on one, so the extra slot cannot just ride along.
        window = stack_obs(history, n_obs_steps)
        obs_dict = dict_apply(
            {k: v for k, v in window.items() if k in shape_meta[phase]["obs"]},
            lambda x: torch.from_numpy(x.astype(np.float32)).to(device),
        )
        with torch.no_grad():
            action = policies[phase].predict_action(obs_dict)["action"][0].cpu().numpy()

        for a in action:
            if step >= budget:
                break
            raw, _, _, _ = wrapper.step(a)
            raw = dict(raw)
            history.append(raw)
            step += 1
            if video_path is not None:
                f = obs_to_frame(raw)
                if f is not None:
                    frames.append(f)

            if sim._check_success():
                success = True
                break
            if phase == "pick" and detector.update(sim):
                handoff_step = step
                if pick_only:
                    break
                phase = "place"
                # cut the chunk: a stale pick action executed after the grasp drags the object
                break
        if success or (pick_only and detector.fired):
            break
        if not pick_only and phase == "pick" and step >= pick_budget:
            phase = "place"
            handoff_step = handoff_step or step

    # Whether the object is *still* held at the end separates a real pick from a policy that
    # grabbed, nudged 2 cm, and dropped it — the detector alone cannot tell those apart.
    still_holding = bool(SU.is_holding_obj(sim))
    dz = float(SU.obj_pos(sim)[2] - z0)

    if video_path is not None and frames:
        # outcome in the filename so the failures are the ones you open; in chained mode the
        # outcome that matters is the task, not whether the grasp fired
        tag = "OK" if (detector.fired if pick_only else success) else "FAIL"
        out = pathlib.Path(str(video_path).replace("$TAG", tag))
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out, frames, fps=20, quality=6, macro_block_size=1)

    wrapper.env.close()
    return {
        "pick_success": bool(detector.fired),
        "task_success": bool(success),
        "handoff_step": handoff_step,
        "steps": step,
        "still_holding": still_holding,
        "final_dz": round(dz, 4),
        "obj": slot_text["obj"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pick_checkpoint", required=True)
    p.add_argument(
        "--place_checkpoint",
        default=None,
        help="omit to evaluate the pick skill alone",
    )
    p.add_argument("--tasks", nargs="+", default=SU.PICK_PLACE_TASKS)
    p.add_argument("--split", default="pretrain")
    p.add_argument("--num_rollouts", type=int, default=50)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default=None)
    p.add_argument(
        "--video_dir",
        default=None,
        help="record rollouts here for diagnosis (filenames carry OK/FAIL)",
    )
    p.add_argument(
        "--video_n",
        type=int,
        default=3,
        help="rollouts to record per task; recording every one is slow and rarely useful",
    )
    args = p.parse_args()

    pick_policy, pick_meta = load_policy(args.pick_checkpoint, args.device)
    policies = {"pick": pick_policy}
    shape_meta = {"pick": pick_meta}

    pick_only = args.place_checkpoint is None
    if not pick_only:
        place_policy, place_meta = load_policy(args.place_checkpoint, args.device)
        policies["place"] = place_policy
        shape_meta["place"] = place_meta

    n_obs_steps = pick_policy.n_obs_steps
    out_path = args.output or ("pick_eval.json" if pick_only else "chained_eval.json")
    print(colored(f"mode: {'PICK ONLY' if pick_only else 'CHAINED pick->place'}", "cyan"))

    encoder = LangEncoder(device="cpu")

    results, per_rollout = {}, {}
    for task in args.tasks:
        horizon = int(get_task_horizon(task=task))
        rollouts = []
        for i in range(args.num_rollouts):
            video_path = None
            if args.video_dir and i < args.video_n:
                video_path = str(
                    pathlib.Path(args.video_dir) / f"{task}_ep{i:02d}_$TAG.mp4"
                )
            try:
                rollouts.append(
                    run_episode(task, args.split, args.seed + i, shape_meta, policies,
                                encoder, horizon, n_obs_steps, args.device,
                                video_path=video_path)
                )
            except Exception as exc:  # a broken scene should not kill the sweep
                print(colored(f"{task} rollout {i} failed: {exc}", "red"))
        if not rollouts:
            continue
        per_rollout[task] = rollouts
        picked = [r for r in rollouts if r["pick_success"]]
        mean = lambda k, rs=rollouts: float(np.mean([r[k] for r in rs]))
        results[task] = {
            "n": len(rollouts),
            "pick_success": mean("pick_success"),
            "still_holding": mean("still_holding"),
            "mean_steps_to_grasp": (
                float(np.mean([r["handoff_step"] for r in picked])) if picked else None
            ),
        }
        if not pick_only:
            results[task]["task_success"] = mean("task_success")
            results[task]["place_given_pick"] = (
                float(np.mean([r["task_success"] for r in picked])) if picked else 0.0
            )
        r = results[task]
        line = (f"{task:32s} pick={r['pick_success']:.2f} hold={r['still_holding']:.2f}")
        if not pick_only:
            line += f" place|pick={r['place_given_pick']:.2f} task={r['task_success']:.2f}"
        print(line)

    if results:
        keys = ["pick_success", "still_holding"] + (
            [] if pick_only else ["place_given_pick", "task_success"]
        )
        results["AVERAGE"] = {
            k: float(np.mean([v[k] for t, v in results.items() if t != "AVERAGE"]))
            for k in keys
        }
        print("\nAVERAGE " + json.dumps(results["AVERAGE"]))

        # per-object breakdown: the documented risk is that handled objects (ladles,
        # measuring cups) fail because fingerpad contact is unavailable for them
        by_obj = {}
        for rs in per_rollout.values():
            for r in rs:
                by_obj.setdefault(r["obj"], []).append(r["pick_success"])
        worst = sorted(by_obj.items(), key=lambda kv: np.mean(kv[1]))[:10]
        print("\nweakest objects (pick_success, n):")
        for obj, vals in worst:
            print(f"  {obj:24s} {np.mean(vals):.2f}  (n={len(vals)})")

    pathlib.Path(out_path).write_text(
        json.dumps({"summary": results, "rollouts": per_rollout}, indent=2)
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
