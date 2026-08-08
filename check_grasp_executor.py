"""
Can the executor reach a pose and pick an object up at all? No detector involved.

This is the analogue of check_eval_harness.py, and it gates everything downstream. A low
`pick_success` from the grasp pipeline could equally mean a bad detector or a servo that
cannot execute *any* pose; this script removes the second explanation by handing the
executor a grasp derived from ground-truth object state.

If the oracle pick rate is poor, the detector's numbers are measuring this servo, not the
grasps, and there is no point running a sweep.

    MUJOCO_GL=egl python check_grasp_executor.py --tasks PickPlaceCounterToCabinet --n 3
    MUJOCO_GL=egl python check_grasp_executor.py --n 3 --output oracle_exec.json
"""

import argparse
import json
import pathlib

import numpy as np
from termcolor import colored

import robocasa  # noqa: F401  (registers the gym envs)
import robocasa.utils.skill_utils as SU
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action

from diffusion_policy.env_runner.robomimic_image_runner import create_env
from eval_chained_pick_place import base_env
from grasp_executor import GraspExecutor
from grasp_geometry import oracle_grasp as _oracle_grasp


def oracle_grasp(env, obj_name="obj"):
    """
    A grasp built from ground-truth geometry (see grasp_geometry.oracle_grasp).

    Returns (pos, R, approach), or None when no approach direction with room for the wrist
    yields a cross-section that fits between the jaws -- a legitimate "this object is not
    graspable from a fixed base" answer rather than an executor failure, recorded as such.
    """
    got = _oracle_grasp(env, obj_name)
    if got is None:
        return None
    (pos, R), approach = got
    return pos, R, approach


def run_episode(task, split, seed, jitter_pos=0.0, jitter_rot=0.0, rng=None):
    env = create_env(split=split, env_name=task, seed=seed)
    env.reset()                      # exactly one reset -- see eval_anygrasp_pick.py
    sim = base_env(env)

    budget = int(0.5 * get_task_horizon(task=task))
    got = oracle_grasp(sim)
    if got is None:
        env.close()
        return {"task": task, "seed": seed, "fired": False, "still_holding": False,
                "final_dz": 0.0, "held_frames": 0, "steps": 0,
                "failure": "no_feasible_grasp", "last_stage": "capture",
                "stage_steps": {}, "stage_err": {}}
    pos, R, approach = got

    if jitter_pos > 0 or jitter_rot > 0:
        rng = rng or np.random.default_rng(seed)
        pos = pos + rng.normal(0, jitter_pos, 3)
        if jitter_rot > 0:
            ax = rng.normal(0, 1, 3)
            ax /= np.linalg.norm(ax)
            ang = rng.normal(0, np.deg2rad(jitter_rot))
            import robosuite.utils.transform_utils as T
            R = T.quat2mat(T.axisangle2quat(ax * ang)) @ R

    ex = GraspExecutor(pos, R, approach)
    detector = SU.GraspMoveDetector()
    z0 = SU.obj_pos(sim)[2]
    held_frames = 0
    step = 0

    while step < budget:
        a = ex.step(sim)
        if a is None:
            break
        env.step(convert_action(a))
        step += 1
        detector.update(sim)
        if SU.is_holding_obj(sim):
            held_frames += 1

    out = {
        "task": task,
        "seed": seed,
        "fired": bool(detector.fired),
        "still_holding": bool(SU.is_holding_obj(sim)),
        "final_dz": round(float(SU.obj_pos(sim)[2] - z0), 4),
        "held_frames": held_frames,
        "steps": step,
        "failure": ex.failure,
        "last_stage": ex.stage,
        "stage_steps": ex.stage_steps,
        "stage_err": ex.stage_err,
    }
    env.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=SU.PICK_PLACE_TASKS)
    p.add_argument("--split", default="pretrain")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--jitter_pos", type=float, default=0.0, help="m, gaussian sigma")
    p.add_argument("--jitter_rot", type=float, default=0.0, help="deg, gaussian sigma")
    p.add_argument("--output", default="oracle_exec.json")
    args = p.parse_args()

    results, rows = {}, []
    for task in args.tasks:
        rs = []
        for i in range(args.n):
            try:
                rs.append(run_episode(task, args.split, args.seed + i,
                                      args.jitter_pos, args.jitter_rot))
            except Exception as exc:
                print(colored(f"{task} rollout {i} failed: {exc}", "red"))
        if not rs:
            continue
        rows.extend(rs)
        fired = float(np.mean([r["fired"] for r in rs]))
        hold = float(np.mean([r["still_holding"] for r in rs]))
        results[task] = {"n": len(rs), "fired": fired, "still_holding": hold}
        colour = "green" if fired >= 0.8 else ("yellow" if fired >= 0.5 else "red")
        print(colored(f"{task:34s} fired={fired:.2f} hold={hold:.2f}", colour))

    if rows:
        overall = float(np.mean([r["fired"] for r in rows]))
        hold = float(np.mean([r["still_holding"] for r in rows]))
        results["AVERAGE"] = {"fired": overall, "still_holding": hold}
        print(f"\nORACLE EXECUTOR  fired={overall:.3f}  still_holding={hold:.3f}  "
              f"(n={len(rows)})")
        # The gate. Below ~0.6 the servo, not the detector, is what a sweep would measure.
        if overall >= 0.8:
            print(colored("EXECUTOR OK - safe to trust detector numbers", "green"))
        elif overall >= 0.6:
            print(colored("EXECUTOR MARGINAL - detector numbers will be depressed", "yellow"))
        else:
            print(colored("EXECUTOR IS THE BOTTLENECK - fix before running any sweep", "red"))
        fails = {}
        for r in rows:
            if not r["fired"]:
                fails[r["failure"] or r["last_stage"]] = fails.get(r["failure"] or r["last_stage"], 0) + 1
        if fails:
            print("failure modes:", json.dumps(fails))

    pathlib.Path(args.output).write_text(
        json.dumps({"summary": results, "rollouts": rows}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
