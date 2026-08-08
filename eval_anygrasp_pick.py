"""
Evaluate a 6-DoF grasp detector on the RoboCasa PickPlace *pick* phase.

Named for the interface, not today's backend: the detector is whatever `serve_grasp.sh` is
serving (graspnet-baseline now, the AnyGrasp SDK once its license lands). The server's
backend id is recorded in the output so a graspnet run and an AnyGrasp run can never be
confused in a file.

Comparability, stated plainly
-----------------------------
Scenes, seeds and budget are identical to the pick-only mode of eval_chained_pick_place.py
(seed = --seed + rollout_index, budget = 0.5 * get_task_horizon), and scoring reuses
SU.GraspMoveDetector / SU.is_holding_obj unchanged, so `pick_success` here is directly
comparable to `pick_eval_e90.json`'s 0.333.

But the two arms are not equally advantaged, and the writeup should say so:

  the grasp arm has an ORACLE advantage -- a segmentation mask tells it which object to
  grasp, mirroring the slot embedding the diffusion policy is handed;
  and a real DISADVANTAGE -- one open-loop grasp attempt, no visual servoing, base fixed.

`pick_success` is the comparable number. `still_holding` is the one that separates a real
grasp from a nudge. `outcome` is what makes a low score interpretable rather than mysterious.

    ./serve_grasp.sh                      # start FIRST
    MUJOCO_GL=egl python eval_anygrasp_pick.py --num_rollouts 10 --output grasp_pick_eval.json
    MUJOCO_GL=egl python eval_anygrasp_pick.py --pose_source oracle   # no detector needed
"""

import argparse
import json
import pathlib
import time

import imageio
import numpy as np
from termcolor import colored

import robocasa  # noqa: F401  (registers the gym envs)
import robocasa.utils.skill_utils as SU
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action

from diffusion_policy.env_runner.robomimic_image_runner import create_env
from eval_chained_pick_place import base_env

import grasp_agent as GA
import grasp_perception as GP
from grasp_executor import GraspExecutor
from grasp_geometry import oracle_grasp

OUTCOMES = ("success", "grasped_then_dropped", "executed_no_contact",
            "unreachable", "no_grasp_proposed")


def plan_grasp(sim, args):
    """Pick a grasp for this scene. Returns (grasp_dict|None, diagnostics)."""
    diag = {"camera": None, "mask_px": {}, "n_points": 0, "n_grasps_raw": 0,
            "n_grasps_kept": 0, "reject": {}, "detect_latency_s": 0.0}

    if args.pose_source == "oracle":
        got = oracle_grasp(sim)
        if got is None:
            return None, diag
        (pos, R), approach = got
        return {"pos": pos, "mat": R, "approach": approach, "score": 1.0,
                "width": 0.0, "depth": 0.0, "obj_dist": 0.0, "index": -1}, diag

    cam, counts = GP.choose_camera(sim, args.cameras)
    diag["camera"], diag["mask_px"] = cam, counts
    if counts.get(cam, 0) == 0:
        return None, diag                       # object not visible from any camera

    gids = GP.object_geom_ids(sim)
    _, dm, seg = GP.capture(sim, cam)
    mask = GP.object_mask(seg, gids)
    obj_cam, obj_world = GP.unproject(sim.sim, cam, dm, mask,
                                      GP.CAPTURE_W, GP.CAPTURE_H)
    if len(obj_world) == 0:
        return None, diag

    # Send the object *and its immediate support*, cropped to a box around the object. Some
    # context is needed -- a floating object with no support surface yields physically silly
    # grasps -- but the whole kitchen starves the detector: uncropped, the 25th percentile of
    # its 64 grasps sat 0.388 m away and nothing survived targeting. Targeting itself happens
    # afterwards in select_grasp, by keeping only grasps on the object's own points.
    radius = GP.crop_radius(obj_cam)
    cloud_cam = GP.scene_cloud(sim.sim, cam, dm, GP.CAPTURE_W, GP.CAPTURE_H,
                               centre_cam=obj_cam.mean(axis=0), radius=radius)
    diag["crop_r"] = round(radius, 3)
    diag["n_points"] = int(len(cloud_cam))

    t0 = time.time()
    grasps = GA.detect(cloud_cam, top_k=args.top_k)
    diag["detect_latency_s"] = round(time.time() - t0, 3)
    if grasps is None:
        return None, diag
    diag["n_grasps_raw"] = int(len(grasps))

    E = GP.CU.get_camera_extrinsic_matrix(sim.sim, cam)
    ranked, reasons = GP.select_grasp(grasps, E, obj_world, sim)
    diag["n_grasps_kept"] = len(ranked)
    diag["reject"] = {k: v for k, v in reasons.items() if v}
    return (ranked[0] if ranked else None), diag


def run_episode(task, split, seed, args, video_path=None):
    env = create_env(split=split, env_name=task, seed=seed)
    # Exactly one reset. RoboCasaGymEnv.__init__ already resets once and the scene comes
    # from this second one -- adding or dropping a reset changes the sampled scene and the
    # seeds stop lining up with pick_eval_e90.json.
    env.reset()
    sim = base_env(env)
    t_wall = time.time()

    budget = int(0.5 * get_task_horizon(task=task))
    detector = SU.GraspMoveDetector()
    z0 = SU.obj_pos(sim)[2]
    slots = SU.make_skill_slots(task, sim.get_ep_meta())

    grasp, diag = plan_grasp(sim, args)

    frames = []
    def record():
        if video_path is None:
            return
        img = sim.sim.render(width=512, height=384,
                             camera_name="robot0_agentview_right")[::-1]
        frames.append(np.asarray(img, dtype=np.uint8))

    steps = 0
    ex = None
    if grasp is not None:
        ex = GraspExecutor(grasp["pos"], grasp["mat"], grasp["approach"])
        record()
        while steps < budget:
            a = ex.step(sim)
            if a is None:
                break
            env.step(convert_action(a))
            steps += 1
            detector.update(sim)
            if steps % 4 == 0:
                record()

    still_holding = bool(SU.is_holding_obj(sim))
    fired = bool(detector.fired)

    # Attribution, evaluated in this order. This is the point of the exercise: it separates
    # "the detector found nothing on the object" from "it found one and the arm missed".
    if grasp is None:
        outcome = "no_grasp_proposed"
    elif ex is not None and ex.failure == "unreachable":
        outcome = "unreachable"
    elif fired and still_holding:
        outcome = "success"
    elif fired and not still_holding:
        outcome = "grasped_then_dropped"
    else:
        outcome = "executed_no_contact"

    if video_path is not None and frames:
        out = pathlib.Path(str(video_path).replace("$TAG", "OK" if fired else "FAIL"))
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out, frames, fps=20, quality=6, macro_block_size=1)

    rec = {
        # baseline-identical keys first, so the two JSONs diff field by field
        "pick_success": fired,
        "task_success": bool(sim._check_success()),
        "handoff_step": detector.t,
        "steps": steps,
        "still_holding": still_holding,
        "final_dz": round(float(SU.obj_pos(sim)[2] - z0), 4),
        "obj": slots["obj"],
        # attribution + provenance
        "outcome": outcome,
        "pose_source": args.pose_source,
        "grasp_score": round(grasp["score"], 4) if grasp else None,
        "grasp_width": round(grasp["width"], 4) if grasp else None,
        "grasp_to_obj_dist": round(grasp["obj_dist"], 4) if grasp else None,
        "stage_steps": ex.stage_steps if ex else {},
        "stage_err": ex.stage_err if ex else {},
        "wall_s": round(time.time() - t_wall, 1),
        **diag,
    }
    env.close()
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=SU.PICK_PLACE_TASKS)
    p.add_argument("--split", default="pretrain")
    p.add_argument("--num_rollouts", type=int, default=10)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--output", default="grasp_pick_eval.json")
    p.add_argument("--video_dir", default=None)
    p.add_argument("--video_n", type=int, default=2)
    p.add_argument("--pose_source", choices=["detector", "oracle"], default="detector",
                   help="oracle builds the grasp from ground-truth geometry and needs no "
                        "server; it is the upper bound the detector is measured against")
    p.add_argument("--cameras", nargs="+", default=GP.AGENTVIEWS)
    p.add_argument("--top_k", type=int, default=64)
    args = p.parse_args()

    backend = "oracle"
    if args.pose_source == "detector":
        health = GA.probe()
        if health is None:
            print(colored(f"grasp server unreachable at {GA.BASE_URL} — "
                          f"start ./serve_grasp.sh first", "red"))
            return
        backend = health.get("backend")
        if health.get("random_weights"):
            # The server is up but has no checkpoint. Its grasps are noise, and a sweep
            # against them would produce a real-looking number that means nothing.
            print(colored("server is running on RANDOM weights (no checkpoint) — "
                          "refusing to score.\nfetch the weights with ./fetch_checkpoint.sh "
                          "and restart the server.", "red"))
            return
        print(colored(f"backend: {backend}  checkpoint={health.get('sha256')}  "
                      f"selftest={health.get('selftest_grasps')}", "cyan"))

    results, per_rollout = {}, {}
    for task in args.tasks:
        rollouts = []
        for i in range(args.num_rollouts):
            vp = None
            if args.video_dir and i < args.video_n:
                vp = str(pathlib.Path(args.video_dir) / f"{task}_ep{i:02d}_$TAG.mp4")
            try:
                rollouts.append(run_episode(task, args.split, args.seed + i, args, vp))
            except Exception as exc:
                print(colored(f"{task} rollout {i} failed: {exc}", "red"))
        if not rollouts:
            continue
        per_rollout[task] = rollouts
        picked = [r for r in rollouts if r["pick_success"]]
        mean = lambda k: float(np.mean([r[k] for r in rollouts]))
        counts = {o: sum(1 for r in rollouts if r["outcome"] == o) for o in OUTCOMES}
        results[task] = {
            "n": len(rollouts),
            "pick_success": mean("pick_success"),
            "still_holding": mean("still_holding"),
            "mean_steps_to_grasp": (float(np.mean([r["handoff_step"] for r in picked]))
                                    if picked else None),
            "outcome_counts": {k: v for k, v in counts.items() if v},
        }
        r = results[task]
        print(f"{task:32s} pick={r['pick_success']:.2f} hold={r['still_holding']:.2f} "
              f"{json.dumps(r['outcome_counts'])}")

    if results:
        results["AVERAGE"] = {
            k: float(np.mean([v[k] for t, v in results.items() if t != "AVERAGE"]))
            for k in ("pick_success", "still_holding")
        }
        print("\nAVERAGE " + json.dumps(results["AVERAGE"]))
        print(colored("baseline (diffusion policy, pick_eval_e90.json): "
                      "pick_success=0.333 still_holding=0.322", "cyan"))

        agg = {}
        for rs in per_rollout.values():
            for r in rs:
                agg[r["outcome"]] = agg.get(r["outcome"], 0) + 1
        print("outcomes:", json.dumps(agg))

    pathlib.Path(args.output).write_text(json.dumps(
        {"backend": backend, "pose_source": args.pose_source,
         "summary": results, "rollouts": per_rollout}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
