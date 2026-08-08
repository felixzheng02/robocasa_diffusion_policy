"""
Is the point cloud geometrically correct? No detector involved.

The project's rule is that a low score must never be ambiguous between a bad model and a
broken harness. This is the perception half of that guarantee (check_grasp_executor.py is
the motion half), and it should pass before any detector number is believed.

Four tests, in increasing order of what they can catch:

A  mask non-empty          -- the geom-id selection actually matches rendered pixels
B  centroid vs ground truth-- rules out a missing flip, non-metric depth, a bad extrinsic
C  cross-camera agreement  -- two independently-posed cameras must unproject the same
                              object to the same place. This needs NO ground truth, and
                              essentially no flip, transpose or double-correction survives
                              it. The strongest single check here.
D  extent is object-scale  -- catches an erosion/outlier regression, where silhouette
                              pixels smear the cloud down the viewing ray

Test B's residual is expected to be a few cm and that is not an error: the cloud is the
visible *surface*, so its centroid sits toward the camera relative to the body origin. What
B rules out are errors that move the centroid by metres, not centimetres.

    MUJOCO_GL=egl python check_grasp_geometry.py --n 2
"""

import argparse
import json
import pathlib

import numpy as np
from termcolor import colored

import robocasa  # noqa: F401
import robocasa.utils.skill_utils as SU

from robocasa.utils.env_helpers import base_env, create_env
import grasp_perception as GP

CENTROID_TOL = 0.08      # m, vs ground truth (surface-only bias is bounded by object size)
CROSS_CAM_TOL = 0.05     # m, between two cameras' independent unprojections
MAX_EXTENT = 0.35        # m, any object bigger than this means the mask leaked

# A camera that catches only a sliver of the object -- 7, 18, 24 pixels through a door gap
# or a reflection -- has a centroid describing a different *part* of the object, and a
# meaningless extent (one such view measured 1.04 m across from 24 pixels). Comparing it
# against a full view is not a geometry test, it is a sampling artefact. Production already
# ignores these: choose_camera() takes the camera with the most object pixels.
MIN_VIEW_PX = 80
MIN_VIEW_FRAC = 0.25     # ...and at least this share of the best camera's count


def check_scene(task, split, seed):
    env = create_env(split=split, env_name=task, seed=seed)
    env.reset()
    sim = base_env(env)

    gids = GP.object_geom_ids(sim)
    gt = SU.obj_pos(sim)
    per_cam, centroids = {}, {}

    for cam in GP.AGENTVIEWS:
        _, dm, seg = GP.capture(sim, cam)
        mask = GP.object_mask(seg, gids)
        n_px = int(mask.sum())
        entry = {"mask_px": n_px}
        if n_px > 0:
            _, world = GP.unproject(sim.sim, cam, dm, mask, GP.CAPTURE_W, GP.CAPTURE_H)
            if len(world):
                c = world.mean(axis=0)
                entry.update(
                    n_points=int(len(world)),
                    centroid_err=round(float(np.linalg.norm(c - gt)), 4),
                    extent=[round(float(v), 3) for v in np.ptp(world, axis=0)],
                    depth_range=[round(float(dm.min()), 3), round(float(dm.max()), 3)],
                )
                centroids[cam] = c
        per_cam[cam] = entry

    # Only cameras with a genuine view take part in the checks (see MIN_VIEW_PX).
    best_px = max((c["mask_px"] for c in per_cam.values()), default=0)
    good = [cam for cam, c in per_cam.items()
            if c["mask_px"] >= max(MIN_VIEW_PX, MIN_VIEW_FRAC * best_px)
            and "centroid_err" in c]
    for cam, c in per_cam.items():
        c["used"] = cam in good

    cross = None
    if len(good) >= 2:
        cross = round(float(max(
            np.linalg.norm(centroids[a] - centroids[b])
            for i, a in enumerate(good) for b in good[i + 1:])), 4)

    fails = []
    if not good:
        # Not automatically a bug: some objects sit deep inside a microwave or oven and are
        # genuinely hard to see. It IS a warning that the task is perception-limited.
        fails.append(f"A: no camera has a usable view (best {best_px} px)")
    else:
        primary = min((per_cam[c] for c in good), key=lambda c: c["centroid_err"])
        if primary["centroid_err"] > CENTROID_TOL:
            fails.append(f"B: centroid off by {primary['centroid_err']} m")
        if max(primary["extent"]) > MAX_EXTENT:
            fails.append(f"D: extent {primary['extent']} — mask leaked / erosion regressed")
        if cross is not None and cross > CROSS_CAM_TOL:
            fails.append(f"C: cameras disagree by {cross} m")

    env.close()
    return {"task": task, "seed": seed, "gt": [round(float(v), 3) for v in gt],
            "cameras": per_cam, "cross_camera_max": cross, "best_px": best_px,
            "used_cameras": good, "pass": not fails, "failures": fails}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=SU.PICK_PLACE_TASKS)
    p.add_argument("--split", default="pretrain")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--output", default="grasp_geometry_check.json")
    args = p.parse_args()

    rows, n_pass = [], 0
    for task in args.tasks:
        for i in range(args.n):
            try:
                r = check_scene(task, args.split, args.seed + i)
            except Exception as exc:
                r = {"task": task, "seed": args.seed + i, "pass": False,
                     "failures": [f"exception: {exc}"]}
            rows.append(r)
            n_pass += bool(r["pass"])
            px = {k.replace("robot0_agentview_", ""): v["mask_px"]
                  for k, v in r.get("cameras", {}).items()}
            tag = colored("PASS", "green") if r["pass"] else colored("FAIL", "red")
            print(f"{tag} {r['task'][:30]:30s} s={r['seed']} px={json.dumps(px)} "
                  f"cross={r.get('cross_camera_max')} {'; '.join(r['failures'])}")

    print(f"\n{n_pass}/{len(rows)} scenes passed")
    if n_pass == len(rows):
        print(colored("GEOMETRY OK — clouds are metric, aligned and object-scale", "green"))
    else:
        print(colored("GEOMETRY SUSPECT — do not trust detector numbers yet", "red"))
    pathlib.Path(args.output).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
