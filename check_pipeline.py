"""
Component-by-component verification of the grasp pipeline.

Every stage gets an explicit expectation and a measured number, checked in isolation, so a
bad end-to-end score can be attributed to a specific component instead of guessed at. This
exists because the opposite approach -- adding a heuristic per observed symptom -- produced
eight tunable constants and a score still well below the learned baseline, which is the
signature of an unfound bug rather than a tuning gap.

    ./serve_grasp.sh                 # needed for stages 6+
    MUJOCO_GL=egl python check_pipeline.py --task PickPlaceCabinetToCounter --seed 100000

Stages
  1 gripper geometry    where the jaws actually are, in the grip_site frame
  2 capture             depth is metric and plausible; buffers agree in shape
  3 object mask         non-empty, and the projected ground-truth centre lands inside it
  4 unprojection        cloud centroid near ground truth; extent is object-scale
  5 crop                still contains the object; enough points for the network
  6 detector output     rotations orthonormal det=+1; scores/widths in range; seeds on cloud
  7 pose conversion     approach is a unit vector; R_eef orthonormal; round trip is exact
  8 enclosure           THE question: does the object lie between the jaws at the pose?
  9 executor            does the servo actually reach a commanded pose?
"""

import argparse
import numpy as np
from termcolor import colored

import robocasa  # noqa: F401
import robocasa.utils.skill_utils as SU
from robocasa.utils.env_utils import convert_action
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from eval_chained_pick_place import base_env

import grasp_perception as GP
import grasp_agent as GA
import grasp_executor as GE
import grasp_wire as W

OK, BAD = colored("PASS", "green"), colored("FAIL", "red")
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {OK if ok else BAD} {name:34s} {detail}")


def jaw_volume(env):
    """
    The gripper's graspable box in the grip_site frame, measured from the model.

    Returns (x_halfspan, z_lo, z_hi) where x is the closing axis and z the approach axis.
    Everything downstream that asks "is the object between the fingers" must use these
    numbers rather than a guessed window.
    """
    sim = env.sim
    r = env.robots[0]
    sid = r.eef_site_id["right"]
    p = np.array(sim.data.site_xpos[sid])
    R = np.array(sim.data.site_xmat[sid]).reshape(3, 3)
    g = r.gripper["right"]
    lo, hi, xs = [], [], []
    for key in ("left_fingerpad", "right_fingerpad"):
        for n in g.important_geoms[key]:
            gid = sim.model.geom_name2id(n)
            c = R.T @ (np.array(sim.data.geom_xpos[gid]) - p)
            s = np.array(sim.model.geom_size[gid])
            lo.append(c[2] - s.max())
            hi.append(c[2] + s.max())
            xs.append(abs(c[0]))
    return float(np.mean(xs)), float(np.min(lo)), float(np.max(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlaceCabinetToCounter")
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--split", default="pretrain")
    args = ap.parse_args()

    env = create_env(split=args.split, env_name=args.task, seed=args.seed)
    env.reset()
    sim = base_env(env)
    gt = SU.obj_pos(sim)
    obj_name = SU.make_skill_slots(args.task, sim.get_ep_meta())["obj"]
    print(f"\n{args.task}  seed={args.seed}  obj={obj_name}  gt={np.round(gt,3)}\n")

    # --- 1 gripper geometry ------------------------------------------------------------
    print("[1] gripper geometry")
    xh, zlo, zhi = jaw_volume(sim)
    check("fingerpad half-span on x", 0.02 < xh < 0.06, f"{xh:.4f} m (jaw opening/2)")
    check("fingerpad span on z (approach)", (zhi - zlo) < 0.06,
          f"z in [{zlo:+.4f}, {zhi:+.4f}] -> span {zhi-zlo:.4f} m")
    print(f"       -> graspable box: |x| < {xh:.3f}, z in [{zlo:.3f}, {zhi:.3f}]")

    # --- 2 capture ---------------------------------------------------------------------
    print("\n[2] capture")
    cam, counts = GP.choose_camera(sim)
    rgb, dm, seg = GP.capture(sim, cam)
    check("shapes agree", rgb.shape[:2] == dm.shape == seg.shape[:2],
          f"rgb{rgb.shape} depth{dm.shape} seg{seg.shape}")
    check("depth is metric & plausible", 0.05 < dm.min() < 5 and dm.max() < 20,
          f"range {dm.min():.3f}..{dm.max():.3f} m   camera={cam}")

    # --- 3 object mask -----------------------------------------------------------------
    print("\n[3] object mask")
    gids = GP.object_geom_ids(sim)
    mask = GP.object_mask(seg, gids)
    check("mask non-empty", mask.sum() > 0, f"{int(mask.sum())} px  (all cams: {counts})")
    P = GP.CU.get_camera_transform_matrix(sim=sim.sim, camera_name=cam,
                                          camera_height=GP.CAPTURE_H,
                                          camera_width=GP.CAPTURE_W)
    hom = P @ np.append(gt, 1.0)
    u, v = hom[0] / hom[2], hom[1] / hom[2]
    inside = (0 <= int(v) < GP.CAPTURE_H and 0 <= int(u) < GP.CAPTURE_W
              and mask[max(0, int(v) - 6):int(v) + 7, max(0, int(u) - 6):int(u) + 7].any())
    check("GT projects into the mask", inside, f"pixel ({u:.0f},{v:.0f})")

    # --- 4 unprojection ----------------------------------------------------------------
    print("\n[4] unprojection")
    obj_cam, obj_world = GP.unproject(sim.sim, cam, dm, mask, GP.CAPTURE_W, GP.CAPTURE_H)
    cerr = float(np.linalg.norm(obj_world.mean(0) - gt)) if len(obj_world) else 9.9
    ext = np.ptp(obj_world, axis=0) if len(obj_world) else np.zeros(3)
    check("centroid near ground truth", cerr < 0.08, f"{cerr:.4f} m  (surface bias expected)")
    check("extent is object-scale", ext.max() < 0.35, f"{np.round(ext,3)}")

    # --- 5 crop ------------------------------------------------------------------------
    print("\n[5] crop")
    r = GP.crop_radius(obj_cam)
    cloud = GP.scene_cloud(sim.sim, cam, dm, GP.CAPTURE_W, GP.CAPTURE_H,
                           centre_cam=obj_cam.mean(0), radius=r)
    E = GP.CU.get_camera_extrinsic_matrix(sim.sim, cam)
    cw = (E[:3, :3] @ cloud.T).T + E[:3, 3]
    d_obj_cloud = np.min(np.linalg.norm(cw - gt, axis=1))
    check("crop still contains the object", d_obj_cloud < 0.10,
          f"nearest cloud pt to GT {d_obj_cloud:.4f} m, radius={r:.3f}")
    check("enough points for the network", len(cloud) >= 512, f"{len(cloud)} points")

    # --- 6 detector --------------------------------------------------------------------
    print("\n[6] detector output")
    gg = GA.detect(cloud, top_k=64)
    if gg is None or len(gg) == 0:
        check("detector returned grasps", False, "none - is serve_grasp.sh running?")
        print(f"\n{sum(results)}/{len(results)} checks passed")
        return
    Rs = gg[:, W.ROT].reshape(-1, 3, 3)
    orth = max(float(np.abs(R @ R.T - np.eye(3)).max()) for R in Rs)
    dets = np.array([np.linalg.det(R) for R in Rs])
    check("rotations orthonormal", orth < 1e-3, f"max |RR^T - I| = {orth:.2e}")
    check("rotations right-handed", np.all(dets > 0.99), f"det in [{dets.min():.4f},{dets.max():.4f}]")
    check("scores positive", gg[:, W.SCORE].min() > 0,
          f"{gg[:,W.SCORE].min():.3f}..{gg[:,W.SCORE].max():.3f}")
    check("widths within GraspNet range", gg[:, W.WIDTH].max() <= 0.1001,
          f"{gg[:,W.WIDTH].min():.3f}..{gg[:,W.WIDTH].max():.3f} m")
    tw = (E[:3, :3] @ gg[:, W.TRANS].T).T + E[:3, 3]
    seed_on_cloud = np.median([np.min(np.linalg.norm(cw - t, axis=1)) for t in tw])
    check("seeds lie on the input cloud", seed_on_cloud < 0.005,
          f"median dist {seed_on_cloud:.5f} m  (pred_decode samples fp2_xyz from it)")

    # --- 7 pose conversion -------------------------------------------------------------
    print("\n[7] pose conversion")
    pos, R_eef, appr, seed = GP.grasp_to_world(gg[0], E)
    check("approach is a unit vector", abs(np.linalg.norm(appr) - 1) < 1e-6,
          f"|a| = {np.linalg.norm(appr):.8f}")
    check("R_eef orthonormal det=+1",
          np.abs(R_eef @ R_eef.T - np.eye(3)).max() < 1e-6 and abs(np.linalg.det(R_eef) - 1) < 1e-6,
          f"det = {np.linalg.det(R_eef):.6f}")
    check("R_eef z-axis == approach", np.allclose(R_eef[:, 2], appr, atol=1e-6),
          f"z-axis {np.round(R_eef[:,2],4)} vs approach {np.round(appr,4)}")

    # --- 8 enclosure: the real question ------------------------------------------------
    print("\n[8] enclosure - does the object lie between the jaws?")
    for label, P_ in (("seed (t)", seed), ("t + depth*a", pos)):
        loc = (obj_world - P_) @ R_eef
        n_true = int(np.sum((np.abs(loc[:, 0]) < xh) & (loc[:, 2] > zlo) & (loc[:, 2] < zhi)))
        n_loose = int(np.sum((np.abs(loc[:, 0]) < xh) & (np.abs(loc[:, 2]) < 0.04)))
        print(f"       {label:14s} in TRUE jaw box: {n_true:5d} pts | "
              f"in my +-0.04 window: {n_loose:5d} pts | "
              f"z of material: {np.percentile(loc[:,2],[5,50,95]).round(4)}")
    loc = (obj_world - pos) @ R_eef
    n_true = int(np.sum((np.abs(loc[:, 0]) < xh) & (loc[:, 2] > zlo) & (loc[:, 2] < zhi)))
    check("material inside the TRUE jaw box", n_true >= 3, f"{n_true} points")

    # --- 9 executor --------------------------------------------------------------------
    print("\n[9] executor")
    tgt_p, tgt_R = GE.eef_pose(sim)
    tgt_p = tgt_p + np.array([0.0, 0.0, -0.05])
    for _ in range(60):
        a, pe, re = GE.servo_action(sim, tgt_p, tgt_R, GE.GRIP_OPEN)
        env.step(convert_action(a))
    fin = float(np.linalg.norm(GE.eef_pose(sim)[0] - tgt_p))
    check("servo reaches a commanded pose", fin < 0.01, f"{fin:.5f} m after 60 steps")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    if not all(results):
        print(colored("PIPELINE HAS A BROKEN COMPONENT - fix before tuning anything", "red"))
    env.close()


if __name__ == "__main__":
    main()
