"""Throwaway: does aligning on the VISIBLE surface bias the grasp toward the camera?

A single-view cloud sees only the front of the object. Centring the jaws on the median
of that visible material should sit systematically in front of the object's true centre,
by roughly half the object's unseen depth -- and the jaw box is only 16 mm deep, so a
1-2 cm bias would miss entirely. That would explain `executed_no_contact` at 73/180.
"""
import numpy as np, robocasa
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from eval_chained_pick_place import base_env
import robocasa.utils.skill_utils as SU
import grasp_perception as GP
from grasp_geometry import object_points

TASKS = ["PickPlaceCounterToStove", "PickPlaceCounterToToasterOven",
         "PickPlaceCabinetToCounter", "PickPlaceMicrowaveToCounter"]
print(f"{'task':28s} {'visible_med':>11s} {'full_med':>9s} {'bias':>7s}  (m along view ray)")
bias = []
for task in TASKS:
    for seed in (100000, 100001):
        env = create_env(split="pretrain", env_name=task, seed=seed)
        env.reset()
        sim = base_env(env)
        cam, _ = GP.choose_camera(sim)
        gids = GP.object_geom_ids(sim)
        _, dm, seg = GP.capture(sim, cam)
        mask = GP.object_mask(seg, gids)
        _, vis = GP.unproject(sim.sim, cam, dm, mask, GP.CAPTURE_W, GP.CAPTURE_H)
        full = object_points(sim)          # ground-truth mesh surface, all sides
        if len(vis) == 0 or len(full) == 0:
            env.close(); continue
        E = GP.CU.get_camera_extrinsic_matrix(sim.sim, cam)
        ray = E[:3, 2] / np.linalg.norm(E[:3, 2])   # camera +z in world = viewing direction
        vm = float(np.median(vis @ ray))
        fm = float(np.median(full @ ray))
        bias.append(fm - vm)
        print(f"{task[:28]:28s} {vm:11.4f} {fm:9.4f} {fm-vm:+7.4f}")
        env.close()
print(f"\nmean bias = {np.mean(bias):+.4f} m   (jaw box is only 0.016 m deep)")
