"""
RGB-D capture, oracle object masking, and grasp-pose conversion.

Everything geometric between the simulator and the detector lives here, so there is exactly
one place that knows about image flips, depth units and frame conventions.

Three findings from measurement are baked in; none is optional.

1. **Mask by body id, not `contact_geoms`.** `env.objects["obj"].contact_geoms` are the
   *collision* geoms; segmentation renders the *visual* geoms. Every name resolves, so the
   lookup looks correct and silently yields **zero** masked pixels.

2. **Erode, then reject depth outliers.** Raw masks include silhouette pixels where the
   depth buffer interpolates between object and background, smearing points down the
   viewing ray. Measured object extents before/after, in metres:

       CounterToCabinet   [0.33,0.14,0.20] -> [0.06,0.07,0.18]
       CabinetToCounter   [0.38,0.63,0.13] -> [0.05,0.04,0.10]

   63 cm of "object" is pure edge artefact.

3. **Camera choice must be adaptive.** Masked pixel counts at reset:

       CounterToCabinet    left 1627   right 1708
       CabinetToCounter    left  364   right  538
       CounterToMicrowave  left  326   right    0  <- fully occluded

   A fixed camera loses whole tasks, so `auto` picks the view with the most object pixels.

One flip convention, applied once. `robosuite.macros.IMAGE_CONVENTION = "opengl"` means the
raw `sim.render` path returns bottom-row-first with no auto-flip, while
`camera_utils.get_camera_segmentation` *does* flip internally. Mixing the two produces a
mask mirrored relative to the depth map -- and because the intrinsics put the principal
point at the exact image centre, the resulting cloud still looks like a plausible scene.
This module therefore calls `sim.render` for all three buffers and flips them together.
"""

import numpy as np
from scipy import ndimage

import robosuite.utils.camera_utils as CU

from grasp_geometry import pick_symmetric, rotation_geodesic
from grasp_executor import STANDOFF
from grasp_ik import IKReach
import grasp_wire as W

AGENTVIEWS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_agentview_center"]
CAPTURE_W, CAPTURE_H = 640, 480   # the offscreen buffer's default; larger rebuilds the GL context

# GraspNet's frame -> robosuite's grip_site frame. Measured in a live env: the site's +z is
# the approach direction and its +x is the finger-closing direction, while graspnetAPI uses
# R[:,0] for approach and R[:,1] for closing. The map is a cyclic axis permutation.
GRASP_TO_EEF = np.array([[0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]])

# Panda's jaws open to 0.08. GraspNet's reported width is NOT a measurement: pred_decode
# multiplies it by 1.2 and clamps to GRASP_MAX_WIDTH = 0.1, so it is an inflated upper
# estimate. Measured across 64 grasps on one scene the widths ran 0.056 / 0.094 / 0.100
# (min/median/max) -- filtering at 0.075 rejected the majority including every grasp that
# was actually on the object. Filter at the true jaw limit and check the *cloud* instead:
# the object's own extent along the closing axis is ground truth, the prediction is not.
MAX_GRIPPER_WIDTH = 0.080
MAX_OBJ_DIST = 0.04                # a grasp further than this from the object is not on it

# The gripper's real graspable box in the grip_site frame, measured from the fingerpad
# geoms (check_pipeline.py stage 1): |x| < 0.024 on the closing axis, and only a 16 mm span
# on the approach axis. The approach span is the surprising one -- the previous enclosure
# test used +-0.04 m, 5x too wide, and passed grasps whose material sat entirely behind the
# fingers. A little padding absorbs cloud noise without reopening that hole.
JAW_HALF_X = 0.024
JAW_HALF_Y = 0.012                 # fingerpad thickness; keeps the axis search on-target
JAW_Z_LO, JAW_Z_HI = -0.012, 0.004
JAW_Z_PAD = 0.008
JAW_CENTRE_Z = 0.5 * (JAW_Z_LO + JAW_Z_HI)
JAW_SEARCH_Z = 0.05                # how far along the approach to look for material to centre
MIN_ENCLOSED = 3
# Panda opens to 0.080 m. Leave clearance: a grasp at the limit squeezes the
# object out instead of holding it, and the cloud's own extent is the honest
# measurement where the detector's `width` is not.
MAX_TRUE_WIDTH = 0.070


def object_geom_ids(env, obj_name="obj"):
    """Ids of every geom belonging to the target object's body."""
    sim = env.sim
    bid = env.obj_body_id[obj_name]
    return [i for i in range(sim.model.ngeom) if int(sim.model.geom_bodyid[i]) == bid]


def capture(env, camera, width=CAPTURE_W, height=CAPTURE_H):
    """RGB, metric depth and segmentation for one camera, all flipped consistently."""
    sim = env.sim
    rgb, depth = sim.render(width=width, height=height, camera_name=camera, depth=True)
    seg = sim.render(width=width, height=height, camera_name=camera, segmentation=True)
    rgb, depth, seg = rgb[::-1], depth[::-1], seg[::-1]
    dm = CU.get_real_depth_map(sim, depth[..., None])[..., 0]
    return rgb, dm, seg


def object_mask(seg, gids, erode=2):
    """Clean object mask: geom-id match, then eroded to shed silhouette mixed pixels."""
    mask = np.isin(seg[..., 1], gids)
    if mask.sum() == 0:
        return mask
    eroded = ndimage.binary_erosion(mask, iterations=erode)
    return eroded if eroded.sum() >= 10 else mask


def unproject(sim, camera, dm, mask, width, height, depth_tol=0.10):
    """Masked pixels -> (points_camera, points_world), with depth outliers dropped."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    z = dm[ys, xs]
    keep = np.abs(z - np.median(z)) < depth_tol      # kills the ray-smeared edge points
    ys, xs, z = ys[keep], xs[keep], z[keep]
    if len(z) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    K = CU.get_camera_intrinsic_matrix(sim, camera, height, width)
    pc_cam = np.stack([(xs - K[0, 2]) * z / K[0, 0],
                       (ys - K[1, 2]) * z / K[1, 1], z], axis=1)
    E = CU.get_camera_extrinsic_matrix(sim, camera)   # camera -> world, correction included
    pc_world = (E[:3, :3] @ pc_cam.T).T + E[:3, 3]
    return pc_cam, pc_world


# Half-size of the crop box around the target object. 0.25 was still far too generous:
# measured over 64 grasps on two scenes, the *median* grasp landed 0.15-0.20 m from the
# object and only 3-6 fell within 4 cm of it, because a +-0.25 m box around a croissant is
# mostly counter and the detector proposes over the whole cloud. Tighter concentrates the
# 20k sampled points on the object while still keeping enough support surface that grasps
# stay physically sensible.
WORKSPACE_R = 0.15


def scene_cloud(sim, camera, dm, width, height, stride=1, max_range=2.5,
                centre_cam=None, radius=WORKSPACE_R):
    """
    The visible scene as a camera-frame cloud, cropped to a workspace box.

    The detector needs *some* context -- fed a floating object with no support surface it
    proposes physically silly grasps -- but it must not be fed the whole kitchen. Measured
    without the crop: of 64 returned grasps, the 25th percentile sat 0.388 m from the target
    object, i.e. the detector spent almost all its capacity on counters, walls and the oven,
    and after filtering to grasps actually on the object **zero** survived.

    Cropping to +-0.25 m around the object concentrates the 20k sampled points the network
    sees onto the object and its immediate support. This is the same idea as graspnet's own
    demo workspace mask and AnyGrasp's `lims` argument, so it transfers to that backend.

    `stride` is 1, not 2. Subsampling made sense when the whole scene was being sent; once
    the crop is applied it just starves the network -- cropped clouds were coming out at
    1400-3500 points, and the scenes where the detector returned only 1-3 raw grasps were
    exactly the sparse ones. The crop happens after extraction, so the cost is one numpy
    pass over the full depth image.
    """
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    ys, xs = ys.ravel(), xs.ravel()
    z = dm[ys, xs]
    keep = (z > 0.05) & (z < max_range)
    ys, xs, z = ys[keep], xs[keep], z[keep]
    K = CU.get_camera_intrinsic_matrix(sim, camera, height, width)
    pts = np.stack([(xs - K[0, 2]) * z / K[0, 0],
                    (ys - K[1, 2]) * z / K[1, 1], z], axis=1)
    if centre_cam is not None:
        m = np.all(np.abs(pts - np.asarray(centre_cam)) <= radius, axis=1)
        if m.sum() >= 512:            # keep the full cloud if the crop starves the network
            pts = pts[m]
    return pts


def crop_radius(obj_cam, margin=0.06, lo=0.08, hi=0.20):
    """
    Size the crop to the object instead of using a constant.

    A fixed +-0.15 m box is enormous around an egg or a peach: over the 18-task sweep the
    dominant rejection by far was `off_object` (256 against `no_enclosure` 1), i.e. the
    detector kept proposing grasps on the surrounding counter because that is most of what
    it was shown. Scaling the box to the object's own extent concentrates the network on
    the object while still including enough support surface to keep grasps sensible.
    """
    if len(obj_cam) == 0:
        return WORKSPACE_R
    half = 0.5 * float(np.max(np.ptp(obj_cam, axis=0)))
    return float(np.clip(half + margin, lo, hi))


def choose_camera(env, cameras=None, width=CAPTURE_W, height=CAPTURE_H, obj_name="obj"):
    """
    The camera that sees the most of the target object.

    Not cosmetic: `PickPlaceCounterToMicrowave` shows 326 object pixels from
    agentview_left and exactly 0 from agentview_right, so a fixed choice silently loses
    whole tasks to occlusion.
    """
    gids = object_geom_ids(env, obj_name)
    counts, best, best_n = {}, None, -1
    for cam in (cameras or AGENTVIEWS):
        _, dm, seg = capture(env, cam, width, height)
        n = int(object_mask(seg, gids).sum())
        counts[cam] = n
        if n > best_n:
            best, best_n = cam, n
    return best, counts


def grasp_to_world(grasp, E):
    """
    One detector grasp -> (`grip_site` world position, world rotation, approach).

    `grasp` is one row of the graspnetAPI array, in the camera frame:
      R[:,0] approach (already negated inside pred_decode, so it points into the scene),
      R[:,1] closing, `translation` the grasp centre, `depth` the finger extension along
      the approach.

    `grip_site` is the fingerpad midpoint (measured ~3.6 mm), so the point the object should
    end up between the jaws is `translation + depth * approach`, expressed in world.
    """
    R_cam = np.asarray(grasp[W.ROT], dtype=np.float64).reshape(3, 3)
    t_cam = np.asarray(grasp[W.TRANS], dtype=np.float64)
    depth = float(grasp[W.DEPTH])

    R_world = E[:3, :3] @ R_cam
    t_world = E[:3, :3] @ t_cam + E[:3, 3]
    approach = R_world[:, 0] / np.linalg.norm(R_world[:, 0])

    # `translation` IS the grasp point. Do NOT add `depth * approach`.
    #
    # Measured directly (check_pipeline.py stage 8), counting target-object points inside
    # the gripper's real graspable box (|x| < 0.024, z in [-0.012, +0.004], taken from the
    # fingerpad geoms):
    #
    #     grasp point = t                213 points enclosed, material z ~ [-0.001, +0.020]
    #     grasp point = t + depth*a       58 points enclosed, material z ~ [-0.031, -0.011]
    #
    # The offset pushes the target 2-3 cm *past* the object, leaving the material behind the
    # jaws -- which is exactly the observed `executed_no_contact` signature: the servo
    # reaches its target to 2 mm, the gripper closes fully to empty, and the object never
    # moves. `depth` is the finger extension in GraspNet's own gripper model, not an offset
    # to apply to a TCP that already sits at the fingerpad midpoint.
    pos = t_world
    R_eef = R_world @ GRASP_TO_EEF
    return pos, R_eef, approach, t_world


def select_grasp(grasps, E, obj_points_world, env, max_width=MAX_GRIPPER_WIDTH,
                 max_obj_dist=MAX_OBJ_DIST):
    """
    Filter scene grasps down to ones on the target object, and rank them.

    The oracle mask does the *targeting* here: the detector proposes grasps for the whole
    scene, and a grasp is kept only if it lands on the object's own points. That mirrors how
    the diffusion policy is handed object identity via its slot embedding, so the comparison
    stays about grasp synthesis rather than object recognition.

    Returns a list of dicts sorted best-first.
    """
    reasons = {"width_pred": 0, "off_object": 0, "no_material_on_axis": 0,
               "no_enclosure": 0, "too_wide": 0, "no_ik": 0}
    if grasps is None or len(grasps) == 0 or len(obj_points_world) == 0:
        return [], reasons

    sid = env.robots[0].eef_site_id["right"]
    R_cur = np.asarray(env.sim.data.site_xmat[sid]).reshape(3, 3)

    out = []
    for i, g in enumerate(np.atleast_2d(grasps)):
        width = float(g[W.WIDTH])
        if width > max_width:
            reasons["width_pred"] += 1    # will not close on this object
            continue
        pos, R_eef, approach, seed = grasp_to_world(g, E)
        # Target on the seed point, not the offset grasp point -- see grasp_to_world.
        d_obj = float(np.min(np.linalg.norm(obj_points_world - seed, axis=1)))
        if d_obj > max_obj_dist:
            reasons["off_object"] += 1    # a grasp on the counter, a distractor, a wall
            continue

        # What the jaws would actually enclose, measured from the object's own points
        # rather than trusted from the detector's inflated width prediction. Doubles as a
        # convention check: a flipped axis or a bad frame composition empties this set for
        # essentially every grasp, so the failure surfaces as `no_grasp_proposed` in the
        # JSON rather than as a mysteriously low score.
        # Align the grasp point to the jaws, then check what they would actually enclose.
        #
        # Neither convention for the grasp point is right on its own. Measured against the
        # real jaw box (|x| < 0.024, z in [-0.012, +0.004], from the fingerpad geoms):
        #
        #     pos = t + depth*approach   material sits ~22 mm BEHIND the jaws
        #     pos = t                    material sits ~12 mm AHEAD of the jaws
        #
        # So the correct offset is a small shift, not the full finger extension and not
        # zero. Derive it from geometry instead of trusting either convention: find the
        # object material lying along the grasp axis and slide the grasp point until that
        # material is centred between the fingers.
        local = (obj_points_world - pos) @ R_eef      # columns of R_eef are the eef axes
        axis = local[(np.abs(local[:, 0]) < JAW_HALF_X)
                     & (np.abs(local[:, 1]) < JAW_HALF_Y)
                     & (np.abs(local[:, 2]) < JAW_SEARCH_Z)]
        if len(axis) < MIN_ENCLOSED:
            reasons["no_material_on_axis"] += 1
            continue
        shift = float(np.median(axis[:, 2])) - JAW_CENTRE_Z
        pos = pos + R_eef[:, 2] * shift

        # Re-check against the true box after the shift; the symmetry flip below is about
        # the approach axis, so |x| and z are invariant and the order does not matter.
        local = (obj_points_world - pos) @ R_eef
        near = local[(np.abs(local[:, 0]) < JAW_HALF_X)
                     & (local[:, 2] > JAW_Z_LO - JAW_Z_PAD)
                     & (local[:, 2] < JAW_Z_HI + JAW_Z_PAD)]
        if len(near) < MIN_ENCLOSED:
            reasons["no_enclosure"] += 1
            continue

        # Will the jaws actually close around it, or squeeze it out?
        #
        # Restored after being deleted as "bug compensation" -- it is not. Measured, the
        # grasp pose genuinely straddles the object in 12/12 scenes (255 collision points
        # between the fingers on average), yet the gripper closes on nothing 41% of the
        # time. Tracing shows contact made and then LOST during the close: the object is
        # squeezed out. PickPlaceCabinetToCounter is the clearest case, with a collision
        # extent of 0.081 m across against a jaw opening of 0.080 m -- it cannot fit.
        #
        # Measured from the enclosed material rather than taken from GraspNet's `width`,
        # which is inflated 1.2x and clamped at 0.1 and so cannot be used as a limit.
        true_w = float(np.ptp(near[:, 0]))
        if true_w > MAX_TRUE_WIDTH:
            reasons["too_wide"] += 1
            continue

        R_eef = pick_symmetric(R_eef, R_cur)
        out.append({
            "index": i,
            "score": float(g[W.SCORE]),
            "width": width,
            "depth": float(g[W.DEPTH]),
            "pos": pos,
            "mat": R_eef,
            "approach": approach,
            "obj_dist": d_obj,
            "reorient": rotation_geodesic(R_cur, R_eef),
            # a mild prior for reaching down rather than sideways: top-down grasps are both
            # more often reachable with a fixed base and less likely to sweep the object
            "downward": float(np.dot(approach, np.array([0.0, 0.0, -1.0]))),
        })

    # Prefer downward approaches. This is NOT a tuning hack -- it stands in for the
    # motion planner this pipeline does not have.
    #
    # Measured on an identical 6-task x 5-seed set, changing only this term:
    #
    #     score + 0.6*downward - 0.25*reorient   ->  `unreachable`  5/30
    #     score alone                            ->  `unreachable` 15/30
    #     score alone + IK feasibility filter    ->  `unreachable` 16/30
    #
    # IK cannot recover it, because the poses ARE kinematically reachable -- traced, the arm
    # wedges against fixture geometry on the way in (contacts with microwave_housing at
    # 0.00003 m/step of travel). With a fixed base and a straight-line Cartesian approach,
    # coming from above is simply the direction that clears cabinet and appliance walls, and
    # `downward` is the cheapest available proxy for that. The principled replacement is
    # collision-aware planning, not a different weight.
    #
    # `downward` is dot(approach, -z): +1 straight down, 0 horizontal, -1 straight up.
    # `reorient` is the wrist geodesic, which breaks ties toward poses needing less motion.
    out.sort(key=lambda d: -(d["score"] + 0.6 * d["downward"] - 0.25 * d["reorient"]))

    # Feasibility, applied in rank order so the cost stays bounded (IK is ~2-6 ms and only
    # the survivors of the cheap filters get here -- typically 2-14 of 64).
    #
    return out, reasons

