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
MAX_TRUE_WIDTH = 0.070             # measured object extent between the jaws, with margin
MAX_OBJ_DIST = 0.04                # a grasp further than this from the object is not on it


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


def scene_cloud(sim, camera, dm, width, height, stride=2, max_range=2.5,
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

    # `pos` is where grip_site should end up; `t_world` is the seed point the grasp was
    # predicted at. They are returned separately because they answer different questions:
    # pred_decode sets grasp_center = fp2_xyz, i.e. a point *sampled from the input cloud*,
    # so `t_world` is the right thing to test "is this grasp on the target object" against
    # (measured: it coincides with object cloud points to 0.0000 m), while `pos` is the
    # right thing to servo to. Conflating them pushed targeting up to 4 cm off the surface.
    pos = t_world + depth * approach
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
    if grasps is None or len(grasps) == 0 or len(obj_points_world) == 0:
        return []

    sid = env.robots[0].eef_site_id["right"]
    R_cur = np.asarray(env.sim.data.site_xmat[sid]).reshape(3, 3)

    out = []
    for i, g in enumerate(np.atleast_2d(grasps)):
        width = float(g[W.WIDTH])
        if width > max_width:
            continue                      # will not close on this object
        pos, R_eef, approach, seed = grasp_to_world(g, E)
        # Target on the seed point, not the offset grasp point -- see grasp_to_world.
        d_obj = float(np.min(np.linalg.norm(obj_points_world - seed, axis=1)))
        if d_obj > max_obj_dist:
            continue                      # a grasp on the counter, a distractor, a wall

        # What the jaws would actually enclose, measured from the object's own points
        # rather than trusted from the detector's inflated width prediction. Doubles as a
        # convention check: a flipped axis or a bad frame composition empties this set for
        # essentially every grasp, so the failure surfaces as `no_grasp_proposed` in the
        # JSON rather than as a mysteriously low score.
        local = (obj_points_world - pos) @ R_eef      # columns of R_eef are the eef axes
        near = local[(np.abs(local[:, 2]) < 0.04) & (np.abs(local[:, 0]) < 0.05)]
        if len(near) < 1:
            continue
        true_w = float(np.ptp(near[:, 0]))            # extent along the closing axis (+-x)
        if true_w > MAX_TRUE_WIDTH:
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
            "true_width": true_w,
            "reorient": rotation_geodesic(R_cur, R_eef),
            # a mild prior for reaching down rather than sideways: top-down grasps are both
            # more often reachable with a fixed base and less likely to sweep the object
            "downward": float(np.dot(approach, np.array([0.0, 0.0, -1.0]))),
        })

    # Rank by reachability first, not raw score. Measured over 12 rollouts, `unreachable`
    # was 4/12 and every one of them had a grasp genuinely *on* the object (obj_dist
    # 0.007-0.030) that the arm simply could not get the wrist to. With the base fixed, a
    # downward approach is far more often reachable than a sideways or upward one, and a
    # small wrist reorientation is more often reachable than a large one -- so a slightly
    # lower-scoring grasp that can actually be executed beats a better one that cannot.
    #
    # `downward` is dot(approach, -z): +1 straight down, 0 horizontal, -1 straight up.
    def rank(d):
        return -(d["score"] + 0.6 * d["downward"] - 0.25 * d["reorient"])

    out.sort(key=rank)
    return out
