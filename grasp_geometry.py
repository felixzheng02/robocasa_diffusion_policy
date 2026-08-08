"""
Pure geometry shared by the oracle executor check and the perception client.

No I/O, no HTTP, no detector: everything here is a function of the MuJoCo model plus a
pose, which is what lets the oracle arm and the detector arm be compared honestly.

The two things it provides:
  - `object_points`: the target object's surface as a world-frame point cloud, read from
    the actual mesh vertices rather than approximated.
  - `top_down_grasp`: a good top-down grasp derived from that cloud.

Why mesh vertices and not `geom_rbound`: rbound is a bounding *sphere* radius, so an AABB
built from it is near-cubic for every object and any "which axis is narrower" decision made
from it is effectively a coin flip. Measured: the rbound oracle picked the wrong closing
axis often enough to fail half of a smoke test whose servo was converging to 0.1 mm.
"""

import mujoco
import numpy as np

# mjtGeom values we can read exactly. Anything else falls back to its bounding sphere.
_MESH, _BOX, _CYLINDER, _SPHERE, _CAPSULE, _ELLIPSOID = 7, 6, 5, 2, 3, 4


def _geom_local_points(model, g, n_primitive=64):
    """Representative points for one geom, in that geom's local frame."""
    gtype = int(model.geom_type[g])
    size = np.asarray(model.geom_size[g], dtype=np.float64)

    if gtype == _MESH:
        mid = int(model.geom_dataid[g])
        adr = int(model.mesh_vertadr[mid])
        num = int(model.mesh_vertnum[mid])
        return np.asarray(model.mesh_vert[adr:adr + num], dtype=np.float64).reshape(-1, 3)

    if gtype == _BOX:
        s = size[:3]
        c = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        return c * s

    if gtype in (_SPHERE, _CAPSULE, _ELLIPSOID, _CYLINDER):
        # a coarse spherical / cylindrical shell is plenty for extent and PCA purposes
        u = np.random.default_rng(0).normal(size=(n_primitive, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        if gtype == _CYLINDER:
            r, h = float(size[0]), float(size[1])
            th = np.linspace(0, 2 * np.pi, n_primitive)
            ring = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
            return np.vstack([np.c_[ring, np.full(len(ring), h)],
                              np.c_[ring, np.full(len(ring), -h)]])
        if gtype == _ELLIPSOID:
            return u * size[:3]
        return u * float(size[0])

    return np.zeros((1, 3))


def object_points(env, obj_name="obj", max_points=4000):
    """
    World-frame surface points of the target object, from its real geometry.

    Uses every geom on the object's body -- the same body-id selection the segmentation
    mask uses, so the oracle and the perceived cloud describe the same thing.
    """
    sim = env.sim
    model, data = sim.model, sim.data
    bid = env.obj_body_id[obj_name]
    out = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid:
            continue
        loc = _geom_local_points(model, g)
        if len(loc) == 0:
            continue
        R = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        p = np.asarray(data.geom_xpos[g], dtype=np.float64)
        out.append((R @ loc.T).T + p)
    if not out:
        return np.zeros((0, 3))
    pts = np.vstack(out)
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts


def slice_minor_axis(xy):
    """Direction of least extent of a 2-D cross-section, and that extent."""
    c = xy - xy.mean(axis=0)
    # PCA via SVD; the last right-singular vector is the least-variance direction
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    minor = Vt[-1]
    proj = c @ minor
    return minor, float(proj.max() - proj.min())


def rotation_geodesic(Ra, Rb):
    """Angle of the rotation taking Ra to Rb, in radians."""
    c = (np.trace(Rb @ Ra.T) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def pick_symmetric(R_target, R_current):
    """
    Choose between a grasp frame and its 180-degree twin about the approach axis.

    A parallel jaw is symmetric, so rotating the closing axis by pi is the *same physical
    grasp*. Picking whichever twin is nearer the current wrist saves up to 90 degrees of
    reorientation, and measured on the oracle check that is the difference between the arm
    reaching the pre-grasp pose and timing out against a 0.45-0.85 rad orientation error.
    Not cosmetic: it moved oracle `unreachable` failures directly into successes.
    """
    flip = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])  # Rz(pi)
    R_alt = R_target @ flip
    return R_alt if (rotation_geodesic(R_current, R_alt)
                     < rotation_geodesic(R_current, R_target)) else R_target


def top_down_grasp(env, obj_name="obj", max_width=0.075, finger_h=0.02,
                   min_pts=20, lo_frac=0.15, hi_frac=0.92, core_r=0.02, min_core=3):
    """
    A top-down grasp on the object's best graspable horizontal cross-section.

    Returns `((pos_world, R_eef_world), approach_world)` or None if no slice fits between
    the jaws.

    Two choices matter here, both learned from measurement:

    - **Which height.** Taking the *highest* slice that fits the jaws sounds right but
      measured poorly: on tall objects it lands on a cap or a thin lip, and the fingers
      close on almost nothing (observed: converged to 3 mm of position error, then
      `held_frames == 0`). Slices are scored instead by how much material they contain
      relative to how wide they are, restricted to the middle band of the object, which
      prefers a solid, narrow purchase over a high one.
    - **Which closing direction.** The cross-section's minor axis, so the jaws span the
      thin direction -- closing along the long axis slides off -- and then `pick_symmetric`
      resolves the sign against the current wrist.

    `grip_site` is the fingerpad midpoint (measured: ~3.6 mm), so the returned position is
    where the object should sit *between* the fingers; no TCP offset is applied.
    """
    pts = object_points(env, obj_name)
    if len(pts) < min_pts:
        return None

    z_lo, z_hi = float(pts[:, 2].min()), float(pts[:, 2].max())
    span = max(z_hi - z_lo, 1e-6)

    best, best_score = None, -np.inf
    z = z_lo + lo_frac * span
    while z <= z_lo + hi_frac * span:
        band = pts[np.abs(pts[:, 2] - z) <= finger_h * 0.5]
        if len(band) >= min_pts:
            minor, width = slice_minor_axis(band[:, :2])
            centre = band[:, :2].mean(axis=0)
            # Reject a centre that sits in empty space. The xy centroid of a *ring* -- a
            # bowl rim, a mug, a pan -- is a hole, and a grasp aimed there converges to a
            # few mm of position error and then closes on nothing. Observed directly:
            # `pre_grasp` error 12 mm, `held_frames == 0`, object undisturbed.
            core = np.sum(np.linalg.norm(band[:, :2] - centre, axis=1) < core_r)
            if width <= max_width and core >= min_core:
                # material per unit jaw opening: rewards a solid, narrow cross-section
                score = len(band) / (width + 0.005)
                if score > best_score:
                    best_score = score
                    best = (centre, z, minor)
        z += 0.005

    if best is None:
        return None
    xy, z, minor = best

    approach = np.array([0.0, 0.0, -1.0])          # straight down into the scene
    pos, R = _eef_frame(np.array([xy[0], xy[1], z]), approach,
                        np.array([minor[0], minor[1], 0.0]))

    sid = env.robots[0].eef_site_id["right"]
    R_cur = np.asarray(env.sim.data.site_xmat[sid]).reshape(3, 3)
    return (pos, pick_symmetric(R, R_cur)), approach


def _side_grasp(env, pts, approach, max_width, core_r, min_core):
    """A grasp along an arbitrary approach: close across the object's thin direction as
    seen from that approach."""
    u, v = _orthonormal_basis(approach)
    centre3 = pts.mean(axis=0)
    proj = np.stack([pts @ u, pts @ v], axis=1)
    minor2, width = slice_minor_axis(proj)
    if width > max_width:
        return None
    c2 = proj.mean(axis=0)
    if np.sum(np.linalg.norm(proj - c2, axis=1) < core_r) < min_core:
        return None
    closing = minor2[0] * u + minor2[1] * v
    return _eef_frame(centre3, approach, closing), width


def oracle_grasp(env, obj_name="obj", max_width=0.075, min_free=0.20,
                 core_r=0.02, min_core=3):
    """
    The best ground-truth grasp for this scene: a top-down grasp when one exists, else the
    best side grasp that has room for the wrist.

    Top-down is tried first and wins whenever it returns anything -- see the comment below,
    which records the measurement that settled it. Side grasps exist because top-down is
    sometimes geometrically impossible: a top-down approach into a drawer has only 0.188 m
    of clearance before the counter above it, and every `PickPlaceDrawerToCounter` oracle
    rollout failed while open-surface tasks scored 1.00.

    Side grasps also matter for the *detector* arm, which is the real reason to keep them:
    AnyGrasp proposes horizontal grasps freely, so the executor has to handle them, and an
    oracle that could only ever go straight down would understate what is achievable and
    misattribute the shortfall to the detector.
    """
    pts = object_points(env, obj_name)
    if len(pts) < 20:
        return None
    centre = pts.mean(axis=0)

    # Top-down first, and *unconditionally* -- no free-space precondition.
    #
    # Gating it on clearance was measured and made things worse: oracle pick rate fell
    # 0.556 -> 0.389 over the same 18 rollouts, with CounterToSink collapsing 1.00 -> 0.33,
    # because the ray test rejected top-down approaches that in fact worked and substituted
    # a cruder centroid-based side grasp. The slice-based top-down grasp is simply better
    # when it exists, so side grasps are a fallback for when it does not, not an
    # alternative to be ranked against it.
    got = top_down_grasp(env, obj_name, max_width=max_width,
                         core_r=core_r, min_core=min_core)
    if got is not None:
        return got

    sid = env.robots[0].eef_site_id["right"]
    R_cur = np.asarray(env.sim.data.site_xmat[sid]).reshape(3, 3)

    # Horizontal candidates, plus a 45-degree tilt. Ranked by how much room the wrist has.
    cands = []
    for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        h = np.array([np.cos(th), np.sin(th), 0.0])
        for tilt in (0.0, -0.5):
            a = h + tilt * np.array([0.0, 0.0, 1.0])
            a /= np.linalg.norm(a)
            cands.append(a)

    best, best_free = None, -np.inf
    for a in cands:
        room = free_space(env, centre, -a, obj_name)
        if room < min_free or room <= best_free:
            continue
        got = _side_grasp(env, pts, a, max_width, core_r, min_core)
        if got is None:
            continue
        (pos, R), _ = got
        best, best_free = ((pos, pick_symmetric(R, R_cur)), a), room

    return best


def free_space(env, point, direction, obj_name="obj"):
    """
    Metres of clear space from `point` along `direction`, ignoring the target object.

    The gripper has to come from somewhere: an approach is only usable if there is a
    corridor behind the grasp for the fingers and wrist. Measured with a MuJoCo ray, which
    is exact and costs nothing. A negative return means the ray hit nothing at all.

    Measured examples at reset -- straight up from the object:
        MicrowaveToCounter  0.339 m (clear; that task's failures are reach, not blocking)
        DrawerToCounter     0.188 m (blocked by the counter directly above the drawer)
    """
    m = env.sim.model._model
    d = env.sim.data._data
    v = np.asarray(direction, dtype=np.float64)
    v = v / np.linalg.norm(v)
    gid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(m, d, np.asarray(point, dtype=np.float64), v,
                         None, 1, env.obj_body_id[obj_name], gid)
    return float("inf") if dist < 0 else float(dist)


def _orthonormal_basis(a):
    """Two unit vectors spanning the plane perpendicular to `a`."""
    a = np.asarray(a, dtype=np.float64)
    a = a / np.linalg.norm(a)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, a)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = seed - np.dot(seed, a) * a
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def _eef_frame(pos, approach, closing):
    """Assemble the world rotation of `grip_site` from an approach and a closing axis.

    Measured in a live env: the site's +z is the approach direction and its +x is the
    finger-closing direction, so those are the two columns we pin.
    """
    z = np.asarray(approach, dtype=np.float64)
    z /= np.linalg.norm(z)
    x = np.asarray(closing, dtype=np.float64) - np.dot(closing, z) * z
    nx = np.linalg.norm(x)
    if nx < 1e-8:                                   # closing axis parallel to approach
        alt = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(alt, z)) > 0.9:
            alt = np.array([0.0, 1.0, 0.0])
        x = alt - np.dot(alt, z) * z
        nx = np.linalg.norm(x)
    x /= nx
    y = np.cross(z, x)
    return np.asarray(pos, dtype=np.float64), np.column_stack([x, y, z])
