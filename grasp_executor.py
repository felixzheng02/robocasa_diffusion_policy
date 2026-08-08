"""
Servo the Panda arm to a 6-DoF grasp pose and lift.

This is the motion half of the grasp evaluation. A grasp detector emits a *pose*; nothing
in robocasa or robosuite will move the arm to one (there is no scripted motion primitive in
either repo), so the executor here is what turns a pose into a pick.

Deliberately runs on the STOCK controller config. The arm is OSC_POSE with
`input_type: "delta"` and `input_ref_frame: "base"`, and switching it to `"absolute"` would
change the env the diffusion-policy baseline was measured in. Keeping it identical is what
makes `pick_success` here comparable to `pick_eval_e90.json`.

The servo law
-------------
With delta input, `output_max = 0.05` and goal_update_mode "achieved", robosuite composes

    scale_action:      scaled  = a * 0.05
    compute_goal_pos:  goal    = world_to_origin_frame(ref_pos) + scaled
    run_controller:    desired = origin_pos + origin_ori @ goal
                             --> desired = ref_pos + origin_ori @ scaled

so commanding `a = clip(origin_ori.T @ (target - ref_pos) / 0.05, -1, 1)` gives

    desired_world = ref_pos + clip(target - ref_pos, +-0.05)

i.e. a *saturated unity-gain* term: once the error is inside 5 cm the controller is handed
the exact target, and the inner OSC impedance (kp=150) closes the rest. There is no outer
gain to tune and no integrator to wind up.

Frames -- all verified in a live env rather than assumed:
  - OSC regulates `gripper0_right_grip_site`; `osc.ref_pos` is that site's world position.
  - `grip_site` sits ~3.6 mm from the fingerpad midpoint, so a grasp point maps to it
    directly with no TCP offset.
  - `osc.origin_pos` / `osc.origin_ori` are the controller's own reference frame. Read them
    off the controller rather than using the `robot0_base_to_eef_*` observations: the two
    happen to coincide for PandaOmron (measured: identical to 0.0), but that is a property
    of this robot's XML, not a guarantee, and the controller's own attributes are correct by
    construction.
  - `origin_ori` is identity in the scenes measured so far, which means a base-frame delta
    and a world-frame delta are numerically equal there. Do NOT rely on that -- the rotation
    is applied explicitly below so a rotated base stays correct.
"""

import numpy as np
import robosuite.utils.transform_utils as T

import robocasa.utils.skill_utils as SU

# --- action layout (robocasa.utils.env_utils.convert_action) ---------------------------
# 0:3 eef_pos | 3:6 eef_rot | 6:7 gripper | 7:11 base_motion | 11:12 control_mode
ACTION_DIM = 12
POS_SCALE = 0.05   # metres per unit action, from output_max in default_pandaomron.json
ROT_SCALE = 0.5    # radians per unit action

# Gripper is thresholded at 0.5 by PandaOmronKeyConverter.unmap_action -- NOT at 0. The
# conventional -1/+1 convention would leave it permanently open, silently.
GRIP_OPEN, GRIP_CLOSE = 0.0, 1.0

# --- stage tuning ----------------------------------------------------------------------
STANDOFF = 0.10        # m back along the approach axis for the pre-grasp pose
LIFT_HEIGHT = 0.15     # m, along world +z (see below)
POS_TOL = 0.015        # m, pre-grasp convergence
ROT_TOL = 0.15         # rad, pre-grasp convergence
APPROACH_POS_TOL = 0.010
SETTLE = 3             # steps the tolerance must hold before advancing
CLOSE_STEPS = 15       # 0.75 s at 20 Hz; Panda finger travel is 0.04 m
STALL_WINDOW = 5       # steps over which to measure a stall
STALL_EPS = 0.001      # m; less movement than this over the window counts as blocked

# Stage budgets. Traced against a real rollout: with NO contacts the eef moves only
# ~0.0093 m per env step, not the 0.05 m the action nominally commands -- OSC impedance
# dynamics set the pace, not the command. A 0.3-0.4 m reach to the standoff pose therefore
# needs 35-50 steps of pure travel before any settling, so the original 120 was far tighter
# than it looked. Raised, with `lift`/`hold` trimmed to compensate: the total must stay under
# the smallest rollout budget in the suite (0.5 * horizon = 225 for the 450-horizon tasks).
STAGE_CAPS = {"pre_grasp": 170, "approach": 60, "close": CLOSE_STEPS, "lift": 40, "hold": 10}

# A jammed arm is not a slow arm. Traced on PickPlaceMicrowaveToCounter, the wrist wedges
# against the microwave housing and per-step motion collapses to 0.00003 m while the
# position error sits at 0.085 -- burning the whole stage budget on a pose it will never
# reach. Detecting that and giving up early is what makes the raised cap affordable.
JAM_WINDOW = 25
JAM_EPS = 0.004        # m of eef travel over the window; below this it is wedged, not slow


def eef_pose(env):
    """World pose of the site OSC actually regulates. Site-based on both halves.

    Not `robot0_eef_quat`: that observation is body-based while `robot0_eef_pos` is
    site-based (a known robosuite inconsistency kept for dataset back-compat), so mixing
    them yields a constant, purposeful-looking wrist offset.
    """
    sid = env.robots[0].eef_site_id["right"]
    return (np.array(env.sim.data.site_xpos[sid]),
            np.array(env.sim.data.site_xmat[sid]).reshape(3, 3))


def _osc(env):
    return env.robots[0].part_controllers["right"]


def servo_action(env, target_pos, target_mat, gripper):
    """One 12-vector driving the eef toward a world-frame pose."""
    osc = _osc(env)
    p_eef, R_eef = eef_pose(env)
    origin_ori = np.array(osc.origin_ori)

    # position error, rotated into the controller's reference frame
    d_pos = origin_ori.T @ (np.asarray(target_pos) - p_eef)

    # orientation error. compute_goal_ori pre-multiplies in the reference frame:
    #     goal_ori = R_err @ current_ori_in_ref
    # so R_err is the *left* error, conjugated from world into the reference frame.
    W = np.asarray(target_mat) @ R_eef.T
    R_err = origin_ori.T @ W @ origin_ori
    d_rot = T.quat2axisangle(T.mat2quat(R_err))

    a = np.zeros(ACTION_DIM, dtype=np.float64)
    a[0:3] = np.clip(d_pos / POS_SCALE, -1.0, 1.0)
    a[3:6] = np.clip(d_rot / ROT_SCALE, -1.0, 1.0)
    a[6] = gripper
    # base_motion stays zero (base is fixed by design) and control_mode stays 0.0 ->
    # base_mode -1 -> goal_update_mode "achieved". Sending >0.5 would switch the goal to
    # update against the *desired* pose, turning the saturated term into an integrator.
    return a, float(np.linalg.norm(d_pos)), float(np.linalg.norm(d_rot))


class GraspExecutor:
    """
    Open-loop-ish grasp: pre-grasp -> approach -> close -> lift -> hold.

    `step()` returns one action per call so the caller keeps ownership of the env loop,
    the step budget and the scoring -- the same shape as the diffusion policy's action
    stream, which is what lets the evaluator treat the two arms identically.
    """

    STAGES = ("pre_grasp", "approach", "close", "lift", "hold", "done")

    def __init__(self, grasp_pos, grasp_mat, approach, standoff=STANDOFF,
                 lift_height=LIFT_HEIGHT):
        self.grasp_pos = np.asarray(grasp_pos, dtype=np.float64)
        self.grasp_mat = np.asarray(grasp_mat, dtype=np.float64)
        # Approach points from the gripper into the scene, so the standoff is *minus* it.
        self.approach = np.asarray(approach, dtype=np.float64)
        self.pre_pos = self.grasp_pos - standoff * self.approach
        # Lift along world +z, not along -approach: the scoring predicate
        # (GraspMoveDetector, MOVE_DZ) is a height test, and retreating along a near
        # horizontal approach would move the object 15 cm without changing its height.
        self.lift_pos = self.grasp_pos + np.array([0.0, 0.0, lift_height])

        self.stage = "pre_grasp"
        self.stage_step = 0
        self.stage_steps = {}
        self.stage_err = {}
        self.failure = None
        self.jammed = False
        self._ok_run = 0
        self._recent = []
        self._jam = []

    # -- helpers -------------------------------------------------------------------
    def _advance(self, env, pos_err, rot_err):
        self.stage_steps[self.stage] = self.stage_step
        self.stage_err[self.stage] = {"pos": round(pos_err, 4), "rot": round(rot_err, 4)}
        self.stage = self.STAGES[self.STAGES.index(self.stage) + 1]
        self.stage_step = 0
        self._ok_run = 0
        self._recent = []
        self._jam = []

    def _stalled(self, p_eef):
        """True when the eef has stopped moving -- i.e. something is physically blocking it.

        This is the real exit condition for `approach`: contact with the object or a shelf
        blocks the arm while the position error stays large, and without this test the
        whole stage cap gets burned on a grasp that was already in position.
        """
        self._recent.append(np.asarray(p_eef, dtype=np.float64))
        if len(self._recent) <= STALL_WINDOW:
            return False
        self._recent.pop(0)
        return float(np.linalg.norm(self._recent[-1] - self._recent[0])) < STALL_EPS

    # -- main ----------------------------------------------------------------------
    def step(self, env):
        """Next action, or None when the sequence is finished."""
        if self.stage == "done":
            return None

        if self.stage == "pre_grasp":
            tgt, grip = self.pre_pos, GRIP_OPEN
        elif self.stage == "approach":
            tgt, grip = self.grasp_pos, GRIP_OPEN
        elif self.stage == "close":
            tgt, grip = self.grasp_pos, GRIP_CLOSE
        elif self.stage == "lift":
            tgt, grip = self.lift_pos, GRIP_CLOSE
        else:  # hold
            tgt, grip = self.lift_pos, GRIP_CLOSE

        a, pos_err, rot_err = servo_action(env, tgt, self.grasp_mat, grip)
        p_eef, _ = eef_pose(env)
        self.stage_step += 1
        cap = STAGE_CAPS[self.stage]

        if self.stage == "pre_grasp":
            if pos_err < POS_TOL and rot_err < ROT_TOL:
                self._ok_run += 1
                if self._ok_run >= SETTLE:
                    self._advance(env, pos_err, rot_err)
            else:
                self._ok_run = 0
            # Wedged against something: stop early rather than spend the whole cap. Only
            # once past the initial acceleration, so a standing start is not read as a jam.
            self._jam.append(np.asarray(p_eef, dtype=np.float64))
            if len(self._jam) > JAM_WINDOW:
                self._jam.pop(0)
                if (self.stage == "pre_grasp" and self.stage_step > JAM_WINDOW
                        and float(np.linalg.norm(self._jam[-1] - self._jam[0])) < JAM_EPS
                        and pos_err > 0.05):
                    self.failure = "unreachable"
                    self.jammed = True
                    self.stage_steps[self.stage] = self.stage_step
                    self.stage_err[self.stage] = {"pos": round(pos_err, 4),
                                                  "rot": round(rot_err, 4)}
                    self.stage = "done"
                    return a
            if self.stage == "pre_grasp" and self.stage_step >= cap:
                # Far away and out of time means the pose was never reachable; close but
                # out of time is just slow, so fall through and try the grasp anyway.
                if pos_err > 0.05:
                    self.failure = "unreachable"
                    self.stage_steps[self.stage] = self.stage_step
                    self.stage_err[self.stage] = {"pos": round(pos_err, 4),
                                                  "rot": round(rot_err, 4)}
                    self.stage = "done"
                else:
                    self._advance(env, pos_err, rot_err)

        elif self.stage == "approach":
            touching = SU.touches_obj(env)
            if pos_err < APPROACH_POS_TOL or touching or self._stalled(p_eef):
                self._advance(env, pos_err, rot_err)
            elif self.stage_step >= cap:
                self._advance(env, pos_err, rot_err)

        else:
            # close / lift / hold are purely time-based. In particular `lift` is NOT cut
            # short when the object is not held: a failed grasp still has to spend the lift
            # so that GraspMoveDetector sees the same window it would on a success, and
            # `still_holding` is read at the true final step.
            if self.stage_step >= cap:
                self._advance(env, pos_err, rot_err)

        return a
