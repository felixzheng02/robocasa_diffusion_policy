"""
Is a grasp pose reachable? A damped least-squares IK solve used as a selection oracle.

Why this exists
---------------
The executor drives a straight Cartesian line to the grasp pose through the stock OSC
controller. That controller resolves Cartesian commands through the Jacobian, so kinematics
are not absent -- but there is no reachability *check*. An unreachable pose is discovered
only after burning the whole pre-grasp stage budget, which showed up as the `unreachable`
bucket (2-4 of 12 rollouts), and every one of those had a grasp genuinely on the object.

Worse, it corrupted *selection*: candidates were ranked by a hand-tuned
`score + 0.6*downward - 0.25*reorient` heuristic that was only ever a proxy for "can the arm
get there". Asking an IK solver is the actual question.

Why not robosuite's IKSolver
----------------------------
`robosuite/utils/ik_utils.py:IKSolver` is a *differential* solver built for streaming teleop
targets: one Jacobian step per call, and it hardcodes `actuator_ids = range(20)` for GR1 with
a TODO. It answers "what joint velocity moves toward this pose", not "does a solution exist".
The iterate-to-convergence loop below is ~40 lines and answers the question directly.

Never disturbs the simulation: all iteration happens on a scratch `MjData` seeded from the
live state, so the caller's env is untouched.
"""

import mujoco
import numpy as np

ARM_QPOS = np.arange(4, 11)      # robot0_joint1..7; verified against model.joint(name).qposadr
POS_TOL = 0.005                  # m
ROT_TOL = 0.10                   # rad
MAX_ITERS = 80
DAMPING = 1e-2
MAX_DQ = 0.30                    # rad per iteration, keeps the linearisation honest


class IKReach:
    """Reachability oracle for one scene. Build once per rollout, query per candidate."""

    def __init__(self, env):
        self.model = env.sim.model._model
        self.site_id = int(env.robots[0].eef_site_id["right"])
        self.dof = ARM_QPOS
        self.lo = self.model.jnt_range[self.model.dof_jntid[self.dof], 0]
        self.hi = self.model.jnt_range[self.model.dof_jntid[self.dof], 1]
        self._qpos0 = np.array(env.sim.data.qpos).copy()
        self._data = mujoco.MjData(self.model)
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def _seed(self, q_arm=None):
        d = self._data
        d.qpos[:] = self._qpos0
        if q_arm is not None:
            d.qpos[self.dof] = q_arm
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return d

    def _errors(self, d, target_pos, target_mat):
        p = d.site(self.site_id).xpos
        R = d.site(self.site_id).xmat.reshape(3, 3)
        e_pos = np.asarray(target_pos) - p
        # left rotation error, as an axis-angle vector
        Rerr = np.asarray(target_mat) @ R.T
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, Rerr.flatten())
        e_rot = np.empty(3)
        mujoco.mju_quat2Vel(e_rot, q, 1.0)
        return e_pos, e_rot

    def _iterate(self, target_pos, target_mat, q_arm=None):
        d = self._seed(q_arm)
        for _ in range(MAX_ITERS):
            e_pos, e_rot = self._errors(d, target_pos, target_mat)
            if np.linalg.norm(e_pos) < POS_TOL and np.linalg.norm(e_rot) < ROT_TOL:
                break
            mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.site_id)
            J = np.vstack([self._jacp[:, self.dof], self._jacr[:, self.dof]])
            e = np.concatenate([e_pos, e_rot])
            # damped least squares: dq = J^T (J J^T + lambda I)^-1 e
            JJt = J @ J.T + DAMPING * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            n = np.linalg.norm(dq)
            if n > MAX_DQ:
                dq *= MAX_DQ / n
            d.qpos[self.dof] = np.clip(d.qpos[self.dof] + dq, self.lo, self.hi)
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
        e_pos, e_rot = self._errors(d, target_pos, target_mat)
        return (float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot)),
                np.array(d.qpos[self.dof]))

    def solve(self, target_pos, target_mat, restarts=2, rng=None):
        """
        Best IK solution for a world-frame pose.

        Seeds from the arm's current configuration first -- that is the configuration the
        executor will actually servo from, so a solution found there is the one most likely
        to be reachable by a straight Cartesian path. Random restarts only run if that fails,
        because a solution in a wildly different branch of configuration space is one the
        Cartesian servo cannot get to anyway.
        """
        best = self._iterate(target_pos, target_mat)
        if best[0] < POS_TOL and best[1] < ROT_TOL:
            return {"ok": True, "pos_err": best[0], "rot_err": best[1], "qpos": best[2]}

        rng = rng or np.random.default_rng(0)
        for _ in range(restarts):
            q0 = rng.uniform(self.lo, self.hi)
            r = self._iterate(target_pos, target_mat, q_arm=q0)
            if r[0] < best[0]:
                best = r
            if best[0] < POS_TOL and best[1] < ROT_TOL:
                break
        return {"ok": bool(best[0] < POS_TOL and best[1] < ROT_TOL),
                "pos_err": best[0], "rot_err": best[1], "qpos": best[2]}

    def reachable(self, pos, mat, pre_pos=None):
        """
        True when the grasp pose is reachable -- and, if given, the pre-grasp pose too.

        Both matter: the executor visits the standoff pose first, and a grasp whose approach
        cannot be staged is no more useful than one that cannot be reached at all.
        """
        g = self.solve(pos, mat)
        if not g["ok"]:
            return False, g
        if pre_pos is not None:
            p = self.solve(pre_pos, mat)
            if not p["ok"]:
                return False, p
        return True, g
