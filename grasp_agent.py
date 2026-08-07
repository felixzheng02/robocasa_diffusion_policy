"""
HTTP client for the grasp detection service. Runs in robocasa_dp.

Mirrors vlm_agent.py deliberately, including the two habits that matter in a long sweep:
`probe()` is called once before the sweep so an unreachable server fails immediately rather
than 200 rollouts in, and every failure inside `detect()` collapses to None so a flaky
request can never raise out of a rollout.
"""

import os

import numpy as np
import requests

import grasp_wire as W

BASE_URL = os.environ.get("GRASP_URL", "http://127.0.0.1:8100")
TIMEOUT = float(os.environ.get("GRASP_TIMEOUT", "120"))


def probe():
    """Server's backend id, or None if it is not reachable/healthy."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def detect(points, colors=None, top_k=64, collision_thresh=0.01):
    """
    Grasps for a camera-frame cloud, as an (M,17) graspnetAPI array. None on any failure.

    The response's `convention` block is asserted rather than assumed: a backend that means
    something different by R has to say so, and this fires instead of the arm silently
    grasping at 90 degrees to the intended axis.
    """
    body = {"schema_version": W.SCHEMA, "frame": "camera",
            "points": W.encode(np.asarray(points).reshape(-1, 3)),
            "top_k": int(top_k), "collision_thresh": float(collision_thresh)}
    if colors is not None:
        body["colors"] = W.encode(np.asarray(colors).reshape(-1, 3))
    try:
        r = requests.post(f"{BASE_URL}/detect", json=body, timeout=TIMEOUT)
        r.raise_for_status()
        out = r.json()
    except Exception:
        return None

    if out.get("convention") != W.CONVENTION:
        raise RuntimeError(
            f"grasp server changed its pose convention: {out.get('convention')} "
            f"!= {W.CONVENTION}. Refusing to convert poses under a stale assumption.")
    if out.get("frame") != "camera":
        raise RuntimeError(f"expected camera-frame grasps, got {out.get('frame')!r}")
    return W.decode(out["grasps"])
