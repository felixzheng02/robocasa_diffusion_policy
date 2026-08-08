"""
The grasp service's wire format. Imported by BOTH conda envs.

This module must stay importable from `grasp` (py3.10 / numpy 1.26) and from
`robocasa_dp` (py3.11 / numpy 2.2.5), so it may depend on numpy, base64 and json and
nothing else. One definition of the schema, no drift between client and server -- the same
argument that keeps the split criterion in robocasa/utils/skill_utils.py rather than
duplicated between the splitter and the evaluator.

Why the payload is a point cloud rather than RGB-D + intrinsics
---------------------------------------------------------------
Both detectors consume an (N,3) camera-frame array natively: AnyGrasp's
`get_grasp(points, ...)` never sees an image, and graspnet-baseline's network input is
`end_points['point_clouds']`. Sending RGB-D would force each backend to re-derive a cloud,
and any difference in that derivation becomes a silent behavioural difference between
backends -- exactly what the shared interface exists to prevent. It also keeps the
depth->cloud conversion on the client, inside robosuite, where it can be checked against
ground truth (see check_grasp_geometry.py).

The response is the raw graspnetAPI 17-column array, which both detectors already emit.
"""

import base64
import numpy as np

SCHEMA = 1

# graspnetAPI Grasp layout. Verified against graspnetAPI/grasp.py property getters.
COLUMNS = ["score", "width", "height", "depth",
           "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22",
           "tx", "ty", "tz", "object_id"]
GRASP_ARRAY_LEN = 17
SCORE, WIDTH, HEIGHT, DEPTH = 0, 1, 2, 3
ROT = slice(4, 13)
TRANS = slice(13, 16)
OBJECT_ID = 16

# The convention block travels with every response and the client asserts it. A future
# backend that means something different by R has to say so, and the assert fires instead
# of the arm quietly grasping at 90 degrees.
CONVENTION = {
    "approach_axis": "R[:,0]",      # points from the gripper into the scene
    "closing_axis": "R[:,1]",
    "translation": "grasp_center",
    "depth_along": "R[:,0]",
    "rotation_layout": "row_major_3x3",
}


def encode(arr):
    """float32 ndarray -> a JSON-safe dict."""
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return {"dtype": "float32",
            "shape": list(arr.shape),
            "b64": base64.b64encode(arr.tobytes()).decode("ascii")}


def decode(d):
    """The inverse of `encode`."""
    if d is None:
        return None
    raw = base64.b64decode(d["b64"])
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def split_rows(gg):
    """(M,17) array -> a list of per-grasp dicts, in the frame the array was expressed in."""
    out = []
    for row in np.atleast_2d(gg):
        out.append({
            "score": float(row[SCORE]),
            "width": float(row[WIDTH]),
            "height": float(row[HEIGHT]),
            "depth": float(row[DEPTH]),
            "rotation": np.asarray(row[ROT], dtype=np.float64).reshape(3, 3),
            "translation": np.asarray(row[TRANS], dtype=np.float64),
            "object_id": int(row[OBJECT_ID]),
        })
    return out
