"""
The grasp detection service. Runs in the `grasp` conda env, never in robocasa_dp.

Same shape as serve_vlm.sh / vlm_agent.py and for the same reason: this env pins
numpy<2 and torch 2.0.1+cu118 to compile a 2019-vintage CUDA extension, while robocasa
hard-asserts numpy==2.2.5. Nothing imports across the boundary -- the eval talks HTTP.

Two backends behind one interface, selected by $GRASP_BACKEND:

  graspnet-baseline   open weights, no license, no MinkowskiEngine  (default, works today)
  anygrasp            the licensed SDK                              (drops in unchanged)

Both consume an (N,3) camera-frame cloud and emit the graspnetAPI 17-column array, so the
client never learns which one answered.

Deliberately does NOT depend on graspnetAPI or open3d. graspnetAPI is used upstream only
for GraspGroup/.nms()/.sort_by_score() -- a thin view over an (N,17) array, an argsort, and
a small NMS -- and it drags in open3d, sklearn, autolab_core and a compiled cython ext.
Reimplementing those few lines in numpy keeps the env to torch+numpy+fastapi, which is the
difference between an env that builds and one that fights.
"""

import base64
import hashlib
import os
import sys
import time
import types

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

import grasp_wire as W

GRASPNET_ROOT = os.environ.get(
    "GRASPNET_ROOT", "/home/felix/Desktop/robocasa_sim/third_party/graspnet-baseline")
CHECKPOINT = os.environ.get(
    "GRASP_CHECKPOINT",
    "/home/felix/Desktop/robocasa_sim/third_party/checkpoints/checkpoint-rs.tar")
BACKEND = os.environ.get("GRASP_BACKEND", "graspnet-baseline")
NUM_POINT = 20000


# --------------------------------------------------------------------------------------
# graspnet-baseline import shim
# --------------------------------------------------------------------------------------
def _install_knn_stub():
    """
    Satisfy the `knn` import without building it.

    graspnet-baseline vendors KNN_CUDA, which uses the TH/THC C API that PyTorch removed
    in 1.11; it cannot compile against torch 2.x without a rewrite. It is also unnecessary
    here -- `knn` is reached only through utils/label_generation.py's
    `process_grasp_labels` / `match_grasp_view_and_label`, which serve training and
    evaluation. The inference path (GraspNet.forward in eval mode -> pred_decode) never
    calls them. The problem is purely that models/graspnet.py imports label_generation at
    module scope to get `batch_viewpoint_params_to_matrix`, which pred_decode *does* need.

    The stub raises rather than returning something plausible: if this assumption is ever
    wrong, the result is a loud traceback, not a silently degraded detector.
    """
    def _knn(*_a, **_k):
        raise RuntimeError(
            "knn was called -- it is stubbed because only training/eval need it. "
            "If inference now needs knn, build it with the THC->ATen substitutions.")

    mod = types.ModuleType("knn_modules")
    mod.knn = _knn
    sys.modules.setdefault("knn_modules", mod)


def _load_graspnet():
    """Import graspnet-baseline (flat sibling imports) and return a loaded eval model."""
    for sub in ("", "models", "utils", "dataset"):
        p = os.path.join(GRASPNET_ROOT, sub) if sub else GRASPNET_ROOT
        if p not in sys.path:
            sys.path.insert(0, p)
    _install_knn_stub()

    from graspnet import GraspNet, pred_decode  # noqa: E402  (needs sys.path first)

    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02,
                   hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # weights_only=False explicitly: the default flipped in torch 2.6 and would refuse
    # this checkpoint outright, so being explicit survives a future bump.
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, pred_decode, device


# --------------------------------------------------------------------------------------
# numpy replacements for the graspnetAPI bits we actually use
# --------------------------------------------------------------------------------------
def nms(gg, trans_thresh=0.03, rot_thresh=np.deg2rad(30.0), keep=200):
    """Greedy NMS over grasps: suppress a lower-scoring grasp that is both near in
    translation and near in rotation to one already kept."""
    if len(gg) == 0:
        return gg
    order = np.argsort(-gg[:, W.SCORE])
    gg = gg[order]
    t = gg[:, W.TRANS]
    R = gg[:, W.ROT].reshape(-1, 3, 3)

    kept = []
    for i in range(len(gg)):
        ok = True
        for j in kept:
            if np.linalg.norm(t[i] - t[j]) > trans_thresh:
                continue
            c = (np.trace(R[j] @ R[i].T) - 1.0) / 2.0
            if np.arccos(np.clip(c, -1.0, 1.0)) < rot_thresh:
                ok = False
                break
        if ok:
            kept.append(i)
        if len(kept) >= keep:
            break
    return gg[kept]


def collision_filter(gg, cloud, thresh=0.01, finger_h=0.02):
    """
    Drop grasps whose closed finger volume already contains scene points.

    Equivalent in spirit to upstream's ModelFreeCollisionDetector, done in numpy: put the
    cloud into each grasp's local frame and count points inside the swept jaw box.
    """
    if len(gg) == 0 or len(cloud) == 0:
        return gg
    out = []
    for row in gg:
        R = row[W.ROT].reshape(3, 3)
        t = row[W.TRANS]
        d, w = float(row[W.DEPTH]), float(row[W.WIDTH])
        local = (cloud - t) @ R          # world -> grasp frame (R's columns are the axes)
        inside = ((local[:, 0] > -0.02) & (local[:, 0] < d) &
                  (np.abs(local[:, 1]) < w / 2 + 0.005) &
                  (np.abs(local[:, 2]) < finger_h / 2))
        # points between the jaws are the *object*; collision means material outside the
        # opening but inside the finger sweep, so only count the shell just beyond width/2
        shell = ((local[:, 0] > -0.02) & (local[:, 0] < d) &
                 (np.abs(local[:, 1]) > w / 2) & (np.abs(local[:, 1]) < w / 2 + 0.01) &
                 (np.abs(local[:, 2]) < finger_h / 2))
        if inside.sum() > 0 and shell.sum() / max(len(cloud), 1) <= thresh:
            out.append(row)
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, W.GRASP_ARRAY_LEN),
                                                                 dtype=np.float32)


# --------------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------------
class GraspNetBaselineDetector:
    name = "graspnet-baseline"

    def __init__(self):
        self.net, self.pred_decode, self.device = _load_graspnet()

    def detect(self, points, colors=None, top_k=64, collision_thresh=0.01):
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(pts) == 0:
            return np.zeros((0, W.GRASP_ARRAY_LEN), dtype=np.float32)
        rng = np.random.default_rng(0)
        if len(pts) >= NUM_POINT:
            idx = rng.choice(len(pts), NUM_POINT, replace=False)
        else:
            idx = np.concatenate([np.arange(len(pts)),
                                  rng.choice(len(pts), NUM_POINT - len(pts), replace=True)])
        sampled = pts[idx]

        end_points = {"point_clouds": torch.from_numpy(sampled[None]).to(self.device)}
        with torch.no_grad():
            end_points = self.net(end_points)
            preds = self.pred_decode(end_points)
        gg = preds[0].detach().cpu().numpy().astype(np.float32)

        gg = gg[gg[:, W.SCORE] > 0]
        gg = nms(gg)
        if collision_thresh > 0:
            gg = collision_filter(gg, pts, thresh=collision_thresh)
        gg = gg[np.argsort(-gg[:, W.SCORE])][:top_k]
        return gg


class AnyGraspDetector:
    """
    Adapter for the licensed AnyGrasp SDK.

    Written now, as a stub, so the shape of the swap is visible from day one: when the
    license arrives this is the only file that changes, and the client is untouched.
    """
    name = "anygrasp"

    def __init__(self):
        from gsnet import AnyGrasp  # noqa: F401  (only importable with a valid license)
        cfg = types.SimpleNamespace(
            checkpoint_path=os.environ["ANYGRASP_CHECKPOINT"],
            max_gripper_width=0.08,   # Panda's limit, not AnyGrasp's 0.1 default
            gripper_height=0.03, top_down_grasp=False, debug=False)
        self.model = AnyGrasp(cfg)
        self.model.load_net()

    def detect(self, points, colors=None, top_k=64, collision_thresh=0.01):
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        cols = (np.asarray(colors, dtype=np.float32).reshape(-1, 3)
                if colors is not None else np.ones_like(pts) * 0.5)
        lims = [float(pts[:, 0].min()), float(pts[:, 0].max()),
                float(pts[:, 1].min()), float(pts[:, 1].max()),
                float(pts[:, 2].min()), float(pts[:, 2].max())]
        gg, _ = self.model.get_grasp(pts, cols, lims=lims,
                                     collision_detection=collision_thresh > 0)
        if gg is None or len(gg) == 0:
            return np.zeros((0, W.GRASP_ARRAY_LEN), dtype=np.float32)
        gg = gg.nms().sort_by_score()[:top_k]
        return np.asarray(gg.grasp_group_array, dtype=np.float32)


BACKENDS = {"graspnet-baseline": GraspNetBaselineDetector, "anygrasp": AnyGraspDetector}


# --------------------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------------------
app = FastAPI()
DETECTOR = None
CKPT_SHA = None


class DetectRequest(BaseModel):
    schema_version: int = W.SCHEMA
    frame: str = "camera"
    points: dict
    colors: dict | None = None
    top_k: int = 64
    collision_thresh: float = 0.01


@app.on_event("startup")
def _startup():
    global DETECTOR, CKPT_SHA
    DETECTOR = BACKENDS[BACKEND]()
    if os.path.exists(CHECKPOINT):
        h = hashlib.sha256()
        with open(CHECKPOINT, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        CKPT_SHA = h.hexdigest()[:16]

    # Self-test: a 6 cm box on a plane. Converts the worst failure available here -- a whole
    # sweep silently scoring `no_grasp_proposed` because a path was wrong -- into a server
    # that refuses to start.
    n = DETECTOR.detect(_selftest_cloud(), top_k=16)
    print(f"[grasp_server] backend={DETECTOR.name} selftest_grasps={len(n)}", flush=True)
    if len(n) == 0:
        raise RuntimeError("self-test found no grasps on a synthetic box -- refusing to serve")
    app.state.selftest = int(len(n))


def _selftest_cloud(n=20000):
    rng = np.random.default_rng(0)
    plane = np.c_[rng.uniform(-0.3, 0.3, n // 2), rng.uniform(-0.3, 0.3, n // 2),
                  np.full(n // 2, 0.6)]
    s = 0.03
    f = rng.uniform(-s, s, (n // 2, 3))
    f[:, 2] = np.abs(f[:, 2])
    box = f + np.array([0.0, 0.0, 0.6 - s])
    cloud = np.vstack([plane, box]).astype(np.float32)
    # camera looks down +z; keep everything in front of the camera
    return cloud


@app.get("/health")
def health():
    return {"backend": DETECTOR.name if DETECTOR else None,
            "checkpoint": CHECKPOINT, "sha256": CKPT_SHA,
            "device": str(next(DETECTOR.net.parameters()).device)
            if hasattr(DETECTOR, "net") else "unknown",
            "schema": W.SCHEMA,
            "selftest_grasps": getattr(app.state, "selftest", None)}


@app.post("/detect")
def detect(req: DetectRequest):
    assert req.frame == "camera", f"expected a camera-frame cloud, got {req.frame!r}"
    t0 = time.time()
    pts = W.decode(req.points)
    cols = W.decode(req.colors) if req.colors else None
    gg = DETECTOR.detect(pts, cols, top_k=req.top_k,
                         collision_thresh=req.collision_thresh)
    return {"schema": W.SCHEMA, "backend": DETECTOR.name, "frame": "camera",
            "grasps": W.encode(gg), "columns": W.COLUMNS, "convention": W.CONVENTION,
            "n_input_points": int(len(pts)),
            "latency_ms": round((time.time() - t0) * 1000, 1)}
