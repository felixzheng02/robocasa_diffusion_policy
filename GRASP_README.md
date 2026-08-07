# Grasp-detector arm for the PickPlace *pick* phase

An analytic 6-DoF grasp detector as an alternative to the slot-conditioned diffusion pick
policy, scored on the same 18 tasks, the same seeds and the same `GraspMoveDetector`, so the
number is directly comparable to `pick_eval_e90.json` (**0.333 `pick_success` / 0.322
`still_holding`** over 180 rollouts).

Backend is **graspnet-baseline** today — same lab as AnyGrasp, open weights, and it avoids
MinkowskiEngine entirely. The **AnyGrasp SDK** drops in behind the same HTTP interface with
no client change once a license arrives (`GRASP_BACKEND=anygrasp`).

## Runbook

```bash
./setup_grasp_env.sh          # conda env + CUDA 11.8 toolkit + torch + clone  (~20 min)
./build_pointnet2.sh          # the CUDA extension; knn is deliberately NOT built
./fetch_checkpoint.sh rs      # weights (see "Known blocker")

./serve_grasp.sh              # start FIRST, port 8100 (coexists with vLLM on 8000)

# verification — run BOTH before believing any number
MUJOCO_GL=egl python check_grasp_geometry.py --n 2      # perception
MUJOCO_GL=egl python check_grasp_executor.py --n 3      # motion (the gate)

MUJOCO_GL=egl python eval_anygrasp_pick.py --num_rollouts 10 --output grasp_pick_eval.json
MUJOCO_GL=egl python eval_anygrasp_pick.py --pose_source oracle   # upper bound, no server
```

`--pose_source oracle` builds the grasp from ground-truth geometry and needs no detector at
all. It is the arm the detector is measured against, and it uses the identical executor and
scoring, so the difference between the two is attributable to grasp synthesis alone.

## Weights

Present and verified: `checkpoint-rs.tar`, sha `60680087c61cba2b`, epoch 18, loss 0.561,
162 tensors / 1.03 M params, **exact** state-dict key match. That key match is also the
cheapest confirmation that `num_view=300 / num_angle=12 / num_depth=4 / cylinder_radius=0.05`
are right — a wrong hyperparameter shows up as mismatched keys, not as bad grasps.

Getting them is the one manual step. Upstream hosts them only on Google Drive and Baidu Pan,
and Drive quota-refuses the file to every programmatic client (gdown 5.x and 6.x, a
hand-rolled confirm-token curl, the Kinect alternate); no HuggingFace or fork mirror exists.
Download in a browser from
<https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view>.

**Do not let it be extracted.** A torch `.tar` is internally a ZIP, so archive managers
unpack it into `archive/data.pkl` + `archive/data/*`, which `torch.load` cannot read. If
`third_party/checkpoints/checkpoint-rs/` is a *directory*, that is what happened — copy the
original file back.

`GRASP_ALLOW_RANDOM_WEIGHTS=1 ./serve_grasp.sh` exercises everything except the weights;
`/health` reports `random_weights: true` and the eval refuses to score against it.

## Current result, and why it is not yet a verdict

**0.083 `pick_success` / 0.000 `still_holding`** (n=12, 4 tasks × 3) against the diffusion
policy's 0.333 / 0.322. This is preliminary and the pipeline is under-tuned.

The attribution is the useful part: `no_grasp_proposed` 6–8/12, `unreachable` 2–4/12 — and in
*every* `no_grasp_proposed` rollout the detector had returned 19–64 grasps that the selection
layer then threw away. **The bottleneck is selection, not the network.**

Four measured fixes took usable-grasp scenes from 4/12 to 7/12:

| fix | why |
|---|---|
| crop the cloud to ±0.15 m around the object | uncropped, the *median* grasp landed 0.39 m away — the detector spent its capacity on counters and walls |
| filter width at Panda's real 0.08, check the object cloud instead | GraspNet's `width` is inflated 1.2× and clamped at 0.1; a 0.075 cutoff killed every on-object grasp |
| target on the **seed** point, not `t + depth·approach` | `pred_decode` sets `grasp_center = fp2_xyz`, sampled *from the input cloud*, so the seed coincides with object points to 0.0000 m |
| server-side collision filtering off | redundant with, and worse-informed than, the client's object-cloud check; cut 64 candidates to 1–3 on sparse clouds (though on its own it changed nothing) |

Remaining failures are scenes where the detector genuinely returns few candidates (croissant
3, liquor 1) or the object is near-invisible (avocado, 559 points).

### The biggest remaining gap: no IK

There is **no IK solve, no reachability check and no path planning**. OSC resolves Cartesian
commands through the Jacobian, so kinematics are not absent, but the executor drives a
straight Cartesian line to the pose and only discovers failure after burning the stage
budget — that is the `unreachable` bucket.

`robosuite/utils/ik_utils.py:IKSolver` (damped least-squares, accepts base-frame targets) is
the principled fix, and the bigger win is at **selection**: ask which of the 64 candidates
have a joint solution rather than ranking them with the current hand-tuned
`score + 0.6·downward − 0.25·reorient` heuristic. It can serve purely as an offline
feasibility oracle while execution still goes through the stock OSC servo, so the env stays
byte-identical and the baseline comparison survives.

**Do not tune selection at n=3 per task.** Loosening a filter was observed to *increase*
`no_grasp_proposed` (6 → 8), because cuDNN nondeterminism flips borderline candidates.

## Why a separate conda env

Identical reasoning to `serve_vlm.sh`. The `grasp` env pins **numpy<2** and
**torch 2.0.1+cu118** so a 2019-vintage CUDA extension will compile, while
`robocasa/__init__.py` hard-asserts `numpy==2.2.5`. The two cannot coexist, and a detector
crash must not take a multi-hour sweep with it. Nothing imports across the boundary — the
eval talks HTTP. `grasp_wire.py` is the one module imported by *both* envs (numpy + base64 +
json only), so the schema cannot drift.

`knn` is deliberately not built: it vendors KNN_CUDA, which uses the TH/THC C API PyTorch
removed in 1.11. It is only reachable through `utils/label_generation.py`
(training/evaluation), but `models/graspnet.py` imports that module at scope, so
`grasp_server.py` installs a stub that **raises** — a wrong assumption surfaces as a
traceback, not a degraded detector.

## Findings that are load-bearing

Each of these silently produces a plausible wrong answer, so none should be "simplified" back:

| finding | what happens if ignored |
|---|---|
| `grip_site`'s approach axis is **+z**, closing axis **±x** (measured, not from docs) | GraspNet's `R[:,0]`/`R[:,1]` map is a *cyclic permutation*; get it wrong and the arm grasps at 90° with no error |
| `grip_site` is the fingerpad midpoint (~3.6 mm) | no TCP offset is needed; adding one puts the jaws past the object |
| mask by `geom_bodyid`, **not** `contact_geoms` | those are *collision* geoms while segmentation renders *visual* ones — every name resolves and the mask comes back **empty** |
| 2 px erosion + depth-median rejection | silhouette pixels smear the cloud down the viewing ray: a 5 cm object measured **63 cm** across |
| adaptive camera choice | `CounterToMicrowave` shows **0** object pixels from `agentview_right`; a fixed camera loses whole tasks |
| gripper closes only above **0.5** | the conventional −1/+1 leaves it open forever, silently |
| symmetry pick (`R` vs `R·Rz(π)`) | a parallel jaw is symmetric; picking the nearer twin moved oracle pick rate **0.222 → 0.556** (n=9, 3 tasks) by removing wrist-reorientation timeouts — `unreachable` failures fell 4 → 1 |
| top-down preferred over side grasps | gating top-down on ray-measured clearance was tried and made it **worse** (0.556 → 0.389, n=18); side grasps are a fallback, not an alternative |

## Oracle executor status — read this before interpreting a detector number

Current, over 6 tasks × 3 seeds (n=18): **`fired` 0.500 / `still_holding` 0.444**.

The average is the least interesting part of it. The split is not:

| | oracle `fired` |
|---|---|
| `CounterToOven` | 1.00 |
| `SinkToCounter` | 1.00 |
| `CounterToSink` | 0.67 |
| `CounterToCabinet` | 0.33 |
| `DrawerToCounter` | **0.00** |
| `MicrowaveToCounter` | **0.00** |

Open surfaces are solved; **enclosed fixtures are not**, and the failures there are almost
entirely `unreachable` — the arm cannot get the wrist to the pre-grasp pose with the base
fixed. All three `MicrowaveToCounter` failures show pre-grasp position errors of
0.085–0.222 m, and the object sits at z = 1.17 m inside a wall-mounted unit. Ray casting
confirms this is *not* an occluded approach corridor (0.339 m of clearance straight up); it
is arm reach.

Two consequences:

1. **This bounds what any detector can score here.** A detector arm cannot beat the oracle
   on tasks the executor cannot reach, so `unreachable` should be read as a property of the
   base-fixed constraint, not of the grasp detector. The `outcome` histogram is what keeps
   the two separable.
2. **Lifting the fixed-base restriction is the single highest-value follow-up** — it targets
   the dominant failure bucket directly. That is a deliberate scope choice, not an oversight.

Caveat: n=3 per task, so per-task rates move in steps of 0.33 and small differences between
executor variants are inside the noise. The open-surface / enclosed-fixture split, however,
reproduced across all three measurement rounds.

## Verified

- **Perception** — cross-camera unprojection agrees to **0.003–0.022 m**. Two independently
  posed cameras landing on the same point is a check that needs no ground truth and that
  essentially no flip, transpose or double-correction survives.
- **Motion** — the servo converges to **0.1 mm / 0.0005 rad**, and a successful oracle pick
  lifts the object a clean **+0.149 m**.
- **Wire** — round trips between numpy 2.2.5 (client) and numpy 1.26 (server); the response's
  `convention` block is asserted on every call, so a backend that redefines `R` fails loudly.

## Reading the output

`grasp_pick_eval.json` keeps `pick_eval_e90.json`'s shape and key names so the two diff field
by field, and adds `outcome` per rollout:

```
success · grasped_then_dropped · executed_no_contact · unreachable · no_grasp_proposed
```

That attribution is the point of the exercise — the baseline's numbers cannot separate "no
grasp was found" from "a good grasp was found and the arm could not reach it".

**Comparability, honestly.** The grasp arm has an *oracle advantage* (a segmentation mask
tells it which object to grasp, mirroring the policy's slot embedding) and a real
*disadvantage* (one open-loop attempt, no visual servoing, base fixed). `pick_success` is
the comparable metric; `still_holding` separates a real grasp from a nudge.
