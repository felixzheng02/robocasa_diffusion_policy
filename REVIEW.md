# How to review this

A guide to checking the grasp-detector arm: read it, run it, and decide whether to believe
the numbers. Ordered so you can stop early if something is wrong.

Budget: ~15 min to read the core, ~30 min to reproduce the verifications, ~20 min for a
smoke eval. The heavy setup (conda env, CUDA extension, weights) is **already done**.

Everything lives on branch `worktree-anygrasp-pick`, pushed. 17 files, 3,756 lines, all new
— **no existing file is modified**, so nothing you already rely on can have broken.

```bash
cd /home/felix/Desktop/robocasa_sim/robocasa_diffusion_policy/.claude/worktrees/anygrasp-pick
git log --oneline main..HEAD          # 7 commits
git diff --stat main..HEAD
```

---

## 1. Read the code (~15 min)

In this order. The first three are where the real thinking is; the rest is plumbing.

| # | file | what to check |
|---|---|---|
| 1 | `grasp_perception.py` | **The frame conversions.** `GRASP_TO_EEF`, `grasp_to_world`, and `select_grasp`'s filters. This is where a silent wrong answer would live. |
| 2 | `grasp_executor.py` | **The servo law.** The module docstring derives why `clip(err/0.05, ±1)` is a saturated unity-gain term. Check that derivation — everything downstream rests on it. |
| 3 | `grasp_geometry.py` | The oracle grasp — my ground-truth baseline. If this is weak, my "upper bound" is not an upper bound and every comparison to it misleads. |
| 4 | `eval_anygrasp_pick.py` | The rollout loop and the `outcome` attribution rules. |
| 5 | `grasp_wire.py`, `grasp_server.py`, `grasp_agent.py` | The service. `grasp_wire.py` is imported by *both* conda envs so the schema cannot drift. |
| 6 | `check_grasp_geometry.py`, `check_grasp_executor.py` | The two verification harnesses. |

Skim `GRASP_README.md` for the runbook and the findings table.

---

## 2. Verify my frame claims yourself (~2 min)

I claim the gripper's approach axis is **+z** and its closing axis **±x**, measured in a live
env rather than taken from docs — and the whole GraspNet→robosuite mapping depends on it.
Don't take my word for it:

```bash
MUJOCO_GL=egl /home/felix/miniforge3/envs/robocasa_dp/bin/python - <<'EOF'
import numpy as np, robocasa
from robocasa.utils.env_utils import create_env
env = create_env(env_name="PickPlaceCounterToCabinet", split="pretrain", seed=100000)
env.reset(); sim = env.sim
sid = env.robots[0].eef_site_id["right"]
R = sim.data.site_xmat[sid].reshape(3,3); p = sim.data.site_xpos[sid]
g = env.robots[0].gripper["right"]
mean = lambda k: np.array([sim.data.geom_xpos[sim.model.geom_name2id(n)]
                           for n in g.important_geoms[k]]).mean(0)
lp, rp = mean("left_fingerpad"), mean("right_fingerpad")
hp = sim.data.body_xpos[sim.model.body_name2id("robot0_right_hand")]
print("approach in SITE frame:", np.round(R.T @ ((p-hp)/np.linalg.norm(p-hp)), 2))
print("closing  in SITE frame:", np.round(R.T @ ((lp-rp)/np.linalg.norm(lp-rp)), 2))
print("grip_site vs fingerpad midpoint (m):", round(float(np.linalg.norm(p-(lp+rp)/2)), 4))
env.close()
EOF
```

Expect `approach ≈ [0,0,1]`, `closing ≈ [±1,0,0]`, offset ≈ 0.0036 m. If any of those differ,
`GRASP_TO_EEF` in `grasp_perception.py` is wrong and every grasp is rotated.

---

## 3. Run the two verification harnesses

These gate everything. The project rule is that a low score must never be ambiguous between
a bad model and a broken harness.

```bash
# perception: is the point cloud metric, aligned, and object-scale?  (~3 min)
MUJOCO_GL=egl /home/felix/miniforge3/envs/robocasa_dp/bin/python check_grasp_geometry.py \
  --tasks PickPlaceCounterToCabinet PickPlaceCabinetToCounter PickPlaceCounterToOven --n 1
```

Expect **PASS** on all three, with `cross=` values around 0.003–0.022. That cross-camera
number is the strongest check here and needs no ground truth: two independently posed cameras
unprojecting the object to the same point is something essentially no flip, transpose or
double-applied axis correction survives.

```bash
# motion: can the executor pick things up given a ground-truth pose? (~10 min)
MUJOCO_GL=egl /home/felix/miniforge3/envs/robocasa_dp/bin/python check_grasp_executor.py \
  --tasks PickPlaceCounterToOven PickPlaceMicrowaveToCounter --n 3
```

Expect `CounterToOven` near 1.00 and `MicrowaveToCounter` at 0.00. **That split is the point** —
open surfaces work, enclosed fixtures fail on `unreachable` because the base is fixed. It
bounds what any detector can score on those tasks.

---

## 4. Run the detector eval

The server must be started **first** — it preallocates GPU memory.

```bash
# terminal 1
cd /home/felix/Desktop/robocasa_sim/robocasa_diffusion_policy/.claude/worktrees/anygrasp-pick
./serve_grasp.sh
# wait for "Uvicorn running on http://127.0.0.1:8100"
curl -s localhost:8100/health | python3 -m json.tool
```

Check `"random_weights": false` and `"sha256": "60680087c61cba2b"`. If `random_weights` is
true the weights are missing and the eval will refuse to score — by design.

```bash
# terminal 2  (~20 min for 12 rollouts)
MUJOCO_GL=egl /home/felix/miniforge3/envs/robocasa_dp/bin/python eval_anygrasp_pick.py \
  --tasks PickPlaceCounterToOven PickPlaceCounterToSink PickPlaceSinkToCounter \
          PickPlaceCounterToCabinet \
  --num_rollouts 3 --output /tmp/my_eval.json --video_dir /tmp/grasp_vids --video_n 1
```

Compare against the oracle arm, which uses the **same executor and scoring** but ground-truth
poses — so the difference isolates grasp synthesis:

```bash
MUJOCO_GL=egl /home/felix/miniforge3/envs/robocasa_dp/bin/python eval_anygrasp_pick.py \
  --tasks PickPlaceCounterToOven --num_rollouts 3 --pose_source oracle --output /tmp/oracle.json
```

---

## 5. Inspect results

```bash
python3 -c "
import json; d=json.load(open('/tmp/my_eval.json'))
print(json.dumps(d['summary'],indent=1))
for t,rs in d['rollouts'].items():
  for r in rs:
    print(f\"{t[:22]:22s} {r['outcome']:20s} raw={r['n_grasps_raw']:3d} kept={r['n_grasps_kept']:2d} \"
          f\"px={r['mask_px'].get(r['camera'],0):5d} obj={r['obj'][:14]:14s} d={r['grasp_to_obj_dist']}\")"
```

**Read `outcome`, not the average.** The whole point of the attribution is that it separates
causes the headline number cannot:

| outcome | meaning |
|---|---|
| `no_grasp_proposed` | nothing survived targeting/filtering — **currently the dominant bucket, and it is my selection layer's fault, not the network's** (`raw` shows the detector returned 19–64) |
| `unreachable` | a grasp was found on the object but the arm could not reach it |
| `executed_no_contact` | reached the pose, closed on nothing |
| `grasped_then_dropped` | grasped and lifted 2 cm, then lost it |

The output JSON keeps `pick_eval_e90.json`'s shape, so:

```bash
python3 -c "
import json
a=json.load(open('/tmp/my_eval.json'))['summary']
b=json.load(open('/home/felix/Desktop/robocasa_sim/robocasa_diffusion_policy/pick_eval_e90.json'))['summary']
for t in a:
  if t in b: print(f'{t[:30]:30s} grasp={a[t][\"pick_success\"]:.2f}  dp={b[t][\"pick_success\"]:.2f}')"
```

Videos land in `/tmp/grasp_vids` with `OK`/`FAIL` in the filename. Watch a `FAIL` — it is
usually obvious within two seconds whether the arm went somewhere sensible.

---

## 6. Is the comparison to the DP baseline even valid?

It is only meaningful if my env construction samples the *same scenes* the diffusion-policy
eval did. I reasoned it does — `create_env` then exactly one `reset()`, same seeds — but
reasoning is not evidence, so I measured it: object position identical to `0.000000` m and the
same sampled object, across two tasks × two seeds, comparing my path against the baseline's
`RobomimicImageWrapper` path.

Re-run it yourself if you want (`_probe_scenes.py` was throwaway, but the check is four lines):
build a scene both ways at one seed and compare `SU.obj_pos` and `make_skill_slots(...)["obj"]`.
If those ever diverge, every cross-comparison in this work is void.

---

## 7. Where to be sceptical

Ranked by how much they'd change your conclusions. I'd want a reviewer to push on all of these.

1. **The selection ranking is guesswork.** `score + 0.6·downward − 0.25·reorient` in
   `grasp_perception.select_grasp` has no principled basis; I picked the weights. Likewise
   `WORKSPACE_R=0.15`, `MAX_OBJ_DIST=0.04`, `MAX_TRUE_WIDTH=0.07`. These are the current
   bottleneck and they should be replaced by IK-based reachability, not re-tuned.

2. **n=3 per task is noise-dominated.** Per-task rates step by 0.33, and I observed a *looser*
   filter producing *more* rejections (6→8) because cuDNN nondeterminism flips borderline
   candidates. Do not read anything into small differences, mine or yours.

3. **The oracle arm is my own construction**, so "the detector is below the oracle" partly
   measures how good my oracle is. It scores 0.500 (n=18), which is not a strong ceiling.

4. **Comparability is arguable.** My arm gets an oracle segmentation mask (which object to
   grasp); the DP policy gets a slot embedding. I think those are comparable forms of task
   conditioning — but my arm also gets one open-loop attempt with no visual servoing and a
   fixed base, which the policy does not suffer. Reasonable people could weigh this differently.

5. **The server self-test is weak.** It only requires ≥1 grasp on a synthetic box, and it
   currently returns exactly 1. It catches a catastrophically wrong checkpoint path, nothing
   subtler.

6. **`collision_thresh` defaults to 0**, i.e. server-side collision filtering is off. I justify
   this (the client's object-cloud check is better informed) and measured that turning it off
   changed nothing, but it does mean nothing rejects a grasp that would drive through a shelf.

7. **Only 4 of 18 tasks have been evaluated with the detector.** No full sweep has run.

---

## 8. If you want to undo any of it

Nothing outside the branch was touched except `CLAUDE.md` (workspace root, not in any repo)
and the new `third_party/` directory. To drop the code entirely:

```bash
git branch -D worktree-anygrasp-pick
git push origin --delete worktree-anygrasp-pick
rm -rf /home/felix/Desktop/robocasa_sim/third_party
/home/felix/miniforge3/bin/conda env remove -n grasp
```
