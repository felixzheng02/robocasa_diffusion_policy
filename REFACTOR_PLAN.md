# Refactor: three projects + one simulator under `robocasa_sim/`

Option **A-split**: keep the four upstream forks as repos, move the VLM and grasp code out
of the diffusion-policy repo into their own repos, and add a directory layer.

## Target layout

```
robocasa_sim/
├── sim/                       # forks; internal layout untouched, remotes intact
│   ├── robosuite/             #   0 project commits
│   ├── robocasa/              #   + env_helpers.py (new), keeps skill_utils.py
│   └── robomimic/             #   1 commit, 7-line diffusers fix
├── diffusion_policy/          # existing repo, minus 24 files
├── vlm_planning/              # NEW repo   — depends on diffusion_policy
├── anygrasp/                  # NEW repo   — depends on sim ONLY
│   └── vendor/
│       ├── graspnet-baseline/ #   pinned clone (gitignored, own .git)
│       └── checkpoints/       #   checkpoint-rs.tar (gitignored, UNREPRODUCIBLE)
├── datasets/                  # 11 GB, unchanged
└── outputs/                   # 79 GB, unchanged
```

`third_party/` disappears; its contents move under `anygrasp/vendor/` since nothing else
uses them.

## Why there is no `common/` package

The only symbols the three projects share are `base_env` and `create_env`, and **neither has
any diffusion-policy content**:

- `create_env` is `gym.make(f"robocasa/{env_name}", split=..., seed=...)` — a wrapper over
  robocasa's own registration (`robocasa/__init__.py` → `wrappers/gym_wrapper.py:register`).
- `base_env` is an 11-line loop unwrapping gym wrappers until `_check_success` is found.

They belong in **robocasa**, which is what registers those envs. Moving them to
`sim/robocasa/robocasa/utils/env_helpers.py` — beside `skill_utils.py`, which all three
projects already import from robocasa — removes the last grasp→DP edge entirely.

`load_policy`, `obs_to_frame` and `stack_obs` stay in the DP repo. VLM keeps importing them,
which is correct: VLM orchestrates DP checkpoints and is legitimately a DP dependent.

## File inventory

**Stays in `diffusion_policy/`** (2 own files + 4 modified fork files + configs + results):
`eval_chained_pick_place.py`, `check_eval_harness.py`,
`diffusion_policy/config/task/robocasa/pretrain_{pick,place}_skill.yaml`,
modified `diffusion_policy/{dataset/lerobot_dataset.py, env/robomimic/robomimic_image_wrapper.py,
model/common/lr_scheduler.py, workspace/train_diffusion_transformer_hybrid_workspace.py}`,
`harness_check.json`, `pick_eval_e90.json`.

**Moves to `vlm_planning/`** (3 files): `vlm_agent.py`, `eval_agentic_pick_place.py`,
`serve_vlm.sh`.

**Moves to `anygrasp/`** (21 files): `grasp_{wire,server,agent,perception,geometry,executor,ik}.py`,
`eval_anygrasp_pick.py`, `check_{grasp_geometry,grasp_executor,pipeline}.py`,
`serve_grasp.sh`, `setup_grasp_env.sh`, `build_pointnet2.sh`, `fetch_checkpoint.sh`,
`GRASP_README.md`, `REVIEW.md`, `REFACTOR_PLAN.md`, and the result JSONs.

**Stays in `sim/robocasa/`**: `utils/skill_utils.py`, `utils/skill_dataset_registry.py`,
`scripts/dataset_scripts/{extract_grasp_signals,select_split_points,materialize_skill_datasets}.py`,
modified `utils/{dataset_registry,lerobot_utils}.py`, plus the new `utils/env_helpers.py`.

## Import edges to rewrite

```
# grasp project (7 sites) -- after this, anygrasp imports nothing from DP
from eval_chained_pick_place import base_env                       -> from robocasa.utils.env_helpers import base_env
from diffusion_policy.env_runner.robomimic_image_runner import create_env
                                                                   -> from robocasa.utils.env_helpers import create_env

# VLM (keeps its DP dependency, but by package not by cwd)
from eval_chained_pick_place import base_env, load_policy, obs_to_frame, stack_obs
                                                                   -> from diffusion_policy.skills.eval_chained import ...

# intra-DP
check_eval_harness.py: from eval_chained_pick_place import base_env -> same package path
```

Sites: `eval_anygrasp_pick.py:44,45`, `check_pipeline.py:32,33`,
`check_grasp_executor.py:28,29`, `check_grasp_geometry.py:36,37`,
`eval_agentic_pick_place.py:52-55,60`, `check_eval_harness.py:38`.

Also: two dead imports at `grasp_perception.py:44-45` (`STANDOFF`, `IKReach` — imported,
never used) should be deleted; that alone removes robocasa/robosuite/mujoco from
`grasp_perception`'s import graph.

## Sequence

The invariant: **every decoupling change lands while files are still where they are.** The
physical move is last and is the most reversible step.

| # | step | verify |
|---|---|---|
| 0 | Record a baseline: `check_eval_harness.py --task PickPlaceCounterToCabinet --n 3 --from_demo_state` | note the result; without it a reorg break is indistinguishable from a pre-existing one |
| 1 | Freeze — no training/eval running (editable installs are live path bindings) | `ps` clean |
| 2 | Commit the dirty anygrasp worktree; push all ahead-branches | `git status` clean in all repos |
| 3 | Back up `checkpoint-rs.tar` outside the tree | file exists elsewhere |
| 4 | Merge `worktree-anygrasp-pick` into DP `main` (21 files, +13,221, **0 deletions**) | free now; becomes rename-vs-add conflicts after any move |
| 5 | Unlock + remove both git worktrees | `git worktree list` shows only main |
| 6 | Add `robocasa/utils/env_helpers.py` with `base_env`/`create_env`; re-export from old locations | `check_eval_harness.py` still passes |
| 7 | Rewrite the 11 import sites to the new paths; delete the 2 dead imports | all three projects' check scripts pass, **still zero files moved** |
| 8 | Fix `setup.py` → `find_namespace_packages(include=["diffusion_policy*"])`; `pip install -e . --no-deps` in `robocasa_dp` | `cd /tmp && python -c "import diffusion_policy.common.pytorch_util"` |
| 9 | Fix `check_eval_harness.py:44` cwd-relative config path to be `__file__`-anchored | runs from `/tmp` |
| 10 | Create `vlm_planning/` and `anygrasp/` repos; `git mv` the 24 files out of DP | each repo's scripts run |
| 11 | Move directories into `sim/`; move `third_party/*` under `anygrasp/vendor/` | — |
| 12 | `pip install -e <newpath> --no-deps` × 5 (robocasa, robosuite in `robocasa` env; robocasa, robosuite, robomimic in `robocasa_dp`) | `import robocasa, robosuite, robomimic` in both envs |
| 13 | Update `serve_grasp.sh` `GRASPNET_ROOT`/`PYTHONPATH`, `grasp_server.py` `CHECKPOINT`, `setup_grasp_env.sh` paths | `./serve_grasp.sh` reaches "Uvicorn running" |
| 14 | Re-run step 0 and compare | identical result |

Steps 6–9 are load-bearing: additive, individually testable, and once they land the move in
step 11 is a `mv` whose only failure mode is a stale editable path, which step 12 fixes
deterministically.

## Hazards

- **Two `locked` git worktrees bind into the DP repo by absolute path, in both directions.**
  A naive `mv` dangles both and makes uncommitted work unreachable by git. Steps 2–5 exist
  to remove this binding before anything moves. `git worktree repair` is the recovery.
- **`outputs/` is 79 GB with no VCS safety net** and is referenced by cwd-relative paths in
  every documented command. If it moves, `mv` on the same filesystem — never `cp` then `rm`.
- **`checkpoint-rs.tar` is not reproducible by script** (Drive quota-refuses every
  programmatic client). Back it up first.
- **`--no-deps` is mandatory for robomimic** — it pulls `numpy<2`, which trips
  `robocasa/__init__.py`'s hard `numpy==2.2.5` assert.
- **`robocasa/macros_private.py` `DATASET_BASE_PATH`** is gitignored with no env-var
  override. Hand-edit if `datasets/` moves.
- **`pip install -e .` in the DP repo currently installs nothing, silently** —
  `find_packages()` returns `[]`. The naive fix (`find_namespace_packages()` with no filter)
  would try to package 79 GB of `outputs/`.
- **`robocasa` is not purely simulator.** It carries 7 project files and
  `dataset_registry.py` mutates global state at import, so every importer inherits 46 skill
  datasets. `sim/` is a slight misnomer the layout accepts.
- **Namespace-package hazard:** with no `__init__.py`, two `diffusion_policy` directories on
  `sys.path` silently merge and resolve by path order.
