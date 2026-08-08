# CLAUDE.md — robocasa_diffusion_policy

Guidance for Claude Code working inside this project. Self-contained: the workspace root
CLAUDE.md keeps only the six-repo overview, the conda-env table, and a pointer here.

## What this project is

RoboCasa's atomic `PickPlace*` human demos are single episodes containing **both** halves of
the task. This project splits them into two independently callable, slot-conditioned
diffusion policies — `pick(obj)` and `place(obj, receptacle)` — then chains them back
together and scores with the original task's own `_check_success`. The central problem is
finding the frame where the pick ends.

Position in the stack: `vlm_planning` depends on **this** repo; this repo depends on
`sim/` (`robosuite`, `robocasa`, `robomimic_robocasa`). `anygrasp` depends on `sim/` only and
imports **zero** diffusion-policy symbols — verified, the string `diffusion_policy` appears
in `anygrasp/` only inside two markdown files.

### The code, as a reviewable diff

Fork point is `4121269` (last upstream merge). Everything since is this project.
`git diff --stat 4121269 HEAD` is the whole change: 13 files, ~2.6k lines.

**New:**

- `eval_chained_pick_place.py` — the chained/pick-only evaluator
- `check_eval_harness.py` — the plumbing check that must pass before any eval number means anything
- `diffusion_policy/skills/{__init__.py,eval_helpers.py}` — `load_policy` / `obs_to_frame` / `stack_obs`
- `diffusion_policy/config/task/robocasa/pretrain_{pick,place}_skill.yaml`
- `pick_eval_e90.json`, `harness_check.json` — recorded results, checked in on purpose

**Modified (five files, not two):**

| file | why |
|---|---|
| `diffusion_policy/dataset/lerobot_dataset.py` | slot conditioning (`SLOT_KEYS`, `_get_slot_embeddings`) |
| `diffusion_policy/env/robomimic/robomimic_image_wrapper.py` | `slot_embs`, `last_raw_obs`, slot obs bounds |
| `diffusion_policy/model/common/lr_scheduler.py` | diffusers 0.39 dropped re-exports |
| `diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py` | LR-scheduler resume fix |
| `setup.py` | `find_namespace_packages` — the package was never installable |

Outside this repo the project also adds `sim/robocasa/robocasa/utils/{skill_utils,skill_dataset_registry,env_helpers}.py`
and the three `scripts/dataset_scripts/` steps below, and modifies
`sim/robocasa/robocasa/utils/{dataset_registry,lerobot_utils}.py` and
`sim/robomimic_robocasa/robomimic/utils/torch_utils.py`.

### `diffusion_policy/skills/eval_helpers.py` — why it exists

`load_policy`, `obs_to_frame` and `stack_obs` used to sit at the top of
`eval_chained_pick_place.py`, which made that file simultaneously an argparse entrypoint and
the import target for five other evaluators: pulling in an 11-line helper executed the whole
script's module body, and it resolved at all only when the process happened to be launched
from the repo root. They are in the package now, so a consumer outside this repo —
`vlm_planning/eval_agentic_pick_place.py` — imports them by a real package path
(`from diffusion_policy.skills.eval_helpers import load_policy, obs_to_frame, stack_obs`).

Only genuinely diffusion-policy-specific helpers belong here. `create_env` and `base_env`
are **not** diffusion-policy code — one wraps robocasa's own gym registration, the other
unwinds the wrapper stack that registration builds — and live in
`sim/robocasa/robocasa/utils/env_helpers.py`. Keeping them here was the last edge tying
`anygrasp` to this repo, and it dragged wandb, torch, h5py, dill and CLIP into an evaluator
that never instantiates a policy, for a seven-line `gym.make`.

### The package is installed, and was not before

`find_packages()` returns `[]` here and always has: 159 `.py` files but only 7 `__init__.py`,
none at depth 1 or 2. `pip install -e .` therefore installed **nothing** while reporting
success, and every script resolved `diffusion_policy` from its cwd. `setup.py` now uses
`find_namespace_packages(include=["diffusion_policy*"])`. **The include filter is
load-bearing:** the bare call returns 119 entries, 60 of which are not diffusion_policy —
`outputs`, `outputs.pick_skill.checkpoints`, `wandb.*`, `media`, `tests` — so it would try to
package the 79 GB of checkpoints under `outputs/`. Verified importable from `/tmp`:

```bash
cd /tmp && python -c "import diffusion_policy.skills.eval_helpers as E; print(E.__file__)"
```

Four packages are editable in `robocasa_dp`: `robocasa`, `robosuite`, `robomimic`,
`diffusion_policy`. **Moving any repo directory breaks its editable install** — loud on
import, invisible to `pip list`, which keeps reporting the stale path. Re-run
`pip install -e <newpath> --no-deps` per env per package.

## Environments

- **`robocasa`** — the data pipeline (steps 1–3 below)
- **`robocasa_dp`** — training and eval; adds hydra/omegaconf/transformers plus robomimic from source

They cannot be merged: `robocasa/__init__.py` hard-asserts `mujoco==3.3.1` and
`numpy==2.2.5`, while `pip install robomimic` drags in `numpy<2` and breaks every
`import robocasa`. PyPI robomimic 0.3.0 also lacks `lang_utils` and `LANG_EMB_KEY`, which
this fork needs — hence the source clone installed `--no-deps`.

```bash
python -c "import robocasa, robomimic, numpy; print(numpy.__version__)"   # must print 2.2.5
```

`export MUJOCO_GL=egl` is required for anything that builds an env (headless render); it
fails at scene creation, not at import. `sim/robocasa/robocasa/macros_private.py` sets
`DATASET_BASE_PATH` to `<workspace>/datasets` and is gitignored local config.

Measured stack in `robocasa_dp`: torch 2.7.1+cu126, diffusers 0.39.0, numpy 2.2.5.

## Commands

### Data pipeline (env `robocasa`, run from `sim/robocasa/`)

**These scripts live in `sim/robocasa/` but are owned by this project.** They exist only to
build the skill datasets these policies train on, and their criterion is shared with this
repo's evaluator (see "The split criterion is shared code"). They sit in `robocasa/` because
they import `robocasa.utils.object_utils`, `playback_utils`, `lerobot_utils` and the dataset
registry, and because the datasets they emit are registry entries that `robocasa` itself must
resolve — not because they belong to the simulator fork. Change the thresholds in
`robocasa/utils/skill_utils.py` and both sides move together; change one side alone and the
training boundary silently drifts from the eval handoff.

Three deliberately separate steps. Step 1 is the only one that touches the simulator, so
keeping it apart means thresholds can be re-tuned and step 2 re-run in seconds.

```bash
python -m robocasa.scripts.dataset_scripts.extract_grasp_signals --split pretrain target --num_procs 6
python -m robocasa.scripts.dataset_scripts.select_split_points   --split pretrain target
python -m robocasa.scripts.dataset_scripts.materialize_skill_datasets --split pretrain target --num_procs 4
```

Steps 1 and 3 run for hours — launch under tmux or `setsid nohup ... &`. Step 2 is fast,
idempotent, and doubles as the split-quality report (per-task counts and reject reasons).

**Measured cost of step 1**, re-derived from the `seconds` field of the 23 signal caches on
disk: **10.7 s/episode** (per-dataset range 8.5–12.0), 4,456 episodes, 13.3 h of worker time,
so ~2.2 h wall at `--num_procs 6`. A fixed scene rebuild dominates that; the dense per-frame
pass over a whole episode is a small fraction of it, which is why `extract_grasp_signals.py`
walks every frame instead of binary-searching for the grasp — binary search would save little
and buy off-by-one bugs.

### Training (env `robocasa_dp`, from this repo)

```bash
python train.py --config-name=train_diffusion_transformer_bs192 \
  task=robocasa/pretrain_pick_skill \
  dataloader.batch_size=64 val_dataloader.batch_size=64 \
  dataloader.num_workers=8 val_dataloader.num_workers=4 \
  training.gradient_accumulate_every=3 \
  training.num_epochs=300 training.checkpoint_every=5 \
  logging.mode=offline hydra.run.dir=outputs/pick_skill
```

Same with `task=robocasa/pretrain_place_skill`, `hydra.run.dir=outputs/place_skill`. One GPU,
so run them sequentially. Every one of those overrides matters: the shipped
`train_diffusion_transformer_bs192.yaml` defaults are `batch_size: 192`,
`gradient_accumulate_every: 1`, `num_epochs: 1000`, `checkpoint_every: 100`.

`bs192` does not fit in 24 GB. The image encoder sees `batch 192 × n_obs_steps 2 × 3 cameras
= 1152 images per forward` (the diffusion horizon of 10 is on the action head, not the
encoder). `64 × 3` accumulation reproduces the benchmark's effective batch.

An "epoch" is `max_train_steps: 500` optimizer steps. **Measured from checkpoint mtimes:
7.6 min/epoch for pick, 8.1 min/epoch for place**, so `checkpoint_every=5` saves roughly
every 40 min and 90 epochs ≈ 11–12 h. `num_epochs=300` is an upper bound, not a target — pick
plateaued by epoch 90.

Measured `train_loss` from `outputs/*/logs.json.txt`:

| | step 0 | ep 10 | ep 50 | ep 90 | ep 95 | ep 125 |
|---|---|---|---|---|---|---|
| pick | 1.223 | 0.088 | 0.056 | 0.048 | 0.047 | — |
| place | 1.236 | 0.099 | 0.064 | 0.054 | 0.053 | 0.049 |

Only 0.056 → 0.048 over epochs 50–90 for pick. `train_action_mse_error`, logged every
`sample_every: 50` epochs, went 0.300 (epoch 0) → 0.041 (epoch 50).

For a smoke run add `training.max_train_steps=200 training.num_epochs=1` (~4 min) — read the
obs-spec line to confirm the slots (`obj_emb` alone for pick, `obj_emb` **and** `recep_emb`
for place); a missing `recep_emb` means the place policy would train blind to its receptacle.

**Resuming** is `training.resume=True`, which loads `outputs/<run>/checkpoints/latest.ckpt`.
It only works because of the workspace patch — see the LR-scheduler gotcha below.

### Evaluation

```bash
MUJOCO_GL=egl python eval_chained_pick_place.py \
  --pick_checkpoint  'outputs/pick_skill/checkpoints/epoch=0095-test_mean_score=-1.000.ckpt' \
  --place_checkpoint 'outputs/place_skill/checkpoints/epoch=0125-test_mean_score=-1.000.ckpt' \
  --num_rollouts 10 --output chained_eval.json
```

Quote checkpoint paths — the `=` in the filename trips zsh. Omit `--place_checkpoint` to
score the pick skill alone (budget becomes `0.5 × horizon`, the same allowance the pick phase
gets in the chain, so the numbers stay comparable). `--video_dir DIR --video_n 1` records
diagnostic rollouts (agentview beside eye-in-hand, `OK`/`FAIL` in the filename). Scene seed is
`--seed + rollout_index`, so runs are comparable across checkpoints. Default
`--tasks` is `SU.PICK_PLACE_TASKS`, all 18.

Cost is dominated by diffusion sampling (`num_inference_steps: 100` per `n_action_steps: 8`
env steps): ~53 s/rollout, so 18 tasks × 10 rollouts ≈ 2.5 h and × 50 is roughly a day.
Levers, in payoff order: fewer inference steps (DDIM), then reusing one env across rollouts of
a task (`run_episode` rebuilds per rollout at ~14 s each).

**Metrics.** `pick_success` = the detector fired; `still_holding` = `is_holding_obj` at the
final step; `place_given_pick` and `task_success` in chained mode. `pick_success` only
requires grasp + 2 cm of motion, so the *gap* between it and `still_holding` is the diagnostic
— a policy that grabs, nudges, and drops scores on the first and fails the second. A high
`pick_success` with low `place_given_pick` is a place-policy problem; a low `pick_success` is
the pick policy or the handoff predicate itself. The evaluator also prints a per-object
breakdown, because the documented risk (correction 2 below) is specifically that handled
objects fail.

Rollouts that raise are caught and skipped rather than killing the sweep, so **check `n` per
task in the output JSON** — a task that silently ran 3 of 10 rollouts still reports a rate.

### Verification

There is no test suite for this project — the `tests/` dir is upstream's. The real check is:

```bash
# ground-truth demo actions replayed through the exact eval path; must fire the detector
MUJOCO_GL=egl python check_eval_harness.py --task PickPlaceCounterToCabinet --n 3 --from_demo_state
```

Run this before reading anything into an eval number — a low score could otherwise equally
mean a broken harness. In one shot it exercises `base_env()`'s unwrapping of the gym wrapper
stack, the three-way action ordering, and `GraspMoveDetector` running online frame by frame.
It writes `harness_check.json` beside the script (`__file__`-anchored, not cwd-relative, since
after the workspace split this is run from several directories).

`--from_demo_state` is required for a meaningful result: it does a full `reset_to` with the
episode's `model.xml.gz` and `ep_meta`, not just `set_state_from_flattened`. A freshly sampled
scene has a different layout and object set, so state vectors differ in length (79 vs 107
qvel observed) and a bare state restore raises. Without the flag the recorded actions are
replayed open-loop into a *different* scene: every code path still runs, but the grasp
succeeding is luck.

The three outcome messages are deliberately distinct — `HARNESS OK`, `PARTIAL` (plumbing
works, misses are open-loop drift), `HARNESS SUSPECT` (ground truth never fires; check action
ordering, then `base_env()`, then the scene).

Recorded baseline in `harness_check.json`: 3/3 fired, on cereal / juice / mayonnaise, at
frames 97 / 81 / 110 of 111 / 93 / 123, final `dz` +0.021 / +0.024 / +0.023.

To watch a split visually (env `robocasa`, from `sim/robocasa/`) — pick must **end** holding
the object, place must **start** holding it:

```bash
MUJOCO_GL=egl python -m robocasa.scripts.dataset_scripts.playback_dataset \
  --dataset datasets/v1.0/pretrain/atomic/PickPlaceCounterToCabinet_pick/20250819/lerobot \
  --n 3 --video_path /tmp/pick.mp4
```

Prefer the eye-in-hand camera when reviewing video — `agentview_center` (what
`playback_dataset` renders by default) is often occluded by a cabinet in these scenes.

Grasp-detector plumbing checks (`check_grasp_geometry.py`, `check_grasp_executor.py`) belong
to the `anygrasp/` project; see `anygrasp/GRASP_README.md`. They are the same idea — a low
score must never be ambiguous between a bad model and broken plumbing.

## Architecture

### The split criterion is shared code, not duplicated logic

`sim/robocasa/robocasa/utils/skill_utils.py` defines what "the pick ended" means and is
imported by **both** the offline splitter and this repo's online evaluator, so the training
boundary and the eval handoff cannot drift apart. `GraspMoveDetector` (in `skill_utils.py`) is
the online form of the rule `split_episode` (in `select_split_points.py`) applies offline.

Thresholds (20 Hz data): `MOVE_DZ=0.02`, `MOVE_WINDOW=60`, `MIN_GRASP_RUN=5`, `GAP_CLOSE=5`,
`PICK_TAIL_PAD=10`, `PLACE_TAIL_PAD=15`, `MIN_SEG_LEN=32` (= horizon 10 + n_obs 2, with slack).

```
t_success = first frame where next.reward > 0
t_grasp   = start of the last contact run beginning before t_success that also passes the move test
t_moved   = first frame within MOVE_WINDOW where |obj_z - obj_z[t_grasp]| >= MOVE_DZ

pick  = [0,       t_moved + PICK_TAIL_PAD]      # ranges inclusive both ends
place = [t_grasp, t_success + PLACE_TAIL_PAD]
```

"Last run before success" and not "first": a human who re-grasps to nudge the object after
placing it would otherwise hand us the adjustment. It is well defined because every
`PickPlace._check_success` conjoins `gripper_obj_far`, so no run can straddle `t_success`.

**The overlap is intentional.** `[t_grasp, t_moved + PICK_TAIL_PAD]` belongs to both segments
— that window is the lift phase, which is where the runtime handoff fires, so the handoff
state is in-distribution for both policies by construction. Anchoring pick's end on `t_moved`
rather than a fixed offset from `t_grasp` is what keeps that true when a lift is slow.
Truncating place at `t_success` drops the idle tail every human demo has after completion,
without which the place policy would be trained to stall. Measured mean overlap on
`CabinetToCounter`: 25.3 frames, 13.4% of the episode.

Four corrections are baked in, each from a measured failure — do not "simplify" them back:

- Grasp uses fingerpad contact (`_check_grasp`), **not** `OU.check_obj_grasped`: the latter
  ANDs contact with `finger qpos < 0.035`, and measured qpos during unambiguous grasps runs
  0.023 (cereal) – 0.041 (juice, mayonnaise), so the fixed threshold silently rejects every
  *wide* object.
- Two-tier contact — fingerpad preferred, any-gripper-contact fallback — because handled
  objects (ladle, measuring cup) are grasped by the handle and register **zero** fingerpad
  frames: all 12 rejected `PickPlaceCounterToDrawer` demos were ladles and measuring cups with
  zero strict-contact frames but 76–169 contiguous loose ones. The displacement test keeps the
  fallback honest — a knuckle brush does not move an object 2 cm.
- The move test is **unsigned**; `CabinetToCounter` *lowers* the object ~45 cm.
- `t_success` comes from the parquet `next.reward` as recorded at collection time, not a
  replayed `_check_success()` — the latter misses **85 of 106** `CounterToBlender` demos
  because `obj_inside_of(th=0.01)` does not survive state restoration.

The first three live in `skill_utils.py`; the fourth is in `select_split_points.py`.
`extract_grasp_signals.py` caches both contact tiers plus a replayed `success` it deliberately
**does not use** — keeping it beside the parquet `reward` is the tripwire that surfaced
correction 4.

**Where to be skeptical:** correction 2 is a genuine loosening. Offline it is safe because
`split_episode` sees the whole trajectory and only falls back when the strict tier yields
nothing. Online, `GraspMoveDetector` must commit frame by frame and approximates this with
`not _ever_strict`. That asymmetry is the most likely source of train/eval drift — if eval
shows low `pick_success` concentrated on handled objects (ladles, measuring cups, spoons),
look here first. This is exactly what the evaluator's per-object breakdown is printed for.

Slots are **never parsed out of `ep_meta["lang"]`**; they come from `ep_meta["object_cfgs"]`
plus the hard-coded 18-row `PICK_PLACE_TARGETS` table in `skill_utils.py`. Several tasks store
a place-only or non-parametric instruction, and `PickPlaceToasterToCounter` never names the
object at all ("Place the toasted item on a plate."). Rows are either a literal receptacle
(`PickPlaceCounterToCabinet: "cabinet"`) or an `obj:<cfg_name>` reference resolved per episode
(`PickPlaceSinkToCounter: "obj:container"`). That table is the most reviewable-by-eye part of
the project and the easiest place for a wrong receptacle to hide — each row should match that
task's `get_ep_meta` in
`sim/robocasa/robocasa/environments/kitchen/atomic/kitchen_pick_place.py`.

### Action ordering — the most dangerous footgun

Three orderings exist and must line up. All three verified against the code:

| where | order |
|---|---|
| parquet `action` column (`meta/modality.json`) | `base_motion` 0:4, `control_mode` 4:5, `eef_pos` 5:8, `eef_rot` 8:11, `gripper` 11:12 |
| what the policy emits (`shape_meta.action.lerobot_keys`) | `[eef_pos, eef_rot, gripper, base_motion, control_mode]` |
| what `env.step` expects (`robocasa.utils.env_utils.convert_action`) | `0:3 eef_pos, 3:6 eef_rot, 6:7 gripper, 7:11 base_motion, 11:12 control_mode` |

The last two agree, which is what matters. Raw parquet order is never used for training — the
loader slices by modality key and `lerobot_keys` reassembles in robosuite order;
`LU.reorder_lerobot_action` does the same for recorded actions via
`ACTION_KEY_ORDERING_HDF5`. The splitter copies the `action` column verbatim, so ordering is
preserved end to end. `abs_action: False`, so `undo_transform_action` never fires. **A
mismatch here produces a flailing policy with no error** — that is what `check_eval_harness.py`
rules out.

Related footgun in `extract_grasp_signals.py`: `action[:, 11]` is the gripper command in
*parquet* order, not policy order. And `next.reward` is a scalar per row, not a 1-element
array (hence the `.tolist()` + `.reshape(-1)`).

**Two fields in that 12-vector are thresholded at 0.5, not 0** —
`PandaOmronKeyConverter.unmap_action` in `robocasa/wrappers/gym_wrapper.py` maps
`gripper_close > 0.5 → +1.0` (close) and everything else to `−1.0` (open), and does the same
for `control_mode → base_mode`. The gym action space declares both as `Box(low=-1, high=1)`,
so the conventional −1/+1 convention type-checks perfectly and leaves the gripper
**permanently open with no error**. Anything hand-writing actions must emit `0.0`/`1.0`.
Keeping `control_mode` at `0.0` also matters, via a two-step gate: `unmap_action` turns it
into `base_mode = −1.0`, and `HybridMobileBase.set_goal` then reads `all_action[-1] > 0` to
pick `goal_update_mode`. At `0.0` you get `"achieved"`, so an OSC delta is measured against the
*current* pose; anything `> 0.5` yields `"desired"`, measuring against the previous goal and
turning a saturated proportional term into an integrator that runs the arm away. Policies
trained on demo data emit the right values; hand-written controllers (the grasp executor) do
not by default.

### Slot conditioning replaces free-form language

The fork's `lang_emb` is replaced by `obj_emb [768]` (pick) and `obj_emb` + `recep_emb`
(place), each embedded separately by the frozen CLIP text encoder
(`CLIPTextModelWithProjection`, `openai/clip-vit-large-patch14`, via
`robomimic.utils.lang_utils.LangEncoder`). Embedding each noun phrase on its own is the whole
point — it lets an argument be swapped at test time without a sentence template. Episodes
carry a structured `task` string that `lerobot_dataset.py` splits and embeds:

```
pick | obj: cereal
place | obj: cereal | recep: cabinet
```

`slots_to_task_string` / `parse_task_string` in `skill_utils.py` are the two ends of that
format, and `parse_task_string` raises on anything malformed rather than half-parsing.

`SLOT_KEYS = {"obj_emb": "obj", "recep_emb": "recep"}`, defined in `lerobot_dataset.py` and
imported by the wrapper and both evaluators, so there is one spelling of the slot names. The
constructor pops slots from `shape_meta.obs`, `_get_slot_embeddings` embeds the unique phrases
once (~100 phrases against thousands of episodes) and shares them, `__getitem__` tiles each
over `n_obs_steps`, and `get_normalizer` gives each an identity normalizer. An assert forbids
declaring both `lang_emb` and slots. A missing slot on any episode is an assert, not a silent
zero.

Because pick and place have **different obs schemas**, they are separate checkpoints with
separate configs — do not load one into the other's workspace. In the chained evaluator this
means the wrapper emits every slot *its* schema declares (the place one), while the policy
input is filtered per phase; the pick policy's normalizer raises on an unexpected `recep_emb`
key. Getting this wrong was a real bug: reassigning `wrapper.slot_embs` per phase left
`get_observation` unable to fill `recep_emb`. The evaluator also has to reset with
**zero-filled placeholder slots** and re-derive the first observation from
`wrapper.last_raw_obs` afterwards, because the slots are only knowable once the scene exists
but `get_observation` needs them present — that is what `last_raw_obs` was added for.

The task configs are copies of `pretrain_human300.yaml` with exactly three functional changes
(verified by diff): `lang_emb` → `obj_emb` (+ `recep_emb` for place) in `shape_meta.obs`,
`dataset.dataset_soup`, and `env_runner.n_test: 50 → 0`. The runner is inert during training
(`rollout_every: null`), but zeroing `n_test` means enabling rollouts fails loudly rather than
quietly scoring a half-skill against full-task success — which would read as 0% forever.

### Registry integration is append-only by construction

`sim/robocasa/robocasa/utils/skill_dataset_registry.py` derives every entry mechanically
rather than listing paths by hand: `build_skill_datasets` walks the existing `PickPlace*`
entries in `ATOMIC_TASK_DATASETS` and rewrites one path segment
(`.../atomic/<Task>/<date>` → `.../atomic/<Task>_<skill>/<date>`), keeping the parent's
`horizon`. `build_skill_task_sets` names only the datasets that exist per split — no
hard-coded lists to drift.

The fold-in lines in `dataset_registry.py` (line ~2972) sit **after** `TASK_SET_REGISTRY`
closes and **before** `DATASET_SOUP_REGISTRY` opens. That position is load-bearing:
`all_atomic_tasks`/`all_tasks` are materialized as *lists* before the update, so every
pre-existing task set and soup stays byte-identical while `get_ds_meta` still finds the new
entries. Verify rather than trust — all four re-checked and current:

```python
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY as S, TASK_SET_REGISTRY as T
len(T["all_atomic_tasks"])        # 65   unpolluted
len(T["all_tasks"])               # 317  unpolluted
len(S["pretrain_human300"])       # 300  unchanged
len(S["pretrain_pick_skill"])     # 18   (also _place; target_* are 5 each)
```

### Data layout and provenance

```
datasets/v1.0/{pretrain,target}/atomic/
├── PickPlaceCounterToCabinet/20250819/
│   ├── lerobot/                       # untouched source
│   └── skill_cache/
│       ├── ..._signals.npz            # step 1: per-frame signals + ep_len offsets
│       ├── ..._signals.json           # slots, failures, seconds
│       └── ..._splits.json            # step 2: env_name, params, summary, episodes, rejected
├── PickPlaceCounterToCabinet_pick/20250819/lerobot/
└── PickPlaceCounterToCabinet_place/20250819/lerobot/
```

Generated datasets are structurally identical to the originals (`data/`, `videos/`, `meta/`,
`extras/`), so `playback_dataset.py`, the dataset soups and the GR00T loader work on them
unmodified. Three things `materialize_skill_datasets.py` does that are worth knowing:

- **Frames are decoded from the source mp4s, not re-rendered.** Re-rendering would re-derive
  pixels through `set_ep_meta` + `gen_textures` + `cam_configs` and could silently diverge
  from the videos the demos shipped with. Actions and sim states are byte-identical slices of
  the source (`df["action"].iloc[t]`, `states[lo:hi+1]`).
- **Four columns are recomputed:** `annotation.human.task_name`,
  `annotation.human.task_description`, `next.done` (true only at `hi`), and `next.reward` —
  place keeps the original signal, **pick synthesises one** from `t_moved` (zeros, then 1.0
  from `t_moved` on), since the task only rewards success after release and pick data would
  otherwise be all-zero.
- **`extras/dataset_meta.json` still names the ORIGINAL task** (`env: PickPlaceCounterToCabinet`,
  not `..._pick`); only `total` is rewritten. Deliberate: playback and the chained evaluator
  must rebuild the real scene. Do not "fix" this.

## Current state (snapshot — re-derive rather than trust)

| | |
|---|---|
| datasets | 46 skill datasets, 8,908 episodes, 11 GB — exactly 4,454 split points × 2 |
| split quality | 4,456 episodes, 4,454 usable, **2 rejected** (0.04%): `CabinetToCounter` ep55, `CounterToDrawer` ep5 |
| pick policy | `outputs/pick_skill`, epoch **95** (`num_epochs=300` run, stopped early); plateaued — 0.056 → 0.048 over epochs 50–90 |
| place policy | `outputs/place_skill`, epoch **125** of a `num_epochs=150` run; its cosine LR has decayed to 6.0e-6, i.e. effectively finished |
| `outputs/` | 79 GB. Each checkpoint is 1.82 GB and `checkpoint_every=5` keeps them all |
| pick-only eval | **33.3% `pick_success`, 32.2% `still_holding`** over 180 rollouts (18 tasks × 10), `pick_eval_e90.json`, epoch-90 checkpoint |
| chained eval | harness verified working (3/3 in `harness_check.json`); **no full sweep run yet** |
| target datasets | built, unused |

Per-task pick spread is wide: `CounterToToasterOven` 0.80 and `CounterToOven` 0.70 against
`DrawerToCounter` 0.00, `CabinetToCounter` / `CounterToDrawer` / `CounterToMicrowave` 0.10.
Recomputed from the rollout records: 60 successes, 120 failures; **115 of the 120 failures
moved the object < 5 mm** (never made contact), and failures averaged **332 steps against 113
for successes** — i.e. budget exhaustion, not fumbled grasps. Conditional
`P(still_holding | pick_success) = 0.933` (56/60); the aggregate ratio 0.322/0.333 = 0.967 is
the number quoted elsewhere, and the two are not the same statistic.

The place policy has **never been scored** — the eval numbers above are pick-only. The
cheapest next result is a chained sweep at `pick@0095` + `place@0125`.

Rejections are recorded with reasons in `..._splits.json`, so they are recoverable if you
disagree with the criterion.

## Gotchas

- **zsh does not word-split unquoted expansions.** `--tasks $TASK_LIST` passes all names as
  one argument and `get_ds_meta` raises instantly. Use a zsh array or `${=VAR}`. Sanity check:
  the first materializer log line must read `N datasets to materialize`, not `1`.
- **The LR scheduler resume bug, and why the workspace is patched.**
  `train_diffusion_transformer_hybrid_workspace.py` builds the cosine schedule with
  `num_training_steps = (max_train_steps × num_epochs) // gradient_accumulate_every`, but
  upstream passed `last_epoch=self.global_step-1`. `global_step` counts *micro*-steps (500 per
  epoch); the scheduler steps once per `gradient_accumulate_every`. Resuming pick at epoch 90
  would therefore have restored `last_epoch=45000` against a 50,000-step cosine —
  **LR 2.5e-6 instead of 8.1e-5, a 32× collapse, with no error**. The fix divides by
  `gradient_accumulate_every`. Both long runs were resumed once and the logged LR is
  continuous across the boundary (8.281e-5 at epoch 85 → 8.241e-5 at 86), which is the check
  to repeat after any resume.
- **Nothing is checkpointed during epoch 0.**
  `train_diffusion_transformer_hybrid_workspace.py:374` gates on
  `self.epoch > 0 and self.epoch % checkpoint_every == 0`, so a short smoke run legitimately
  produces no `checkpoints/` dir — `outputs/smoke_{pick,place}/` on disk are exactly that. On a
  long run, verify `checkpoints/` exists after ~40 min rather than assuming; a multi-hour run
  that turns out not to be saving is the expensive failure.
- **Validation never runs at all.** The whole `# run validation` block in
  `train_diffusion_transformer_hybrid_workspace.py` (~lines 333–348) is commented out
  upstream, so `val_every: 50` and `max_val_steps` are dead config and `val_ratio: 0.02` in the
  task configs buys nothing. Confirmed: **zero** `val_loss` entries across 49,837 (pick) and
  64,583 (place) logged rows, i.e. 95 and 125 epochs. Do not wait for a validation curve, and
  do not read its absence as "too early" — uncommenting the block is what it would take.
  `train_loss` and the `sample_every` action-MSE are the only signals.
- **`materialize_skill_datasets.py` is not resumable** — it `rmtree`s and rewrites every
  output. Resume at dataset granularity via `--tasks`. A completeness check must look at
  `episodes.jsonl` line count plus `stats.json`/`tasks.jsonl`/`extras/dataset_meta.json`,
  **not** `info.json` or directory existence — those are written at dataset *creation*, so a
  killed job leaves them on an empty dataset.
- **`LeRobotDataset` never shuts down its `AsyncImageWriter` pool.** A worker handling many
  datasets in sequence leaked 4 processes per job; live writers climbed to 45 (target) and 67
  (pretrain), exhausting RAM and all 7 GB of swap. It scales with job *turnover*, not
  concurrency, so lowering `--num_procs` alone never fixes it. `process_dataset` calls
  `stop_image_writer()` in a `finally` and the pool is `IMAGE_WRITER_PROCESSES=2 ×
  IMAGE_WRITER_THREADS=4` — keep both. Live process count is `2 × 2 × --num_procs`. If live
  writers climb past ~25 during a run, this broke.
- **Train `place` only on the full soup.** `recep_emb` is constant within a task (14 distinct
  values across the 18-task pretrain soup: blender, bowl, cabinet, counter, drawer, fridge
  drawer, fridge shelf, microwave, oven, pan, plate, sink, stand mixer bowl, toaster oven), so
  a single-task dataset gives the place policy no receptacle variation at all.
- **`--split target` is the same 18 env names, not new tasks** — 5 of them, with disjoint
  scenes and object instances. Pretrain vocabulary is **94 objects × 14 receptacles**; target
  is **105 × 4** (cabinet, counter, pan, plate) and introduces **13 object categories unseen in
  pretrain**: cheese grater, cupcake, donut, glass cup, jar, jug, ketchup, mustard, oil/vinegar
  bottle, strainer, straw, turkey slice, wine. Target datasets are built but unused so far —
  keeping them out of pretraining is what makes a target number meaningful. Target entries use
  `filter_key: 500_demos` and `horizon: 750` against pretrain's `100_demos` / `450`.
- **`horizon` means two different things in two registries.** `DATASET_SOUP_REGISTRY` entries
  carry `horizon: 450` (pretrain) / `750` (target) alongside `filter_key` — that is the
  training-side number. But `get_task_horizon(task=...)`, which is what both eval scripts use
  as the **rollout budget**, reads `ATOMIC_TASK_DATASETS[task]["horizon"]`, a single per-task
  value with no split argument. Measured across the 18 PickPlace tasks: 450 ×4, 600 ×7, 750
  ×5, 900 ×1, 1050 ×1 (`CounterToCabinet` 750, `CounterToMicrowave` 1050). So a pretrain
  rollout is usually *not* 450 steps, and per-rollout wall clock varies more than twofold
  across tasks — worth remembering before extrapolating a sweep's runtime from one task's
  timing. Note also `horizon: 10` in the training config is the diffusion action horizon, a
  third unrelated meaning.
- **The DP fork carries a 2022 dependency set.** Two `diffusers` imports were already patched:
  `diffusion_policy/model/common/lr_scheduler.py` and
  `sim/robomimic_robocasa/robomimic/utils/torch_utils.py` — the latter *inside*
  `lr_scheduler_from_optim_params`, so it only fires at optimizer creation and survives a
  standalone import, which is why its traceback pointed at the wrong repo. Both cases are the
  same shape: `diffusers.optimization` used to re-export `Union`/`Optional`/`Optimizer` under
  the pinned 0.11.1 and 0.39 does not. Expect the same on paths not yet exercised. Fix by
  importing the symbol from its real home — downgrading is not an option, since the fork's
  pins (`numpy=1.23.3`, `torch=1.12.1`) conflict with robocasa's `numpy==2.2.5`.
