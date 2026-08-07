"""
Agentic pick -> place evaluation: a VLM plans the skill sequence and monitors execution.

Where eval_chained_pick_place.py hard-codes [pick, place], reads its slots from ground
truth and hands off on SU.GraspMoveDetector, this script asks a locally served Qwen2.5-VL
to do all three jobs:

    planner  reads ep_meta["lang"] plus the scene's candidate phrases and emits a skill
             sequence. Called once, and again after every interrupt.
    monitor  looks at one frame every --monitor_every action chunks and returns
             continue / advance / interrupt. This is what moves the cursor.

GraspMoveDetector is still updated on every step, but purely as an observer: the
`gm_agreement` field in each rollout records where it fired against where the VLM chose
to advance, which is the number that answers "should the VLM own the handoff at all".

The monitor call is asynchronous. It is fired after predict_action so its latency hides
under the eight env steps that follow, and resolved inside the chunk loop so an `advance`
can cut the chunk — a stale pick action executed after the handoff drags the object, which
is the same reason eval_chained_pick_place.py breaks on its detector.

Three arms share this file and the baseline, all at the same seeds:

    baseline      eval_chained_pick_place.py    ground-truth slots, detector handoff
    oracle-slots  this file, --oracle_slots     ground-truth slots, VLM handoff
    full agentic  this file                     VLM slots,          VLM handoff

Example:
    MUJOCO_GL=egl python eval_agentic_pick_place.py \\
        --pick_checkpoint  outputs/pick_skill/checkpoints/latest.ckpt \\
        --place_checkpoint outputs/place_skill/checkpoints/latest.ckpt \\
        --tasks PickPlaceCounterToCabinet --num_rollouts 1
"""

import argparse
import collections
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import imageio
import numpy as np
import torch
from termcolor import colored

import robocasa  # noqa: F401  (registers the gym envs)
import robocasa.utils.skill_utils as SU
import vlm_agent as VA
from robomimic.utils.lang_utils import LangEncoder

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.lerobot_dataset import SLOT_KEYS
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from robocasa.utils.dataset_registry_utils import get_task_horizon

# The obs pipeline is imported, not forked, so it is literally identical across the arms
# being compared.
from eval_chained_pick_place import base_env, load_policy, obs_to_frame, stack_obs

# Chunks between monitor calls. One chunk is 8 env steps (~1.3 s wall clock), so 2 puts a
# decision request in flight roughly every 2.6 s against a ~2 s round trip.
MONITOR_EVERY = 2
# Steps a freshly started skill runs before its first monitor call. Without this the first
# frame after a handoff still shows the pick pose and draws an immediate spurious decision.
GRACE_STEPS = 32
MAX_REPLANS = 2
# Steps to keep running after the plan is exhausted. Every PickPlace _check_success ANDs
# OU.gripper_obj_far, so success only registers once the gripper retreats — without this
# tail a VLM that advances on "the object is in the cabinet" scores a false failure.
TAIL_STEPS = 20


def plan_summary(plan):
    """'pick(cereal) -> place(cereal, cabinet)' — the one-line form used in logs."""
    return " -> ".join(
        f"{s['skill']}({s['obj']})" if s["recep"] is None
        else f"{s['skill']}({s['obj']}, {s['recep']})"
        for s in plan
    )


def run_episode(env_name, split, seed, shape_meta, policies, encoder, horizon,
                n_obs_steps, device, monitor_every=MONITOR_EVERY, oracle_slots=False,
                video_path=None):
    """One agentic rollout. Returns a dict of outcome flags and the decision trace."""
    t_start = time.time()

    # The place schema is the superset, so the wrapper always emits both slots and phase
    # selection happens on the policy input instead. Handing the wrapper a phase-shaped
    # dict is the bug CLAUDE.md records.
    wrapper_meta = shape_meta["place"]
    wrapper = RobomimicImageWrapper(
        env=create_env(split=split, env_name=env_name, seed=seed),
        shape_meta=wrapper_meta,
        init_state=None,
        render_obs_key="robot0_agentview_right_image",
    )
    dim = wrapper_meta["obs"]["obj_emb"]["shape"][0]
    zero = np.zeros(dim, dtype=np.float32)
    wrapper.slot_embs = {k: zero for k in SLOT_KEYS if k in wrapper_meta["obs"]}
    wrapper.reset()
    sim = base_env(wrapper)

    ep_meta = sim.get_ep_meta()
    lang = ep_meta.get("lang", "")
    vocab = VA.scene_vocab(ep_meta)
    truth = SU.make_skill_slots(env_name, ep_meta)  # scoring only, never fed to the VLM

    # A replan re-embeds phrases it has already seen, and the CLIP pass runs on CPU.
    emb_cache = {}

    def emb(phrase):
        if phrase not in emb_cache:
            emb_cache[phrase] = encoder.get_lang_emb(phrase).numpy()
        return emb_cache[phrase]

    def apply_step(spec):
        """Point the wrapper at one plan step, filling every slot key its schema declares."""
        full = {
            "obj_emb": emb(spec["obj"]),
            "recep_emb": emb(spec["recep"]) if spec["recep"] else zero,
        }
        wrapper.slot_embs = {k: v for k, v in full.items() if k in wrapper_meta["obs"]}

    history = collections.deque(maxlen=n_obs_steps)
    first_obs = dict(wrapper.get_observation(wrapper.last_raw_obs))
    history.append(first_obs)

    frames = []

    def record(obs):
        if video_path is None:
            return
        f = obs_to_frame(obs)
        if f is not None:
            frames.append(f)

    record(first_obs)

    # ---- plan -------------------------------------------------------------------
    if oracle_slots:
        plan = [
            {"skill": "pick", "obj": truth["obj"], "recep": None},
            {"skill": "place", "obj": truth["obj"], "recep": truth["target"]},
        ]
        reason = "oracle"
    else:
        out = VA.plan(lang, obs_to_frame(first_obs), vocab)
        if out is None:
            wrapper.env.close()
            return {"end_reason": "planner_failed", "obj": truth["obj"],
                    "task_success": False, "pick_success": False,
                    "still_holding": False, "steps": 0, "handoff_step": None,
                    "final_dz": 0.0, "wall_s": round(time.time() - t_start, 1)}
        plan, reason = out["plan"], out["reason"]

    plans_log = [{"step": 0, "plan": plan, "reason": reason}]
    apply_step(plan[0])

    # ---- rollout ----------------------------------------------------------------
    detector = SU.GraspMoveDetector()  # observer only; never gates control flow
    budget = horizon                   # identical to the baseline, deliberately
    skill_budget = int(0.5 * horizon)

    plan_id = cursor = replans = 0
    step = skill_step = chunks_since_fire = 0
    pending = None          # (future, plan_id, cursor, step_fired)
    tail_until = None       # set when the plan is exhausted; disables monitoring
    success = replan_capped = False
    end_reason = None
    decisions, latencies = [], []
    monitor_calls = monitor_errors = 0
    vlm_advance_step = None  # where the VLM ended the first pick step
    z0 = SU.obj_pos(sim)[2]

    def monitor_call(panel, spec, grip):
        # The scene's other objects are what "grasped the wrong thing" is measured against.
        distractors = [o for o in vocab["objects"] if o != spec["obj"]]
        t0 = time.time()
        d, sentence = VA.monitor(panel, spec["skill"], spec["obj"], spec["recep"], grip,
                                 distractors=distractors)
        return d, sentence, time.time() - t0

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        while step < budget:
            if tail_until is not None and step >= tail_until:
                break
            decision = None
            cur = plan[cursor]

            window = stack_obs(history, n_obs_steps)
            obs_dict = dict_apply(
                {k: v for k, v in window.items() if k in shape_meta[cur["skill"]]["obs"]},
                lambda x: torch.from_numpy(x.astype(np.float32)).to(device),
            )
            with torch.no_grad():
                action = policies[cur["skill"]].predict_action(obs_dict)["action"][0].cpu().numpy()

            # Fire after predict_action: the 1.3 s diffusion call is already spent, so
            # firing earlier would only add to the decision's staleness.
            if (pending is None and tail_until is None
                    and chunks_since_fire >= monitor_every and skill_step >= GRACE_STEPS):
                grip = float(np.sum(history[-1]["robot0_gripper_qpos"]))
                pending = (pool.submit(monitor_call, obs_to_frame(history[-1]), cur, grip),
                           plan_id, cursor, step)
                chunks_since_fire = 0

            for a in action:
                if step >= budget or (tail_until is not None and step >= tail_until):
                    break
                raw, _, _, _ = wrapper.step(a)
                raw = dict(raw)
                history.append(raw)
                step += 1
                skill_step += 1
                detector.update(sim)
                record(raw)

                if sim._check_success():
                    success = True
                    end_reason = "success"
                    break

                # Resolve inside the chunk so advance can cut it.
                if pending is not None and pending[0].done():
                    fut, pid, pcur, fired = pending
                    pending = None
                    monitor_calls += 1
                    try:
                        d, sentence, lat = fut.result()
                    except Exception:
                        d, sentence, lat = None, None, 0.0
                    latencies.append(lat)
                    if d is None:
                        monitor_errors += 1
                    elif pid == plan_id and pcur == cursor:
                        decisions.append({
                            "step_fired": fired, "step_applied": step, "lag": step - fired,
                            "cursor": pcur, "skill": cur["skill"], "decision": d,
                            "saw": sentence,   # the raw sentence the decision was mapped from
                        })
                        if d in ("advance", "interrupt"):
                            decision = d
                            break
                    # else: the skill it referred to already ended — drop it.

            chunks_since_fire += 1
            if success:
                break
            if tail_until is not None:
                continue  # tail runs the last skill open-loop until the deadline

            # A VLM that always answers `continue` degrades to the baseline schedule
            # rather than hanging.
            if skill_step >= skill_budget:
                decision = "advance"
                decisions.append({
                    "step_fired": None, "step_applied": step, "lag": None,
                    "cursor": cursor, "skill": cur["skill"], "decision": "advance_forced",
                })

            if decision == "advance":
                if cur["skill"] == "pick" and vlm_advance_step is None:
                    vlm_advance_step = step
                cursor += 1
                skill_step = chunks_since_fire = 0
                pending = None
                if cursor < len(plan):
                    apply_step(plan[cursor])
                elif replans < MAX_REPLANS:
                    decision = "interrupt"  # nothing left to run; try a new plan
                else:
                    end_reason = "plan_exhausted"
                    tail_until = step + TAIL_STEPS
                    cursor = len(plan) - 1
                    apply_step(plan[cursor])
                    continue

            if decision == "interrupt":
                if replans >= MAX_REPLANS:
                    replan_capped = True  # stop honouring interrupts; budget ends it
                else:
                    replans += 1
                    plan_id += 1  # invalidates anything still in flight
                    situation = (f"{cur['skill']}({cur['obj']}) had not finished by step "
                                 f"{step} of {budget}")
                    out = VA.plan(lang, obs_to_frame(history[-1]), vocab, situation=situation)
                    if out is None:
                        end_reason = "replan_failed"
                        break
                    plan, cursor = out["plan"], 0
                    plans_log.append({"step": step, "plan": plan, "reason": out["reason"]})
                    skill_step = chunks_since_fire = 0
                    pending = None
                    apply_step(plan[0])
    finally:
        # Per rollout, so a straggler cannot be mistaken for the next rollout's decision.
        pool.shutdown(wait=False, cancel_futures=True)

    if end_reason is None:
        end_reason = "budget" if step >= budget else "tail"

    still_holding = bool(SU.is_holding_obj(sim))
    dz = float(SU.obj_pos(sim)[2] - z0)

    if video_path is not None and frames:
        tag = "OK" if success else "FAIL"
        out_path = pathlib.Path(str(video_path).replace("$TAG", tag))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out_path, frames, fps=20, quality=6, macro_block_size=1)

    wrapper.env.close()

    # Scored against the *initial* plan, like slot_obj_correct — these two measure how
    # well the planner grounded the instruction, which a later replan would muddy.
    initial = plans_log[0]["plan"]
    first_place = next((s for s in initial if s["skill"] == "place"), None)
    return {
        # --- keys the baseline also returns, same meaning, so the JSONs diff directly.
        # One difference worth knowing: the baseline only updates the detector while
        # phase == "pick", so its detector.t stops at the handoff. Here it runs the whole
        # episode, which makes detector_step an honest number.
        "pick_success": bool(detector.fired),
        "task_success": bool(success),
        "handoff_step": vlm_advance_step,   # where the VLM moved off pick
        "steps": step,
        "still_holding": still_holding,
        "final_dz": round(dz, 4),
        "obj": truth["obj"],
        # --- agentic
        "end_reason": end_reason,
        "instruction": lang,
        "plan": plan_summary(initial),
        "plans": plans_log,
        "n_replans": replans,
        "replan_capped": replan_capped,
        "plan_shape_ok": [s["skill"] for s in initial] == ["pick", "place"],
        "slot_obj_correct": initial[0]["obj"] == truth["obj"],
        "slot_recep_correct": bool(first_place and first_place["recep"] == truth["target"]),
        "obj_vocab_size": len(vocab["objects"]),
        "decisions": decisions,
        "gm_agreement": {
            "vlm_advance_step": vlm_advance_step,
            "detector_step": detector.t,
            "delta": (vlm_advance_step - detector.t
                      if vlm_advance_step is not None and detector.t is not None else None),
        },
        "monitor_calls": monitor_calls,
        "monitor_errors": monitor_errors,
        "mean_monitor_latency_s": round(float(np.mean(latencies)), 2) if latencies else None,
        "wall_s": round(time.time() - t_start, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pick_checkpoint", required=True)
    p.add_argument("--place_checkpoint", required=True,
                   help="required: the agentic loop needs both skills available")
    p.add_argument("--tasks", nargs="+", default=SU.PICK_PLACE_TASKS)
    p.add_argument("--split", default="pretrain")
    p.add_argument("--num_rollouts", type=int, default=50)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="agentic_eval.json")
    p.add_argument("--monitor_every", type=int, default=MONITOR_EVERY,
                   help="action chunks between monitor calls (1 chunk = 8 env steps)")
    p.add_argument("--oracle_slots", action="store_true",
                   help="skip the planner and use ground-truth slots; isolates the monitor")
    p.add_argument("--video_dir", default=None)
    p.add_argument("--video_n", type=int, default=3)
    args = p.parse_args()

    served = VA.probe()
    if served is None:
        raise SystemExit(
            f"no VLM at {VA.BASE_URL} — start it first:  ./serve_vlm.sh\n"
            "(it must be running before this script: vLLM preallocates its memory pool)"
        )
    print(colored(f"VLM: {served}", "cyan"))
    print(colored(f"mode: {'ORACLE SLOTS' if args.oracle_slots else 'FULL AGENTIC'}"
                  f"  monitor_every={args.monitor_every} chunks", "cyan"))

    pick_policy, pick_meta = load_policy(args.pick_checkpoint, args.device)
    place_policy, place_meta = load_policy(args.place_checkpoint, args.device)
    policies = {"pick": pick_policy, "place": place_policy}
    shape_meta = {"pick": pick_meta, "place": place_meta}
    n_obs_steps = pick_policy.n_obs_steps

    encoder = LangEncoder(device="cpu")

    results, per_rollout = {}, {}
    for task in args.tasks:
        horizon = int(get_task_horizon(task=task))
        rollouts = []
        for i in range(args.num_rollouts):
            video_path = None
            if args.video_dir and i < args.video_n:
                video_path = str(pathlib.Path(args.video_dir) / f"{task}_ep{i:02d}_$TAG.mp4")
            try:
                rollouts.append(
                    run_episode(task, args.split, args.seed + i, shape_meta, policies,
                                encoder, horizon, n_obs_steps, args.device,
                                monitor_every=args.monitor_every,
                                oracle_slots=args.oracle_slots, video_path=video_path)
                )
            except Exception as exc:  # a broken scene should not kill the sweep
                print(colored(f"{task} rollout {i} failed: {exc}", "red"))
        if not rollouts:
            continue
        per_rollout[task] = rollouts
        mean = lambda k, rs=rollouts: float(np.mean([bool(r.get(k)) for r in rs]))
        picked = [r for r in rollouts if r["pick_success"]]
        deltas = [r["gm_agreement"]["delta"] for r in rollouts
                  if r.get("gm_agreement", {}).get("delta") is not None]
        results[task] = {
            "n": len(rollouts),
            "pick_success": mean("pick_success"),
            "still_holding": mean("still_holding"),
            "task_success": mean("task_success"),
            "place_given_pick": (float(np.mean([r["task_success"] for r in picked]))
                                 if picked else 0.0),
            "slot_obj_correct": mean("slot_obj_correct"),
            "slot_recep_correct": mean("slot_recep_correct"),
            "plan_shape_ok": mean("plan_shape_ok"),
            "mean_replans": float(np.mean([r.get("n_replans", 0) for r in rollouts])),
            "median_gm_delta": float(np.median(deltas)) if deltas else None,
            "monitor_errors": int(sum(r.get("monitor_errors", 0) for r in rollouts)),
        }
        r = results[task]
        print(f"{task:32s} pick={r['pick_success']:.2f} hold={r['still_holding']:.2f} "
              f"place|pick={r['place_given_pick']:.2f} task={r['task_success']:.2f} "
              f"slot_obj={r['slot_obj_correct']:.2f} gm_delta={r['median_gm_delta']}")

    if results:
        keys = ["pick_success", "still_holding", "place_given_pick", "task_success",
                "slot_obj_correct", "slot_recep_correct", "plan_shape_ok"]
        results["AVERAGE"] = {
            k: float(np.mean([v[k] for t, v in results.items() if t != "AVERAGE"]))
            for k in keys
        }
        print("\nAVERAGE " + json.dumps(results["AVERAGE"]))

        errs = sum(v.get("monitor_errors", 0) for t, v in results.items() if t != "AVERAGE")
        if errs:
            print(colored(f"WARNING: {errs} monitor calls failed — a run where the server "
                          f"was flaky must not be read as a policy result", "yellow"))

    pathlib.Path(args.output).write_text(
        json.dumps({"summary": results, "rollouts": per_rollout}, indent=2)
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
