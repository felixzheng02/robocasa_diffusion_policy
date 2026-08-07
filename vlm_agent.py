"""
VLM planner and monitor for the agentic pick -> place demo.

Talks to a locally served Qwen2.5-VL through vLLM's OpenAI-compatible endpoint. Two
prompts share one transport:

    plan()     once per episode (and again on a replan): natural-language instruction
               plus the scene's candidate phrases -> a skill sequence with slot arguments
    monitor()  every few action chunks: one frame plus the gripper opening
               -> continue / advance / interrupt

Both go through vLLM's guided decoding (`response_format: json_schema`), so the planner
*cannot* emit a slot phrase outside the scene's vocabulary and the monitor cannot emit a
decision outside the three. That matters more than it looks: the policies are conditioned
on CLIP embeddings of exact phrases, so a free-form "cupboard" instead of "cabinet" would
land somewhere the place policy never saw and read as a policy failure rather than a
grounding one.

Deliberately dependency-free beyond what the robocasa_dp env already carries (requests,
PIL). The server lives in its own conda env because vLLM's torch pin fights this one's
numpy==2.2.5 / mujoco==3.3.1, and because a VLM OOM must not take a multi-hour sweep with
it.
"""

import base64
import io
import json
import os

import requests
from PIL import Image

import robocasa.utils.skill_utils as SU

BASE_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1")

# Set by probe(). vLLM keys requests by the exact served id, which depends on which
# checkpoint was passed to `vllm serve`, so we ask rather than hard-code.
MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

DECISIONS = ("continue", "advance", "interrupt")


def probe():
    """
    Return the model id the server is actually serving, or None if it is unreachable.

    Doubles as the startup health check: calling this before a sweep turns "the server was
    down for 200 of 210 monitor calls" from a silently bad result into a refusal to start.
    """
    global MODEL
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=5)
        r.raise_for_status()
        MODEL = r.json()["data"][0]["id"]
        return MODEL
    except Exception:
        return None


def _chat(messages, schema, timeout, max_tokens):
    """
    One constrained turn. Returns the parsed object, or None on any failure.

    Every failure mode collapses to None on purpose — the caller runs inside a rollout and
    must never take an exception from a flaky HTTP call. The caller counts the Nones.
    """
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": schema},
        },
    }
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception:
        return None


def to_jpeg_b64(panel, quality=80):
    """HWC uint8 -> data URI. JPEG rather than PNG: at 256px the artefacts are invisible
    to the model and the payload is ~6x smaller, which shows up in prefill latency."""
    buf = io.BytesIO()
    Image.fromarray(panel).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _msg(text, panel):
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": to_jpeg_b64(panel)}},
            {"type": "text", "text": text},
        ],
    }


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------


def scene_vocab(ep_meta):
    """
    Candidate slot phrases for one episode: {"objects": [...], "receptacles": [...]}.

    Objects come from this scene's object_cfgs, so the list is the target plus its one to
    three distractors and chance level is 1/len — worth logging alongside any accuracy
    number.

    Receptacles are the *training* vocabulary (the plain values of PICK_PLACE_TARGETS),
    not the scene's fixtures. ep_meta["fixtures"] lists dozens of walls, floors and
    counters, and anything outside PICK_PLACE_TARGETS is an out-of-distribution CLIP
    embedding for the place policy. The six tasks whose target is a container object
    rather than a fixture ("obj:container", "obj:plate") are covered by the union with
    the scene's own objects.
    """
    objects = []
    for cfg in ep_meta.get("object_cfgs", []):
        lang = SU._format_cat_as_lang(cfg.get("info", {}).get("cat", ""))
        if lang and lang not in objects:
            objects.append(lang)

    receptacles = sorted(
        {t for t in SU.PICK_PLACE_TARGETS.values() if not t.startswith("obj:")}
    )
    receptacles += [o for o in objects if o not in receptacles]
    return {"objects": objects, "receptacles": receptacles}


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

PLAN_PROMPT = """You are the planner for a kitchen robot with exactly two skills:
  pick(obj)          grasp obj and lift it clear of whatever it is resting on
  place(obj, recep)  put the currently held obj into or onto recep

Instruction: "{instruction}"

Objects in this scene: {objects}
Receptacles: {receptacles}

The image shows the scene now. Room camera on the left, gripper camera on the right.

Emit the shortest skill sequence that satisfies the instruction. Use the listed phrases
verbatim. Set recep to null on pick steps. Every place step must repeat the obj it is
placing.{situation}"""

SITUATION_PROMPT = """

The previous attempt was aborted: {situation}
Re-plan from the robot's current state as shown in the image."""


def plan(instruction, panel, vocab, situation=None, timeout=20.0):
    """
    Choose a skill sequence. Returns {"reason": str, "plan": [{skill, obj, recep}]}, or
    None if the server did not answer.

    `reason` is first in the schema on purpose. Guided decoding emits fields in schema
    order, so a short justification is generated *before* the plan commits to a slot —
    cheap chain-of-thought where it is free (one call per episode). It is what gives the
    model a chance on the five tasks whose instruction never names the object, e.g.
    PickPlaceToasterToCounter's "Place the toasted item on a plate."
    """
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["reason", "plan"],
        "properties": {
            "reason": {"type": "string", "maxLength": 160},
            "plan": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["skill", "obj", "recep"],
                    "properties": {
                        "skill": {"enum": list(SU.SKILLS)},
                        "obj": {"enum": vocab["objects"]},
                        # null is a member rather than the field being optional: guided
                        # decoding cannot express "required iff skill == place", so the
                        # key is always present and pick steps are normalised below.
                        "recep": {"enum": vocab["receptacles"] + [None]},
                    },
                },
            },
        },
    }
    text = PLAN_PROMPT.format(
        instruction=instruction,
        objects=", ".join(vocab["objects"]),
        receptacles=", ".join(vocab["receptacles"]),
        situation="" if situation is None else SITUATION_PROMPT.format(situation=situation),
    )
    out = _chat([_msg(text, panel)], schema, timeout, max_tokens=256)
    if out is None:
        return None
    for step in out["plan"]:
        if step["skill"] == "pick":
            step["recep"] = None
    return out


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------
#
# The monitor is the one call that is deliberately NOT schema-constrained, and the reason
# is measured. Asked the identical question about the identical frame, this model answers
#
#     guided decoding, {gripper_contains: str}   -> "nothing"   0/6 correct
#     free text, one sentence                    -> "a Kellogg's cereal box"  5/6 correct
#
# on the final frames of pick-skill demo episodes, where the object is grasped by
# construction. vLLM's grammar backend collapses this quantized model's perception, so the
# perceptual question is asked in free text and mapped to a decision in Python below.
#
# Answer length also moves the operating point, measured over 12 demo episodes (one empty
# frame and one holding frame each):
#
#     max_tokens=12, terse noun     -> names object on  4/10 holding, "nothing" on 7/10 empty
#     max_tokens=80, one sentence   -> says holding on 11/12 holding, but ALSO on 8/12 empty
#
# So "did it say holding?" is not a usable rule — it fires on two thirds of empty grippers.
# Requiring the sentence to NAME THE TARGET is:
#
#     advance on a real grasp        5/12
#     false advance on empty gripper 2/12
#
# That is the trade this loop wants. A false advance hands `place` an empty gripper and
# loses the episode outright; a missed advance only burns steps until SKILL_BUDGET forces
# the handoff, which is exactly the baseline's fixed schedule. Sensitivity is the honest
# weak point of the demo — see the resolution note in monitor().

PICK_QUESTION = (
    "The left half is a room camera, the right half is a camera mounted on the robot's "
    "gripper. The robot is trying to pick up the {obj}. What object, if any, is the "
    "gripper currently holding? One sentence."
)

PLACE_QUESTION = (
    "The left half is a room camera, the right half is a camera mounted on the robot's "
    "gripper. The robot is trying to put the {obj} into/onto the {recep}. What object, if "
    "any, is the gripper currently holding? One sentence."
)

# Phrases that mean "the gripper is empty".
_EMPTY = (
    "empty", "nothing", "not holding", "no object", "not currently holding",
    "does not appear to be holding", "isn't holding", "does not seem to be holding",
)
_STOP = {"the", "a", "an", "of", "and", "with"}


def _names(sentence, phrase):
    """True when `sentence` mentions any content word of `phrase`."""
    words = [w for w in phrase.lower().split() if w not in _STOP and len(w) > 2]
    return any(w in sentence for w in words)


def monitor(panel, skill, obj, recep, grip, distractors=(), timeout=8.0):
    """
    One decision about the skill in flight. Returns (decision, sentence); (None, None) if
    the server did not answer.

    The mapping is deliberately asymmetric, for the cost reason in the comment above:

        pick   names the target            -> advance    (the grasp happened)
               names a different scene obj -> interrupt  (grasped the wrong thing)
               otherwise                   -> continue
        place  reports an empty gripper    -> advance    (released onto the receptacle)
               names a different scene obj -> interrupt
               otherwise                   -> continue

    `grip` is accepted and logged by the caller but not put in the prompt: adding it moved
    no decisions in testing, and every token in the prompt is prefill latency.

    Known limitation: sensitivity is ~40% at the 256x256 camera resolution the policies
    train on. The wrist crop is simply small. Rendering a larger monitor-only view with
    sim.render(height=384, width=384, camera_name="robot0_eye_in_hand") is the obvious
    next lever and costs nothing per step, since it would only run on monitor ticks.
    """
    q = (PICK_QUESTION if skill == "pick" else PLACE_QUESTION).format(obj=obj, recep=recep)
    body = {
        "model": MODEL,
        "messages": [_msg(q, panel)],
        "max_tokens": 80,
        "temperature": 0,
    }
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        sentence = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None, None

    s = sentence.lower()
    empty = any(k in s for k in _EMPTY)
    wrong = (not empty) and any(_names(s, d) for d in distractors if not _names(obj, d))

    if skill == "pick":
        decision = "advance" if (not empty and _names(s, obj)) else (
            "interrupt" if wrong else "continue")
    else:
        decision = "advance" if empty else ("interrupt" if wrong else "continue")
    return decision, sentence
