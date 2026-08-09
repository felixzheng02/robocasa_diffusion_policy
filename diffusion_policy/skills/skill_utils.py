"""
The pick/place split criterion, and the slot machinery built on it.

This is *methodology*, not simulation: what counts as a completed pick, how long a segment
is padded, how a task string is formed. It is imported by the offline splitter **and** by
every online evaluator, which is the point -- the training boundary and the eval handoff
cannot drift apart if there is only one definition of both.

The physical predicates it rests on (`is_holding_obj`, `obj_pos`, ...) belong to the
simulator and are re-exported here so existing callers keep working unchanged.
"""

import numpy as np

from robocasa.scripts.dataset_scripts.playback_utils import _format_cat_as_lang
from robocasa.utils.pick_place_tasks import (   # noqa: F401  (re-exported for callers)
    PICK_PLACE_TARGETS,
    PICK_PLACE_TASKS,
    SKILLS,
)
from robocasa.utils.predicates import (          # noqa: F401  (re-exported for callers)
    gripper_obj_dist,
    is_holding_obj,
    obj_pos,
    touches_obj,
)

MOVE_DZ = 0.02  # metres the object's height must change to count as picked up
MOVE_WINDOW = 60  # frames after the grasp in which that must happen (3.0 s)
MIN_GRASP_RUN = 5  # ignore grasp blips shorter than this
GAP_CLOSE = 5  # bridge dropouts up to this many frames
PICK_TAIL_PAD = 10  # frames kept after t_moved in the pick segment
PLACE_TAIL_PAD = 15  # frames kept after t_success in the place segment
MIN_SEG_LEN = 32  # reject degenerate segments (horizon 10 + n_obs 2, with slack)

# per-frame signals cached by extract_grasp_signals.py and consumed by select_split_points.py
SIGNAL_KEYS = ["held", "held_loose", "obj_z", "gdist", "success", "reward", "grip_cmd"]


# ---------------------------------------------------------------------------
# grasp / move predicates
# ---------------------------------------------------------------------------

class GraspMoveDetector:
    """
    Decides, live during a rollout, the moment a pick has succeeded.

    This is the online twin of the offline rule `split_episode` applies to recorded demos.
    Both live in this module on purpose: the condition that ends a pick segment in the
    training data and the condition that hands off from the pick policy to the place policy
    at rollout time are the same code, so they cannot drift apart.

    The rule in words: the object counts as picked once it has been held continuously for
    `MIN_GRASP_RUN` steps **and** its height has moved by `MOVE_DZ` from wherever it was when
    the hold began. Requiring both is what separates a real lift from a gripper closing on
    empty air or brushing a distractor.

    Once fired it stays fired -- this is a latch, not a per-frame predicate. Feed it every
    step of a rollout with `update`.

    Attributes
    ----------
    obj_name : str
        Which object in the scene is being watched.
    z0 : float or None
        Height in metres latched the first time the object was held; `None` until then.
        Never re-latched, so a regrasp does not move the reference.
    run : int
        How many consecutive steps the object has been held. Reset to 0 whenever the hold
        breaks.
    fired : bool
        Whether the pick condition has been met. Once True it never returns to False.
    t : int or None
        The step index at which it fired, counted from the first `update` call; `None` if it
        never fired.

    Notes
    -----
    The displacement test is deliberately **unsigned**. Half the PickPlace tasks carry the
    object *downwards* -- cabinet to counter drops it about 45 cm -- and an upward-only test
    would throw those demos away. A failed grasp or a grasp on a distractor still leaves the
    target object's height unchanged, so the test keeps its original meaning.
    """

    def __init__(self, obj_name="obj"):
        """
        Create a detector that has not yet seen any steps.

        Inputs
        ------
        obj_name : str
            Which object in the scene to watch. Defaults to `"obj"`, the manipulation target
            in every PickPlace task.

        Outputs
        -------
        None
            Initialises the instance in place.

        Procedure
        ---------
        1. Store the object name to watch.
        2. Set the latched height `z0` to None -- nothing has been held yet.
        3. Zero the consecutive-hold counter `run`.
        4. Set `fired` False and the fire step `t` to None.
        5. Zero the internal step counter used to stamp `t`.
        6. Set the `_ever_strict` flag False, which is what gates the loose contact fallback.
        """
        self.obj_name = obj_name
        self.z0 = None
        self.run = 0
        self.fired = False
        self.t = None
        self._step = 0
        self._ever_strict = False

    def update(self, env):
        """
        Feed one simulation step to the detector and report whether the pick has succeeded.

        Call this exactly once per env step, in order, for the whole rollout. Skipping steps
        corrupts both the consecutive-hold counter and the reported fire time.

        Inputs
        ------
        env : robosuite Kitchen env
            Already unwrapped. Read at its current step; nothing is stepped or mutated.

        Outputs
        -------
        fired : bool
            True once the pick condition has been met, on this step or any earlier one.
            Also readable afterwards as `self.fired`, with the step index in `self.t`.

        Procedure
        ---------
        1. Test strict fingerpad contact and remember whether it has *ever* held (`_ever_strict`).
        2. Take the object as held if strict contact fires now, or -- only for an episode that
           has never once shown strict contact -- if any gripper geom touches it.
        3. Read the object's current height.
        4. If held, increment the consecutive-hold counter and latch `z0` on the first hold
           ever; if not held, reset the counter to 0 but leave `z0` alone.
        5. If not already fired, the hold has lasted at least `MIN_GRASP_RUN` steps, and a
           height has been latched, fire when the height has moved at least `MOVE_DZ` in
           either direction, stamping `t` with the current step index.
        6. Advance the internal step counter and return the latch.

        Notes
        -----
        Step 2 is where the online rule approximates the offline one and is the most likely
        source of train/eval drift. Offline, `split_episode` sees the whole trajectory and
        falls back to loose contact only when the strict tier yielded nothing for the entire
        episode. Online we must commit frame by frame, so `not _ever_strict` stands in for
        "strict contact never happens" using only the past. If eval shows low `pick_success`
        concentrated on handled objects -- ladles, measuring cups, spoons -- look here first.
        """
        # Same two tiers the splitter uses: prefer fingerpad contact, fall back to any
        # gripper contact for handled objects that can never satisfy it. The displacement
        # requirement below is what keeps the fallback honest.
        strict = is_holding_obj(env, self.obj_name)
        self._ever_strict |= strict
        held = strict or (not self._ever_strict and touches_obj(env, self.obj_name))
        z = obj_pos(env, self.obj_name)[2]

        if held:
            self.run += 1
            if self.z0 is None:
                self.z0 = z
        else:
            self.run = 0

        if not self.fired and self.run >= MIN_GRASP_RUN and self.z0 is not None:
            if abs(z - self.z0) >= MOVE_DZ:
                self.fired = True
                self.t = self._step

        self._step += 1
        return self.fired


# ---------------------------------------------------------------------------
# trajectory helpers (pure numpy, used by select_split_points.py)
# ---------------------------------------------------------------------------


def close_gaps(mask, max_gap=GAP_CLOSE):
    """
    Bridge short dropouts in a boolean signal.

    Contact detection flickers: a genuine continuous grasp produces a mask that drops to
    False for a frame or two at a time. Closing those gaps first is what lets `true_runs`
    report one long grasp instead of a dozen fragments.

    Inputs
    ------
    mask : array-like of bool, `(T,)`
        One boolean per frame, e.g. the per-frame `held` signal. Not modified.
    max_gap : int
        Longest run of False that may be filled in, in frames. Defaults to `GAP_CLOSE` (5,
        i.e. 0.25 s at 20 Hz). A gap longer than this is left alone.

    Outputs
    -------
    closed : np.ndarray, `(T,)`, bool
        A new array. Same as the input except that False runs of at most `max_gap` frames
        lying strictly between two True values are set True. Leading and trailing False runs
        are never filled, since they have no True on both sides.

    Procedure
    ---------
    1. Copy the input into a fresh bool array so the caller's data is untouched.
    2. Find the indices of every True value.
    3. Return the copy unchanged if there are fewer than two -- nothing can be bridged.
    4. For each neighbouring pair of True indices, measure the distance between them and,
       if they are not already adjacent and the gap is at most `max_gap` frames, set
       everything between them True.
    5. Return the filled mask.
    """
    mask = np.asarray(mask, dtype=bool).copy()
    (idx,) = np.where(mask)
    if len(idx) < 2:
        return mask
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            mask[a:b] = True
    return mask


def true_runs(mask, min_len=1):
    """
    Find every maximal stretch of consecutive True values in a boolean signal.

    Used to turn a per-frame contact mask into candidate grasp episodes, which the splitter
    then filters down to the one run that ends the pick.

    Inputs
    ------
    mask : array-like of bool, `(T,)`
        One boolean per frame. Usually the output of `close_gaps`. Not modified.
    min_len : int
        Discard any run shorter than this many frames. Defaults to 1, which keeps everything.
        The splitter passes `MIN_GRASP_RUN` to drop contact blips.

    Outputs
    -------
    runs : list[tuple[int, int]]
        `(start, end_exclusive)` pairs, in increasing order, covering each maximal True run
        of at least `min_len` frames. Half-open, so a run's length is `end - start` and
        `mask[start:end]` selects it. Empty list if the mask is all False.

    Procedure
    ---------
    1. Coerce the input to a bool array.
    2. Pad a False onto each end, so runs touching a boundary still produce edges.
    3. Difference the padded array: +1 marks a False-to-True edge, -1 a True-to-False edge.
    4. Read the +1 positions as run starts and the -1 positions as exclusive run ends.
    5. Pair them up in order, keep only pairs spanning at least `min_len` frames, and return
       them as plain ints.
    """
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


# ---------------------------------------------------------------------------
# slot extraction
# ---------------------------------------------------------------------------

# task -> the receptacle the object ends up in/on. A plain string is used verbatim as the
# noun phrase; "obj:<name>" means "use the category of the object cfg called <name>", which
# is how tasks whose receptacle is a movable object (a pan, a plate, a bowl) get filled in.
#
# Granularity matches what each task's _check_success actually tests, and the wording
# follows each task's own get_ep_meta in
# robocasa/environments/kitchen/atomic/kitchen_pick_place.py.
#
# There is no source slot: pick is pick(obj). The object is identified by category, and
# where it currently sits is visible in the images.


def _obj_cat_lang(ep_meta, cfg_name):
    """
    Look up one object's category in an episode's metadata and phrase it as English.

    Inputs
    ------
    ep_meta : dict
        An episode's metadata. Only `ep_meta["object_cfgs"]` is read: a list of per-object
        dicts, each with a `"name"` (its role in the scene, e.g. `"obj"`, `"container"`) and
        an `"info"` sub-dict carrying a `"cat"` category string.
    cfg_name : str
        The role name to find, e.g. `"obj"` or `"plate"`.

    Outputs
    -------
    lang : str or None
        The category as a noun phrase, e.g. `"peanut butter"` from a `peanut_butter`
        category. `None` if no cfg has that name, or if the one found has no category --
        callers treat both as "this episode cannot fill that slot".

    Procedure
    ---------
    1. Walk the episode's object cfgs, defaulting to an empty list if the key is missing.
    2. Stop at the first whose `"name"` matches `cfg_name`.
    3. Pull its category string out of `info`, defaulting to empty.
    4. Format that category as a noun phrase and return it.
    5. Return None if the loop finds no match.
    """
    for cfg in ep_meta.get("object_cfgs", []):
        if cfg.get("name") == cfg_name:
            return _format_cat_as_lang(cfg.get("info", {}).get("cat", ""))
    return None


def _resolve(spec, ep_meta):
    """
    Turn one `PICK_PLACE_TARGETS` table entry into the noun phrase for this episode.

    Table entries come in two forms because a receptacle is sometimes a fixture that is the
    same in every episode (a cabinet) and sometimes a movable object drawn per episode from
    a category pool (whatever container happens to be in the sink).

    Inputs
    ------
    spec : str
        A table value. Either a literal noun phrase used verbatim (`"cabinet"`), or an
        `"obj:<cfg_name>"` reference meaning "use the category of the object cfg called
        `<cfg_name>` in this episode" (`"obj:container"`).
    ep_meta : dict
        The episode's metadata, read only when `spec` is an `obj:` reference.

    Outputs
    -------
    phrase : str
        The receptacle as a bare noun phrase, ready to embed.

    Raises
    ------
    ValueError
        If `spec` references an object cfg this episode does not have. Loud on purpose: a
        silently empty receptacle would train the place policy against a blank slot.

    Procedure
    ---------
    1. If `spec` does not start with `"obj:"`, return it unchanged -- it is already a phrase.
    2. Otherwise strip the prefix to get the cfg name and look up that object's category.
    3. Raise `ValueError` if the episode has no such object.
    4. Return the resolved category phrase.
    """
    if spec.startswith("obj:"):
        lang = _obj_cat_lang(ep_meta, spec[4:])
        if not lang:
            raise ValueError(f"episode has no object cfg named {spec[4:]!r}")
        return lang
    return spec


def make_skill_slots(env_name, ep_meta):
    """
    Work out the conditioning slots for one episode: what is picked, and where it goes.

    These two noun phrases are what replace free-form language in this project. Each is
    embedded separately by the frozen CLIP text encoder, which is what lets an argument be
    swapped at test time without a sentence template.

    Inputs
    ------
    env_name : str
        The task class name, e.g. `"PickPlaceCounterToCabinet"`. Must be one of the 18 keys
        in `PICK_PLACE_TARGETS`.
    ep_meta : dict
        The episode's metadata, carrying `"object_cfgs"`.

    Outputs
    -------
    slots : dict[str, str]
        Exactly two keys, both bare noun phrases:
        - `"obj"` -- the object being manipulated, e.g. `"beer"`.
        - `"target"` -- the receptacle it ends up in or on, e.g. `"cabinet"`.
        The pick skill uses `"obj"` alone; place uses both.

    Raises
    ------
    KeyError
        If `env_name` is not a PickPlace task.
    ValueError
        If the episode has no object cfg named `"obj"`, or the receptacle cannot be resolved.

    Procedure
    ---------
    1. Reject any task name absent from the `PICK_PLACE_TARGETS` table.
    2. Look up the category of the cfg named `"obj"` -- the manipulation target.
    3. Raise if that object is missing; an episode with no object cannot be split.
    4. Resolve this task's table entry into a receptacle phrase for this episode.
    5. Return both as a dict.

    Notes
    -----
    `ep_meta["lang"]` is never parsed. Five of the eighteen tasks store a place-only or
    non-parametric sentence, and `PickPlaceToasterToCounter` never names the object at all
    ("Place the toasted item on a plate."). Everything here comes from `object_cfgs` plus the
    hard-coded table instead.
    """
    if env_name not in PICK_PLACE_TARGETS:
        raise KeyError(f"{env_name} is not a PickPlace task")

    obj = _obj_cat_lang(ep_meta, "obj")
    if not obj:
        raise ValueError("episode has no object cfg named 'obj'")

    return {"obj": obj, "target": _resolve(PICK_PLACE_TARGETS[env_name], ep_meta)}


# ---------------------------------------------------------------------------
# the structured task string carried through the LeRobot datasets
# ---------------------------------------------------------------------------



def slots_to_task_string(skill, slots):
    """
    Encode a skill and its slots as the one-line `task` string stored with each episode.

    LeRobot datasets carry a single free-text `task` field per episode, so the structured
    slots have to be packed into it. This function and `parse_task_string` are the two ends
    of that format::

        pick  | obj: beer
        place | obj: beer | recep: cabinet

    Inputs
    ------
    skill : str
        Either `"pick"` or `"place"`. Anything else fails an assertion.
    slots : dict[str, str]
        As returned by `make_skill_slots`. `"obj"` is always read; `"target"` is read only
        for place.

    Outputs
    -------
    task : str
        The encoded string. Two fields for pick, three for place, separated by `" | "`.

    Procedure
    ---------
    1. Assert the skill is one we know.
    2. For pick, emit the skill name and the object slot.
    3. For place, emit the skill name, the object slot, and the receptacle slot.

    Notes
    -----
    The dataset class splits this back apart and embeds each noun phrase separately, which is
    what makes the object and the receptacle independently swappable at test time.
    """
    assert skill in SKILLS, skill
    if skill == "pick":
        return f"pick | obj: {slots['obj']}"
    return f"place | obj: {slots['obj']} | recep: {slots['target']}"


def parse_task_string(task):
    """
    Decode an episode's `task` string back into its skill and slots.

    The exact inverse of `slots_to_task_string`. Every malformed input raises rather than
    half-parsing, because a slot that silently comes back wrong would condition a policy on
    the wrong noun with no visible error.

    Inputs
    ------
    task : str
        A task string such as `"pick | obj: beer"` or
        `"place | obj: beer | recep: cabinet"`. Whitespace around each field is tolerated.

    Outputs
    -------
    skill : str
        Either `"pick"` or `"place"`.
    obj : str
        The object noun phrase.
    recep : str or None
        The receptacle noun phrase for place; always `None` for pick.

    Raises
    ------
    ValueError
        If the skill name is unknown, a field is missing or misspelled, or the field count
        does not match the skill (2 for pick, 3 for place).

    Procedure
    ---------
    1. Split on `"|"` and strip whitespace from each field.
    2. Read the skill from the first field and reject anything not in `SKILLS`.
    3. Require the second field to start with `"obj:"`, and take its value.
    4. For pick, require exactly two fields and return with `recep` as None.
    5. For place, require exactly three fields with the third starting `"recep:"`, and
       return its value as the receptacle.
    """
    parts = [p.strip() for p in task.split("|")]
    skill = parts[0]
    if skill not in SKILLS:
        raise ValueError(f"unknown skill in task string: {task!r}")
    if not parts[1].startswith("obj:"):
        raise ValueError(f"malformed skill task string: {task!r}")
    obj = parts[1][len("obj:"):].strip()

    if skill == "pick":
        if len(parts) != 2:
            raise ValueError(f"pick task string must have exactly one slot: {task!r}")
        return skill, obj, None

    if len(parts) != 3 or not parts[2].startswith("recep:"):
        raise ValueError(f"malformed place task string: {task!r}")
    return skill, obj, parts[2][len("recep:"):].strip()
