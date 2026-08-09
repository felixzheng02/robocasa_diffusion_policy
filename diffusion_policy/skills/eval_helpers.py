"""
Diffusion-policy helpers shared by the skill evaluators.

These were defined at the top of `eval_chained_pick_place.py`, which made that file both an
argparse entrypoint and the import target for five other evaluators -- so pulling in an
11-line helper executed the whole script's module body, and it resolved at all only when the
process happened to be launched from the repo root.

They live in the package now so a consumer outside this repo -- the VLM planner, which
orchestrates diffusion-policy checkpoints -- can import them by a real package path.

Only genuinely diffusion-policy-specific helpers belong here. `create_env` and `base_env`
are NOT diffusion-policy code (one wraps robocasa's own gym registration, the other unwinds
the wrapper stack that registration builds) and live in `robocasa.utils.env_helpers`.
"""

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace


def load_policy(checkpoint, device):
    """
    Load a trained diffusion policy from a checkpoint file, ready for inference.

    A checkpoint stores both the weights and the full hydra config it was trained with, so
    this also hands back the `shape_meta` describing the observation and action keys the
    policy expects. The caller needs that to know which slots to feed it (`obj_emb` for the
    pick policy, `obj_emb` + `recep_emb` for place).

    Inputs
    ------
    checkpoint : str
        Path to a `.ckpt` file written by the training workspace.
    device : str or torch.device
        Where to put the model, e.g. `"cuda:0"` or `"cpu"`.

    Outputs
    -------
    policy : BaseImagePolicy
        The model, moved to `device` and switched to eval mode. This is the EMA copy when
        the run trained with `use_ema`, otherwise the plain weights.
    shape_meta : dict
        The `cfg.task.shape_meta` block as plain Python (not OmegaConf). Has an `obs` sub-dict
        mapping each observation key to its `shape` and `type`, and an `action` entry.
        Interpolations are left unresolved.

    Procedure
    ---------
    1. Read the checkpoint file with `dill` as the unpickler, giving a payload dict.
    2. Pull the training config out of `payload["cfg"]`.
    3. Build the workspace object whose class is named by `cfg._target_`.
    4. Restore every saved key into that workspace with `load_payload`.
    5. Choose `workspace.ema_model` if the run used EMA, otherwise `workspace.model`.
    6. Move the chosen model to `device` and call `.eval()` on it.
    7. Return the model alongside `cfg.task.shape_meta` converted to a plain dict.
    """
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=None)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(torch.device(device))
    policy.eval()

    return policy, OmegaConf.to_container(cfg.task.shape_meta, resolve=False)


def obs_to_frame(obs):
    """
    Build one diagnostic video frame: the third-person view beside the wrist view.

    Both cameras are needed to tell the failure modes apart -- agentview shows whether the arm
    went to the right place at all, eye-in-hand shows whether it was aligned on the object and
    simply mistimed the gripper.

    Inputs
    ------
    obs : dict[str, np.ndarray]
        One observation. Only two keys are read, in this order:
        `"robot0_agentview_right_image"` and `"robot0_eye_in_hand_image"`. Each is either
        `(C, H, W)` or `(T, C, H, W)` float in [0, 1]. Any other key is ignored, and a
        missing camera key is skipped rather than raising.

    Outputs
    -------
    frame : np.ndarray, `(H, W * n_panels, 3)`, uint8 in [0, 255], or None
        The panels joined left to right. `n_panels` is however many of the two camera keys
        were present. Returns `None` when neither was found, which the caller must handle --
        it means no video can be written for this step.

    Procedure
    ---------
    1. Walk the two camera keys in a fixed order so panels never swap places between frames.
    2. Skip any key that is not in `obs`.
    3. If an image carries a time axis (`ndim == 4`), keep only its last step.
    4. Move the channel axis to the end, turning `(C, H, W)` into `(H, W, C)`.
    5. Rescale the [0, 1] floats to [0, 255], clip, and cast to uint8.
    6. Concatenate the collected panels along the width axis, or return `None` if there are none.
    """
    panels = []
    for key in ("robot0_agentview_right_image", "robot0_eye_in_hand_image"):
        if key not in obs:
            continue
        img = obs[key]
        if img.ndim == 4:  # (T, C, H, W) -> last step
            img = img[-1]
        img = np.moveaxis(img, 0, -1)  # CHW -> HWC
        panels.append((img * 255).clip(0, 255).astype(np.uint8))
    return np.concatenate(panels, axis=1) if panels else None


def stack_obs(history, n_obs_steps):
    """
    Turn a rolling history of single-step observations into one batched policy input.

    The policy always wants exactly `n_obs_steps` frames of context. At the very start of a
    rollout there are fewer than that, so the oldest frame is repeated to fill the gap -- the
    policy sees a stationary scene rather than a short sequence.

    Inputs
    ------
    history : collections.deque[dict[str, np.ndarray]]
        Per-step observation dicts, oldest first. Every dict must carry the same keys.
        Must not be empty. **Mutated in place**: if it is shorter than `n_obs_steps`, copies
        of its oldest entry are pushed onto the front (see step 1).
    n_obs_steps : int
        How many frames of context the policy expects. Comes from the training config.

    Outputs
    -------
    stacked : dict[str, np.ndarray]
        Same keys as one history entry. Each value has shape `(1, n_obs_steps, *feature_shape)`
        -- a leading batch axis of 1, then time, then whatever that key's own shape is
        (`(3, 128, 128)` for a camera, `(768,)` for a slot embedding). Dtype is unchanged.

    Procedure
    ---------
    1. While the history holds fewer than `n_obs_steps` entries, duplicate its oldest entry
       onto the front until it is long enough.
    2. Take the last `n_obs_steps` entries, so a longer history is trimmed to the newest window.
    3. For each key, stack that key across the window into `(n_obs_steps, *feature_shape)`.
    4. Insert a leading batch axis of size 1 and return the resulting dict.
    """
    while len(history) < n_obs_steps:
        history.appendleft(history[0])
    window = list(history)[-n_obs_steps:]
    return {
        key: np.stack([step[key] for step in window])[None]
        for key in window[0]
    }
