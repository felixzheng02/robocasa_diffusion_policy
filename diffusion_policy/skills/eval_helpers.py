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
    """Load a trained policy and the shape_meta it was trained with."""
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
    One diagnostic frame: third-person view beside the wrist view.

    Both are needed to tell the failure modes apart -- agentview shows whether the arm went
    to the right place at all, eye-in-hand shows whether it was aligned on the object and
    simply mistimed the gripper.
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
    """deque of per-step obs dicts -> {key: (1, n_obs_steps, ...)}."""
    while len(history) < n_obs_steps:
        history.appendleft(history[0])
    window = list(history)[-n_obs_steps:]
    return {
        key: np.stack([step[key] for step in window])[None]
        for key in window[0]
    }
