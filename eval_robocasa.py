"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""
import gymnasium as gym
import robocasa
from diffusion_policy.common.pytorch_util import dict_apply

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from omegaconf import OmegaConf
import copy

import os
import pathlib
import argparse
import hydra
import torch
import dill
import wandb
import json
from termcolor import colored
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY


def eval_task(checkpoint, base_output_dir, device, task, num_rollouts, num_envs, split, overwrite):
    if base_output_dir is None:
        base_output_dir = os.path.join(os.path.dirname(checkpoint), "../evals", os.path.basename(checkpoint).replace(".ckpt", ""), split)

    output_dir = os.path.join(base_output_dir, task)

    out_path = os.path.join(output_dir, 'eval_log.json')
    if overwrite is False and os.path.exists(out_path):
        # click.confirm(f"Output path {out_path} already exists! Overwrite?", abort=True)
        print(f"Eval stats path {out_path} already exists! Skipping.")
        return

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg = copy.deepcopy(OmegaConf.to_container(cfg))
    cfg["task"]["env_runner"]["env_kwargs"] = {
        "split": split,
        "seed": 1111111,
        "env_name": task,
    }
    cfg = OmegaConf.create(cfg)

    horizon = get_task_horizon(task=task)
    
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_test = num_rollouts

    # set dataset path and horizon
    cfg.task.env_runner.max_steps = int(horizon)
    cfg.task.env_runner.n_envs = num_envs

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    try_num = 1
    MAX_TRIES = 5
    while try_num <= MAX_TRIES:
        env_runner = None
        runner_log = None
        # try:
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir)
        runner_log = env_runner.run(policy)
        # except Exception as e:
        #     print(f"Excpetion in env_runner (try {try_num})")
        #     print(e)
        #     print()
        #     try_num += 1
        #     continue
        
        break
    
    # dump log to json
    if runner_log is not None:
        json_log = dict()
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                json_log[key] = value._path
            else:
                json_log[key] = value
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    # close and delete everything
    if env_runner is not None:
        env_runner.close()
    del policy
    del workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-o', '--output_dir', default=None)
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('-t', '--task_set', required=True, nargs='+')
    parser.add_argument('-n', '--num_rollouts', default=50, type=int)
    parser.add_argument('-e', '--num_envs', default=5, type=int)
    parser.add_argument('-s', '--split', required=True)
    args = parser.parse_args()

    all_tasks = []
    for task_soup_i in args.task_set:
        all_tasks += TASK_SET_REGISTRY[task_soup_i]
    all_tasks = set(all_tasks)
    
    for task_i, task in enumerate(all_tasks):
        print(colored(f"[{task_i+1}/{len(all_tasks)}] running evals for {task}", "yellow"))
        eval_task(args.checkpoint, args.output_dir, args.device, task, args.num_rollouts, args.num_envs, args.split, overwrite=False)

if __name__ == '__main__':
    main()