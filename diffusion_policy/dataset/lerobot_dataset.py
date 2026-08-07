import os
import h5py
import numpy as np
import random
import json
import math
from copy import deepcopy
from contextlib import contextmanager
from collections import OrderedDict
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.pytorch_util import dict_apply
import robomimic.utils.torch_utils as TorchUtils
from tqdm import tqdm
import robomimic.utils.tensor_utils as TensorUtils
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
import copy

import robomimic.utils.lang_utils as LangUtils
from robomimic.macros import LANG_EMB_KEY

import torch.utils.data
import torch
from typing import Dict, List


from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
from robocasa.utils.skill_utils import parse_task_string

# Structured conditioning slots for the pick / place skill datasets. Each maps a shape_meta
# obs key to the part of the episode's task string it is embedded from. A config declares
# whichever slots that skill takes: pick uses obj_emb alone, place uses both.
SLOT_KEYS = {"obj_emb": "obj", "recep_emb": "recep"}


from robocasa.utils.groot_utils.groot_dataset import LeRobotSingleDataset, LE_ROBOT_MODALITY_FILENAME, ModalityConfig, LE_ROBOT_EPISODE_FILENAME, LeRobotMixtureDataset
import pathlib

def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values,
    maintaining the order: video, state, action, annotation
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Initialize dictionary with ordered keys
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict

class LerobotDataset(LeRobotSingleDataset, BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            filter_key=None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            lang_encoder=None,
            del_lang_encoder_after_init=True,
        ):

        assert n_obs_steps and n_obs_steps > 0
        self.abs_action = abs_action
        assert not self.abs_action, "abs_action is not supported in LerobotDataset"
        dataset_path = pathlib.Path(dataset_path)
        delta_indices = list(range(-n_obs_steps+1, horizon - n_obs_steps + 1))
        delta_indices_obs = list(range(-n_obs_steps+1, 1))
        assert len(delta_indices_obs) == n_obs_steps, \
            f"delta_indices_obs length {len(delta_indices_obs)} != n_obs_steps {n_obs_steps}"   
        modality_keys_dict = get_modality_keys(dataset_path)
        video_modality_keys = modality_keys_dict["video"]
        language_modality_keys = modality_keys_dict["annotation"]
        state_modality_keys = modality_keys_dict["state"]
        action_modality_keys = modality_keys_dict["action"]
        state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]
        modality_configs = {
            "video": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=video_modality_keys,  # we will include all video modalities
            ),
            "state": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=state_modality_keys,
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=action_modality_keys,
            ),
        }

        LeRobotSingleDataset.__init__(
            self,
            dataset_path=dataset_path,
            filter_key=filter_key,
            embodiment_tag="oxe_droid",
            modality_configs=modality_configs,
        )
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths
        rgb_keys = dict()
        lowdim_keys = dict()
        obs_shape_meta = copy.deepcopy(shape_meta['obs'])
        self.lang_emb = obs_shape_meta.pop('lang_emb', None)
        # structured slots, used by the pick / place skill datasets instead of lang_emb
        self.slot_specs = {
            key: obs_shape_meta.pop(key) for key in list(SLOT_KEYS) if key in obs_shape_meta
        }
        assert not (self.lang_emb is not None and self.slot_specs), \
            "use either lang_emb or the structured slots, not both"
        if self.lang_emb is not None or self.slot_specs:
            assert language_modality_keys, "Language modality keys should not be empty if lang_emb is defined"
            self._lang_encoder = lang_encoder
            if self.slot_specs:
                self._get_slot_embeddings()
            else:
                self._get_lang_embeddings()
            if del_lang_encoder_after_init:
                del self._lang_encoder
                self._lang_encoder = None
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys[key] = attr["lerobot_keys"]
            elif type == 'low_dim':
                lowdim_keys[key] = attr["lerobot_keys"]
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.action_info = self.shape_meta['action']
        self.lerobot_action_keys = self.action_info['lerobot_keys']
        self.action_size = self.action_info['shape'][0]
    
    def _get_lang_embeddings(self):
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        if self._lang_encoder is None:
            self._lang_encoder = LangUtils.LangEncoder(
                    device=device,
            )
        self._demo_id_to_demo_lang_emb = {}
    
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        id2remark = {e["episode_index"]: e["tasks"][0] for e in episode_metadata}
        
        for ep_batch in tqdm(np.array_split(self.trajectory_ids, int(math.ceil(len(self.trajectory_ids) / 64)))):
            # get language embedding
            lang_batch = [id2remark[ep] for ep in ep_batch]
            emb_batch = self._lang_encoder.get_lang_emb(lang_batch)
            emb_batch = TensorUtils.to_numpy(emb_batch)
            for batch_idx, ep in enumerate(ep_batch):
                self._demo_id_to_demo_lang_emb[ep] = emb_batch[batch_idx]

    def _get_slot_embeddings(self):
        """
        Embed each conditioning slot separately.

        Episodes in the skill datasets carry a structured task string
        ("pick | obj: beer", "place | obj: beer | recep: cabinet"). Each slot's noun phrase
        is embedded on its own, which is what lets obj and receptacle be swapped
        independently at test time. The phrase vocabulary is tiny relative to the number of
        episodes, so embed the unique phrases once and share them.
        """
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        if self._lang_encoder is None:
            self._lang_encoder = LangUtils.LangEncoder(device=device)

        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        id2task = {e["episode_index"]: e["tasks"][0] for e in episode_metadata}

        # episode -> {slot name: noun phrase}
        ep_slots = {}
        for ep in self.trajectory_ids:
            _, obj, recep = parse_task_string(id2task[ep])
            slots = {"obj": obj, "recep": recep}
            missing = [
                key for key in self.slot_specs if slots[SLOT_KEYS[key]] is None
            ]
            assert not missing, f"episode {ep} task {id2task[ep]!r} has no {missing}"
            ep_slots[ep] = slots

        phrases = sorted({
            ep_slots[ep][SLOT_KEYS[key]]
            for ep in self.trajectory_ids
            for key in self.slot_specs
        })
        embeddings = {}
        for batch in np.array_split(phrases, int(math.ceil(len(phrases) / 64))):
            batch = list(batch)
            emb = TensorUtils.to_numpy(self._lang_encoder.get_lang_emb(batch))
            embeddings.update(zip(batch, emb))

        self._demo_id_to_slot_emb = {
            ep: {
                key: embeddings[ep_slots[ep][SLOT_KEYS[key]]] for key in self.slot_specs
            }
            for ep in self.trajectory_ids
        }

    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # threadpool_limits(1)

        # super call to get data
        data = LeRobotSingleDataset.__getitem__(self, idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key, lerobot_keys in self.rgb_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[lerobot_key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
        for key, lerobot_keys in self.lowdim_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            obs_dict[key] = data[lerobot_key][T_slice].astype(np.float32)

        if self.lang_emb is not None:
            trajectory_id, _ = self.all_steps[idx]
            lang_emb = self._demo_id_to_demo_lang_emb[trajectory_id]
            obs_dict[LANG_EMB_KEY] = np.tile(
                lang_emb,
                (self.n_obs_steps, 1)
            ).astype(np.float32)
        elif self.slot_specs:
            trajectory_id, _ = self.all_steps[idx]
            for key, emb in self._demo_id_to_slot_emb[trajectory_id].items():
                obs_dict[key] = np.tile(emb, (self.n_obs_steps, 1)).astype(np.float32)

        action_concat = []

        for lr_key in self.lerobot_action_keys:
            if lr_key in data:
                action_concat.append(data[lr_key])
            else:
                raise ValueError(f"Key {lr_key} not found in data")
        

        action_concat = np.concatenate(action_concat, axis=-1)
        assert action_concat.shape[-1] == self.action_size, \
            f"action_concat shape mismatch: {action_concat.shape[-1]} != {self.action_size}"
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action_concat.astype(np.float32))
        }

        return torch_data
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # Almost same as robomimic_replay_image_dataset.py
        normalizer = LinearNormalizer()
        assert not self.abs_action, "normalizer for abs_action is not supported in LerobotDataset"


        scale = np.ones((self.action_size), dtype=np.float32)
        offset = np.zeros((self.action_size), dtype=np.float32)
        normalizer['action'] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={}, #stat
        )

        # obs
        for key, lerobot_keys in self.lowdim_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            # strip "state." prefix
            lerobot_key = lerobot_key.replace("state.", "")
            stat = self._metadata.statistics.state[lerobot_key].model_dump()
            for k, v in stat.items():
                if type(v) is np.ndarray:
                    stat[k] = v.astype(np.float32)

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('sin'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('cos'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer
        
        # lang_emb / structured slots: passed through unnormalized
        embedding_specs = (
            {LANG_EMB_KEY: self.lang_emb} if self.lang_emb is not None else self.slot_specs
        )
        for emb_key, spec in embedding_specs.items():
            dim = int(np.prod(spec["shape"]))
            normalizer[emb_key] = SingleFieldLinearNormalizer.create_manual(
                scale=np.ones((dim,), dtype=np.float32),
                offset=np.zeros((dim,), dtype=np.float32),
                input_stats_dict={}, #stat
            )

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

class LerobotCotrainingDataset(LeRobotMixtureDataset, BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_paths: List[str] | None = None,
            dataset_soup=None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0, # validation not implemented yet,
            ds_weights=None,
            ds_weights_alpha=0.40,
            metadata_config: dict = {
            "percentile_mixing_method": "weighted_average",
        } 
        ):
        # exactly one of dataset_paths or dataset_soup must be defined
        assert (dataset_paths == None) + (dataset_soup == None) == 1

        if dataset_soup is not None:
            dataset_soup_list = copy.deepcopy(DATASET_SOUP_REGISTRY[dataset_soup])
        else:
            dataset_soup_list = [
                {"path": ds_path, "filter_key": None}
                for ds_path in dataset_paths
            ]

        for i in range(len(dataset_soup_list)):
            ds_path = dataset_soup_list[i]["path"]
            if not os.path.isabs(ds_path):
                # hack: fill in robocasa base dataset path
                from robocasa.macros import DATASET_BASE_PATH
                dataset_soup_list[i]["path"] = os.path.join(DATASET_BASE_PATH, ds_path)
            
            dataset_soup_list[i]["ds_weight"] = dataset_soup_list[i].get("ds_weight", None)

        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        lang_encoder = LangUtils.LangEncoder(device=device)
        datasets = [
            LerobotDataset(
                shape_meta=shape_meta,
                dataset_path=ds_meta["path"],
                filter_key=ds_meta["filter_key"],
                horizon=horizon,
                pad_after=pad_after,
                pad_before=pad_before,
                n_obs_steps=n_obs_steps,
                abs_action=abs_action,
                rotation_rep=rotation_rep,
                use_legacy_normalizer=use_legacy_normalizer,
                use_cache=use_cache,
                seed=seed,
                val_ratio=val_ratio,
                lang_encoder=lang_encoder,
                del_lang_encoder_after_init=False,
            ) for ds_meta in dataset_soup_list
        ]
        del lang_encoder
        self.abs_action = abs_action
        assert not self.abs_action, "abs_action is not supported in LerobotCotrainingDataset"
        assert ds_weights is None or len(ds_weights) == len(datasets), \
            f"ds_weights length {len(ds_weights)} != datasets length {len(datasets)}"
        
        if ds_weights is None and all(ds_meta["ds_weight"] is not None for ds_meta in dataset_soup_list):
            ds_weights = [ds_meta["ds_weight"] for ds_meta in dataset_soup_list]
        
        if not ds_weights:
            ds_weights = np.array([np.power(len(dataset), ds_weights_alpha) for dataset in datasets])
            # the groot dataloader requires that at least one dataset has weight 1.0
            ds_weights = ds_weights / ds_weights[0]
        print("dataset weights:", ds_weights)
        
        dataset_mixture = list(zip(datasets, ds_weights))
        # set balance_dataset_weights to False, since we are calculating weights ourselves
        LeRobotMixtureDataset.__init__(self,  data_mixture=dataset_mixture, mode="train",  balance_dataset_weights=False, balance_trajectory_weights=False, metadata_config=metadata_config)
        rgb_keys = dict()
        lowdim_keys = dict()
        obs_shape_meta = copy.deepcopy(shape_meta['obs'])
        self.lang_emb = obs_shape_meta.pop('lang_emb', None)
        self.slot_specs = {
            key: obs_shape_meta.pop(key) for key in list(SLOT_KEYS) if key in obs_shape_meta
        }
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys[key] = attr["lerobot_keys"]
            elif type == 'low_dim':
                lowdim_keys[key] = attr["lerobot_keys"]
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.action_info = self.shape_meta['action']
        self.lerobot_action_keys = self.action_info['lerobot_keys']
        self.action_size = self.action_info['shape'][0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset, trajectory_name, step = self.sample_step(idx)
        global_ds_index = self.to_global_index(dataset, trajectory_name, step)
        return dataset.__getitem__(global_ds_index)

    def to_global_index(self, dataset, trajectory_id: int, base_index: int) -> int:
        """Convert (trajectory_id, base_index) → global index for a given dataset"""
        traj_idx = dataset.get_trajectory_index(trajectory_id) 
        g_idx = int(dataset.start_indices[traj_idx] + base_index)
        # # TODO: remove
        # assert g_idx == dataset.all_steps.index((trajectory_id, base_index)), \
        #     f"g_idx {g_idx} != dataset.all_steps.index({trajectory_id}, {base_index})"
        return g_idx
    
    def __len__(self):
        return np.sum(self.dataset_lengths)

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # Almost same as robomimic_replay_image_dataset.py
        normalizer = LinearNormalizer()
        assert not self.abs_action, "normalizer for abs_action is not supported in LerobotDataset"

        # tag should be same for all datasets
        tag = self.datasets[0].tag
        # TODO, look into how these vals are affected in original code
        all_stats = self.merged_metadata[tag].statistics

        scale = np.ones((self.action_size), dtype=np.float32)
        offset = np.zeros((self.action_size), dtype=np.float32)
        normalizer['action'] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={}, #stat
        )


        for key, lerobot_keys in self.lowdim_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            # strip "state." prefix
            lerobot_key = lerobot_key.replace("state.", "")
            stat = all_stats.state[lerobot_key].model_dump()
            for k, v in stat.items():
                if type(v) is np.ndarray:
                    stat[k] = v.astype(np.float32)

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('sin'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('cos'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer
        # lang_emb / structured slots: passed through unnormalized
        embedding_specs = (
            {LANG_EMB_KEY: self.lang_emb} if self.lang_emb is not None else self.slot_specs
        )
        for emb_key, spec in embedding_specs.items():
            dim = int(np.prod(spec["shape"]))
            normalizer[emb_key] = SingleFieldLinearNormalizer.create_manual(
                scale=np.ones((dim,), dtype=np.float32),
                offset=np.zeros((dim,), dtype=np.float32),
                input_stats_dict={}, #stat
            )

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

