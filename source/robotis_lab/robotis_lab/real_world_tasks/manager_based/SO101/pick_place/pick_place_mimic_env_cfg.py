# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .joint_pos_env_cfg import SO101BottlePickPlaceEnvCfg, SO101BottlePickPlaceTable1EnvCfg


def _configure_mimic(cfg) -> None:
    cfg.datagen_config.name = "pick_and_place_the_bottle_in_the_basket"
    cfg.datagen_config.generation_guarantee = True
    cfg.datagen_config.generation_keep_failed = True
    cfg.datagen_config.generation_num_trials = 10
    cfg.datagen_config.generation_select_src_per_subtask = True
    cfg.datagen_config.generation_transform_first_robot_pose = False
    cfg.datagen_config.generation_interpolate_from_last_target_pose = True
    cfg.datagen_config.generation_relative = True
    cfg.datagen_config.max_num_failures = 25
    cfg.datagen_config.seed = 42

    subtask_configs = []
    # First subtask: Grasp the bottle
    subtask_configs.append(
        SubTaskConfig(
            object_ref="bottle",
            subtask_term_signal="grasp_bottle",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.003,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description="Grasp bottle",
            next_subtask_description="Place bottle in basket",
        )
    )
    # Second subtask: Place bottle in basket
    subtask_configs.append(
        SubTaskConfig(
            object_ref="basket",
            subtask_term_signal="bottle_in_basket",
            subtask_term_offset_range=(5, 15),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.001,
            num_interpolation_steps=10,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description="Place bottle in basket",
            next_subtask_description="Task complete",
        )
    )
    subtask_configs.append(
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="random",
            selection_strategy_kwargs={},
            action_noise=0.0001,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
    )
    cfg.subtask_configs["SO101"] = subtask_configs


@configclass
class SO101PickPlaceMimicEnvCfg(SO101BottlePickPlaceEnvCfg, MimicEnvCfg):
    """Configuration for the SO101 pick_place task with mimic environment."""

    def __post_init__(self):
        super().__post_init__()
        _configure_mimic(self)


@configclass
class SO101PickPlaceTable1MimicEnvCfg(SO101BottlePickPlaceTable1EnvCfg, MimicEnvCfg):
    """Configuration for the SO101 pick_place task with mimic environment (Table1 assets)."""

    def __post_init__(self):
        super().__post_init__()
        _configure_mimic(self)
