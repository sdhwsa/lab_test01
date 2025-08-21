# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import (
    agents,
    ik_rel_env_cfg,
)

from .mimic_env import PickPlaceFFWBG2MimicEnv
from .mimic_env_cfg import PickPlaceFFWBG2MimicEnvCfg

gym.register(
    id="RobotisLab-PickPlace-FFW-BG2-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.PickPlaceFFWBG2EnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="RobotisLab-PickPlace-FFW-BG2-Mimic-v0",
    entry_point="robotis_lab.simulation_tasks.manager_based.FFW_BG2.pick_place:PickPlaceFFWBG2MimicEnv",
    kwargs={
        "env_cfg_entry_point": mimic_env_cfg.PickPlaceFFWBG2MimicEnvCfg,
    },
    disable_env_checker=True,
)
