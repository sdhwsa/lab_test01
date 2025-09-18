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

"""
Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import numpy as np

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def euler_from_quat_xyz_stable(quat):
    """
    Convert quaternion to Euler XYZ while keeping roll in [-pi/2, pi/2]
    and avoiding 180° yaw flips.
    """
    w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]

    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2*(w*y - z*x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi/2), torch.asin(sinp))

    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # force roll into [-pi/2, pi/2]
    mask = roll.abs() > torch.pi/2
    roll = torch.where(mask, roll - torch.sign(roll)*torch.pi, roll)
    yaw = torch.where(mask, yaw + torch.pi, yaw)

    # normalize yaw
    yaw = (yaw + torch.pi) % (2*torch.pi) - torch.pi

    return torch.stack([roll, pitch, yaw], dim=-1)

def task_done(env: ManagerBasedRLEnv, bottle_cfg: SceneEntityCfg, yaw_threshold: float = 0.2) -> torch.Tensor:
    """
    Success = bottle rotated ~180° (yaw near ±π), and lifted above ground.
    """

    bottle: RigidObject = env.scene[bottle_cfg.name]
    pos = bottle.data.root_pos_w
    quat = bottle.data.root_quat_w  # (w,x,y,z)

    euler = euler_from_quat_xyz_stable(quat)
    yaw = euler[:, 2]  # Extract yaw

    # --- yaw success condition ---
    # Only accept if near ±pi (not near 0!)
    yaw_ok = (yaw > torch.pi - yaw_threshold) | (yaw < -torch.pi + yaw_threshold)

    # --- position success condition ---
    z_ok = pos[:, 2] > 0.0

    done = yaw_ok & z_ok

    return done
