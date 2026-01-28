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

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from robotis_lab.real_world_tasks.manager_based.OMY.pick_place.mdp import omy_pick_place_events
from robotis_lab.real_world_tasks.manager_based.SO101.pick_place.pick_place_env_cfg import PickPlaceEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotis_lab.assets.robots.SO101 import SO101_CFG  # isort: skip
from robotis_lab.assets.object.robotis_omy_table import OMY_TABLE_CFG
from robotis_lab.assets.object.sdh_omx_table1 import SDH_OMX_TABLE1_CFG
from robotis_lab.assets.object.plastic_bottle import PLASTIC_BOTTLE_CFG
from robotis_lab.assets.object.plastic_basket import PLASTIC_BASKET_CFG
from robotis_lab.assets.object.sdh_basket1 import SDH_BASKET1_CFG
from robotis_lab.assets.object.sdh_listerine import SDH_LISTERINE_CFG
from robotis_lab.assets.object.sdh_vaseline import SDH_VASELINE_CFG

import math


def _apply_robot_x_offset(scene, delta_x: float) -> None:
    pos = scene.robot.init_state.pos if scene.robot.init_state.pos is not None else [0.0, 0.0, 0.0]
    pos = list(pos)
    pos[0] += delta_x
    scene.robot.init_state.pos = pos


@configclass
class EventCfg:
    """Configuration for events."""

    init_so101_arm_pose = EventTerm(
        func=omy_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [-0.13, -1.73, 1.62, 1.29, -0.032, 0.022],
        },
    )

    randomize_so101_joint_state = EventTerm(
        func=omy_pick_place_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_bottle_positions = EventTerm(
        func=omy_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (-0.22, -0.15), "y": (-0.08, 0.22), "z": (0.05, 0.05), "yaw": (-math.pi / 2 - 0.90, -math.pi / 2 + 0.90)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("bottle")],
        },
    )

    randomize_basket_positions = EventTerm(
        func=omy_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, -0.15), "y": (-0.3, -0.2), "z": (0.03, 0.03), "roll": (-math.pi / 2, -math.pi / 2), "pitch": (math.pi, math.pi), "yaw": (0.0, 0.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("basket")],
        },
    )

    randomize_scene_light = EventTerm(
        func=omy_pick_place_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": (1000.0, 3000.0),
            "color_range": ((0.7, 1.0), (0.7, 1.0), (0.7, 1.0)),
            "asset_cfg": SceneEntityCfg("light"),
        },
    )

    randomize_top_camera = EventTerm(
        func=omy_pick_place_events.randomize_camera_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cam_top"),
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
            "convention": "ros",
        },
    )


@configclass
class SO101BottlePickPlaceEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set SO101 as robot
        self.scene.robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        # _apply_robot_x_offset(self.scene, 0.35)
        

        # Set table
        self.scene.table = OMY_TABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Table")

        self.scene.bottle = PLASTIC_BOTTLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Bottle")

        self.scene.basket = PLASTIC_BASKET_CFG.replace(prim_path="{ENV_REGEX_NS}/Basket")

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Wrist camera on link5 for SO101
        self.scene.cam_wrist = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/so101/link5/cam_wrist",
            update_period=0.0,
            height=480,
            width=848,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=11.8, focus_distance=200.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.00236, -0.05351, -0.00602),
                rot=(0.72276, 0.0421, 0.0442, -0.6884),
                convention="isaac",
            )
        )
        self.scene.cam_top = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Table/cam_top",
            update_period=0.0,
            height=480,
            width=848,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0, focus_distance=200.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.7),
                rot=(0.7071068, 0.0, 0.0, -0.7071068),
                convention="isaac",
            )
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/so101/link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/so101/link7",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )


@configclass
class SO101BottlePickPlaceTable1EnvCfg(SO101BottlePickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Swap table asset for the scanned table
        self.scene.table = SDH_OMX_TABLE1_CFG.replace(prim_path="{ENV_REGEX_NS}/Table")
        # Swap bottle and basket assets for the custom ones
        self.scene.bottle = SDH_LISTERINE_CFG.replace(prim_path="{ENV_REGEX_NS}/Bottle")
        # self.scene.bottle = SDH_VASELINE_CFG.replace(prim_path="{ENV_REGEX_NS}/Bottle")
        self.scene.basket = SDH_BASKET1_CFG.replace(prim_path="{ENV_REGEX_NS}/Basket")
        # Shift robot for the scanned table setup
        
        #  절대위치
        self.scene.robot.init_state.pos = [-0.35, 0.0, 0.0]
        
        #  상대위치
        # _apply_robot_x_offset(self.scene, 0.35)

        
