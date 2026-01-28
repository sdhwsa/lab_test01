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

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from robotis_lab.assets.robots.SO101_dual import SO101_L_CFG, SO101_R_CFG
from robotis_lab.assets.object.sdh_omx_table1 import SDH_OMX_TABLE1_CFG
from robotis_lab.assets.object.sdh_basket1 import SDH_BASKET1_CFG
from robotis_lab.assets.object.sdh_listerine import SDH_LISTERINE_CFG
from robotis_lab.real_world_tasks.manager_based.OMY.pick_place.mdp import omy_pick_place_events
from robotis_lab.real_world_tasks.manager_based.SO101.dual_pick_place.pick_place_env_cfg import DualPickPlaceEnvCfg

import math


SO101_L_ROOT = "so101_L"
SO101_R_ROOT = "so101_R"


@configclass
class EventCfg:
    """Configuration for events."""

    init_so101_l_arm_pose = EventTerm(
        func=omy_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "asset_cfg": SceneEntityCfg("robot_l"),
        },
    )

    init_so101_r_arm_pose = EventTerm(
        func=omy_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "asset_cfg": SceneEntityCfg("robot_r"),
        },
    )

    randomize_so101_l_joint_state = EventTerm(
        func=omy_pick_place_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot_l"),
        },
    )

    randomize_so101_r_joint_state = EventTerm(
        func=omy_pick_place_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot_r"),
        },
    )

    randomize_bottle_positions = EventTerm(
        func=omy_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, 0.0), "y": (-0.05, 0.2), "z": (0.05, 0.05),
                           "yaw": (-math.pi / 2 - 0.50, -math.pi / 2 + 0.50)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("bottle")],
        },
    )

    randomize_basket_positions = EventTerm(
        func=omy_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, -0.15), "y": (-0.4, -0.25), "z": (0.01, 0.01),
                           "roll": (-math.pi / 2, -math.pi / 2), "pitch": (math.pi, math.pi), "yaw": (0.0, 0.0)},
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
class SO101DualBottlePickPlaceEnvCfg(DualPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()

        self.scene.robot_l = SO101_L_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotL")
        self.scene.robot_l.spawn.semantic_tags = [("class", "robot_l")]
        self.scene.robot_r = SO101_R_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotR")
        self.scene.robot_r.spawn.semantic_tags = [("class", "robot_r")]

        self.scene.table = SDH_OMX_TABLE1_CFG.replace(prim_path="{ENV_REGEX_NS}/Table")
        self.scene.bottle = SDH_LISTERINE_CFG.replace(prim_path="{ENV_REGEX_NS}/Bottle")
        self.scene.basket = SDH_BASKET1_CFG.replace(prim_path="{ENV_REGEX_NS}/Basket")

        self.scene.plane.semantic_tags = [("class", "ground")]

        self.scene.cam_wrist_l = CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/RobotL/{SO101_L_ROOT}/link5/cam_wrist_l",
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
            ),
        )
        self.scene.cam_wrist_r = CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/RobotR/{SO101_R_ROOT}/link5/cam_wrist_r",
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
            ),
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
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame_l = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/RobotL/{SO101_L_ROOT}/link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/RobotL/{SO101_L_ROOT}/link7",
                    name="end_effector_l",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )
        self.scene.ee_frame_r = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/RobotR/{SO101_R_ROOT}/link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/RobotR/{SO101_R_ROOT}/link7",
                    name="end_effector_r",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )


@configclass
class SO101DualBottlePickPlaceTable1EnvCfg(SO101DualBottlePickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot_l.init_state.pos = [-0.35, 0.3, 0.0]
        self.scene.robot_r.init_state.pos = [-0.35, -0.15, 0.0]
