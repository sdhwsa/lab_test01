from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from robotis_lab.real_world_tasks.manager_based.SO101.pick_place import mdp


@configclass
class ObjectTableDualSceneCfg(InteractiveSceneCfg):
    """Scene with two SO101 robots and a table object."""

    robot_l: ArticulationCfg = MISSING
    robot_r: ArticulationCfg = MISSING
    ee_frame_l: FrameTransformerCfg = MISSING
    ee_frame_r: FrameTransformerCfg = MISSING

    table: AssetBaseCfg = MISSING

    cam_wrist_l: CameraCfg = MISSING
    cam_wrist_r: CameraCfg = MISSING
    cam_top: CameraCfg = MISSING

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    print("ActionsCfg: dual arm_action/gripper_action will be set by agent env cfg")
    arm_action_l: mdp.ActionTermCfg = MISSING
    gripper_action_l: mdp.ActionTermCfg = MISSING
    arm_action_r: mdp.ActionTermCfg = MISSING
    gripper_action_r: mdp.ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)

        joint_pos_l = ObsTerm(
            func=mdp.joint_pos_name,
            params={"joint_names": ["joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "gripper_joint_1_L"],
                    "asset_name": "robot_l"},
        )
        joint_vel_l = ObsTerm(
            func=mdp.joint_vel_name,
            params={"joint_names": ["joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "gripper_joint_1_L"],
                    "asset_name": "robot_l"},
        )
        joint_pos_r = ObsTerm(
            func=mdp.joint_pos_name,
            params={"joint_names": ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "gripper_joint_1_R"],
                    "asset_name": "robot_r"},
        )
        joint_vel_r = ObsTerm(
            func=mdp.joint_vel_name,
            params={"joint_names": ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "gripper_joint_1_R"],
                    "asset_name": "robot_r"},
        )
        cam_wrist_l = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_wrist_l"), "data_type": "rgb", "normalize": False},
        )
        cam_wrist_r = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_wrist_r"), "data_type": "rgb", "normalize": False},
        )
        cam_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_top"), "data_type": "rgb", "normalize": False},
        )
        ee_frame_state_l = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame_l"), "robot_cfg": SceneEntityCfg("robot_l")},
        )
        ee_frame_state_r = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame_r"), "robot_cfg": SceneEntityCfg("robot_r")},
        )
        joint_pos_target_l = ObsTerm(
            func=mdp.joint_pos_target_name,
            params={"joint_names": ["joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "gripper_joint_1_L"],
                    "asset_name": "robot_l"},
        )
        joint_pos_target_r = ObsTerm(
            func=mdp.joint_pos_target_name,
            params={"joint_names": ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "gripper_joint_1_R"],
                    "asset_name": "robot_r"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_bottle = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot_l"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame_l"),
                "object_cfg": SceneEntityCfg("bottle"),
            },
        )

        bottle_in_basket = ObsTerm(
            func=mdp.bottle_in_basket,
            params={
                "bottle_cfg": SceneEntityCfg("bottle"),
                "basket_cfg": SceneEntityCfg("basket"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.task_done,
        params={"bottle_cfg": SceneEntityCfg("bottle"), "basket_cfg": SceneEntityCfg("basket"), "distance_threshold": 0.1},
    )

    bottle_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("bottle")},
    )


@configclass
class DualPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the dual SO101 pick and place environment."""

    scene: ObjectTableDualSceneCfg = ObjectTableDualSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    recorders: RecordTerm = RecordTerm()

    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 30.0
        self.sim.dt = 0.01
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

    def init_action_cfg(self, mode: str):
        print(f"Initializing action configuration for device: {mode}")
        if mode in ["record", "inference"]:
            self.actions.arm_action_l = mdp.JointPositionActionCfg(
                asset_name="robot_l",
                joint_names=["joint[1-5]_L"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.gripper_action_l = mdp.JointPositionActionCfg(
                asset_name="robot_l",
                joint_names=["gripper_joint_1_L"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.arm_action_r = mdp.JointPositionActionCfg(
                asset_name="robot_r",
                joint_names=["joint[1-5]_R"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.gripper_action_r = mdp.JointPositionActionCfg(
                asset_name="robot_r",
                joint_names=["gripper_joint_1_R"],
                scale=1.0,
                use_default_offset=False,
            )
        elif mode in ["teleop_ik", "mimic_ik"]:
            self.actions.arm_action_l = DifferentialInverseKinematicsActionCfg(
                asset_name="robot_l",
                joint_names=["joint[1-5]_L"],
                body_name="link7",
                controller=DifferentialIKControllerCfg(
                    command_type="pose", ik_params={"lambda_val": 0.05},
                    ik_method="dls",
                    use_relative_mode=(mode == "teleop_ik"),
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            )
            self.actions.gripper_action_l = mdp.JointPositionActionCfg(
                asset_name="robot_l",
                joint_names=["gripper_joint_1_L"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.arm_action_r = DifferentialInverseKinematicsActionCfg(
                asset_name="robot_r",
                joint_names=["joint[1-5]_R"],
                body_name="link7",
                controller=DifferentialIKControllerCfg(
                    command_type="pose", ik_params={"lambda_val": 0.05},
                    ik_method="dls",
                    use_relative_mode=(mode == "teleop_ik"),
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            )
            self.actions.gripper_action_r = mdp.JointPositionActionCfg(
                asset_name="robot_r",
                joint_names=["gripper_joint_1_R"],
                scale=1.0,
                use_default_offset=False,
            )
        else:
            self.actions.arm_action_l = None
            self.actions.gripper_action_l = None
            self.actions.arm_action_r = None
            self.actions.gripper_action_r = None
