import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robotis_lab.assets.robots import ROBOTIS_LAB_ASSETS_DATA_DIR


SO101_L_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOTIS_LAB_ASSETS_DATA_DIR}/robots/SO101/SO101_L.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1_L": 0.0,
            "joint2_L": 0.0,
            "joint3_L": 0.0,
            "joint4_L": 0.0,
            "joint5_L": 0.0,
            "gripper_joint_1_L": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-5]_L"],
            velocity_limit_sim=4.8,
            effort_limit_sim=100.0,
            stiffness=400.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint_1_L"],
            velocity_limit_sim=4.8,
            effort_limit_sim=30.0,
            stiffness=20.0,
            damping=5.0,
        ),
    },
)

SO101_R_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOTIS_LAB_ASSETS_DATA_DIR}/robots/SO101/SO101_R.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1_R": 0.0,
            "joint2_R": 0.0,
            "joint3_R": 0.0,
            "joint4_R": 0.0,
            "joint5_R": 0.0,
            "gripper_joint_1_R": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-5]_R"],
            velocity_limit_sim=4.8,
            effort_limit_sim=100.0,
            stiffness=400.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint_1_R"],
            velocity_limit_sim=4.8,
            effort_limit_sim=30.0,
            stiffness=20.0,
            damping=5.0,
        ),
    },
)
