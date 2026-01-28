import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robotis_lab.assets.robots import ROBOTIS_LAB_ASSETS_DATA_DIR

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOTIS_LAB_ASSETS_DATA_DIR}/robots/SO101/SO101.usd",
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
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "gripper_joint_1": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-5]"],
            velocity_limit_sim=4.8,
            effort_limit_sim=100.0,
            stiffness=400.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint_1"],
            velocity_limit_sim=4.8,
            effort_limit_sim=30.0,
            stiffness=20.0,
            damping=5.0,
        ),
    },
)
