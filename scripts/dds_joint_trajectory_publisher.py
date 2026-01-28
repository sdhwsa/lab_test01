#!/usr/bin/env python3
import argparse
import math
import time

from robotis_dds_python.idl.builtin_interfaces.msg import Duration_, Time_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_, JointTrajectoryPoint_
from robotis_dds_python.tools.topic_manager import TopicManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish DDS JointTrajectory_ messages for Isaac Sim teleop testing."
    )
    parser.add_argument(
        "--robot-type",
        choices=["OMY", "OMX", "SO101"],
        default=None,
        help="Use a preset joint order for the robot type.",
    )
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    parser.add_argument(
        "--topic",
        default="leader/joint_trajectory",
        help="DDS topic name (robotis_dds_python maps to rt/<topic>).",
    )
    parser.add_argument(
        "--joint-names",
        nargs="+",
        default=None,
        help="Joint names in publish order.",
    )
    parser.add_argument(
        "--positions",
        nargs="*",
        type=float,
        default=None,
        help="Base joint positions (radians). Default: all zeros.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
        help="Publish rate in Hz.",
    )
    parser.add_argument(
        "--sine-amplitude",
        type=float,
        default=0.0,
        help="Amplitude for sine motion (radians). 0 means constant positions.",
    )
    parser.add_argument(
        "--sine-frequency",
        type=float,
        default=0.2,
        help="Frequency for sine motion (Hz).",
    )
    return parser.parse_args()


def _build_header() -> Header_:
    now = time.time()
    sec = int(now)
    nanosec = int((now - sec) * 1e9)
    return Header_(stamp=Time_(sec=sec, nanosec=nanosec), frame_id="")


def main() -> None:
    args = _parse_args()
    if args.joint_names:
        joint_names = list(args.joint_names)
    elif args.robot_type == "OMY":
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "rh_r1_joint"]
    elif args.robot_type == "OMX":
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper_joint_1"]
    elif args.robot_type == "SO101":
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper_joint_1"]
    else:
        raise SystemExit("Provide --joint-names or --robot-type.")

    if args.positions is None:
        base_positions = [0.0] * len(joint_names)
    else:
        if len(args.positions) > len(joint_names):
            raise SystemExit(
                f"--positions length {len(args.positions)} is longer than joint count {len(joint_names)}"
            )
        base_positions = list(args.positions)
        if len(base_positions) < len(joint_names):
            base_positions.extend([0.0] * (len(joint_names) - len(base_positions)))

    writer = TopicManager(args.domain_id).topic_writer(args.topic, JointTrajectory_)
    dt = 1.0 / args.rate
    t0 = time.time()

    while True:
        t = time.time() - t0
        if args.sine_amplitude == 0.0:
            positions = base_positions
        else:
            offset = args.sine_amplitude * math.sin(2.0 * math.pi * args.sine_frequency * t)
            positions = [p + offset for p in base_positions]

        point = JointTrajectoryPoint_(
            positions=positions,
            velocities=[],
            accelerations=[],
            effort=[],
            time_from_start=Duration_(sec=0, nanosec=int(dt * 1e9)),
        )
        msg = JointTrajectory_(header=_build_header(), joint_names=joint_names, points=[point])
        writer.write(msg)
        time.sleep(dt)


if __name__ == "__main__":
    main()
