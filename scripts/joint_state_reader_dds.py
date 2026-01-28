#!/usr/bin/env python3

import argparse
import os
import struct
import sys
import time

import serial

try:
    from robotis_dds_python.idl.builtin_interfaces.msg import Duration_, Time_
    from robotis_dds_python.idl.std_msgs.msg import Header_
    from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_, JointTrajectoryPoint_
    from robotis_dds_python.tools.topic_manager import TopicManager
except ModuleNotFoundError as exc:
    raise SystemExit(
        "robotis_dds_python not found. Install it or set PYTHONPATH to the "
        "robotis_lab/third_party/robotis_dds_python path."
    ) from exc


class JointStateDDSBridge:
    """Read SO101 joint ticks and publish JointTrajectory over DDS."""

    def __init__(
        self,
        port: str,
        baud: int,
        topic: str,
        joint_names: list[str],
        rate_hz: float,
        domain_id: int,
    ):
        self.port = port
        self.baud = baud
        self.topic = topic
        self.joint_names = joint_names
        self.rate_hz = rate_hz
        self.domain_id = domain_id
        self.serial_port = None

        self._connect()

        manager = TopicManager(domain_id=self.domain_id)
        self.writer = manager.topic_writer(self.topic, JointTrajectory_)

    def _connect(self) -> None:
        try:
            self.serial_port = serial.Serial(self.port, self.baud, timeout=0.1)
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            time.sleep(0.1)
            print(f"[DDS] Connected to {self.port} @ {self.baud}")
        except Exception as exc:
            print(f"[DDS] Failed to connect to {self.port}: {exc}")
            self.serial_port = None

    def _read_servo_position(self, servo_id: int) -> int | None:
        if not self.serial_port:
            return None

        try:
            length = 4
            instruction = 0x02
            address = 0x38
            read_length = 0x02
            checksum = (~(servo_id + length + instruction + address + read_length)) & 0xFF
            cmd = bytes([0xFF, 0xFF, servo_id, length, instruction, address, read_length, checksum])

            self.serial_port.reset_input_buffer()
            self.serial_port.write(cmd)
            time.sleep(0.002)
            response = self.serial_port.read(8)

            if len(response) < 7:
                return None
            if response[0] != 0xFF or response[1] != 0xFF:
                return None
            if response[2] != servo_id:
                return None

            pos = struct.unpack("<H", response[5:7])[0]
            if 0 <= pos <= 4095:
                return pos
        except Exception:
            return None
        return None

    @staticmethod
    def _ticks_to_radians(ticks: int | None, fallback: float) -> float:
        if ticks is None:
            return fallback
        normalized = (ticks - 2048) / 2048.0
        return normalized * 3.14159

    def _build_message(self, positions: list[float]) -> JointTrajectory_:
        now = time.time()
        stamp = Time_(sec=int(now), nanosec=int((now % 1) * 1e9))
        header = Header_(stamp=stamp, frame_id="base_link")
        point = JointTrajectoryPoint_(
            positions=list(positions),
            velocities=[],
            accelerations=[],
            effort=[],
            time_from_start=Duration_(sec=0, nanosec=0),
        )
        return JointTrajectory_(header=header, joint_names=list(self.joint_names), points=[point])

    def run(self) -> None:
        period = 1.0 / self.rate_hz
        last_positions = [0.0] * len(self.joint_names)
        print(f"[DDS] Publishing to {self.topic} @ {self.rate_hz} Hz")

        while True:
            if not self.serial_port:
                self._connect()
                time.sleep(0.5)
                continue

            positions = []
            for idx in range(len(self.joint_names)):
                servo_id = idx + 1
                ticks = self._read_servo_position(servo_id)
                positions.append(self._ticks_to_radians(ticks, last_positions[idx]))
                time.sleep(0.01)

            last_positions = positions
            msg = self._build_message(positions)
            self.writer.write(msg)
            time.sleep(period)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO101 DDS joint trajectory publisher.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for SO101.")
    parser.add_argument("--baud", type=int, default=1000000, help="Serial baud rate.")
    parser.add_argument("--topic", default="leader/joint_trajectory", help="DDS topic name.")
    parser.add_argument(
        "--joint-names",
        nargs="+",
        default=["joint1", "joint2", "joint3", "joint4", "joint5", "gripper_joint_1"],
        help="Joint name order for JointTrajectory.",
    )
    parser.add_argument("--rate", type=float, default=20.0, help="Publish rate (Hz).")
    parser.add_argument("--domain-id", type=int, default=int(os.getenv("ROS_DOMAIN_ID", 151)))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bridge = JointStateDDSBridge(
        port=args.port,
        baud=args.baud,
        topic=args.topic,
        joint_names=args.joint_names,
        rate_hz=args.rate,
        domain_id=args.domain_id,
    )
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("\n[DDS] Stopped.")


if __name__ == "__main__":
    main()
