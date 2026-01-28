#!/usr/bin/env python3
import argparse
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration  # ROS2 builtin_interfaces

from robotis_dds_python.idl.builtin_interfaces.msg import Duration_ as DDSDuration, Time_ as DDSTime
from robotis_dds_python.idl.std_msgs.msg import Header_ as DDSHeader
from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_ as DDSJointTrajectory
from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectoryPoint_ as DDSJointTrajectoryPoint
from robotis_dds_python.tools.topic_manager import TopicManager


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Bridge JointState -> ROS2 JointTrajectory -> DDS JointTrajectory_."
    )
    # Input JointState
    parser.add_argument("--js-topic", default="/joint_states")
    # Output control topic (ROS2)
    parser.add_argument("--ros-topic", default="/leader/joint_trajectory")
    # DDS topic
    parser.add_argument("--dds-topic", default="leader/joint_trajectory")
    parser.add_argument("--domain-id", type=int, default=None)
    parser.add_argument(
        "--rmw-implementation",
        default=None,
        help="Optional ROS 2 RMW implementation (e.g., rmw_cyclonedds_cpp).",
    )
    parser.add_argument("--node-name", default="js_to_ros2_to_dds_joint_trajectory_bridge")

    # Control parameters
    parser.add_argument("--rate", type=float, default=2.0, help="Publish rate (Hz) for /leader/joint_trajectory")
    parser.add_argument("--tfs-sec", type=int, default=0)
    parser.add_argument("--tfs-ns", type=int, default=100_000_000, help="time_from_start nanosec (default 0.1s)")

    return parser.parse_args()


def _to_dds(msg: JointTrajectory) -> DDSJointTrajectory:
    stamp = msg.header.stamp
    header = DDSHeader(
        stamp=DDSTime(sec=stamp.sec, nanosec=stamp.nanosec),
        frame_id=msg.header.frame_id or "",
    )

    points = []
    for p in msg.points:
        # 안전하게 빈 배열 처리
        points.append(
            DDSJointTrajectoryPoint(
                positions=list(p.positions) if p.positions else [],
                velocities=list(p.velocities) if p.velocities else [],
                accelerations=list(p.accelerations) if p.accelerations else [],
                effort=list(p.effort) if p.effort else [],
                time_from_start=DDSDuration(sec=p.time_from_start.sec, nanosec=p.time_from_start.nanosec),
            )
        )

    return DDSJointTrajectory(header=header, joint_names=list(msg.joint_names), points=points)


class Bridge(Node):
    """
    Sub:  /joint_states (sensor_msgs/JointState)
    Pub:  /leader/joint_trajectory (trajectory_msgs/JointTrajectory)   <-- 컨트롤러 호환
    DDS:  leader/joint_trajectory (robotis_dds_python IDL JointTrajectory_)
    """

    def __init__(self, js_topic: str, ros_topic: str, dds_topic: str, domain_id: int | None,
                 node_name: str, rate_hz: float, tfs: Duration):
        super().__init__(node_name)

        # Domain
        if domain_id is not None:
            os.environ["ROS_DOMAIN_ID"] = str(domain_id)

        # DDS writer
        self.dds_writer = TopicManager(domain_id=domain_id).topic_writer(dds_topic, DDSJointTrajectory)

        # QoS: /joint_states는 BEST_EFFORT인 경우가 많아서 구독 QoS를 맞춰줌
        qos_sub = QoSProfile(depth=10)
        qos_sub.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_sub.durability = DurabilityPolicy.VOLATILE

        self.latest_js: JointState | None = None
        self.sub = self.create_subscription(JointState, js_topic, self._on_js, qos_sub)

        # Control publisher (ROS2)
        self.pub = self.create_publisher(JointTrajectory, ros_topic, 10)

        self.tfs = tfs

        # Timer-based publish (너가 --rate 2에서 잘 됐다고 해서 기본 2Hz)
        period = 1.0 / max(rate_hz, 1e-6)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(f"Sub JointState: {js_topic}")
        self.get_logger().info(f"Pub ROS2 JointTrajectory: {ros_topic} @ {rate_hz} Hz (tfs={tfs.sec}s {tfs.nanosec}ns)")
        self.get_logger().info(f"Bridge to DDS: {dds_topic}")

    def _on_js(self, msg: JointState):
        self.latest_js = msg

    def _make_traj(self, js: JointState) -> JointTrajectory:
        jt = JointTrajectory()

        # header: stamp/frame_id가 꼭 필요하진 않지만 채워두면 좋음
        jt.header.stamp = self.get_clock().now().to_msg()
        jt.header.frame_id = js.header.frame_id if js.header.frame_id else "base_link"

        jt.joint_names = list(js.name)

        pt = JointTrajectoryPoint()
        pt.positions = list(js.position)

        # velocity/effort는 컨트롤러에 따라 선택이지만, 있으면 길이 맞을 때만 넣어줌
        if js.velocity and len(js.velocity) == len(js.position):
            pt.velocities = list(js.velocity)

        pt.time_from_start = self.tfs
        jt.points = [pt]
        return jt

    def _tick(self):
        if self.latest_js is None:
            return
        js = self.latest_js
        if not js.name or not js.position:
            return

        # ROS2 control 토픽으로 publish
        jt = self._make_traj(js)
        self.pub.publish(jt)

        # 동시에 DDS로도 write
        try:
            self.dds_writer.write(_to_dds(jt))
        except Exception as exc:
            self.get_logger().error(f"DDS write failed: {exc}")


def main():
    args = _parse_args()

    if args.rmw_implementation:
        os.environ["RMW_IMPLEMENTATION"] = args.rmw_implementation
    if args.domain_id is not None:
        os.environ.setdefault("ROS_DOMAIN_ID", str(args.domain_id))

    rclpy.init()

    tfs = Duration(sec=args.tfs_sec, nanosec=args.tfs_ns)
    node = Bridge(
        js_topic=args.js_topic,
        ros_topic=args.ros_topic,
        dds_topic=args.dds_topic,
        domain_id=args.domain_id,
        node_name=args.node_name,
        rate_hz=args.rate,
        tfs=tfs,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
