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

"""Record demonstrations for dual SO101 environments with DDS teleop."""

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
import os
import threading
import time

import gymnasium as gym
import torch
from pynput.keyboard import Listener

from isaaclab.app import AppLauncher

from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.tools.topic_manager import TopicManager

import sys


class RateLimiter:
    def __init__(self, hz: float):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class DualDdsTeleop:
    """DDS teleoperation for two independent leader topics."""

    def __init__(
        self,
        env: "ManagerBasedRLEnv",
        leader_topic_l: str,
        leader_topic_r: str,
        joint_names_l: list[str],
        joint_names_r: list[str],
        asset_name_l: str = "robot_l",
        asset_name_r: str = "robot_r",
    ):
        self.env = env
        self.asset_name_l = asset_name_l
        self.asset_name_r = asset_name_r
        self.joint_names_l = list(joint_names_l)
        self.joint_names_r = list(joint_names_r)
        self.domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}
        self._lock = threading.Lock()
        self._cmd_l = {}
        self._cmd_r = {}

        topic_manager = TopicManager(domain_id=self.domain_id)
        self.reader_l = topic_manager.topic_reader(leader_topic_l, JointTrajectory_)
        self.reader_r = topic_manager.topic_reader(leader_topic_r, JointTrajectory_)

        self.thread_l = threading.Thread(
            target=self._subscriber_loop,
            args=(self.reader_l, self._cmd_l),
            daemon=True,
        )
        self.thread_r = threading.Thread(
            target=self._subscriber_loop,
            args=(self.reader_r, self._cmd_r),
            daemon=True,
        )
        self.thread_l.start()
        self.thread_r.start()

        self.listener = Listener(on_press=self._on_press)
        self.listener.start()
        self._keyboard_controls()

    def _keyboard_controls(self):
        print("\n[Control] Press keys to control the robot:")
        print("[N] Save successful episode and proceed to the next one")
        print("[R] Skip failed episode (not saved) and proceed to the next one")
        print("[B] Start recording the current episode")

    def _on_press(self, key):
        try:
            if key.char == "b":
                self._started = True
                self._reset_state = False
            elif key.char == "r":
                self._started = False
                self._reset_state = True
                self._call_callback("R")
            elif key.char == "n":
                self._started = False
                self._reset_state = True
                self._call_callback("N")
        except AttributeError:
            pass

    def _call_callback(self, key):
        if key in self._additional_callbacks:
            self._additional_callbacks[key]()

    def add_callback(self, key: str, func):
        self._additional_callbacks[key] = func

    def reset(self):
        self._reset_state = False

    def publish_observations(self):
        # No-op for DDS-only teleop in this script.
        return

    def _subscriber_loop(self, reader, cmd_store):
        try:
            while True:
                for msg in reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self._lock:
                            cmd_store.update(joint_dict)
        except Exception as e:
            print(f"Subscriber thread exception: {e}")
        finally:
            try:
                reader.Close()
            except Exception:
                pass

    def _get_current_positions(self, asset_name: str, joint_names: list[str]) -> dict[str, float]:
        asset = self.env.scene[asset_name]
        obs_joint_name = asset.data.joint_names
        all_positions = asset.data.joint_pos.squeeze(0).tolist()
        if isinstance(all_positions[0], list):
            all_positions = [p for sub in all_positions for p in sub]
        return {name: all_positions[obs_joint_name.index(name)] for name in joint_names}

    def get_action(self):
        if self._reset_state:
            self._reset_state = False
            return {"reset": True}
        if not self._started:
            return None

        with self._lock:
            pos_l = self._get_current_positions(self.asset_name_l, self.joint_names_l)
            pos_r = self._get_current_positions(self.asset_name_r, self.joint_names_r)
            pos_l.update(self._cmd_l)
            pos_r.update(self._cmd_r)

        positions = [pos_l[name] for name in self.joint_names_l] + [pos_r[name] for name in self.joint_names_r]
        return torch.tensor(positions, device=self.env.device, dtype=torch.float32).unsqueeze(0)


def _parse_args():
    parser = argparse.ArgumentParser(description="Dual SO101 teleop recorder.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
    parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Output dataset file path.")
    parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record.")
    parser.add_argument("--leader_topic_left", type=str, default="leader_left/joint_trajectory")
    parser.add_argument("--leader_topic_right", type=str, default="leader_right/joint_trajectory")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args_cli = _parse_args()
    app_launcher = AppLauncher(vars(args_cli))
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.managers import TerminationTermCfg, DatasetExportMode
    import robotis_lab
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from recorder_manager.recorder_manager import StreamingRecorderManager

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.init_action_cfg("record")
    env_cfg.seed = args_cli.seed

    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if not hasattr(env_cfg.terminations, "success"):
        setattr(env_cfg.terminations, "success", None)
    env_cfg.terminations.success = TerminationTermCfg(
        func=lambda env: torch.zeros(1, dtype=torch.bool, device=env.device)
    )

    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args_cli.dataset_file) or "."
    env_cfg.recorders.dataset_filename = os.path.basename(args_cli.dataset_file)
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE

    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    del env.recorder_manager
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
    env.recorder_manager.flush_steps = 100
    env.recorder_manager.compression = "lzf"
    env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE

    joint_names_l = ["joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "gripper_joint_1_L"]
    joint_names_r = ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "gripper_joint_1_R"]
    teleop_interface = DualDdsTeleop(
        env=env,
        leader_topic_l=args_cli.leader_topic_left,
        leader_topic_r=args_cli.leader_topic_right,
        joint_names_l=joint_names_l,
        joint_names_r=joint_names_r,
    )

    should_reset_recording_instance = False
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    should_reset_task_success = False
    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)

    rate_limiter = RateLimiter(args_cli.step_hz)

    env.reset()
    teleop_interface.reset()

    current_recorded_demo_count = 0
    start_record_state = False

    while simulation_app.is_running():
        with torch.inference_mode():
            teleop_interface.publish_observations()
            actions = teleop_interface.get_action()

            if should_reset_task_success:
                should_reset_task_success = False
                env.termination_manager.set_term_cfg(
                    "success",
                    TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
                )
                env.termination_manager.compute()
                try:
                    for env_id, ep in getattr(env.recorder_manager, "_episodes", {}).items():
                        if ep is not None and not ep.is_empty():
                            ep.success = True
                except Exception as e:
                    print(f"Warning: Failed to mark episodes as successful: {e}")

                env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                env.recorder_manager.export_episodes(from_step=False)
                env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
                if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

            if should_reset_recording_instance:
                try:
                    env.recorder_manager._clear_episode_cache()
                except Exception as e:
                    print(f"Warning: Failed to clear episode cache: {e}")
                env.reset()
                should_reset_recording_instance = False
                if start_record_state:
                    print("Stop Recording!!!")
                start_record_state = False
                env.termination_manager.set_term_cfg(
                    "success",
                    TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
                )
                print(f"Resetting recording instance. Current recorded demo count: {current_recorded_demo_count}")
                if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break
            elif actions is None:
                env.render()
            else:
                if isinstance(actions, dict):
                    if "reset" in actions:
                        env.render()
                        continue
                else:
                    if actions.ndim == 1:
                        actions = actions.unsqueeze(0)
                    if not start_record_state:
                        print("Start Recording!!!")
                        start_record_state = True
                    env.step(actions)
            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
