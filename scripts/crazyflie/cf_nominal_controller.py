#!/usr/bin/env python3

import rclpy
import numpy as np

# from refinecbf_ros2.config import Config
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nominal_controller import NominalController
from refinecbf_ros2.srv import HighLevelCommand
from refinecbf_ros2.action import Calibration
from config import Config
from collections import deque
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from threading import Lock


class CrazyflieNominalControl(NominalController):
    def __init__(self):
        super().__init__("cf_nominal_control", hj_setup=False)
        self.declare_parameters(
            "",
            [
                ("services.target_position", rclpy.Parameter.Type.STRING),
                ("services.calibrate_controller", rclpy.Parameter.Type.STRING),
                ("actions.calibrate_controller", rclpy.Parameter.Type.STRING),
            ],
        )
        self.dynamics = self.config.dynamics
        # Control bounds

        self.declare_parameters(
            "",
            [
                ("limits.max_thrust", self.control_config["limits"]["max_thrust"]),
                ("limits.min_thrust", self.control_config["limits"]["min_thrust"]),
                ("limits.max_roll", self.control_config["limits"]["max_roll"]),
                ("limits.max_pitch", self.control_config["limits"]["max_pitch"]),
            ],
        )
        self.max_thrust = self.get_parameter("limits.max_thrust").value
        self.min_thrust = self.get_parameter("limits.min_thrust").value
        self.max_roll = self.get_parameter("limits.max_roll").value
        self.max_pitch = self.get_parameter("limits.max_pitch").value
        self.safety_controls_idis = self.config.safety_controls

        self.target_position_topic = self.get_parameter("services.target_position").value
        self.target_position_service = self.create_service(
            HighLevelCommand, self.target_position_topic, self.target_position_callback
        )

        self.calibrate_controller_topic = self.get_parameter("actions.calibrate_controller").value
        self.calibrate_controller_cb_group = MutuallyExclusiveCallbackGroup()
        self.calibrate_controller_action = ActionServer(self, Calibration, self.calibrate_controller_topic,
                                                        self.calibrate_controller_callback,
                                                        callback_group=self.calibrate_controller_cb_group)
        self.target = self.control_config["nominal"]["goal"]["coordinates"]
        self.gain = np.array(self.control_config["nominal"]["goal"]["gain"])
        self.u_target = self.control_config["nominal"]["goal"]["u_ref"]
        self.state = np.zeros_like(self.target)

        self.calibration_lock = Lock()

        # Initialize parameters
        umin = np.array([-self.max_roll, -self.max_pitch, -np.inf, self.min_thrust])
        umax = np.array([self.max_roll, self.max_pitch, np.inf, self.max_thrust])
        umin[self.safety_controls_idis] = np.array(self.config.control_space["lo"])
        umax[self.safety_controls_idis] = np.array(self.config.control_space["hi"])
        self.umin = umin
        self.umax = umax
        # self.controller = lambda x, t: np.clip(self.u_target + self.gain @ (x - self.target), umin, umax)
        self.controller = lambda x, t: self.controller_actual(x, t)
        self.state_buffer = deque([], int(0.2 * self.controller_rate))  # Last 0.2 seconds of states as average
        self.avg_state = np.zeros_like(self.target)
        self.start_controller()
        
    def controller_actual(self, x, t):
        random_thrust_offset = np.random.uniform(0.0, 0.0)
        return np.clip(self.u_target + self.gain @ (x - self.target), self.umin, self.umax) + np.array([0.0, 0.0, 0.0, random_thrust_offset])

    def calibrate_controller_callback(self, goal_handle):
        self.get_logger().info(f"Current u_target: {self.u_target[3]}")
        # self.calibration_state = np.array(goal_handle.request.position.value)
        self.calibration_state = np.zeros(3)
        self.calibration_state = self.state[:3]
        with self.calibration_lock:
            self.target[:3] = self.calibration_state
        self.get_logger().info(f"Calibrating controlller, at position: {self.calibration_state}")
        import time
        time.sleep(3)  # sleep for 5 seconds to get to and stabilize at goal
        for _ in range(4):
            deviation_z = self.avg_state[2] - self.calibration_state[2]
            self.get_logger().info("Calibration deviation: {:.2f}".format(deviation_z))
            thrust_offset = self.gain[3, 2] * deviation_z
            self.get_logger().info("Thrust offset: {:.2f}".format(thrust_offset))
            with self.calibration_lock:
                self.u_target[3] += thrust_offset
                self.get_logger().info("New thrust target {:.2f}".format(self.u_target[3]))
            time.sleep(5)

        goal_handle.succeed()
        result = Calibration.Result()
        self.get_logger().info("Calibration complete, target thrust {:.2f}".format(self.u_target[3]))
        result.response = "Calibration complete"
        return result

    def process_state(self, state):
        super().process_state(state)
        self.state_buffer.append(state)
        self.avg_state = np.mean(self.state_buffer, axis=0)

    def target_position_callback(self, request, response):
        value = np.array(request.position.value)
        self.target[:3] = value
        self.get_logger().info(f"New target position: {self.target[:3]}")
        response.response = "Success"
        return response


def main(args=None):
    rclpy.init(args=args)
    controller = CrazyflieNominalControl()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(controller)
    try:
        executor.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
