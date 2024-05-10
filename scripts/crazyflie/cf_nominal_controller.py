#!/usr/bin/env python3

import rclpy
import numpy as np

# from refinecbf_ros2.config import Config
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nominal_controller import NominalController
from refinecbf_ros2.srv import HighLevelCommand
from config import Config


class CrazyflieNominalControl(NominalController):
    def __init__(self):
        super().__init__("cf_nominal_control", hj_setup=False)
        self.declare_parameters(
            "",
            [
                ("services.target_position", rclpy.Parameter.Type.STRING),
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

        self.target = self.control_config["nominal"]["goal"]["coordinates"]
        self.gain = self.control_config["nominal"]["goal"]["gain"]
        self.u_target = self.control_config["nominal"]["goal"]["u_ref"]
        self.state = np.zeros_like(self.target)

        # Initialize parameters
        umin = np.array([-self.max_roll, -self.max_pitch, -np.inf, self.min_thrust])
        umax = np.array([self.max_roll, self.max_pitch, np.inf, self.max_thrust])
        umin[self.safety_controls_idis] = np.array(self.config.control_space["lo"])
        umax[self.safety_controls_idis] = np.array(self.config.control_space["hi"])
        self.controller = lambda x, t: np.clip(self.u_target + self.gain @ (x - self.target), umin, umax)
        self.start_controller()

    def target_position_callback(self, request, response):
        value = np.array(request.position.value)
        self.get_logger().info(f"New target position: {value}")
        self.target[:3] = value
        self.get_logger().info(f"New target position: {self.target[:3]}")
        response.response = "Success"
        return response


def main(args=None):
    rclpy.init(args=args)
    controller = CrazyflieNominalControl()

    try:
        while rclpy.ok():
            rclpy.spin(controller)
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
