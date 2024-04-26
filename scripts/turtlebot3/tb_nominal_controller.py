#!/usr/bin/env python3

import rclpy
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from turtlebot3.hjr_nominal_control import NominalControlHJ
from turtlebot3.pd_nominal_control import NominalControlPD
from refinecbf_ros2.srv import HighLevelCommand
from nominal_controller import NominalController
from config import Config
from ament_index_python.packages import get_package_share_directory
import yaml


class TurtlebotNominalControl(NominalController):
    def __init__(self):
        super().__init__("tb_nominal_control")
        self.declare_parameters(
            "",
            [
                ("controller_type", self.control_config["controller_type"]),
            ],
        )
        self.max_vel = self.control_config["limits"]["max_vel"]
        self.min_vel = self.control_config["limits"]["min_vel"]
        self.max_omega = self.control_config["limits"]["max_omega"]
        umin = np.array([self.min_vel, -self.max_omega])
        umax = np.array([self.max_vel, self.max_omega])
        self.target = np.array(self.control_config["nominal"]["goal"]["coordinates"])
        self.controller_type = self.get_parameter("controller_type").value
        if self.controller_type == "HJR":
            self.config = Config(self, hj_setup=True)
            self.declare_parameter("env_config_file", rclpy.Parameter.Type.STRING)
            env_config_file = self.get_parameter("env_config_file").value
            with open(os.path.join(get_package_share_directory("refinecbf_ros2"), "config", env_config_file), "r") as f:
                self.env_config = yaml.safe_load(f)
            self.umax_hjr = np.array(self.env_config["control_space"]["hi"])
            self.umin_hjr = np.array(self.env_config["control_space"]["lo"])
            self.padding = np.array(self.control_config["nominal"]["goal"]["padding"])
            self.max_time = self.control_config["nominal"]["goal"]["max_time"]
            self.time_intervals = self.control_config["nominal"]["goal"]["time_intervals"]
            self.solver_accuracy = self.control_config["nominal"]["goal"]["solver_accuracy"]
            assert self.solver_accuracy in ["low", "medium", "high", "very_high"]
            self.hj_dynamics = self.config.hj_dynamics
            self.grid = self.config.grid
            self.controller_prep = NominalControlHJ(
                self.hj_dynamics,
                self.grid,
                final_time=self.max_time,
                time_intervals=self.time_intervals,
                solver_accuracy=self.solver_accuracy,
                target=self.target,
                padding=self.padding,
            )
            self.get_logger().info("Solving for nominal control, nominal control default is 0")
            self.controller = lambda x, t: np.zeros(self.dynamics.control_dims)
            self.controller = self.controller_prep.get_nominal_control

        elif self.controller_type == "PD":
            self.config = Config(self, hj_setup=False)
            self.controller = NominalControlPD(target=self.target, umin=umin, umax=umax).get_nominal_control

        else:
            raise NotImplementedError(f"{self.controller_type} is not a valid controller type")

        self.get_logger().info("Nominal controller ready!")


def main(args=None):
    rclpy.init(args=args)
    controller = TurtlebotNominalControl()

    try:
        while rclpy.ok():
            rclpy.spin(controller)
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
