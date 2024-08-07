#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from refinecbf_ros2.msg import HiLoArray
from refinecbf_ros2.srv import ModifyEnvironment
from config import Config
import numpy as np


class ModifyEnvironmentServer(Node):
    def __init__(self):
        super().__init__("modify_environment_node")

        # Load configuration
        config = Config(self, hj_setup=False)
        self.disturbance_space = config.disturbance_space
        self.control_space = config.control_space
        self.actuation_update_list = config.actuation_updates_list
        self.actuation_idx = 0
        self.disturbance_update_list = config.disturbance_updates_list
        self.disturbance_idx = 0

        # Set up publishers
        self.declare_parameters(
            "",
            [
                ("topics.actuation_update", rclpy.Parameter.Type.STRING),
                ("topics.disturbance_update", rclpy.Parameter.Type.STRING),
                ("topics.simulated_disturbance_update", rclpy.Parameter.Type.STRING),
                ("services.modify_environment", rclpy.Parameter.Type.STRING),
            ],
        )
        actuation_update_topic = self.get_parameter("topics.actuation_update").value
        self.actuation_update_pub = self.create_publisher(HiLoArray, actuation_update_topic, 1)

        disturbance_update_topic = self.get_parameter("topics.disturbance_update").value
        self.disturbance_update_pub = self.create_publisher(HiLoArray, disturbance_update_topic, 1)

        simulated_disturbance_update_topic = self.get_parameter("topics.simulated_disturbance_update").value
        self.simulated_disturbance_update_pub = self.create_publisher(HiLoArray, simulated_disturbance_update_topic, 1)

        # Set up services
        modify_environment_service = self.get_parameter("services.modify_environment").value
        self.srv = self.create_service(ModifyEnvironment, modify_environment_service, self.handle_modified_environment)

    def update_disturbances(self):
        if self.disturbance_idx >= len(self.disturbance_update_list):
            self.get_logger().warn("No more disturbances to update, no update sent")
        else:
            key = list(self.disturbance_update_list.keys())[self.disturbance_idx]
            disturbance_space = self.disturbance_update_list[key]
            hi = np.array(disturbance_space["hi"])
            lo = np.array(disturbance_space["lo"])
            self.disturbance_idx += 1
            self.get_logger().info(f"Updating disturbances with key: {key}")
            self.disturbance_update_pub.publish(HiLoArray(hi=hi, lo=lo))
            if disturbance_space.get("simulate") is not None:
                hi_sim = np.array(disturbance_space["simulate"]["hi"])
                lo_sim = np.array(disturbance_space["simulate"]["lo"])
            else:
                hi_sim = hi
                lo_sim = lo
            self.simulated_disturbance_update_pub.publish(HiLoArray(hi=hi_sim, lo=lo_sim))

    def update_actuation(self):
        if self.actuation_idx >= len(self.actuation_update_list):
            self.get_logger().warn("No more actuations to update, no update sent")
        else:
            control_space = self.actuation_update_list[self.actuation_idx]
            hi = np.array(control_space["hi"])
            lo = np.array(control_space["lo"])
            self.actuation_idx += 1
            self.actuation_update_pub.publish(HiLoArray(hi=hi, lo=lo))

    def handle_modified_environment(self, request, response):
        """
        To add disturbances, paste the following in a terminal:
          ros2 service call /env/modify_environment refinecbf_ros2/srv/ModifyEnvironment "{'modification': 'dist'}"
          ros2 service call /env/modify_environment refinecbf_ros2/srv/ModifyEnvironment "{'modification': 'action'}"

        """
        modification_request = request.modification
        if modification_request == "dist":
            self.update_disturbances()
            response.status = "Disturbance Updated"
        elif modification_request == "action":
            self.update_actuation()
            response.status = "Actuation Updated"
        else:
            response.status = "Invalid modification request"
        return response


def main(args=None):
    rclpy.init(args=args)
    modify_environment_server = ModifyEnvironmentServer()
    rclpy.spin(modify_environment_server)
    modify_environment_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
