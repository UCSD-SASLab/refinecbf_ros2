#!/usr/bin/env python3

import rclpy
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from refinecbf_ros2.msg import Array
from refinecbf_ros2.srv import HighLevelCommand
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from template.hw_interface import BaseInterface
from ament_index_python.packages import get_package_share_directory
import yaml



class TurtlebotInterface(BaseInterface):
    """
    This class converts the state and control messages from the SafetyFilterNode to the correct type
    for the Turtlebots.
    Each HW platform should have its own Interface node
    """

    state_msg_type = Odometry
    control_out_msg_type = Twist
    external_control_msg_type = Twist

    def __init__(self):
        super().__init__("turtlebot_interface")
        self.declare_parameter("control_config_file", rclpy.Parameter.Type.STRING)
        control_config_file = self.get_parameter("control_config_file").value
        with open(os.path.join(get_package_share_directory("refinecbf_ros2"), control_config_file)) as f:
            control_config = yaml.safe_load(f)

        self.declare_parameters(
            "",
            [
                ("buffer_time_external_control", control_config["external"]["buffer_time"]),
                ("buffer_time_mod_external_control", control_config["external"]["mod_buffer_time"]),
                ("limits.max_vel", control_config["limits"]["max_vel"]),
                ("limits.min_vel", control_config["limits"]["min_vel"]),
                ("limits.max_omega", control_config["limits"]["max_omega"]),
            ]
        )

        # Initialize external control parameters
        self.external_control = None
        self.buffer_time_external_control = self.get_parameter("buffer_time_external_control").value  # seconds
        self.buffer_time_mod_external_control = self.get_parameter("buffer_time_mod_external_control").value  # seconds

        self.max_vel = self.get_parameter("limits.max_vel").value
        self.min_vel = self.get_parameter("limits.min_vel").value
        self.max_omega = self.get_parameter("limits.max_omega").value
        self.is_running = False

    def handle_high_level_command(self, request, response):
        if request.command == "start":
            if self.is_running:
                response.response = "Already running (start command ignored)"
            else:
                self.is_running = True
                response.response = "Turtlebot can move now"
        elif request.command == "end":
            if self.is_running:
                self.is_running = False
                response.response = "Turtlebot stopping"
                # FIXME: Make sure to send zero commands to the robot
            else:
                response.response = "Already stopped (end command ignored)"
        elif request.command == "goto":
            raise NotImplementedError("goto command not implemented")
        else: 
            response.response = "actions not implemented ({} command ignored)".format(request.command)
        return response

    def callback_state(self, state_in_msg):
        w = state_in_msg.pose.pose.orientation.w
        x = state_in_msg.pose.pose.orientation.x
        y = state_in_msg.pose.pose.orientation.y
        z = state_in_msg.pose.pose.orientation.z

        # Convert Quaternion to Yaw
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (np.power(y, 2) + np.power(z, 2))) + np.pi / 2 # FIXME: why is this necessary? I think it has something to do with the odom and rviz coordinate frames
        yaw = np.arctan2(np.sin(yaw),np.cos(yaw)) # Remap yaw to -pi to pi range

        state_out_msg = Array()
        state_out_msg.value = [state_in_msg.pose.pose.position.x, state_in_msg.pose.pose.position.y, yaw]
        self.state_pub.publish(state_out_msg)

    def process_safe_control(self, control_in_msg):
        control_in = control_in_msg.value
        control_out_msg = self.control_out_msg_type()
        control_out_msg.linear.x = np.minimum(self.max_vel, np.maximum(self.min_vel, control_in[1]))
        control_out_msg.linear.y = 0.0
        control_out_msg.linear.z = 0.0

        control_out_msg.angular.x = 0.0
        control_out_msg.angular.y = 0.0
        control_out_msg.angular.z = np.minimum(-self.max_omega, np.maximum(self.max_omega, control_in[0]))
        return control_out_msg

    def process_external_control(self, control_in_msg):
        # When nominal control comes through the HW interface, it is a Twist message
        control_out_msg = Array()
        control_out_msg.value = [control_in_msg.angular.z, control_in_msg.linear.x]
        new_val = np.array(control_out_msg.value)
        if (self.external_control is None) or (not np.allclose(self.external_control, new_val, atol=1e-1, rtol=1e-1)):
            # If the external control has changed, then reset the external control mod timestamp
            self.external_control_mod_ts = self.get_clock().now().nanoseconds
            self.external_control = new_val
        
        self.external_control = control_out_msg
        self.buffer_time_external_control = self.get_clock.now().nanoseconds
        return control_out_msg
    
    def process_disturbance(self, disturbance_msg):
        disturbance_in = disturbance_msg.value
        disturbance_out_msg = self.disturbance_out_msg_type()
        raise NotImplementedError("Override to process the disturbance message")
        return disturbance_out_msg

    def override_nominal_control(self):
        curr_time = self.get_clock().now().nanoseconds

        # Determine if external control should be published
        return (
            self.external_control is not None
            and (curr_time - self.buffer_time_external_control) * 1e9 <= self.external_control_time_buffer
            and (curr_time - self.external_control_mod_ts) * 1e9 <= self.buffer_time_mod_external_control
        )


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()