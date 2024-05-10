#!/usr/bin/env python3

import rclpy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from refinecbf_ros2.msg import Array
from refinecbf_ros2.srv import HighLevelCommand
import sys
import os
import rowan
from crazyflie_interfaces.srv import NotifySetpointsStop, Land, Takeoff

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from template.hw_interface import BaseInterface
from ament_index_python.packages import get_package_share_directory
import yaml


class CrazyflieInterface(BaseInterface):
    """
    This class converts the state and control messages from the SafetyFilterNode to the correct type
    for the Crazyflies.
    """

    state_msg_type = Odometry
    control_out_msg_type = Twist
    external_control_msg_type = Twist
    disturbance_out_msg_type = Odometry  # TEMP

    def __init__(self):
        super().__init__("crazyflie_interface")
        # In flight flag setup
        self.declare_parameters(
            "",
            [
                ("topics.in_flight", rclpy.Parameter.Type.STRING),
                ("topics.external_setpoint", "/external_setpoint"),
                ("services.target_position", rclpy.Parameter.Type.STRING),
                ("services.takeoff", rclpy.Parameter.Type.STRING),
                ("services.land", rclpy.Parameter.Type.STRING),
                ("services.stop_setpoints", rclpy.Parameter.Type.STRING),
            ],
        )
        self.in_flight_flag_topic = self.get_parameter("topics.in_flight").value
        self.is_in_flight = False
        self.zero_control_out_msg = self.control_out_msg_type()
        self.zero_control_out_msg.linear.x = 0.0
        self.zero_control_out_msg.linear.y = 0.0
        self.zero_control_out_msg.linear.z = 0.0
        self.zero_control_out_msg.angular.z = 0.0

        takeoff_service = self.get_parameter("services.takeoff").value
        self.takeoffService = self.create_client(Takeoff, takeoff_service)
        self.takeoffService.wait_for_service()

        land_service = self.get_parameter("services.land").value
        self.landService = self.create_client(Land, land_service)
        self.landService.wait_for_service()

        stop_setpoints_service = self.get_parameter("services.stop_setpoints").value
        self.notifySetpointsStopService = self.create_client(NotifySetpointsStop, stop_setpoints_service)
        self.notifySetpointsStopService.wait_for_service()

        # External setpoint setup
        self.declare_parameter("control.external_setpoint_buffer", 5.0)
        self.external_setpoint_time_buffer = self.get_parameter("control.external_setpoint_buffer").value
        self.external_setpoint_ts = None
        self.external_setpoint = None

        self.target_position_topic = self.get_parameter("services.target_position").value
        self.lqr_target_service = self.create_client(HighLevelCommand, self.target_position_topic)

        # Control bounds
        self.declare_parameter("control_config_file", rclpy.Parameter.Type.STRING)
        control_config_file = self.get_parameter("control_config_file").value
        with open(os.path.join(get_package_share_directory("refinecbf_ros2"), "config", control_config_file), "r") as f:
            control_config = yaml.safe_load(f)
        self.declare_parameters(
            "",
            [
                ("control.limits.max_thrust", control_config["limits"]["max_thrust"]),
                ("control.limits.min_thrust", control_config["limits"]["min_thrust"]),
                ("control.limits.max_roll", control_config["limits"]["max_roll"]),
                ("control.limits.max_pitch", control_config["limits"]["max_pitch"]),
                ("control.limits.max_yawrate", control_config["limits"]["max_yawrate"]),
            ],
        )
        self.max_thrust = self.get_parameter("control.limits.max_thrust").value
        self.min_thrust = self.get_parameter("control.limits.min_thrust").value
        self.max_roll = self.get_parameter("control.limits.max_roll").value
        self.max_pitch = self.get_parameter("control.limits.max_pitch").value
        self.max_yawrate = self.get_parameter("control.limits.max_yawrate").value

        self.takeoff_timer = None

        self.declare_parameter("backend", rclpy.Parameter.Type.STRING)
        backend = self.get_parameter("backend").value
        self.negate_yaw = -1 if backend == "sim" else 1  # TODO: Temporary hack to fix

        self.init_subscribers()

    def handle_high_level_command(self, request, response):
        if request.command == "start":
            if self.is_in_flight:
                response.response = "Already in Flight (start command ignored)"
            else:
                req = Takeoff.Request()
                req.group_mask = 0  # all groups?
                req.height = 1.0
                duration = 3.0
                req.duration = rclpy.duration.Duration(seconds=duration).to_msg()
                self.takeoffService.call_async(req)
                self.takeoff_timer = self.create_timer(10.0, self.toggle_in_flight_flag)
                response.response = "Taking Off"
        elif request.command == "end":
            if not self.is_in_flight:
                response.response = "Not in flight (end command ignored)"
            else:
                # 1. Stop sending low level control commands
                self.is_in_flight = False
                # 2. Inform drone no more low level commands
                req = NotifySetpointsStop.Request()
                req.remain_valid_millisecs = 10
                req.group_mask = 0  # all groups?
                self.notifySetpointsStopService.call_async(req)  # FIXME: Ensure this stops all the commands
                # 3. Send the land command multiple times to ensure the drone lands
                req = Land.Request()
                req.group_mask = 0  # all groups?
                req.height = 0.05
                req.duration = rclpy.duration.Duration(seconds=3.0).to_msg()
                # Repeat the land command to ensure the drone lands
                for _ in range(2):
                    self.landService.call_async(req)
                    # sleep
                    rclpy.spin_once(self, timeout_sec=0.1)  # FIXME: Does this work?
                response.response = "Landing"

        elif request.command == "goto":
            if not self.is_in_flight:
                response.response = "Not in flight (goto command ignored)"
            else:
                # Send external setpoint control to the drone nominal controller
                self.lqr_target_service.call_async(request)
                response.response = "New target set: x={}, y={}, z={}".format(*np.array(request.position.value))
        else:
            response.response = "actions not implemented (no impact)"
        return response

    def callback_state(self, msg):
        state_out_msg = Array()
        euler_angles = rowan.to_euler(
            (
                [
                    msg.pose.pose.orientation.w,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                ]
            ),
            "xyz",
        )
        # FIXME: Unsure about euler angles, only know that z is -yaw
        # self.get_logger().info("Euler angles: {}".format(np.degrees(euler_angles)), throttle_duration_sec=0.5)
        new_state = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            self.negate_yaw * euler_angles[2],
        ]
        state_out_msg.value = new_state
        # TODO: Conversion from quaternion to euler angles
        self.state_pub.publish(state_out_msg)

    def process_safe_control(self, control_in_msg):
        control_in = control_in_msg.value
        control_out_msg = self.control_out_msg_type()
        rpyt = np.array([np.arctan(control_in[0]), control_in[1], control_in[2], control_in[3]])
        rpyt_converted = self.convert_and_clip_control_output(rpyt)
        control_out_msg.linear.y = rpyt_converted[0]
        control_out_msg.linear.x = rpyt_converted[1]
        control_out_msg.angular.z = rpyt_converted[2]
        control_out_msg.linear.z = rpyt_converted[3]
        # self.get_logger().info("Sending control message {}".format(control_out_msg), throttle_duration_sec=0.5)
        return control_out_msg

    def convert_and_clip_control_output(self, rpyt):
        rpyt_converted = rpyt.copy()
        rpyt_converted[:3] = np.degrees(rpyt[:3])
        rpyt_converted[3] = rpyt[3] * 4096  # (0-16) -> (0-65535)
        min_values = np.array([-self.max_roll, -self.max_pitch, -self.max_yawrate, self.min_thrust])
        max_values = np.array([self.max_roll, self.max_pitch, self.max_yawrate, self.max_thrust])
        rpyt_converted = np.minimum(max_values, np.maximum(min_values, rpyt_converted))
        return rpyt_converted

    def process_external_control(self, control_in_msg):
        self.external_control_robot = control_in_msg
        control = control_in_msg.control
        control_out_msg = Array()
        control_out_msg.value = [np.tan(control.roll), control.pitch, control.yaw_dot, control.thrust]
        return control_out_msg

    def process_disturbance(self, disturbance_in_msg):
        disturbance_in = disturbance_in_msg.value
        disturbance_out_msg = self.disturbance_out_msg_type()
        disturbance_out_msg.pose.pose.position.y = disturbance_in[0]
        disturbance_out_msg.pose.pose.position.z = disturbance_in[1]
        disturbance_out_msg.twist.twist.linear.y = disturbance_in[2]
        disturbance_out_msg.twist.twist.linear.z = disturbance_in[3]
        return disturbance_out_msg

    def override_safe_control(self):
        return not self.is_in_flight  # If not in flight, override

    def toggle_in_flight_flag(self):
        self.get_logger().info("In flight flag toggled")
        if not self.is_in_flight:
            # Initialize low level control
            for _ in range(5):  
                self.safe_control_pub.publish(self.zero_control_out_msg)
        self.is_in_flight = not self.is_in_flight
        self.destroy_timer(self.takeoff_timer)


def main(args=None):
    rclpy.init(args=args)
    node = CrazyflieInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
