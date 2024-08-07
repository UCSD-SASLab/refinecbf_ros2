#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from refinecbf_ros2.msg import Array
from refinecbf_ros2.srv import HighLevelCommand
from utils import load_parameters


class BaseInterface(Node):
    """
    BaseInterface is an abstract base class that converts the state and control messages
    from the SafetyFilterNode to the correct type for the crazyflies. Each hardware platform
    should have its own Interface node that subclasses this base class.

    Attributes:
    - state_msg_type: The ROS message type for the robot's state.
    - control_out_msg_type: The ROS message type for the robot's safe control.
    - external_control_msg_type: The ROS message type for the robot's external control.

    Subscribers:
    - ~topics/robot_state: Subscribes to the robot's state.
    - ~topics/cbf_safe_control: Subscribes to the safe control messages.
    - ~topics/robot_external_control: Subscribes to the external control messages.

    Publishers:
    - state_pub (~topics/cbf_state): Publishes the converted state messages.
    - safe_control_pub (~topics/robot_safe_control): Publishes the converted safe control messages.
    - external_control_pub (~topics/cbf_external_control): Publishes the converted external control messages.
    """

    state_msg_type = None
    control_out_msg_type = None
    external_control_msg_type = None
    disturbance_out_msg_type = None

    def __init__(self, node_name="base_interface"):
        super().__init__(node_name)
        # Get topics from parameters
        self.declare_parameters(
            "",
            [
                ("topics.robot_state", rclpy.Parameter.Type.STRING),
                ("topics.cbf_state", rclpy.Parameter.Type.STRING),
                ("topics.robot_safe_control", rclpy.Parameter.Type.STRING),
                ("topics.cbf_safe_control", rclpy.Parameter.Type.STRING),
                ("topics.robot_external_control", rclpy.Parameter.Type.STRING),
                ("topics.cbf_external_control", rclpy.Parameter.Type.STRING),
                ("services.highlevel_command", rclpy.Parameter.Type.STRING),
            ],
        )

        self.declare_parameter("robot", rclpy.Parameter.Type.STRING)
        self.declare_parameter("exp", rclpy.Parameter.Type.INTEGER)

        # Generate the update get parameters
        self.robot_state_topic = self.get_parameter("topics.robot_state").value
        cbf_state_topic = self.get_parameter("topics.cbf_state").value
        self.state_pub = self.create_publisher(Array, cbf_state_topic, 1)

        robot_safe_control_topic = self.get_parameter("topics.robot_safe_control").value
        self.cbf_safe_control_topic = self.get_parameter("topics.cbf_safe_control").value
        self.safe_control_pub = self.create_publisher(self.control_out_msg_type, robot_safe_control_topic, 1)

        self.robot_external_control_topic = self.get_parameter("topics.robot_external_control").value
        cbf_external_control_topic = self.get_parameter("topics.cbf_external_control").value
        self.external_control_pub = self.create_publisher(Array, cbf_external_control_topic, 1)

        high_level_command_srv = self.get_parameter("services.highlevel_command").value
        self.create_service(HighLevelCommand, high_level_command_srv, self.handle_high_level_command)

        # Check for disturbance topic
        env_config = load_parameters(self.get_parameter("robot").value, self.get_parameter("exp").value, "env")

        self.number_disturbance_dims = env_config["disturbance_space"]["n_dims"]
        if not self.number_disturbance_dims == 0:
            self.declare_parameters(
                "",
                [
                    ("topics.robot_disturbance", rclpy.Parameter.Type.STRING),
                    ("topics.simulated_disturbance", rclpy.Parameter.Type.STRING),
                ],
            )
            robot_disturbance_topic = self.get_parameter("topics.robot_disturbance").value
            self.simulated_disturbance_topic = self.get_parameter("topics.simulated_disturbance").value
            self.disturbance_pub = self.create_publisher(self.disturbance_out_msg_type, robot_disturbance_topic, 1)

    def init_subscribers(self):
        self.create_subscription(self.state_msg_type, self.robot_state_topic, self.callback_state, 1)
        self.create_subscription(Array, self.cbf_safe_control_topic, self.callback_safe_control, 1)
        self.create_subscription(
            self.external_control_msg_type, self.robot_external_control_topic, self.callback_external_control, 1
        )
        if not self.number_disturbance_dims == 0:
            self.create_subscription(Array, self.simulated_disturbance_topic, self.callback_disturbance, 1)

    def callback_state(self, state_msg):
        """
        Callback for the state subscriber. This method should be implemented in a subclass.

        Args:
            state_msg: The incoming state message.
        """
        raise NotImplementedError("Must be subclassed")

    def handle_high_level_command(self, request, response):
        response.response = "actions not implemented (no impact)"
        return response

    def callback_safe_control(self, control_in_msg):
        """
        Callback for the safe control subscriber. This method should be implemented in a subclass.
        Should call self.override_safe_control()

        Args:
            control_msg: The incoming control message.
        """
        control_out_msg = self.process_safe_control(control_in_msg)
        assert isinstance(control_out_msg, self.control_out_msg_type), "Override to process the safe control message"
        if not self.override_safe_control():
            self.safe_control_pub.publish(control_out_msg)

    def callback_external_control(self, control_in_msg):
        """
        Callback for the external control subscriber. This method should be implemented in a subclass.
        Typical usage:
        - Process incoming control message to Array
        - Call self.override_nominal_control(control_msg)
        Args:
            control_msg: The incoming control message.
        """
        if self.override_safe_control():
            assert isinstance(control_in_msg, self.control_out_msg_type)
            self.safe_control_pub.publish(
                control_in_msg
            )  # TODO: Fix with the not in flight (or something similar flag)
        control_out_msg = self.process_external_control(control_in_msg)
        assert isinstance(control_out_msg, Array), "Override to process the external control message"
        if self.override_nominal_control():
            self.external_control_pub.publish(control_out_msg)

    def callback_disturbance(self, disturbance_msg):
        disturbance_out_msg = self.process_disturbance(disturbance_msg)
        assert isinstance(
            disturbance_out_msg, self.disturbance_out_msg_type
        ), "Override to process the disturbance message"
        self.disturbance_pub.publish(disturbance_out_msg)

    def process_external_control(self, control_in_msg):
        raise NotImplementedError("Must be subclassed")

    def process_safe_control(self, control_in_msg):
        raise NotImplementedError("Must be subclassed")

    def process_disturbance(self, disturbance_msg):
        raise NotImplementedError("Must be subclassed")

    def convert_and_clip_control_output(self, control_in_msg):
        return control_in_msg

    def override_safe_control(self):
        """
        Checks if the robot should override the safe control. Defaults to False.
        Should be overriden if the robot has to be able to be taken over by user.

        Returns:
            True if the robot should override the safe control, False otherwise.
        """
        return False

    def override_nominal_control(self):
        """
        Checks if the robot should override the nominal control. Defaults to False.
        Should be overriden if we would like interactive experiments. (e.g. geofencing)

        Returns:
            True if the robot should override the nominal control, False otherwise.
        """
        return False
