#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import jax
import jax.numpy as jnp

# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose
from example_interfaces.msg import Bool
from refinecbf_ros2.msg import ValueFunctionMsg, Array, Obstacles
from config import Config
import numpy as np
import matplotlib.pyplot as plt
from utils import load_parameters


class Visualization(Node):

    def __init__(self):
        super().__init__("visualization_node")

        # Config:
        self.config = Config(self, hj_setup=True)
        self.state_safety_idis = self.config.safety_states
        self.grid = self.config.grid
        # Control Dict with Goal Params:
        control_config = load_parameters(self.get_parameter("robot").value, self.get_parameter("exp").value, "control")
        self.nominal_control_dict = control_config["nominal"]

        self.goal = np.array(self.nominal_control_dict["goal"]["coordinates"])
        if "path_planning" in self.nominal_control_dict:
            self.intermediate_target = self.goal.copy()
        # Subscriber for SDF and VF:
        self.declare_parameters(
            "",
            [
                ("topics.sdf_update", rclpy.Parameter.Type.STRING),
                ("topics.vf_update", rclpy.Parameter.Type.STRING),
                ("topics.obstacle_update", rclpy.Parameter.Type.STRING),
                ("topics.cbf_state", rclpy.Parameter.Type.STRING),
                ("topics.obstacle_marker", rclpy.Parameter.Type.STRING),
                ("topics.sdf_marker", rclpy.Parameter.Type.STRING),
                ("topics.vf_marker", rclpy.Parameter.Type.STRING),
                ("topics.goal_marker", rclpy.Parameter.Type.STRING),
                ("topics.target_state", rclpy.Parameter.Type.STRING),
                ("topics.intermediate_target_marker", rclpy.Parameter.Type.STRING),
                ("topics.intermediate_target_state", rclpy.Parameter.Type.STRING),
            ],
        )

        self.declare_parameter("vf_update_method", "pubsub")

        self.vf_update_method = self.get_parameter("vf_update_method").value
        sdf_update_topic = self.get_parameter("topics.sdf_update").value
        vf_topic = self.get_parameter("topics.vf_update").value

        if self.vf_update_method == "pubsub":
            self.sdf_update_sub = self.create_subscription(
                ValueFunctionMsg, sdf_update_topic, self.callback_sdf_pubsub, 1
            )
            self.vf_update_sub = self.create_subscription(ValueFunctionMsg, vf_topic, self.callback_vf_pubsub, 1)
        elif self.vf_update_method == "file":
            self.sdf_update_sub = self.create_subscription(Bool, sdf_update_topic, self.callback_sdf_file, 1)
            self.vf_update_sub = self.create_subscription(Bool, vf_topic, self.callback_vf_file, 1)
        else:
            raise NotImplementedError("{} is not a valid vf update method".format(self.vf_update_method))

        obstacle_update_topic = self.get_parameter("topics.obstacle_update").value
        self.obstacle_update_sub = self.create_subscription(
            Obstacles, obstacle_update_topic, self.callback_obstacle, 1
        )
        self.active_obstacle_names = []

        # Subscriber for Robot State:
        cbf_state_topic = self.get_parameter("topics.cbf_state").value
        self.state_sub = self.create_subscription(Array, cbf_state_topic, self.callback_state, 1)

        # Publisher for Marker messages
        obstacle_marker_topic = self.get_parameter("topics.obstacle_marker").value
        self.obstacle_marker_publisher = self.create_publisher(Marker, obstacle_marker_topic, 1)

        # Publisher for SDF
        sdf_marker_topic = self.get_parameter("topics.sdf_marker").value
        self.sdf_marker_publisher = self.create_publisher(Marker, sdf_marker_topic, 1)

        # Publisher for VF
        vf_marker_topic = self.get_parameter("topics.vf_marker").value
        self.vf_marker_publisher = self.create_publisher(Marker, vf_marker_topic, 1)

        # Publisher for Goal:
        goal_marker_topic = self.get_parameter("topics.goal_marker").value
        self.goal_marker_publisher = self.create_publisher(Marker, goal_marker_topic, 1)
        goal_update_topic = self.get_parameter("topics.target_state").value
        self.goal_marker_update_sub = self.create_subscription(Array, goal_update_topic, self.callback_goal_marker, 1)

        # Publisher for Intermediate Target:
        intermediate_target_marker_topic = self.get_parameter("topics.intermediate_target_marker").value
        self.intermediate_target_marker_publisher = self.create_publisher(Marker, intermediate_target_marker_topic, 1)
        intermediate_target_update_topic = self.get_parameter("topics.intermediate_target_state").value
        self.intermediate_target_update_sub = self.create_subscription(Array, intermediate_target_update_topic, self.callback_intermediate_target, 1)
        
        # load Obstacle and Boundary dictionaries
        self.obstacle_dict = self.config.obstacle_list
        self.boundary_dict = self.config.boundary_env

    def clip_state(self, state):
        return np.clip(state, np.array(self.grid.domain.lo) + 0.01, np.array(self.grid.domain.hi) - 0.01)

    def obstacle_marker(self, obstacle, obstacle_marker_id):
        raise NotImplementedError("Must Be Subclassed")

    def sdf_marker(self, points, sdf_marker_id):
        raise NotImplementedError("Must Be Subclassed")

    def vf_marker(self, points, vf_marker_id):
        raise NotImplementedError("Must Be Subclassed")

    def zero_level_set_contour(self, vf):
        raise NotImplementedError("Must Be Subclassed")

    def goal_marker(self, goal, goal_marker_id):
        raise NotImplementedError("Must Be Subclassed")

    def add_obstacles(self):
        obstacle_marker_id = 1
        if len(self.obstacle_dict) != 0:
            for name, obstacle in self.obstacle_dict.items():
                marker = self.obstacle_marker(obstacle, obstacle_marker_id, name in self.active_obstacle_names)
                self.obstacle_marker_publisher.publish(marker)
                obstacle_marker_id += 1

    def update_sdf_contour(self):
        sdf_marker_id = 100
        array_points = self.zero_level_set_contour(self.sdf)
        for i, points in enumerate(array_points):
            marker = self.sdf_marker(points, sdf_marker_id + i)
            self.sdf_marker_publisher.publish(marker)

    def update_vf_contour(self):
        vf_marker_id = 200
        array_points = self.zero_level_set_contour(self.vf)
        for i, points in enumerate(array_points):
            marker = self.vf_marker(points, vf_marker_id + i)
            self.vf_marker_publisher.publish(marker)

    def update_goal(self):
        goal_marker_id = 300
        marker = self.goal_marker(self.goal, goal_marker_id)
        self.goal_marker_publisher.publish(marker)

    def update_intermediate_target(self):
        intermediate_target_marker_id = 300
        marker = self.goal_marker(self.intermediate_target, intermediate_target_marker_id)
        self.intermediate_target_marker_publisher.publish(marker)

    def callback_sdf_pubsub(self, sdf_msg):
        self.sdf = np.array(sdf_msg.vf).reshape(self.config.grid_shape)

    def callback_sdf_file(self, sdf_msg):
        if not sdf_msg.data:
            return
        self.sdf = np.array(np.load("./sdf.npy")).reshape(self.config.grid_shape)

    def callback_vf_pubsub(self, vf_msg):
        self.vf = np.array(vf_msg.vf).reshape(self.config.grid_shape)

    def callback_vf_file(self, vf_msg):
        if not vf_msg.data:
            return
        self.vf = np.load("vf.npy").reshape(self.config.grid_shape)

    def callback_obstacle(self, obstacle_msg):
        self.active_obstacle_names = obstacle_msg.obstacle_names

    def callback_goal_marker(self, goal_msg):
        self.goal = np.array(goal_msg.value)
        self.get_logger().info(f"Goal Updated: {self.goal}")

    def callback_intermediate_target(self, target_msg):
        self.intermediate_target = np.array(target_msg.value)

    def callback_state(self, state_msg):
        self.robot_state = jnp.reshape(np.array(state_msg.value)[self.state_safety_idis], (-1, 1)).T
        if hasattr(self, "vf"):
            self.update_vf_contour()
        if hasattr(self, "sdf"):
            self.update_sdf_contour()
        if hasattr(self, "obstacle_dict"):
            self.add_obstacles()
        if hasattr(self, "goal"):
            self.update_goal()
        if hasattr(self, "intermediate_target"):
            self.update_intermediate_target()
