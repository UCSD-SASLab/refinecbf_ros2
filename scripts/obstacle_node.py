#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from example_interfaces.msg import Bool
from refinecbf_ros2.msg import Array, ValueFunctionMsg, Obstacles
from refinecbf_ros2.srv import ActivateObstacle
import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
import matplotlib.pyplot as plt
from config import Config


class ObstacleNode(Node):
    def __init__(self):
        # Following publishers:
        # - /env/obstacle_update
        # Following subscribers:
        # - /state
        super().__init__("obstacle_node")

        # Config:
        config = Config(self, hj_setup=True)
        self.dynamics = config.dynamics
        self.grid = config.grid
        self.detection_obstacles = config.detection_obstacles
        self.service_obstacles = config.service_obstacles
        self.active_obstacles = config.active_obstacles
        self.update_obstacles = config.update_obstacles
        self.active_obstacle_names = config.active_obstacle_names
        self.boundary = config.boundary
        self.safety_states_idis = config.safety_states
        self.robot_state = None
        # Parameter and Publisher setup:
        self.declare_parameter("vf_update_method", "pubsub")
        self.vf_update_method = self.get_parameter("vf_update_method").value
        self.declare_parameters(
            "",
            [
                ("topics.sdf_update", rclpy.Parameter.Type.STRING),
                ("topics.obstacle_update", rclpy.Parameter.Type.STRING),
                ("topics.cbf_state", rclpy.Parameter.Type.STRING),
                ("services.activate_obstacle", rclpy.Parameter.Type.STRING),
            ],
        )
        sdf_update_topic = self.get_parameter("topics.sdf_update").value
        if self.vf_update_method == "pubsub":
            self.sdf_update_pub = self.create_publisher(ValueFunctionMsg, sdf_update_topic, 10)
        elif self.vf_update_method == "file":
            self.sdf_update_pub = self.create_publisher(Bool, sdf_update_topic, 10)
        else:
            raise NotImplementedError(f"{self.vf_update_method} is not a valid vf update method")

        obstacle_update_topic = self.get_parameter("topics.obstacle_update").value
        self.obstacle_update_pub = self.create_publisher(Obstacles, obstacle_update_topic, 10)

        # Subscribers:
        cbf_state_topic = self.get_parameter("topics.cbf_state").value
        self.subscription = self.create_subscription(Array, cbf_state_topic, self.callback_state, 10)

        # Services:
        activate_obstacle_service = self.get_parameter("services.activate_obstacle").value
        self.srv = self.create_service(ActivateObstacle, activate_obstacle_service, self.handle_activate_obstacle)

        # Timer setup for running obstacle detection at fixed rate
        detection_rate = self.declare_parameter("obstacle_detection_rate", 1.0).value  # Detection rate in Hz
        self.detection_timer = self.create_timer(1.0 / detection_rate, self.obstacle_detection)
        # Initialize and Update Obstacles
        self.update_sdf()
        self.update_active_obstacles()
        self.startTime = self.get_clock().now().seconds_nanoseconds()[0]

    def obstacle_detection(self):
        updatesdf = False
        for obstacle in self.detection_obstacles:
            if obstacle not in self.active_obstacles:
                if (
                    self.robot_state is not None
                    and obstacle.distance_to_obstacle(self.robot_state) <= obstacle.detectionRadius
                ):
                    self.get_logger().info("Obstacle Detected: {}".format(obstacle.obstacleName))
                    self.active_obstacles.append(obstacle)
                    self.active_obstacle_names.append(obstacle.obstacleName)
                    updatesdf = True
        for obstacle in self.update_obstacles:
            if obstacle not in self.active_obstacles:
                when_to_update = self.get_clock().now().seconds_nanoseconds()[0] - self.startTime
                if when_to_update >= obstacle.updateTime:
                    self.get_logger().info("Obstacle appeared: {}".format(obstacle.obstacleName))
                    self.active_obstacles.append(obstacle)
                    self.active_obstacle_names.append(obstacle.obstacleName)
                    updatesdf = True

        if updatesdf:
            self.update_sdf()
            self.update_active_obstacles()

    def update_sdf(self):
        sdf = hj.utils.multivmap(self.build_sdf(), jnp.arange(self.grid.ndim))(self.grid.states)
        self.get_logger().info("Share Safe SDF {:.2f}".format(((sdf >= 0).sum() / sdf.size) * 100))
        if self.vf_update_method == "pubsub":
            self.sdf_update_pub.publish(ValueFunctionMsg(vf=sdf.flatten()))
        else:  # self.vf_update_method == "file"
            np.save("./sdf.npy", sdf)
            self.sdf_update_pub.publish(Bool(data=True))

    def update_active_obstacles(self):
        self.obstacle_update_pub.publish(Obstacles(obstacle_names=self.active_obstacle_names))

    def callback_state(self, state_msg):
        self.robot_state = np.array(state_msg.value)[self.safety_states_idis]

    def build_sdf(self):
        def sdf(x):
            sdf_val = self.boundary.boundary_sdf(x)
            for obstacle in self.active_obstacles:
                obstacle_sdf = obstacle.obstacle_sdf(x)
                sdf_val = jnp.minimum(sdf_val, obstacle_sdf)
            return sdf_val

        return sdf

    def handle_activate_obstacle(self, request, response):
        obstacle_index = request.obstacle_number
        if obstacle_index >= len(self.service_obstacles):
            response.output = "Invalid Obstacle Number"
        elif self.service_obstacles[obstacle_index] in self.active_obstacles:
            response.output = "Obstacle Already Active"
        else:
            self.active_obstacles.append(self.service_obstacles[obstacle_index])
            self.update_sdf()
            response.output = "Obstacle Activated"
        return response


def main(args=None):
    rclpy.init(args=args)
    obstacle_node = ObstacleNode()

    rclpy.spin(obstacle_node)
    obstacle_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
