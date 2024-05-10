#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from refinecbf_ros2.msg import Array, HiLoArray
from config import Config


class DisturbanceNode(Node):
    """
    This node is responsible for generating disturbances for simulator purposes. Hardware does not have this issue.
    """

    def __init__(self):
        super().__init__("disturbance_node")
        config = Config(self)
        self.declare_parameters(
            "",
            [
                ("topics.simulated_disturbance", rclpy.Parameter.Type.STRING),
                ("topics.disturbance_update", rclpy.Parameter.Type.STRING),
                ("topics.cbf_state", rclpy.Parameter.Type.STRING),
            ],
        )

        self.declare_parameters("", [("beta_skew", 1.0), ("seed", 0), ("rate", 20)])

        disturbance_topic = self.get_parameter("topics.simulated_disturbance").value
        disturbance_update_topic = self.get_parameter("topics.disturbance_update").value
        state_topic = self.get_parameter("topics.cbf_state").value

        self.pub_disturbance = self.create_publisher(Array, disturbance_topic, 10)

        self.disturbance_update_sub = self.create_subscription(
            HiLoArray, disturbance_update_topic, self.callback_disturbance_update, 10
        )

        self.state_sub = self.create_subscription(Array, state_topic, self.callback_state, 10)
        self.disturbance_dims = config.disturbance_space["n_dims"]

        if not self.disturbance_dims == 0:
            self.disturbance_lo = np.array(config.disturbance_space["lo"])
            self.disturbance_hi = np.array(config.disturbance_space["hi"])

        self.dynamics = config.dynamics
        self.state = None
        self.state_initialized = False

        self.beta_skew = self.get_parameter("beta_skew").value
        self.seed = self.get_parameter("seed").value
        self.random_state = np.random.default_rng(seed=self.seed)
        self.rate = self.get_parameter("rate").value

        self.timer = self.create_timer(1.0 / self.rate, self.run)  # Triggers the run method at the specified rate

    def run(self):
        if self.state_initialized:
            per_state_disturbance_msg = Array()
            per_state_disturbance = self.compute_disturbance()
            per_state_disturbance_msg.value = per_state_disturbance.tolist()
            self.pub_disturbance.publish(per_state_disturbance_msg)

    def compute_disturbance(self):
        if self.disturbance_dims == 0:
            return np.zeros_like(self.state)
        disturbance = (
            self.random_state.beta(self.beta_skew, self.beta_skew, size=self.disturbance_dims)
            * (self.disturbance_hi - self.disturbance_lo)
            + self.disturbance_lo
        )
        per_state_disturbance = self.dynamics.disturbance_matrix(self.state, 0.0) @ disturbance
        return per_state_disturbance

    def callback_disturbance_update(self, msg):
        self.disturbance_lo = np.array(msg.lo)
        self.disturbance_hi = np.array(msg.hi)
        self.disturbance_dims = len(self.disturbance_lo)

    def callback_state(self, msg):
        self.state = np.array(msg.value)
        self.state_initialized = True


def main(args=None):
    rclpy.init(args=args)
    disturbance_node = DisturbanceNode()
    rclpy.spin(disturbance_node)
    disturbance_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
