#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp
from refinecbf_ros2.msg import ValueFunctionMsg, HiLoArray

# Ensure the following imports are compatible with ROS2 or appropriately adapted
from config import Config, QuadraticCBF
from refine_cbfs import HJControlAffineDynamics, TabularControlAffineCBF
from example_interfaces.msg import Bool
import threading
import time
from utils import load_parameters, load_array


class HJReachabilityNode(Node):
    """
    HJReachabilityNode is a ROS node that computes the Hamilton-Jacobi reachability for a robot.

    Subscribers:
    - disturbance_update_sub (~topics/disturbance_update): Updates the disturbance.
    - actuation_update_sub (~topics/actuation_update): Updates the actuation.
    - sdf_update_sub (~topics/sdf_update): Updates the obstacles.

    Publishers:
    - vf_pub (~topics/vf_update): Publishes the value function.
    """

    def __init__(self) -> None:
        """
        Initializes the HJReachabilityNode. It sets up ROS subscribers for disturbance, actuation, and obstacle updates,
        and a publisher for the value function. It also initializes the Hamilton-Jacobi dynamics and the value function.
        """
        # Load configuration
        super().__init__("hj_reachability_node")
        self.config = Config(self, hj_setup=True)
        # Initialize dynamics, grid, and Hamilton-Jacobi dynamics
        self.dynamics = self.config.dynamics
        self.grid = self.config.grid
        self.hj_dynamics = self.config.hj_dynamics
        self.control_space = self.hj_dynamics.control_space
        self.disturbance_space = self.hj_dynamics.disturbance_space

        # Topics (parameters)
        self.declare_parameters(
            "",
            [
                ("topics.sdf_update", rclpy.Parameter.Type.STRING),
                ("topics.disturbance_update", rclpy.Parameter.Type.STRING),
                ("topics.actuation_update", rclpy.Parameter.Type.STRING),
                ("topics.vf_update", rclpy.Parameter.Type.STRING),
            ],
        )

        self.declare_parameter("vf_update_method", "pubsub")
        self.declare_parameter("vf_update_accuracy", "high")
        self.declare_parameter("vf_initialization_method", "sdf")
        self.declare_parameter("initial_vf_file", "None")
        self.declare_parameter("update_vf_online", True)

        control_config = load_parameters(self.get_parameter("robot").value, self.get_parameter("exp").value, "control")

        self.vf_update_method = self.get_parameter("vf_update_method").value
        self.vf_update_accuracy = self.get_parameter("vf_update_accuracy").value
        # Initialize a lock for thread-safe value function updates
        # Get initial safe space and setup solver
        self.sdf_update_topic = self.get_parameter("topics.sdf_update").value
        self.sdf_available = False

        if self.vf_update_method == "pubsub":
            self.sdf_subscriber = self.create_subscription(
                ValueFunctionMsg, self.sdf_update_topic, self.callback_sdf_update_pubsub, 10
            )
        elif self.vf_update_method == "file":
            self.sdf_subscriber = self.create_subscription(
                Bool, self.sdf_update_topic, self.callback_sdf_update_file, 10
            )
        else:
            raise NotImplementedError(f"{self.vf_update_method} is not a valid vf update method")

        # Wait while not sdf update topic received
        self.first_message_received = threading.Event()
        self.spin_thread = threading.Thread(target=self.spin)
        self.spin_thread.start()
        self.first_message_received.wait()

        self.brt = lambda sdf_values: lambda t, x: jnp.minimum(x, sdf_values)
        self.solver_settings = hj.SolverSettings.with_accuracy(
            self.vf_update_accuracy, value_postprocessor=self.brt(self.sdf_values)
        )

        self.vf_initialization_method = self.get_parameter("vf_initialization_method").value
        if self.vf_initialization_method == "sdf":
            self.vf = self.sdf_values.copy()
        elif self.vf_initialization_method == "cbf":
            cbf_params = control_config["initial_cbf"]
            original_cbf = QuadraticCBF(self.dynamics, cbf_params["Parameters"], test=False)
            tabular_cbf = TabularControlAffineCBF(self.dynamics, params={}, test=False, grid=self.grid)
            tabular_cbf.tabularize_cbf(original_cbf)
            self.vf = tabular_cbf.vf_table.copy()
        elif self.vf_initialization_method == "file":
            self.vf = load_array(self.get_parameter("robot").value, self.get_parameter("exp").value, "vf")
            if self.vf.ndim == self.grid.ndim + 1:
                self.vf = self.vf[-1]
            assert self.vf.shape == tuple(self.config.grid_shape), "vf file is not compatible with grid size"
        else:
            raise NotImplementedError("{} is not a valid initialization method".format(self.vf_initialization_method))
        self.get_logger().info(f"Share of safe cells: {np.sum(self.vf >= 0) / self.vf.size:.3f}")

        # Set up value function publisher
        self.vf_topic = self.get_parameter("topics.vf_update").value

        # Log a warning if the update flag is not set
        self.update_vf_flag = self.get_parameter("update_vf_online").value
        if not self.update_vf_flag:
            self.get_logger().warn("Value function is not being updated")

        # Publishers depending on the update method
        if self.vf_update_method == "pubsub":
            self.vf_pub = self.create_publisher(ValueFunctionMsg, self.vf_topic, 10)
        else:  # self.vf_update_method == "file"
            self.vf_pub = self.create_publisher(Bool, self.vf_topic, 10)

        # Subscribers setup
        disturbance_update_topic = self.get_parameter("topics.disturbance_update").value
        self.disturbance_update_sub = self.create_subscription(
            HiLoArray, disturbance_update_topic, self.callback_disturbance_update, 10
        )

        actuation_update_topic = self.get_parameter("topics.actuation_update").value
        self.actuation_update_sub = self.create_subscription(
            HiLoArray, actuation_update_topic, self.callback_actuation_update, 10
        )

        # Start updating the value function
        self.publish_initial_vf()
        self.update_vf()  # This method spins indefinitely

    def spin(self):
        rclpy.spin(self)

    def publish_initial_vf(self):
        # ROS2 uses a slightly different API for waiting for subscribers
        self.get_logger().info("Number of subscribers: {}".format(self.vf_pub.get_subscription_count()))
        while self.vf_pub.get_subscription_count() < 2:
            self.get_logger().info("HJR node: Waiting for subscribers to connect")
            time.sleep(1)
        if self.vf_update_method == "pubsub":
            msg = ValueFunctionMsg()
            msg.vf = self.vf.flatten().tolist()  # Ensure data is in a suitable format
            self.vf_pub.publish(msg)
        else:  # self.vf_update_method == "file"
            np.save("vf.npy", self.vf.copy())
            self.vf_pub.publish(Bool(data=True))  # Publish a Bool message indicating completion

    def callback_disturbance_update(self, msg):
        """
        Callback for the disturbance update subscriber.

        Args:
            msg (HiLoArray): The incoming disturbance update message.

        This method updates the disturbance space and the dynamics.
        """
        max_disturbance = msg.hi
        min_disturbance = msg.lo
        self.disturbance_space = hj.sets.Box(lo=jnp.array(min_disturbance), hi=jnp.array(max_disturbance))
        self.update_dynamics()  # FIXME:Check whether this is required or happens automatically

    def callback_actuation_update(self, msg):
        """
        Callback for the actuation update subscriber.

        Args:
            msg (HiLoArray): The incoming actuation update message.

        This method updates the control space and the dynamics.
        """
        max_control = msg.hi
        min_control = msg.lo
        self.control_space = hj.sets.Box(lo=jnp.array(min_control), hi=jnp.array(max_control))
        self.update_dynamics()  # FIXME:Check whether this is required or happens automatically

    def callback_sdf_update_pubsub(self, msg):
        """
        Callback for the obstacle update subscriber.

        Args:
            msg (ValueFunctionMsg): The incoming obstacle update message.

        This method updates the obstacle and the solver settings.
        """
        self.get_logger().info("SDF update received")
        self.sdf_values = jnp.array(msg.vf).reshape(self.config.grid_shape)
        if not self.first_message_received.is_set():
            self.first_message_received.set()
        else:
            self.solver_settings = hj.SolverSettings.with_accuracy(
                self.vf_update_accuracy, value_postprocessor=self.brt(self.sdf_values)
            )

    def callback_sdf_update_file(self, msg):
        self.get_logger().info("SDF update received")
        if not msg.data:
            return
        self.sdf_values = np.array(np.load("sdf.npy")).reshape(self.config.grid_shape)
        if not self.first_message_received.is_set():
            self.first_message_received.set()
        else:
            self.solver_settings = hj.SolverSettings.with_accuracy(
                self.vf_update_accuracy, value_postprocessor=self.brt(self.sdf_values)
            )

    def update_dynamics(self):
        """
        Updates the Hamilton-Jacobi dynamics based on the current control and disturbance spaces.
        """
        self.hj_dynamics = HJControlAffineDynamics(
            self.dynamics,
            control_space=self.control_space,
            disturbance_space=self.disturbance_space,
        )

    def update_vf(self):
        """
        Continuously updates the value function and publishes it as long as the node is running and the update flag is set.
        """
        while rclpy.ok():
            if self.update_vf_flag:
                self.get_logger().info(f"Share of safe cells: {np.sum(self.vf >= 0) / self.vf.size:.3f}", throttle_duration_sec=5.0)
                time_now = self.get_clock().now().seconds_nanoseconds()[0]
                new_values = hj.step(
                    self.solver_settings,
                    self.hj_dynamics,
                    self.grid,
                    0.0,
                    self.vf.copy(),
                    -0.5,
                    progress_bar=False,
                )
                elapsed_time = self.get_clock().now().seconds_nanoseconds()[0] - time_now
                self.get_logger().info(f"Time taken to calculate vf: {elapsed_time:.2f}", throttle_duration_sec=5.0)
                self.vf = new_values
                if self.vf_update_method == "pubsub":
                    self.vf_pub.publish(ValueFunctionMsg(vf=self.vf.flatten().tolist()))
                else:  # self.vf_update_method == "file"
                    np.save("vf.npy", self.vf)
                    self.vf_pub.publish(Bool(data=True))


def main(args=None):
    rclpy.init(args=args)
    hj_reachability_node = HJReachabilityNode()
    rclpy.spin(hj_reachability_node)
    hj_reachability_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
