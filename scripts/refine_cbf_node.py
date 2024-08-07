#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import jax
import jax.numpy as jnp
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')
from refinecbf_ros2.msg import ValueFunctionMsg, Array, HiLoArray
from refinecbf_ros2.srv import ProcessState
from example_interfaces.msg import Bool, Float32
from refine_cbfs import TabularControlAffineCBF
from cbf_opt import ControlAffineASIF, SlackifiedControlAffineASIF
from config import Config
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import matplotlib.pyplot as plt
from utils import load_parameters
import threading


class SafetyFilterNode(Node):
    """
    Docstring for the ROS2 version of the SafetyFilterNode
    """

    def __init__(self):
        super().__init__("safety_filter_node")
        self.initialized_safety_filter = False  # Initialization flag for the callback

        self.config = Config(self, hj_setup=True)
        self.dynamics = self.config.dynamics
        self.grid = self.config.grid
        self.safety_states_idis = self.config.safety_states
        self.safety_controls_idis = self.config.safety_controls
        self.vf_update_callback_group = MutuallyExclusiveCallbackGroup()
        self.other_callback_group = ReentrantCallbackGroup()
        self.service_callback_group = ReentrantCallbackGroup()
        # Parameters (for topics)
        self.declare_parameters(
            "",
            [
                ("topics.vf_update", rclpy.Parameter.Type.STRING),
                ("topics.cbf_state", rclpy.Parameter.Type.STRING),
                ("topics.cbf_nominal_control", rclpy.Parameter.Type.STRING),
                ("topics.cbf_safe_control", rclpy.Parameter.Type.STRING),
                ("topics.actuation_update", rclpy.Parameter.Type.STRING),
                ("topics.disturbance_update", rclpy.Parameter.Type.STRING),
                ("topics.value_function", rclpy.Parameter.Type.STRING),
                ("services.find_closest_safe_state", rclpy.Parameter.Type.STRING),
            ],
        )

        self.declare_parameter("safety_filter_active", True)
        self.declare_parameter("vf_update_method", "pubsub")

        control_config = load_parameters(self.get_parameter("robot").value, self.get_parameter("exp").value, "control")

        self.declare_parameter("control.asif.gamma", control_config["asif"]["gamma"])
        self.declare_parameter("control.asif.slack", control_config["asif"]["slack"])

        # Subscribers
        self.vf_update_method = self.get_parameter("vf_update_method").value
        vf_topic = self.get_parameter("topics.vf_update").value
        if self.vf_update_method == "pubsub":
            self.vf_sub = self.create_subscription(ValueFunctionMsg, vf_topic, self.callback_vf_update_pubsub, 
                                                   1, callback_group=self.vf_update_callback_group)
        elif self.vf_update_method == "file":
            self.vf_sub = self.create_subscription(Bool, vf_topic, self.callback_vf_update_file, 
                                                   1, callback_group=self.vf_update_callback_group)
        else:
            raise NotImplementedError(f"{self.vf_update_method} is not a valid vf update method")

        state_topic = self.get_parameter("topics.cbf_state").value
        self.state_sub = self.create_subscription(Array, state_topic, self.callback_state, 
                                                  1, callback_group=self.other_callback_group)
        self.state = None

        # CBF setup
        gamma = self.get_parameter("control.asif.gamma").value
        slackify_safety_constraint = self.get_parameter("control.asif.slack").value
        self.get_logger().info(f"Using gamma: {gamma}, slackify: {slackify_safety_constraint}")
        alpha = lambda x: gamma * x
        self.active_buffer_cbf = TabularControlAffineCBF(self.dynamics, grid=self.grid, alpha=alpha)
        self.back_buffer_cbf = TabularControlAffineCBF(self.dynamics, grid=self.grid, alpha=alpha)
        self.lock = threading.Lock()

        if slackify_safety_constraint:
            self.safety_filter_solver = SlackifiedControlAffineASIF(self.dynamics, self.active_buffer_cbf)
        else:  # Enforce strict inequality
            self.safety_filter_solver = ControlAffineASIF(self.dynamics, self.active_buffer_cbf)

        # Control limits
        self.safety_filter_solver.umin = np.array(self.config.control_space["lo"])
        self.safety_filter_solver.umax = np.array(self.config.control_space["hi"])
        if self.config.disturbance_space["n_dims"] != 0:
            self.safety_filter_solver.dmin = np.array(self.config.disturbance_space["lo"])
            self.safety_filter_solver.dmax = np.array(self.config.disturbance_space["hi"])

        # Control subscriptions
        nom_control_topic = self.get_parameter("topics.cbf_nominal_control").value
        self.nominal_control_sub = self.create_subscription(Array, nom_control_topic, self.callback_safety_filter, 
                                                            1, callback_group=self.other_callback_group)

        filtered_control_topic = self.get_parameter("topics.cbf_safe_control").value
        self.pub_filtered_control = self.create_publisher(Array, filtered_control_topic, 1)

        self.declare_parameter("sensing_online", rclpy.Parameter.Type.BOOL)
        self.sensing_online = self.get_parameter("sensing_online").value
        
        if self.sensing_online:
            actuation_update_topic = self.get_parameter("topics.actuation_update").value
            self.actuation_update_sub = self.create_subscription(
                HiLoArray, actuation_update_topic, self.callback_actuation_update, 
                1, callback_group=self.other_callback_group
            )

            # Optional disturbance updates
            if self.config.disturbance_space["n_dims"] != 0:
                disturbance_update_topic = self.get_parameter("topics.disturbance_update").value
                self.disturbance_update_sub = self.create_subscription(
                    HiLoArray, disturbance_update_topic, self.callback_disturbance_update, 
                    1, callback_group=self.other_callback_group
                )
        find_safe_state_service = self.get_parameter("services.find_closest_safe_state").value
        self.create_service(ProcessState, find_safe_state_service, self.find_closest_safe_state_service,
                            callback_group=self.service_callback_group)

        # Value function publishing
        value_function_topic = self.get_parameter("topics.value_function").value
        self.value_function_pub = self.create_publisher(Float32, value_function_topic, 1)

        self.safety_filter_active = self.get_parameter("safety_filter_active").value
        if self.safety_filter_active:
            self.initialized_safety_filter = False
            self.safety_filter_solver.setup_optimization_problem()
            self.get_logger().info("Safety filter is used, but not initialized yet")
        else:
            self.initialized_safety_filter = True
            self.safety_filter_solver = lambda state, nominal_control: nominal_control
            self.get_logger().warn("No safety filter, be careful!")

    def callback_actuation_update(self, msg):
        self.safety_filter_solver.umin = np.array(msg.lo)
        self.safety_filter_solver.umax = np.array(msg.hi)

    def callback_disturbance_update(self, msg):
        self.safety_filter_solver.dmin = np.array(msg.lo)
        self.safety_filter_solver.dmax = np.array(msg.hi)
        self.get_logger().info(f"Updated disturbance bounds: {self.safety_filter_solver.dmin}, {self.safety_filter_solver.dmax}")

    def callback_vf_update_file(self, vf_msg):
        if not vf_msg.data:
            return
        try:
            self.back_buffer_cbf.vf_table = np.load("vf.npy").reshape(self.config.grid_shape)
        except (ValueError, EOFError):
            self.get_logger().info("Value function file not found, waiting for next update")
            import time
            time.sleep(0.01)
            self.back_buffer_cbf.vf_table = np.load("vf.npy").reshape(self.config.grid_shape)
        with self.lock:
            self.swap_buffers()
        if not self.initialized_safety_filter:
            self.get_logger().info("Initialized safety filter")
            self.initialized_safety_filter = True
    
    def find_closest_safe_state_service(self, request, response):
        """
        Service to find closest safe state to requested target state
        """
        desired_state = jnp.array(request.desired_state.value)
        weighting = jnp.array(request.weighting.value)
        if (not self.safety_filter_active) or self.back_buffer_cbf.vf(desired_state, 0.0) >= 0.0:
            self.get_logger().warn("Current state is already safe")
            processed_state = Array()
            processed_state.value = desired_state.tolist()
            response.processed_state = processed_state
            return response

        with self.lock:
            contour = plt.contour(
                self.grid.coordinate_vectors[0],
                self.grid.coordinate_vectors[1],
                self.back_buffer_cbf.vf_table[:, :, self.grid.nearest_index(desired_state)[2], self.grid.nearest_index(desired_state)[3]].T,
                levels=[0.5],
            )
        array_points = [path.vertices for path in contour.collections[0].get_paths()][0]
        boundary_states = np.concatenate([array_points, np.zeros_like(array_points)], axis=1)
        closest_state = boundary_states[np.argmin(np.linalg.norm(weighting * (boundary_states- desired_state), axis=1))]
        processed_state = Array()
        processed_state.value = closest_state.tolist()
        response.processed_state = processed_state
        return response

    def swap_buffers(self):
        if self.active_buffer_cbf.vf_table is None:
            self.active_buffer_cbf.vf_table = self.back_buffer_cbf.vf_table.copy()
        else:
            self.active_buffer_cbf, self.back_buffer_cbf = self.back_buffer_cbf, self.active_buffer_cbf
        self.safety_filter_solver.cbf = self.active_buffer_cbf

    def callback_vf_update_pubsub(self, vf_msg):
        self.back_buffer_cbf.vf_table = np.array(vf_msg.vf).reshape(self.config.grid_shape)
        with self.lock:
            self.swap_buffers()
        if not self.initialized_safety_filter:
            self.get_logger().info("Initialized safety filter")
            self.initialized_safety_filter = True

    def callback_safety_filter(self, control_msg):
        nom_control = np.array(control_msg.value)
        if self.state is None:
            self.get_logger().info("State not set yet, no control published", throttle_duration_sec=5.0)
            return
        if not self.initialized_safety_filter:
            safety_control_msg = control_msg
            self.get_logger().warn(
                "Safety filter not initialized yet, outputting nominal control", throttle_duration_sec=2.0
            )
        else:
            nom_control_active = nom_control[self.safety_controls_idis]
            safety_control_msg = Array()
            if hasattr(self.safety_filter_solver, "cbf"):
                vf = np.array(self.safety_filter_solver.cbf.vf(self.state.copy(), 0.0)).item()
                self.value_function_pub.publish(Float32(data=vf))
            safety_control_active = self.safety_filter_solver(
                self.state.copy(), nominal_control=np.array([nom_control_active])
            )
            safety_control = nom_control.copy()
            safety_control[self.safety_controls_idis] = safety_control_active[0]
            safety_control_msg.value = safety_control.tolist()

        self.pub_filtered_control.publish(safety_control_msg)

    def callback_state(self, state_msg):
        self.state = np.array(state_msg.value)[self.safety_states_idis]


def main(args=None):
    rclpy.init(args=args)
    safety_filter = SafetyFilterNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(safety_filter)
    try:
        executor.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
