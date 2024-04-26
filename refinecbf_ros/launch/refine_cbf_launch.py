import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, OpaqueFunction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.conditions import IfCondition

package_name = 'refinecbf_ros2'

def generate_launch_description():
    topics_config = os.path.join(get_package_share_directory(package_name), 'config', 'topics_config.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'safety_filter_active', default_value='True',
            description='Activate refinecbf based safety filter'),
        DeclareLaunchArgument(
            'update_vf_online', default_value='True',
            description='Update value function online using HJ Reachability'),
        DeclareLaunchArgument(
            'vf_initialization_method', default_value='sdf',
            description='Value function initialization method'),
        DeclareLaunchArgument(
            'vf_update_method', default_value='file',
            description='Message parsing method for VF update'),
        DeclareLaunchArgument(
            'CBF_parameter_file', default_value=None,
            description='CBF parameter file'),
        DeclareLaunchArgument(
            'initial_vf_file', default_value=None,
            description='Initial value function file'),
        DeclareLaunchArgument(
            'vf_update_accuracy', default_value='high',
            description='Accuracy of HJ Reachability computation'),
        DeclareLaunchArgument(
            'env_config_file', default_value='detection_env.yaml',
            description='Environment config file'),
        DeclareLaunchArgument(
            'control_config_file', default_value='crazyflie_control.yaml',
            description='Control config file'),
        Node(
            package='refinecbf_ros2',
            executable='refine_cbf_node.py',
            name='safety_filter_node',
            output='screen',
            parameters=[
                topics_config,
                {'safety_filter_active': LaunchConfiguration('safety_filter_active'),
                 'vf_update_method': LaunchConfiguration('vf_update_method'),
                 'env_config_file': LaunchConfiguration('env_config_file'),
                 'control_config_file': LaunchConfiguration('control_config_file'),}
            ]),
        Node(
            package='refinecbf_ros2',
            executable='hj_reachability_node.py',
            name='hj_reachability_node',
            output='screen',
            parameters=[
                topics_config,
                {'update_vf_online': LaunchConfiguration('update_vf_online'),
                 'vf_initialization_method': LaunchConfiguration('vf_initialization_method'),
                 'vf_update_accuracy': LaunchConfiguration('vf_update_accuracy'),
                 'vf_update_method': LaunchConfiguration('vf_update_method'),
                 'env_config_file': LaunchConfiguration('env_config_file'),
                 'initial_vf_file': LaunchConfiguration('initial_vf_file'),
                 'CBF_parameter_file': LaunchConfiguration('CBF_parameter_file'),
                }
                
            ],
            condition=IfCondition(LaunchConfiguration('safety_filter_active')),
            ),
        Node(
            package='refinecbf_ros2',
            executable='modify_environment.py',
            name='modify_environment_node',
            output='screen',
            parameters=[
                topics_config,
                {'env_config_file': LaunchConfiguration('env_config_file'),
                }
            ],
        )
        ])