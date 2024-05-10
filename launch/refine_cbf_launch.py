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
            'safety_filter_active',
            description='Activate refinecbf based safety filter'),
        DeclareLaunchArgument(
            'update_vf_online',
            description='Update value function online using HJ Reachability'),
        DeclareLaunchArgument(
            'vf_initialization_method',
            description='Value function initialization method'),
        DeclareLaunchArgument(
            'vf_update_method',
            description='Message parsing method for VF update'),
        DeclareLaunchArgument(
            'vf_update_accuracy',
            description='Accuracy of HJ Reachability computation'),
        DeclareLaunchArgument(
            'robot',
            description='Robot name'),
        DeclareLaunchArgument(
            'exp',
            description='Which experiment to run'),


        Node(
            package='refinecbf_ros2',
            executable='refine_cbf_node.py',
            name='safety_filter_node',
            output='screen',
            parameters=[
                topics_config,
                {'safety_filter_active': LaunchConfiguration('safety_filter_active'),
                 'vf_update_method': LaunchConfiguration('vf_update_method'),
                 'robot': LaunchConfiguration('robot'),
                 'exp': LaunchConfiguration('exp'),
                 }
            ]),
        Node(
            package='refinecbf_ros2',
            executable='obstacle_node.py',
            output='screen',
            parameters=[topics_config,
                        {'robot': LaunchConfiguration('robot'),
                         'vf_update_method': LaunchConfiguration('vf_update_method'),
                         'exp': LaunchConfiguration('exp'),
                         },
                        ]
        ),
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
                 'robot': LaunchConfiguration('robot'),
                 'exp': LaunchConfiguration('exp'),
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
                {'robot': LaunchConfiguration('robot'),
                 'exp': LaunchConfiguration('exp'),
                }
            ],
        )
        ])