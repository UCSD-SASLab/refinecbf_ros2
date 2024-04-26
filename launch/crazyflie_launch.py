import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
import yaml

package_name = 'refinecbf_ros2'


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def generate_launch_description():
    topics_config_path = os.path.join(get_package_share_directory(package_name), 'config', 'topics_config.yaml')
    topics_config = load_yaml(topics_config_path)['/**']['ros__parameters']

    cf_topics_config_path = os.path.join(get_package_share_directory(package_name), 'config', 'crazyflie_topics_config.yaml')
    cf_topics_config = load_yaml(cf_topics_config_path)['/**']['ros__parameters']
    # crazyflie_topics_config_path = FindPackageShare('your_package_name') + '/config/crazyflie_topics.yaml'
    # control_config_path = FindPackageShare('your_package_name') + '/config/control_config.yaml'
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
            'backend', default_value='sim',
            description='cpp / cflib / sim backend for crazyflie'),
        DeclareLaunchArgument(
            'vf_update_method', default_value='pubsub',
            description='Message parsing method for VF update'),
        DeclareLaunchArgument(
            'vf_update_accuracy', default_value='high',
            description='Accuracy of HJ Reachability computation'),
        DeclareLaunchArgument(
            'env_config_file', default_value='detection_env.yaml',
            description='Environment config file'),
        DeclareLaunchArgument(
            'control_config_file', default_value='crazyflie_control.yaml',
            description='Control config file'),
        DeclareLaunchArgument(
            'CBF_parameter_file', default_value='crazyflie_CBF_params.yaml',
            description='CBF parameter file'),
        DeclareLaunchArgument(
            'initial_vf_file', default_value='target_values_judy.npy',
            description='Initial VF file'),
        DeclareLaunchArgument(
            'robot_number', default_value='0',
            description='Robot number for experiments'),
        # Nodes configurations
        Node(
            package='refinecbf_ros2',
            executable='cf_nominal_controller.py',
            name='cf_nominal_control',
            output='screen',
            parameters=[topics_config_path,
                        cf_topics_config_path,
                        {'env_config_file': LaunchConfiguration('env_config_file'),
                         'control_config_file': LaunchConfiguration('control_config_file'),
                         },
                        ],
        ),
        Node(
            package='refinecbf_ros2',
            executable='cf_hw_interface.py',
            name='cf_hw_interface',
            output='screen',
            parameters=[topics_config_path,
                        cf_topics_config_path,
                        {'control_config_file': LaunchConfiguration('control_config_file'),
                         'env_config_file': LaunchConfiguration('env_config_file'),
                         'backend': LaunchConfiguration('backend'),
                        },
                        ]
        ),
        Node(
            package='refinecbf_ros2',
            executable='obstacle_node.py',
            output='screen',
            parameters=[topics_config_path,
                        {'vf_update_method': LaunchConfiguration('vf_update_method'),
                         'env_config_file': LaunchConfiguration('env_config_file')},
                        ]
        ),
        Node(
            package='refinecbf_ros2',
            executable='disturbance_node.py',
            name='disturbance_node',
            output='screen',
            parameters=[topics_config_path,
                        {'env_config_file': LaunchConfiguration('env_config_file')}]

        ),
        Node(
            package='refinecbf_ros2',
            executable='cf_visualization.py',
            name='cf_visualization',
            output='screen',
            parameters=[topics_config_path,
                        {'vf_update_method': LaunchConfiguration('vf_update_method'),
                         'env_config_file': LaunchConfiguration('env_config_file'),
                         'control_config_file': LaunchConfiguration('control_config_file')},
                        ]
        ),
        # Include other launch files
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare(package_name),
                    'launch',
                    'refine_cbf_launch.py'
                ])
            ]),
            launch_arguments={
                topics_config_path: topics_config_path,
                'env_config_file': LaunchConfiguration('env_config_file'),
                'control_config_file': LaunchConfiguration('control_config_file'),
                'safety_filter_active': LaunchConfiguration('safety_filter_active'),
                'update_vf_online': LaunchConfiguration('update_vf_online'),
                'vf_initialization_method': LaunchConfiguration('vf_initialization_method'),
                'vf_update_method': LaunchConfiguration('vf_update_method'),
                'vf_update_accuracy': LaunchConfiguration('vf_update_accuracy'),
                'CBF_parameter_file': LaunchConfiguration('CBF_parameter_file'),
                'initial_vf_file': LaunchConfiguration('initial_vf_file'),
            }.items()
        ),
        GroupAction(
            actions=[
                SetRemap('/cf231/odom', topics_config['topics']['robot_state']),
                SetRemap('/cf231/land', cf_topics_config['services']['land']),
                SetRemap('/cf231/takeoff', cf_topics_config['services']['takeoff']),
                SetRemap('/cf231/notify_setpoints_stop', cf_topics_config['services']['stop_setpoints']),
                SetRemap('/cf231/cmd_vel_legacy', topics_config['topics']['robot_safe_control']),

                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        PathJoinSubstitution([
                            FindPackageShare('crazyflie'),
                            'launch',
                            'launch.py'
                        ])
                    ]),
                    launch_arguments={
                        'backend': LaunchConfiguration('backend'),
                    }.items(),
                )
            ]
        )
    ])
