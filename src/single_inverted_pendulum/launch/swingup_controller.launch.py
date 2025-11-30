#!/usr/bin/env python3
"""
Launch file for Single Pendulum Swing-Up Controller
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    # Package share directory
    pkg_share = FindPackageShare('single_inverted_pendulum')
    
    # Config file path
    params_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'single_pendulum_swingup_params.yaml'
    ])
    
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=params_file,
        description='Path to parameter file'
    )
    
    # Controller node
    controller_node = Node(
        package='single_inverted_pendulum',
        executable='single_pendulum_swingup_controller.py',
        name='swingup_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        emulate_tty=True,
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        params_file_arg,
        controller_node,
    ])
