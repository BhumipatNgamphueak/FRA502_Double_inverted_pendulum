#!/usr/bin/env python3
"""
Launch file for Swing-Up + Balance Controller
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    pkg_share = FindPackageShare('double_inverted_pendulum')
    
    params_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'trajectory_tracking_params.yaml'
    ])
    
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
    
    controller_node = Node(
        package='double_inverted_pendulum',
        executable='trajectory_tracking_controller.py',
        name='swingup_balance_controller',
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