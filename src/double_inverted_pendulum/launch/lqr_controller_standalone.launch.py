#!/usr/bin/env python3
"""
Launch file for LQR Stabilization Controller
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for LQR stabilization controller"""
    
    # Get package share directory
    # NOTE: Replace 'double_invert_pendulum_control' with your actual package name
    pkg_share = FindPackageShare('double_inverted_pendulum')
    
    # Path to parameters file
    params_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'lqr_params.yaml'
    ])
    
    # Declare launch arguments
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
    
    # LQR Stabilization Node
    lqr_stabilization_node = Node(
        package='double_inverted_pendulum',
        executable='lqr_controller_node.py',
        name='lqr_stabilization_controller',
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
        lqr_stabilization_node,
    ])