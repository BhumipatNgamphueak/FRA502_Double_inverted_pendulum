#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Launch file for LQR controller only (without Gazebo)
    
    Use this when you want to run the controller separately from the simulation.
    Make sure the simulation is already running before launching this.
    """
    
    # Configuration file
    lqr_config = PathJoinSubstitution([
        FindPackageShare('double_inverted_pendulum'),
        'config',
        'lqr_params.yaml'
    ])
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=lqr_config,
        description='Path to LQR configuration file'
    )
    
    # LQR Controller Node
    lqr_controller = Node(
        package='double_inverted_pendulum',
        executable='lqr_controller_node.py',
        name='lqr_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        config_file_arg,
        lqr_controller
    ])
