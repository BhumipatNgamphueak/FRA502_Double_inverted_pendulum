#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """
    Launch file for LQR-controlled double inverted pendulum with Gazebo
    
    This launch file:
    1. Includes the full Gazebo simulation launch
    2. Starts the LQR controller node
    3. Loads LQR parameters from config file
    """
    
    # Get package directories
    controller_pkg = get_package_share_directory('double_inverted_pendulum')
    sim_pkg = get_package_share_directory('double_invert_pendulum_simulation')
    
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
    
    # Include Gazebo simulation launch
    # This starts: Gazebo, robot_state_publisher, controllers, RViz
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sim_pkg, 'launch', 'simulation_full_launch.py')
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )
    
    # LQR Controller Node
    lqr_controller = Node(
        package='double_inverted_pendulum',
        executable='lqr_controller_node.py',
        name='lqr_controller',
        output='screen',
        parameters=[
            lqr_config,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        emulate_tty=True,
        respawn=False
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        gazebo_launch,
        lqr_controller
    ])
