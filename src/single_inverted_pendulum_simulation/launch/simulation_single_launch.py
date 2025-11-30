#!/usr/bin/env python3
"""
Launch file for Single Inverted Pendulum in Gazebo
FIXED: Added delays for proper controller loading
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    # Package directories
    desc_pkg = FindPackageShare('single_inverted_pendulum_description')
    sim_pkg = FindPackageShare('single_inverted_pendulum_simulation')
    gazebo_ros_share = FindPackageShare('ros_gz_sim')
    
    # Paths
    xacro_file = PathJoinSubstitution([
        desc_pkg,
        'robot',
        'visual',
        'single_inverted_pendulum.xacro'
    ])
    
    world_file = PathJoinSubstitution([
        sim_pkg,
        'worlds',
        'empty.sdf'
    ])
    
    controller_config = PathJoinSubstitution([
        sim_pkg,
        'config',
        'controller_single.yaml'
    ])
    
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=world_file,
        description='World file'
    )
    
    # Robot description
    robot_description = Command(['xacro ', xacro_file])
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )
    
    # Gazebo server
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                gazebo_ros_share,
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': ['-r -v4 ', LaunchConfiguration('world')],
            'on_exit_shutdown': 'true'
        }.items()
    )
    
    # Spawn robot
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'single_inverted_pendulum',
            '-z', '0.0'
        ],
        output='screen'
    )
    
    # FIXED: Add delay before loading controllers
    # Give Gazebo and ros2_control time to initialize (3 seconds)
    load_joint_state_broadcaster = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster', 
                          '--controller-manager', '/controller_manager'],
                output='screen'
            )
        ]
    )
    
    # FIXED: Load effort controller after joint_state_broadcaster (add 2 more seconds)
    load_effort_controller = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['effort_controller',
                          '--controller-manager', '/controller_manager'],
                output='screen'
            )
        ]
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        world_arg,
        robot_state_publisher,
        gazebo,
        spawn_entity,
        load_joint_state_broadcaster,
        load_effort_controller
    ])