#!/usr/bin/env python3
"""
RDIP Controller - ULTIMATE FIX
- MUCH stronger balance control for β (second pendulum)
- Proper angle wrapping for BOTH α and β
- Emergency stop feature
- Restart capability
"""

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String, Float64, Bool
from geometry_msgs.msg import Vector3Stamped, PointStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import os
from ament_index_python.packages import get_package_share_directory


class SwingUpBalanceController(Node):
    
    MODE_SETTLING = 0
    MODE_SWINGUP = 1
    MODE_BALANCE = 2
    MODE_FALLEN = 3
    MODE_EMERGENCY_STOP = 4  # NEW: Emergency stop mode
    
    def __init__(self):
        super().__init__('swingup_balance_controller')
        
        self._declare_parameters()
        self._get_parameters()
        self._load_trajectory()
        self._load_lqr_gains()
        self._compute_balance_gains()
        
        self.current_state = np.zeros(6)
        self.state_received = False
        
        self.mode = self.MODE_SETTLING
        self.trajectory_time = 0.0
        self.theta_integral = 0.0
        
        self.settle_start_time = None
        self.settle_check_start = None
        self.balance_start_time = None
        self.fall_time = None
        self.trajectory_start_time = None
        
        # Emergency stop flag
        self.emergency_stop_requested = False
        
        # For publishing
        self.last_tau = 0.0
        self.last_x_ref = np.zeros(7)
        self.last_x_error = np.zeros(7)
        
        # Publishers/Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, self.joint_states_topic, self.joint_state_callback, 10)
        
        # NEW: Emergency stop subscriber
        self.emergency_stop_sub = self.create_subscription(
            Bool, '/emergency_stop', self.emergency_stop_callback, 10)
        
        self.effort_pub = self.create_publisher(
            Float64MultiArray, self.effort_command_topic, 10)
        self.mode_pub = self.create_publisher(String, '/controller_mode', 10)
        
        self.ref_alpha_pub = self.create_publisher(Float64, '/reference/alpha', 10)
        self.ref_beta_pub = self.create_publisher(Float64, '/reference/beta', 10)
        self.ref_theta_pub = self.create_publisher(Float64, '/reference/theta', 10)
        self.error_pub = self.create_publisher(PointStamped, '/tracking/error', 10)
        self.wrapped_state_pub = self.create_publisher(PointStamped, '/joint_states_wrapped', 10)
        
        if self.publish_debug_info:
            self.debug_pub = self.create_publisher(Vector3Stamped, self.debug_topic, 10)
        
        self.control_period = 1.0 / self.control_frequency
        self.control_timer = self.create_timer(self.control_period, self.control_loop)
        
        self.node_start_time = None
        self._log_controller_info()

    def _declare_parameters(self):
        self.declare_parameter('control_frequency', 1000.0)
        self.declare_parameter('trajectory_states_file', 'rdip_trajectory_states.csv')
        self.declare_parameter('trajectory_control_file', 'rdip_trajectory_control.csv')
        self.declare_parameter('lqr_gains_file', 'rdip_lqr_gains_timevarying.csv')
        self.declare_parameter('max_torque', 50.0)
        self.declare_parameter('enable_saturation', True)
        self.declare_parameter('joint_names',
            ['revolute_joint', 'first_pendulum_joint', 'second_pendulum_joint'],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY))
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('effort_command_topic', '/effort_controller/commands')
        self.declare_parameter('publish_debug_info', True)
        self.declare_parameter('debug_topic', '/trajectory_tracking_debug')
        
        self.declare_parameter('settling_position_threshold', 0.3)
        self.declare_parameter('settling_velocity_threshold', 0.2)
        self.declare_parameter('settling_duration', 0.3)
        self.declare_parameter('max_settling_time', 5.0)
        self.declare_parameter('startup_delay', 2.0)
        
        self.declare_parameter('use_trajectory_gains_for_balance', True)
        self.declare_parameter('balance_K_theta', 5.0)
        self.declare_parameter('balance_K_alpha', 100.0)     # Increased
        self.declare_parameter('balance_K_beta', -200.0)     # Increased (more negative)
        self.declare_parameter('balance_K_theta_dot', 3.0)
        self.declare_parameter('balance_K_alpha_dot', 5.0)   # Increased
        self.declare_parameter('balance_K_beta_dot', 5.0)    # Increased
        self.declare_parameter('balance_K_integral', 10.0)   # Increased
        
        # NEW: Stronger multipliers for balance gains from trajectory
        self.declare_parameter('balance_gain_multiplier_alpha', 2.0)  # Make α gains 2× stronger
        self.declare_parameter('balance_gain_multiplier_beta', 3.0)   # Make β gains 3× stronger!
        
        self.declare_parameter('fall_threshold', 0.5)
        self.declare_parameter('fall_grace_period', 1.0)
        self.declare_parameter('auto_restart', True)
        self.declare_parameter('restart_delay', 3.0)
        
        self.declare_parameter('enable_early_balance', True)
        self.declare_parameter('early_balance_alpha_threshold', 0.2)
        self.declare_parameter('early_balance_beta_threshold', 0.2)
        self.declare_parameter('early_balance_velocity_threshold', 2.0)
    
    def _get_parameters(self):
        self.control_frequency = self.get_parameter('control_frequency').value
        self.traj_states_file = self.get_parameter('trajectory_states_file').value
        self.traj_control_file = self.get_parameter('trajectory_control_file').value
        self.lqr_gains_file = self.get_parameter('lqr_gains_file').value
        self.max_torque = self.get_parameter('max_torque').value
        self.enable_saturation = self.get_parameter('enable_saturation').value
        self.joint_names = self.get_parameter('joint_names').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.effort_command_topic = self.get_parameter('effort_command_topic').value
        self.publish_debug_info = self.get_parameter('publish_debug_info').value
        self.debug_topic = self.get_parameter('debug_topic').value
        
        self.settle_pos_thresh = self.get_parameter('settling_position_threshold').value
        self.settle_vel_thresh = self.get_parameter('settling_velocity_threshold').value
        self.settle_duration = self.get_parameter('settling_duration').value
        self.max_settle_time = self.get_parameter('max_settling_time').value
        self.startup_delay = self.get_parameter('startup_delay').value
        
        self.use_trajectory_gains_for_balance = self.get_parameter('use_trajectory_gains_for_balance').value
        self.bal_K_theta = self.get_parameter('balance_K_theta').value
        self.bal_K_alpha = self.get_parameter('balance_K_alpha').value
        self.bal_K_beta = self.get_parameter('balance_K_beta').value
        self.bal_K_theta_dot = self.get_parameter('balance_K_theta_dot').value
        self.bal_K_alpha_dot = self.get_parameter('balance_K_alpha_dot').value
        self.bal_K_beta_dot = self.get_parameter('balance_K_beta_dot').value
        self.bal_K_integral = self.get_parameter('balance_K_integral').value
        
        self.balance_mult_alpha = self.get_parameter('balance_gain_multiplier_alpha').value
        self.balance_mult_beta = self.get_parameter('balance_gain_multiplier_beta').value
        
        self.fall_threshold = self.get_parameter('fall_threshold').value
        self.fall_grace_period = self.get_parameter('fall_grace_period').value
        self.auto_restart = self.get_parameter('auto_restart').value
        self.restart_delay = self.get_parameter('restart_delay').value
        
        self.enable_early_balance = self.get_parameter('enable_early_balance').value
        self.early_balance_alpha_thresh = self.get_parameter('early_balance_alpha_threshold').value
        self.early_balance_beta_thresh = self.get_parameter('early_balance_beta_threshold').value
        self.early_balance_vel_thresh = self.get_parameter('early_balance_velocity_threshold').value
    
    def _load_trajectory(self):
        try:
            pkg_share = get_package_share_directory('double_inverted_pendulum')
            traj_dir = os.path.join(pkg_share, 'trajectories')
            states_path = os.path.join(traj_dir, self.traj_states_file)
            control_path = os.path.join(traj_dir, self.traj_control_file)
            
            if not os.path.exists(states_path):
                states_path = self.traj_states_file
                control_path = self.traj_control_file
            
            states_df = pd.read_csv(states_path)
            control_df = pd.read_csv(control_path)
            
            self.traj_time = states_df['time'].values
            self.traj_states = states_df[['theta', 'alpha', 'beta', 
                                          'theta_dot', 'alpha_dot', 
                                          'beta_dot', 'int_theta']].values
            self.traj_control = control_df['u'].values
            self.T_total = self.traj_time[-1]
            
            self.alpha_start = self.traj_states[0, 1]
            self.beta_start = self.traj_states[0, 2]
            self.alpha_end = self.traj_states[-1, 1]
            self.beta_end = self.traj_states[-1, 2]
            
            self.get_logger().info(f'Trajectory: {self.T_total:.2f}s')
            self.get_logger().info(f'  Start: α={self.alpha_start:.3f}, β={self.beta_start:.3f}')
            self.get_logger().info(f'  End:   α={self.alpha_end:.3f}, β={self.beta_end:.3f}')
        except Exception as e:
            self.get_logger().error(f'Failed to load trajectory: {e}')
            raise
    
    def _load_lqr_gains(self):
        try:
            pkg_share = get_package_share_directory('double_inverted_pendulum')
            gains_path = os.path.join(pkg_share, 'trajectories', self.lqr_gains_file)
            if not os.path.exists(gains_path):
                gains_path = self.lqr_gains_file
            if os.path.exists(gains_path):
                gains_df = pd.read_csv(gains_path)
                self.lqr_gains = gains_df[['K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7']].values
                self.use_time_varying_lqr = True
                self.get_logger().info('Loaded time-varying LQR gains')
            else:
                self.use_time_varying_lqr = False
        except Exception as e:
            self.use_time_varying_lqr = False
            self.get_logger().warn(f'Could not load LQR gains: {e}')
    
    def _compute_balance_gains(self):
        """
        CRITICAL FIX: Strengthen balance gains, especially for β!
        """
        if self.use_trajectory_gains_for_balance and self.use_time_varying_lqr:
            # Start with trajectory end gains
            K_base = self.lqr_gains[-1, :].copy()
            
            # CRITICAL: Multiply α and β gains to make balance stronger
            K_base[1] *= self.balance_mult_alpha  # K_alpha (2×)
            K_base[2] *= self.balance_mult_beta   # K_beta (3×) - MUCH stronger!
            K_base[4] *= self.balance_mult_alpha  # K_alpha_dot (2×)
            K_base[5] *= self.balance_mult_beta   # K_beta_dot (3×)
            
            self.K_balance = K_base
            
            self.get_logger().info('Balance gains (trajectory × multipliers):')
        else:
            self.K_balance = np.array([
                self.bal_K_theta, 
                self.bal_K_alpha, 
                self.bal_K_beta,
                self.bal_K_theta_dot, 
                self.bal_K_alpha_dot, 
                self.bal_K_beta_dot,
                self.bal_K_integral
            ])
            self.get_logger().info('Balance gains (manual):')
        
        self.get_logger().info(f'  K = [{self.K_balance[0]:.1f}, {self.K_balance[1]:.1f}, '
                              f'{self.K_balance[2]:.1f}, {self.K_balance[3]:.1f}, '
                              f'{self.K_balance[4]:.1f}, {self.K_balance[5]:.1f}, '
                              f'{self.K_balance[6]:.1f}]')
        self.get_logger().info(f'  |K_beta| = {abs(self.K_balance[2]):.1f} (should be >2000!)')
    
    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop requests"""
        if msg.data and not self.emergency_stop_requested:
            self.emergency_stop_requested = True
            self.get_logger().warn('⚠ EMERGENCY STOP REQUESTED!')
    
    def joint_state_callback(self, msg: JointState):
        try:
            idx_arm = msg.name.index(self.joint_names[0])
            idx_pend1 = msg.name.index(self.joint_names[1])
            idx_pend2 = msg.name.index(self.joint_names[2])
            
            self.current_state = np.array([
                msg.position[idx_arm],
                msg.position[idx_pend1],
                msg.position[idx_pend2],
                msg.velocity[idx_arm] if len(msg.velocity) > idx_arm else 0.0,
                msg.velocity[idx_pend1] if len(msg.velocity) > idx_pend1 else 0.0,
                msg.velocity[idx_pend2] if len(msg.velocity) > idx_pend2 else 0.0
            ])
            
            self.state_received = True
        except (ValueError, IndexError):
            pass

    def wrap_to_pi(self, angle):
        """Wrap angle to [-π, π]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def angle_diff(self, target, current):
        """
        Compute shortest angular distance.
        CRITICAL for both α AND β!
        """
        diff = target - current
        return self.wrap_to_pi(diff)
    
    def is_at_hanging(self):
        """Check if at hanging position"""
        alpha = self.wrap_to_pi(self.current_state[1])
        beta = self.wrap_to_pi(self.current_state[2])
        alpha_dot = self.current_state[4]
        beta_dot = self.current_state[5]
        
        alpha_err = abs(self.angle_diff(self.alpha_start, alpha))
        beta_err = abs(self.angle_diff(self.beta_start, beta))
        
        pos_ok = (alpha_err < self.settle_pos_thresh) and (beta_err < self.settle_pos_thresh)
        vel_ok = (abs(alpha_dot) < self.settle_vel_thresh) and (abs(beta_dot) < self.settle_vel_thresh)
        return pos_ok and vel_ok
    
    def is_at_upup(self):
        """Check if at inverted position"""
        alpha = self.wrap_to_pi(self.current_state[1])
        beta = self.wrap_to_pi(self.current_state[2])
        
        alpha_err = abs(self.angle_diff(self.alpha_end, alpha))
        beta_err = abs(self.angle_diff(self.beta_end, beta))
        return (alpha_err < self.fall_threshold) and (beta_err < self.fall_threshold)
    
    def has_fallen(self):
        """Check if fallen (stricter for β!)"""
        if self.mode != self.MODE_BALANCE:
            return False
        if self.balance_start_time is None:
            return False
        now = self.get_clock().now().nanoseconds / 1e9
        if (now - self.balance_start_time) < self.fall_grace_period:
            return False
        
        alpha = self.wrap_to_pi(self.current_state[1])
        beta = self.wrap_to_pi(self.current_state[2])
        
        alpha_err = abs(self.angle_diff(self.alpha_end, alpha))
        beta_err = abs(self.angle_diff(self.beta_end, beta))
        
        # CRITICAL: β must stay very close to target!
        return (alpha_err > self.fall_threshold) or (beta_err > self.fall_threshold * 0.8)
    
    def is_near_goal(self):
        """Check if near goal for early balance transition"""
        alpha = self.wrap_to_pi(self.current_state[1])
        beta = self.wrap_to_pi(self.current_state[2])
        alpha_dot = self.current_state[4]
        beta_dot = self.current_state[5]
        
        alpha_err = abs(self.angle_diff(self.alpha_end, alpha))
        beta_err = abs(self.angle_diff(self.beta_end, beta))
        
        pos_ok = (alpha_err < self.early_balance_alpha_thresh) and \
                 (beta_err < self.early_balance_beta_thresh)
        vel_ok = (abs(alpha_dot) < self.early_balance_vel_thresh) and \
                 (abs(beta_dot) < self.early_balance_vel_thresh)
        return pos_ok and vel_ok

    def get_trajectory_reference(self, t):
        if t >= self.T_total: 
            return self.traj_states[-1, :].copy(), 0.0
        elif t <= 0.0: 
            return self.traj_states[0, :].copy(), self.traj_control[0]
        
        idx = np.searchsorted(self.traj_time, t)
        if idx == 0: 
            return self.traj_states[0, :].copy(), self.traj_control[0]
        if idx >= len(self.traj_states): 
            idx = len(self.traj_states) - 1
        
        t0, t1 = self.traj_time[idx-1], self.traj_time[idx]
        ratio = (t - t0) / (t1 - t0)
        
        x_ref = (1 - ratio) * self.traj_states[idx-1, :] + ratio * self.traj_states[idx, :]
        
        idx_ctrl = min(idx, len(self.traj_control) - 1)
        idx_ctrl_prev = min(idx - 1, len(self.traj_control) - 1)
        u_ref = (1 - ratio) * self.traj_control[idx_ctrl_prev] + ratio * self.traj_control[idx_ctrl]
        
        return x_ref, u_ref
    
    def get_trajectory_lqr_gain(self, t):
        if not self.use_time_varying_lqr:
            return self.K_balance
        idx = np.searchsorted(self.traj_time, t)
        if idx == 0: 
            return self.lqr_gains[0, :]
        if idx >= len(self.lqr_gains): 
            return self.lqr_gains[-1, :]
        t0, t1 = self.traj_time[idx-1], self.traj_time[idx]
        ratio = (t - t0) / (t1 - t0)
        return (1 - ratio) * self.lqr_gains[idx-1, :] + ratio * self.lqr_gains[idx, :]

    def compute_swingup_control(self):
        """Swing-up control with proper angle wrapping"""
        x_ref, u_ff = self.get_trajectory_reference(self.trajectory_time)
        K = self.get_trajectory_lqr_gain(self.trajectory_time)
        
        theta = self.current_state[0]
        alpha = self.current_state[1]
        beta = self.current_state[2]
        theta_dot = self.current_state[3]
        alpha_dot = self.current_state[4]
        beta_dot = self.current_state[5]
        
        x_error = np.zeros(7)
        
        x_error[0] = x_ref[0] - theta
        
        # CRITICAL: Wrapped angle differences
        x_error[1] = self.angle_diff(x_ref[1], alpha)
        x_error[2] = self.angle_diff(x_ref[2], beta)
        
        x_error[3] = x_ref[3] - theta_dot
        x_error[4] = x_ref[4] - alpha_dot
        x_error[5] = x_ref[5] - beta_dot
        
        self.theta_integral += x_error[0] * self.control_period
        self.theta_integral = np.clip(self.theta_integral, -5.0, 5.0)
        x_error[6] = self.theta_integral
        
        u_fb = np.dot(K, x_error)
        tau = u_ff + u_fb
        
        return tau, x_ref, x_error
    
    def compute_balance_control(self):
        """Balance control with STRONG β feedback"""
        theta_ref = 0.0
        alpha_ref = self.alpha_end
        beta_ref = self.beta_end
        
        theta = self.current_state[0]
        alpha = self.current_state[1]
        beta = self.current_state[2]
        theta_dot = self.current_state[3]
        alpha_dot = self.current_state[4]
        beta_dot = self.current_state[5]
        
        # CRITICAL: Wrapped errors for angles
        theta_error = theta_ref - theta
        alpha_error = self.angle_diff(alpha_ref, alpha)
        beta_error = self.angle_diff(beta_ref, beta)
        
        self.theta_integral += theta_error * self.control_period
        self.theta_integral = np.clip(self.theta_integral, -5.0, 5.0)
        
        x_error = np.array([
            theta_error, alpha_error, beta_error,
            -theta_dot, -alpha_dot, -beta_dot,
            self.theta_integral
        ])
        
        tau = np.dot(self.K_balance, x_error)
        x_ref = np.array([theta_ref, alpha_ref, beta_ref, 0, 0, 0, 0])
        
        return tau, x_ref, x_error

    def reset_controller(self):
        """Reset to initial state"""
        self.mode = self.MODE_SETTLING
        self.settle_start_time = None
        self.settle_check_start = None
        self.theta_integral = 0.0
        self.balance_start_time = None
        self.node_start_time = None
        self.last_tau = 0.0
        self.emergency_stop_requested = False
        self.get_logger().info('🔄 Controller reset')

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        
        if self.node_start_time is None:
            self.node_start_time = now
        
        tau = 0.0
        x_ref = self.last_x_ref.copy()
        x_error = self.last_x_error.copy()
        
        # EMERGENCY STOP: Immediately zero torque and stop
        if self.emergency_stop_requested:
            if self.mode != self.MODE_EMERGENCY_STOP:
                self.mode = self.MODE_EMERGENCY_STOP
                self.get_logger().error('🛑 EMERGENCY STOP ACTIVATED!')
            tau = 0.0
            self._publish_all(tau, x_ref, x_error)
            return
        
        if not self.state_received:
            pass
        
        elif self.mode == self.MODE_SETTLING:
            x_ref = np.array([0, self.alpha_start, self.beta_start, 0, 0, 0, 0])
            x_error = np.zeros(7)
            
            if (now - self.node_start_time) >= self.startup_delay:
                if self.settle_start_time is None:
                    self.settle_start_time = now
                    alpha_w = self.wrap_to_pi(self.current_state[1])
                    beta_w = self.wrap_to_pi(self.current_state[2])
                    self.get_logger().info(f'Settling... α={alpha_w:.3f}, β={beta_w:.3f}')
                
                if self.is_at_hanging():
                    if self.settle_check_start is None:
                        self.settle_check_start = now
                    elif (now - self.settle_check_start) >= self.settle_duration:
                        self.mode = self.MODE_SWINGUP
                        self.trajectory_start_time = now
                        self.trajectory_time = 0.0
                        self.theta_integral = 0.0
                        self.get_logger().info('🚀 SWING-UP STARTED!')
                else:
                    self.settle_check_start = None
                
                if self.settle_start_time and (now - self.settle_start_time) > self.max_settle_time:
                    self.get_logger().warn('⏱ Settling timeout - starting anyway')
                    self.mode = self.MODE_SWINGUP
                    self.trajectory_start_time = now
                    self.trajectory_time = 0.0
        
        elif self.mode == self.MODE_SWINGUP:
            self.trajectory_time = now - self.trajectory_start_time
            
            should_balance = False
            if self.enable_early_balance and self.is_near_goal():
                should_balance = True
                self.get_logger().info(f'⚖️ BALANCE MODE (early) at t={self.trajectory_time:.2f}s')
            elif self.trajectory_time >= self.T_total:
                should_balance = True
                self.get_logger().info('⚖️ BALANCE MODE (trajectory complete)')
            
            if should_balance:
                self.mode = self.MODE_BALANCE
                self.balance_start_time = now
                self.theta_integral = 0.0
                tau, x_ref, x_error = self.compute_balance_control()
            else:
                tau, x_ref, x_error = self.compute_swingup_control()
                
                if int(self.trajectory_time * 2) % 1 == 0 and \
                   self.trajectory_time - int(self.trajectory_time) < 0.02:
                    alpha = self.current_state[1]
                    beta = self.current_state[2]
                    self.get_logger().info(
                        f't={self.trajectory_time:.2f}s: τ={tau:.2f}, '
                        f'α={alpha:.2f}(err={x_error[1]:.3f}), '
                        f'β={beta:.2f}(err={x_error[2]:.3f})'
                    )
            
            if self.enable_saturation:
                tau = np.clip(tau, -self.max_torque, self.max_torque)
        
        elif self.mode == self.MODE_BALANCE:
            if self.has_fallen():
                self.get_logger().warn('💥 FALLEN!')
                self.mode = self.MODE_FALLEN
                self.fall_time = now
                tau = 0.0
            else:
                tau, x_ref, x_error = self.compute_balance_control()
                if self.enable_saturation:
                    tau = np.clip(tau, -self.max_torque, self.max_torque)
                
                if self.balance_start_time:
                    bal_time = now - self.balance_start_time
                    if int(bal_time) % 2 == 0 and bal_time - int(bal_time) < 0.01:
                        beta = self.current_state[2]
                        self.get_logger().info(
                            f'⚖️ Balance {bal_time:.0f}s: τ={tau:.2f}, β_err={x_error[2]:.3f}'
                        )
        
        elif self.mode == self.MODE_FALLEN:
            tau = 0.0
            if self.auto_restart and self.fall_time and (now - self.fall_time) > self.restart_delay:
                self.get_logger().info('🔄 Auto-restart...')
                self.reset_controller()
        
        self.last_tau = tau
        self.last_x_ref = x_ref
        self.last_x_error = x_error
        
        self._publish_all(tau, x_ref, x_error)
    
    def _publish_all(self, tau, x_ref, x_error):
        effort_msg = Float64MultiArray()
        effort_msg.data = [float(tau), 0.0, 0.0]
        self.effort_pub.publish(effort_msg)
        
        mode_msg = String()
        mode_names = ['SETTLING', 'SWINGUP', 'BALANCE', 'FALLEN', 'EMERGENCY_STOP']
        if self.mode == self.MODE_SWINGUP:
            mode_msg.data = f'SWINGUP_{self.trajectory_time:.2f}s'
        elif self.mode == self.MODE_BALANCE and self.balance_start_time:
            now = self.get_clock().now().nanoseconds / 1e9
            mode_msg.data = f'BALANCE_{now - self.balance_start_time:.1f}s'
        else:
            mode_msg.data = mode_names[self.mode]
        self.mode_pub.publish(mode_msg)
        
        self.ref_theta_pub.publish(Float64(data=float(x_ref[0])))
        self.ref_alpha_pub.publish(Float64(data=float(x_ref[1])))
        self.ref_beta_pub.publish(Float64(data=float(x_ref[2])))
        
        error_msg = PointStamped()
        error_msg.header.stamp = self.get_clock().now().to_msg()
        error_msg.header.frame_id = 'error'
        error_msg.point.x = float(x_error[0])
        error_msg.point.y = float(x_error[1])
        error_msg.point.z = float(x_error[2])
        self.error_pub.publish(error_msg)
        
        state_msg = PointStamped()
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.header.frame_id = 'wrapped'
        state_msg.point.x = float(self.wrap_to_pi(self.current_state[0]))
        state_msg.point.y = float(self.wrap_to_pi(self.current_state[1]))
        state_msg.point.z = float(self.wrap_to_pi(self.current_state[2]))
        self.wrapped_state_pub.publish(state_msg)
        
        if self.publish_debug_info:
            debug_msg = Vector3Stamped()
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.vector.x = float(tau)
            debug_msg.vector.y = float(x_error[1])
            debug_msg.vector.z = float(x_error[2])
            self.debug_pub.publish(debug_msg)

    def _log_controller_info(self):
        self.get_logger().info('='*60)
        self.get_logger().info('🎯 RDIP CONTROLLER - ULTIMATE FIX')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Trajectory: {self.T_total:.2f}s')
        self.get_logger().info(f'Max torque: {self.max_torque:.1f} Nm')
        self.get_logger().info('Features:')
        self.get_logger().info('  ✅ Proper angle wrapping for α AND β')
        self.get_logger().info('  ✅ Strengthened β balance control')
        self.get_logger().info('  ✅ Emergency stop via /emergency_stop topic')
        self.get_logger().info('='*60)


def main(args=None):
    rclpy.init(args=args)
    node = SwingUpBalanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()