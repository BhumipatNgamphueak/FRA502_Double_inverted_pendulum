#!/usr/bin/env python3
"""
IMPROVED Swing-Up and LQR Stabilization Controller
FIXES: Zero velocity startup, better energy pumping

KEY IMPROVEMENTS:
1. Initial kick to start motion from rest
2. Modified control law that works at zero velocity
3. Better energy calculation using actual URDF parameters
4. Automatic mode switching with hysteresis
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Vector3Stamped


class ImprovedSwingUpLQR(Node):
    """
    Improved two-phase controller with robust swing-up
    """
    
    def __init__(self):
        super().__init__('improved_swingup_lqr')
        
        # Declare and get parameters
        self._declare_parameters()
        self._get_parameters()
        
        # Initialize state
        self.current_state = np.zeros(6)  # [θ, α, β, θ̇, α̇, β̇]
        self.state_received = False
        
        # Controller state
        self.controller_mode = 'SWING_UP'
        self.stabilization_counter = 0
        self.stabilization_samples = int(self.stabilization_time * self.control_frequency)
        
        # Initial kick state
        self.kick_counter = 0
        self.kick_duration = int(0.2 * self.control_frequency)  # 0.2 second kick
        self.kick_applied = False
        
        # LQR Gain Matrix
        self.K = np.array([
            self.K_theta, self.K_alpha, self.K_beta,
            self.K_theta_dot, self.K_alpha_dot, self.K_beta_dot
        ])
        
        self._log_controller_config()
        
        # Create subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, self.joint_states_topic, self.joint_state_callback, 10
        )
        
        # Create publishers
        self.effort_pub = self.create_publisher(
            Float64MultiArray, self.effort_command_topic, 10
        )
        self.mode_pub = self.create_publisher(String, '/controller_mode', 10)
        
        if self.publish_debug_info:
            self.debug_pub = self.create_publisher(
                Vector3Stamped, self.debug_topic, 10
            )
        
        # Create timer
        self.control_period = 1.0 / self.control_frequency
        self.control_timer = self.create_timer(self.control_period, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('IMPROVED SWING-UP + LQR CONTROLLER READY')
        self.get_logger().info('='*70)
        self.get_logger().info('Starting with INITIAL KICK to overcome friction...')
        self.get_logger().info('='*70)
    
    def _declare_parameters(self):
        """Declare all parameters"""
        # Control
        self.declare_parameter('control_frequency', 100.0)
        
        # System parameters (FROM YOUR ACTUAL URDF!)
        self.declare_parameter('M1', 0.0997302210483473)  # URDF: first_pendulum mass
        self.declare_parameter('M2', 0.00680601326910171)  # URDF: second_pendulum mass
        self.declare_parameter('l1', 0.0646186151454772)  # URDF: first_pendulum CoM
        self.declare_parameter('l2', 0.0299439674255459)  # URDF: second_pendulum CoM
        self.declare_parameter('L1', 0.1432)  # URDF: first_pendulum length
        self.declare_parameter('I1_xx', 0.000174588068807012)  # URDF
        self.declare_parameter('I2_xx', 1.8911424822564e-06)  # URDF
        self.declare_parameter('g', 9.81)
        
        # LQR Gains
        self.declare_parameter('K_theta', 3.162278)
        self.declare_parameter('K_alpha', -2075.754)
        self.declare_parameter('K_beta', -1406.793)
        self.declare_parameter('K_theta_dot', 4.751476)
        self.declare_parameter('K_alpha_dot', -171.095)
        self.declare_parameter('K_beta_dot', -136.876)
        
        # Swing-up parameters (INCREASED for better performance)
        self.declare_parameter('energy_gain', 50.0)  # Increased from 15
        self.declare_parameter('damping_gain', 1.5)  # Increased damping
        self.declare_parameter('initial_kick_torque', 3.0)  # Initial kick
        self.declare_parameter('swing_up_max_torque', 10.0)  # Increased max
        
        # Switching thresholds
        self.declare_parameter('switch_to_lqr_angle', 0.3)
        self.declare_parameter('switch_to_lqr_velocity', 2.0)
        self.declare_parameter('stabilization_time', 0.5)
        self.declare_parameter('fall_angle_threshold', 0.6)
        
        # Control limits
        self.declare_parameter('lqr_max_torque', 5.0)
        self.declare_parameter('enable_saturation', True)
        
        # Topics
        self.declare_parameter('joint_names', [
            'revolute_joint', 'first_pendulum_joint', 'second_pendulum_joint'
        ])
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('effort_command_topic', '/effort_controller/commands')
        self.declare_parameter('publish_debug_info', True)
        self.declare_parameter('debug_topic', '/swingup_lqr_debug')
    
    def _get_parameters(self):
        """Get all parameters"""
        self.control_frequency = self.get_parameter('control_frequency').value
        
        # System
        self.M1 = self.get_parameter('M1').value
        self.M2 = self.get_parameter('M2').value
        self.l1 = self.get_parameter('l1').value
        self.l2 = self.get_parameter('l2').value
        self.L1 = self.get_parameter('L1').value
        self.I1_xx = self.get_parameter('I1_xx').value
        self.I2_xx = self.get_parameter('I2_xx').value
        self.g = self.get_parameter('g').value
        
        # LQR
        self.K_theta = self.get_parameter('K_theta').value
        self.K_alpha = self.get_parameter('K_alpha').value
        self.K_beta = self.get_parameter('K_beta').value
        self.K_theta_dot = self.get_parameter('K_theta_dot').value
        self.K_alpha_dot = self.get_parameter('K_alpha_dot').value
        self.K_beta_dot = self.get_parameter('K_beta_dot').value
        
        # Swing-up
        self.energy_gain = self.get_parameter('energy_gain').value
        self.damping_gain = self.get_parameter('damping_gain').value
        self.initial_kick_torque = self.get_parameter('initial_kick_torque').value
        self.swing_up_max_torque = self.get_parameter('swing_up_max_torque').value
        
        # Switching
        self.switch_angle = self.get_parameter('switch_to_lqr_angle').value
        self.switch_velocity = self.get_parameter('switch_to_lqr_velocity').value
        self.stabilization_time = self.get_parameter('stabilization_time').value
        self.fall_angle_thresh = self.get_parameter('fall_angle_threshold').value
        
        # Limits
        self.lqr_max_torque = self.get_parameter('lqr_max_torque').value
        self.enable_saturation = self.get_parameter('enable_saturation').value
        
        # Topics
        self.joint_names = self.get_parameter('joint_names').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.effort_command_topic = self.get_parameter('effort_command_topic').value
        self.publish_debug_info = self.get_parameter('publish_debug_info').value
        self.debug_topic = self.get_parameter('debug_topic').value
    
    def _log_controller_config(self):
        """Log configuration"""
        self.get_logger().info('')
        self.get_logger().info('='*70)
        self.get_logger().info('CONTROLLER CONFIGURATION')
        self.get_logger().info('='*70)
        self.get_logger().info(f'SWING-UP: energy_gain={self.energy_gain:.1f}, ' +
                             f'damping_gain={self.damping_gain:.1f}')
        self.get_logger().info(f'STABILIZATION: LQR with max_torque={self.lqr_max_torque:.1f} N·m')
        self.get_logger().info('='*70)
    
    def joint_state_callback(self, msg):
        """Process joint states"""
        try:
            state = np.zeros(6)
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    state[i] = msg.position[idx]
                    state[i+3] = msg.velocity[idx]
            
            self.current_state = state
            self.state_received = True
        except Exception as e:
            self.get_logger().error(f'joint_state_callback error: {e}')
    
    # ========================================================================
    # ENERGY CALCULATIONS
    # ========================================================================
    
    def _compute_energy(self, state):
        """Compute total system energy"""
        θ, α, β, θ_dot, α_dot, β_dot = state
        
        # Kinetic energy
        KE_1 = 0.5 * self.I1_xx * α_dot**2
        KE_2 = 0.5 * self.I2_xx * β_dot**2
        v1 = self.L1 * θ_dot
        KE_trans = 0.5 * (self.M1 + self.M2) * v1**2
        KE = KE_1 + KE_2 + KE_trans
        
        # Potential energy (reference: hanging down = 0)
        h1 = self.l1 * (1 + np.cos(α))
        h2 = self.L1 * (1 + np.cos(α)) + self.l2 * (1 + np.cos(α + β))
        PE = self.M1 * self.g * h1 + self.M2 * self.g * h2
        
        return KE + PE
    
    def _compute_desired_energy(self):
        """Energy at upright position"""
        h1_up = 2 * self.l1
        h2_up = 2 * self.L1 + 2 * self.l2
        return self.M1 * self.g * h1_up + self.M2 * self.g * h2_up
    
    # ========================================================================
    # IMPROVED SWING-UP CONTROL
    # ========================================================================
    
    def swing_up_control(self, state):
        """
        IMPROVED swing-up control with:
        1. Initial kick for zero-velocity startup
        2. Modified control law
        3. Better energy pumping
        """
        θ, α, β, θ_dot, α_dot, β_dot = state
        
        # PHASE 1: Initial kick to overcome static friction
        if not self.kick_applied:
            if self.kick_counter < self.kick_duration:
                self.kick_counter += 1
                return self.initial_kick_torque  # Constant kick
            else:
                self.kick_applied = True
                self.get_logger().info('Initial kick complete. Starting energy pumping...')
        
        # PHASE 2: Energy-based control
        E_current = self._compute_energy(state)
        E_desired = self._compute_desired_energy()
        ΔE = E_desired - E_current
        
        # IMPROVED control law (works even at zero velocity!)
        # Use cos(α) directly as control direction when velocity is low
        if abs(θ_dot) < 0.1:
            # Low velocity: use potential energy gradient
            control_signal = -np.sin(α) * np.cos(α)  # Pumps energy efficiently
        else:
            # Normal velocity: standard energy pumping
            control_signal = np.sign(θ_dot * np.cos(α))
        
        u_energy = self.energy_gain * ΔE * control_signal
        
        # Damping near upright
        dist_from_up = (abs(np.arctan2(np.sin(α - np.pi), np.cos(α - np.pi))) +
                       abs(np.arctan2(np.sin(β - np.pi), np.cos(β - np.pi))))
        
        proximity_weight = np.exp(-5 * dist_from_up)
        u_damping = -self.damping_gain * proximity_weight * θ_dot
        
        u = u_energy + u_damping
        
        return u
    
    # ========================================================================
    # LQR CONTROL
    # ========================================================================
    
    def lqr_control(self, state):
        """LQR stabilization around upright"""
        θ, α, β, θ_dot, α_dot, β_dot = state
        
        # Error from upright
        α_error = np.arctan2(np.sin(α - np.pi), np.cos(α - np.pi))
        β_error = np.arctan2(np.sin(β - np.pi), np.cos(β - np.pi))
        
        x_error = np.array([θ, α_error, β_error, θ_dot, α_dot, β_dot])
        u = -self.K @ x_error
        
        return float(u)
    
    # ========================================================================
    # MODE SWITCHING
    # ========================================================================
    
    def check_switch_to_lqr(self, state):
        """Check if ready for LQR"""
        θ, α, β, θ_dot, α_dot, β_dot = state
        
        α_dist = abs(np.arctan2(np.sin(α - np.pi), np.cos(α - np.pi)))
        β_dist = abs(np.arctan2(np.sin(β - np.pi), np.cos(β - np.pi)))
        
        angle_ok = (α_dist < self.switch_angle and β_dist < self.switch_angle)
        velocity_ok = (abs(α_dot) < self.switch_velocity and abs(β_dot) < self.switch_velocity)
        
        return angle_ok and velocity_ok
    
    def check_switch_to_swingup(self, state):
        """Check if fallen"""
        θ, α, β, θ_dot, α_dot, β_dot = state
        
        α_dist = abs(np.arctan2(np.sin(α - np.pi), np.cos(α - np.pi)))
        β_dist = abs(np.arctan2(np.sin(β - np.pi), np.cos(β - np.pi)))
        
        return (α_dist > self.fall_angle_thresh or β_dist > self.fall_angle_thresh)
    
    # ========================================================================
    # MAIN CONTROL LOOP
    # ========================================================================
    
    def control_loop(self):
        """Main control loop"""
        if not self.state_received:
            self.get_logger().warn('Waiting for joint states...', throttle_duration_sec=2.0)
            return
        
        try:
            x = self.current_state
            
            # Mode switching
            if self.controller_mode == 'SWING_UP':
                if self.check_switch_to_lqr(x):
                    self.stabilization_counter += 1
                    if self.stabilization_counter >= self.stabilization_samples:
                        self.controller_mode = 'STABILIZE'
                        self.stabilization_counter = 0
                        self.get_logger().info('='*70)
                        self.get_logger().info('✓ SWITCHING TO LQR STABILIZATION!')
                        self.get_logger().info('='*70)
                else:
                    self.stabilization_counter = 0
                
                tau = self.swing_up_control(x)
                max_torque = self.swing_up_max_torque
                
            else:  # STABILIZE
                if self.check_switch_to_swingup(x):
                    self.controller_mode = 'SWING_UP'
                    self.kick_applied = False  # Reset for re-kick
                    self.kick_counter = 0
                    self.get_logger().warn('='*70)
                    self.get_logger().warn('⚠ FALLEN! Returning to SWING-UP')
                    self.get_logger().warn('='*70)
                    tau = self.swing_up_control(x)
                    max_torque = self.swing_up_max_torque
                else:
                    tau = self.lqr_control(x)
                    max_torque = self.lqr_max_torque
            
            # Saturation
            if self.enable_saturation:
                tau = np.clip(tau, -max_torque, max_torque)
            
            # Publish
            self._publish_effort(tau)
            self._publish_mode()
            
            if self.publish_debug_info:
                self._publish_debug(x, tau)
                
        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')
    
    def _publish_effort(self, tau):
        """Publish effort command"""
        msg = Float64MultiArray()
        msg.data = [float(tau), 0.0, 0.0]
        self.effort_pub.publish(msg)
    
    def _publish_mode(self):
        """Publish mode"""
        msg = String()
        msg.data = self.controller_mode
        self.mode_pub.publish(msg)
    
    def _publish_debug(self, state, tau):
        """Publish debug"""
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'swingup_lqr_debug'
        msg.vector.x = float(tau)
        msg.vector.y = float(state[1])  # α
        msg.vector.z = float(state[2])  # β
        self.debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImprovedSwingUpLQR()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()