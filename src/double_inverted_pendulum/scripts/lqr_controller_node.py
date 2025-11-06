#!/usr/bin/env python3
"""
================================================================
TANGENTIAL Swing-Up and LQR Stabilization Controller
================================================================
Target Equilibrium: EP3 (Up-Up) - Maximum Potential Energy
  - alpha = pi (Pendulum 1 UP)
  - beta  = 0  (Pendulum 2 UP, aligned with P1)

Physics Model:
- TANGENTIAL swing (matches URDF and MATLAB EOMs)
- Convention: alpha=0 (DOWN), beta=0 (ALIGNED)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Vector3Stamped
# Import parameter types for explicit declaration
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


class TangentialSwingUpLQR_EP3(Node):
    """
    Controller for TANGENTIAL double pendulum.
    Targets EP3 (Up-Up) state: alpha=pi, beta=0.
    """
    
    def __init__(self):
        super().__init__('tangential_swingup_lqr_ep3')
        
        # Declare and get parameters
        self._declare_parameters()
        self._get_parameters()
        
        # Initialize state
        # [theta, alpha, beta, theta_dot, alpha_dot, beta_dot]
        self.current_state = np.zeros(6)  
        self.state_received = False
        
        # Controller state
        self.controller_mode = 'SWING_UP'
        self.stabilization_counter = 0
        self.stabilization_samples = int(self.stabilization_time * self.control_frequency)
        
        # Initial kick state
        self.kick_counter = 0
        self.kick_duration = int(0.2 * self.control_frequency)  # 0.2 second kick
        self.kick_applied = False
        
        # LQR Gain Matrix for EP3 (Up-Up)
        self.K_ep3 = np.array([
            self.K_theta, self.K_alpha, self.K_beta,
            self.K_theta_dot, self.K_alpha_dot, self.K_beta_dot
        ])
        
        # Desired energy at EP3 (alpha=pi, beta=0)
        self.E_desired_ep3 = self._compute_desired_energy()
        
        self._log_controller_config()
        
        # Subscribers and Publishers
        self.joint_state_sub = self.create_subscription(
            JointState, self.joint_states_topic, self.joint_state_callback, 10
        )
        self.effort_pub = self.create_publisher(
            Float64MultiArray, self.effort_command_topic, 10
        )
        self.mode_pub = self.create_publisher(String, '/controller_mode', 10)
        
        if self.publish_debug_info:
            self.debug_pub = self.create_publisher(
                Vector3Stamped, self.debug_topic, 10
            )
        
        # Control Timer
        self.control_period = 1.0 / self.control_frequency
        self.control_timer = self.create_timer(self.control_period, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('TANGENTIAL SWING-UP + LQR CONTROLLER (EP3 GOAL)')
        self.get_logger().info(f'TARGET STATE: alpha=pi ({np.pi:.2f}), beta=0.0')
        self.get_logger().info(f'Desired Energy (EP3): {self.E_desired_ep3:.3f} J')
        self.get_logger().info('='*70)

    
    def _declare_parameters(self):
        """Declare all parameters from the YAML file"""
        self.declare_parameter('control_frequency', 100.0)
        
        # System Parameters (from MATLAB EOM)
        self.declare_parameter('g', 9.81)
        self.declare_parameter('M_arm', 0.1)
        self.declare_parameter('l1', 0.1)
        self.declare_parameter('l2', 0.1)
        self.declare_parameter('l3', 0.1)
        self.declare_parameter('I_arm', 0.01)
        self.declare_parameter('M1', 0.1)
        self.declare_parameter('l4', 0.1)
        self.declare_parameter('l5', 0.1)
        self.declare_parameter('I1', 0.01)
        self.declare_parameter('M2', 0.1)
        self.declare_parameter('l6', 0.1)
        self.declare_parameter('I2', 0.01)
        
        # LQR Gains (for EP3)
        self.declare_parameter('K_theta', 0.0)
        self.declare_parameter('K_alpha', 0.0)
        self.declare_parameter('K_beta', 0.0)
        self.declare_parameter('K_theta_dot', 0.0)
        self.declare_parameter('K_alpha_dot', 0.0)
        self.declare_parameter('K_beta_dot', 0.0)
        
        # Swing-up
        self.declare_parameter('energy_gain', 10.0)
        self.declare_parameter('damping_gain', 1.0)
        self.declare_parameter('initial_kick_torque', 1.0)
        self.declare_parameter('swing_up_max_torque', 3.0)
        
        # Switching (for EP3)
        self.declare_parameter('switch_to_lqr_angle_alpha', 0.3)
        self.declare_parameter('switch_to_lqr_angle_beta', 0.3)
        self.declare_parameter('switch_to_lqr_velocity', 2.0)
        self.declare_parameter('stabilization_time', 0.5)
        self.declare_parameter('fall_angle_threshold_alpha', 0.6)
        self.declare_parameter('fall_angle_threshold_beta', 0.6)
        
        # Limits
        self.declare_parameter('lqr_max_torque', 5.0)
        self.declare_parameter('enable_saturation', True)
        
        # Topics
        # =================================================================
        # *** FIXED ***
        # Provide a non-empty list as the default value to correctly
        # infer the type as STRING_ARRAY, avoiding the BYTE_ARRAY error.
        # =================================================================
        self.declare_parameter(
            'joint_names', 
            ['revolute_joint', 'first_pendulum_joint', 'second_pendulum_joint'],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        )
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('effort_command_topic', '/effort_controller/commands')
        self.declare_parameter('publish_debug_info', True)
        self.declare_parameter('debug_topic', '/swingup_lqr_debug')
    
    def _get_parameters(self):
        """Get all parameters"""
        self.control_frequency = self.get_parameter('control_frequency').value
        
        # System
        self.g = self.get_parameter('g').value
        self.M_arm = self.get_parameter('M_arm').value
        self.l1 = self.get_parameter('l1').value
        self.l2 = self.get_parameter('l2').value
        self.l3 = self.get_parameter('l3').value
        self.I_arm = self.get_parameter('I_arm').value
        self.M1 = self.get_parameter('M1').value
        self.l4 = self.get_parameter('l4').value
        self.l5 = self.get_parameter('l5').value
        self.I1 = self.get_parameter('I1').value
        self.M2 = self.get_parameter('M2').value
        self.l6 = self.get_parameter('l6').value
        self.I2 = self.get_parameter('I2').value
        
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
        self.switch_angle_a = self.get_parameter('switch_to_lqr_angle_alpha').value
        self.switch_angle_b = self.get_parameter('switch_to_lqr_angle_beta').value
        self.switch_velocity = self.get_parameter('switch_to_lqr_velocity').value
        self.stabilization_time = self.get_parameter('stabilization_time').value
        self.fall_angle_a = self.get_parameter('fall_angle_threshold_alpha').value
        self.fall_angle_b = self.get_parameter('fall_angle_threshold_beta').value

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
        self.get_logger().info('='*70)
        self.get_logger().info('TANGENTIAL CONTROLLER CONFIGURATION (EP3 GOAL)')
        self.get_logger().info(f'  freq: {self.control_frequency} Hz')
        self.get_logger().info(f'  LQR K_alpha: {self.K_alpha:.2f}, K_beta: {self.K_beta:.2f}')
        self.get_logger().info(f'  Swing-Up: E_gain={self.energy_gain:.1f}, Kick={self.initial_kick_torque:.1f} Nm')
        self.get_logger().info(f'  Switch (alpha): < {self.switch_angle_a} rad')
        self.get_logger().info(f'  Switch (beta):  < {self.switch_angle_b} rad')
        self.get_logger().info('='*70)
    
    def joint_state_callback(self, msg):
        """Process joint states"""
        try:
            state = np.zeros(6)
            name_map = {name: i for i, name in enumerate(msg.name)}
            
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in name_map:
                    idx = name_map[joint_name]
                    state[i] = msg.position[idx]
                    state[i+3] = msg.velocity[idx]
            
            self.current_state = state
            self.state_received = True
        except Exception as e:
            self.get_logger().error(f'joint_state_callback error: {e}')
    
    # ========================================================================
    # TANGENTIAL ENERGY CALCULATIONS (from MATLAB EOM)
    # Convention: alpha=0 (DOWN), beta=0 (ALIGNED)
    # ========================================================================
    
    def _compute_potential_energy(self, state):
        """V_total = V_arm + V1 + V2"""
        _, alpha, beta, _, _, _ = state
        
        # V_arm = M_arm*g*l1
        V_arm = self.M_arm * self.g * self.l1
        
        # V1 = M1*g*z1 = M1*g*(l1 - l4*cos(alpha))
        V1 = self.M1 * self.g * (self.l1 - self.l4 * np.cos(alpha))
        
        # V2 = M2*g*z2 = M2*g*(l1 - l5*cos(alpha) - l6*cos(alpha + beta))
        V2 = self.M2 * self.g * (self.l1 - self.l5 * np.cos(alpha) - self.l6 * np.cos(alpha + beta))
        
        return V_arm + V1 + V2

    def _compute_kinetic_energy(self, state):
        """T_total = T_arm + T_pendulum1 + T_pendulum2"""
        theta, alpha, beta, theta_dot, alpha_dot, beta_dot = state
        
        # T_arm = (1/2)*M_arm*(l2*theta_dot)^2 + (1/2)*I_arm*theta_dot^2
        T_arm = 0.5 * self.M_arm * (self.l2 * theta_dot)**2 + \
                0.5 * self.I_arm * theta_dot**2
        
        # T_pendulum1 = T_trans_1 + T_rot_1
        # T_rot_1 = (1/2)*I1*(alpha_dot^2 + (theta_dot*sin(alpha))^2)
        T_rot_1 = 0.5 * self.I1 * (alpha_dot**2 + (theta_dot * np.sin(alpha))**2)
        
        # T_trans_1 = (1/2)*M1*(v_x1^2 + v_y1^2 + v_z1^2)
        # v_x1 = -l3*sin(theta)*theta_dot - l4*cos(alpha)*sin(theta)*alpha_dot - l4*sin(alpha)*cos(theta)*theta_dot
        # v_y1 =  l3*cos(theta)*theta_dot + l4*cos(alpha)*cos(theta)*alpha_dot - l4*sin(alpha)*sin(theta)*theta_dot
        # v_z1 =  l4*sin(alpha)*alpha_dot
        v_x1 = -self.l3*np.sin(theta)*theta_dot - self.l4*np.cos(alpha)*np.sin(theta)*alpha_dot - self.l4*np.sin(alpha)*np.cos(theta)*theta_dot
        v_y1 =  self.l3*np.cos(theta)*theta_dot + self.l4*np.cos(alpha)*np.cos(theta)*alpha_dot - self.l4*np.sin(alpha)*np.sin(theta)*theta_dot
        v_z1 =  self.l4*np.sin(alpha)*alpha_dot
        T_trans_1 = 0.5 * self.M1 * (v_x1**2 + v_y1**2 + v_z1**2)
        
        T_pendulum1 = T_trans_1 + T_rot_1
        
        # T_pendulum2 = T_trans_2 + T_rot_2
        # T_rot_2 = (1/2)*I2*((alpha_dot+beta_dot)^2 + (theta_dot*sin(alpha+beta))^2)
        T_rot_2 = 0.5 * self.I2 * ((alpha_dot + beta_dot)**2 + (theta_dot * np.sin(alpha + beta))**2)
        
        # T_trans_2 = (1/2)*M2*(v_x2^2 + v_y2^2 + v_z2^2)
        ad = alpha_dot
        bd = beta_dot
        td = theta_dot
        a = alpha
        b = beta
        th = theta
        s = np.sin
        c = np.cos
        
        v_x2 = -self.l3*s(th)*td - self.l5*c(a)*s(th)*ad - self.l5*s(a)*c(th)*td - self.l6*c(a+b)*s(th)*(ad+bd) - self.l6*s(a+b)*c(th)*td
        v_y2 =  self.l3*c(th)*td + self.l5*c(a)*c(th)*ad - self.l5*s(a)*s(th)*td + self.l6*c(a+b)*c(th)*(ad+bd) - self.l6*s(a+b)*s(th)*td
        v_z2 =  self.l5*s(a)*ad + self.l6*s(a+b)*(ad+bd)
        T_trans_2 = 0.5 * self.M2 * (v_x2**2 + v_y2**2 + v_z2**2)
        
        T_pendulum2 = T_trans_2 + T_rot_2
        
        return T_arm + T_pendulum1 + T_pendulum2

    def _compute_total_energy(self, state):
        return self._compute_kinetic_energy(state) + self._compute_potential_energy(state)

    def _compute_desired_energy(self):
        """Energy at EP3 (alpha=pi, beta=0)"""
        # State: [theta=0, alpha=pi, beta=0, theta_dot=0, alpha_dot=0, beta_dot=0]
        state_ep3 = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])
        # KE at rest is 0
        PE_ep3 = self._compute_potential_energy(state_ep3)
        return PE_ep3
    
    # ========================================================================
    # SWING-UP CONTROL (Targeting EP3)
    # ========================================================================
    
    def swing_up_control(self, state):
        """
        Energy-based swing-up controller.
        Pumps energy into the system to reach E_desired_ep3.
        """
        theta, alpha, beta, theta_dot, alpha_dot, beta_dot = state
        
        # PHASE 1: Initial kick to overcome static friction
        if not self.kick_applied:
            if self.kick_counter < self.kick_duration:
                self.kick_counter += 1
                return self.initial_kick_torque  # Constant kick
            else:
                self.kick_applied = True
                self.get_logger().info('Initial kick complete. Starting energy pumping...')
        
        # PHASE 2: Energy-based control
        E_current = self._compute_total_energy(state)
        E_desired = self.E_desired_ep3
        ΔE = E_desired - E_current
        
        # Use total angular velocity of pendulums for control signal
        # This is more robust than just theta_dot
        # We want to pump energy based on P1's motion
        
        # Heuristic: Use alpha_dot to determine push direction
        # This is a common and effective swing-up strategy
        # u = k * (E_d - E) * sign(alpha_dot * cos(alpha))
        # We will use theta_dot as it's the actuated variable
        
        # This is a robust energy pumping law:
        # push in the direction of velocity when pendulum is rising
        # push against velocity when pendulum is falling
        # A simpler version: u = k * delta_E * sign(theta_dot * cos(alpha))
        # Let's use the main pendulum's (alpha's) velocity and position
        
        control_signal = np.sign(alpha_dot * np.cos(alpha))
        
        u_energy = self.energy_gain * ΔE * control_signal
        
        # Damping near the arm's zero position (optional, but good)
        u_damping = -self.damping_gain * theta_dot
        
        u = u_energy + u_damping
        
        return u
    
    # ========================================================================
    # LQR CONTROL (Targeting EP3)
    # ========================================================================
    
    def lqr_control(self, state):
        """LQR stabilization around EP3 (alpha=pi, beta=0)"""
        theta, alpha, beta, theta_dot, alpha_dot, beta_dot = state
        
        # Error from upright (alpha=pi, beta=0)
        # Use np.arctan2 for correct wrapping
        alpha_error = np.arctan2(np.sin(alpha - np.pi), np.cos(alpha - np.pi))
        beta_error = np.arctan2(np.sin(beta), np.cos(beta))
        
        x_error = np.array([theta, alpha_error, beta_error, theta_dot, alpha_dot, beta_dot])
        u = -self.K_ep3 @ x_error
        
        return float(u)
    
    # ========================================================================
    # MODE SWITCHING (Targeting EP3)
    # ========================================================================
    
    def check_switch_to_lqr(self, state):
        """Check if ready for LQR at EP3"""
        theta, alpha, beta, theta_dot, alpha_dot, beta_dot = state
        
        alpha_dist = abs(np.arctan2(np.sin(alpha - np.pi), np.cos(alpha - np.pi)))
        beta_dist = abs(np.arctan2(np.sin(beta), np.cos(beta)))
        
        angle_ok = (alpha_dist < self.switch_angle_a and beta_dist < self.switch_angle_b)
        velocity_ok = (abs(alpha_dot) < self.switch_velocity and abs(beta_dot) < self.switch_velocity)
        
        return angle_ok and velocity_ok
    
    def check_switch_to_swingup(self, state):
        """Check if fallen from EP3"""
        theta, alpha, beta, theta_dot, alpha_dot, beta_dot = state
        
        alpha_dist = abs(np.arctan2(np.sin(alpha - np.pi), np.cos(alpha - np.pi)))
        beta_dist = abs(np.arctan2(np.sin(beta), np.cos(beta)))
        
        return (alpha_dist > self.fall_angle_a or beta_dist > self.fall_angle_b)
    
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
                        self.get_logger().info('✓ SWITCHING TO LQR STABILIZATION (EP3)!')
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
        # Only apply torque to the first joint (revolute_joint)
        msg.data = [float(tau), 0.0, 0.0]
        self.effort_pub.publish(msg)
    
    def _publish_mode(self):
        """Publish mode"""
        msg = String()
        msg.data = self.controller_mode
        self.mode_pub.publish(msg)
    
    def _publish_debug(self, state, tau):
        """Publish debug info: torque, alpha, beta"""
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'swingup_lqr_debug'
        msg.vector.x = float(tau)
        msg.vector.y = float(state[1])  # α
        msg.vector.z = float(state[2])  # β
        self.debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TangentialSwingUpLQR_EP3()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()