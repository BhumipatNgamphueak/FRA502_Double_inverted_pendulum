#!/usr/bin/env python3
"""
SINGLE PENDULUM - ENERGY REGULATED SWING-UP
===========================================
FIX: Stop pumping energy when close to target, allow LQR to capture

Key changes:
1. Gradually reduce energy pumping as E → E_desired
2. Looser velocity threshold (3.0 rad/s) for switching
3. Early LQR engagement when energy is sufficient
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
import time

class LowPassFilter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False
        
    def update(self, new_val):
        if not self.initialized:
            self.value = new_val
            self.initialized = True
        else:
            self.value = self.alpha * new_val + (1.0 - self.alpha) * self.value
        return self.value

class SinglePendulumLoopController(Node):
    
    def __init__(self):
        super().__init__('swingup_controller')
        
        self.control_frequency = 100.0
        self.control_period = 1.0 / self.control_frequency
        
        # Physical parameters
        self.g = 9.81
        self.M1 = 0.190
        self.l1 = 0.072
        self.l_arm = 0.157
        
        # ============================================================
        # SWING-UP PARAMETERS (ENERGY REGULATED)
        # ============================================================
        self.energy_gain = 2.55        
        self.damping_gain = 0.11
        self.pump_sign = -1.0
        
        # NEW: Energy regulation - reduce pumping when close to target
        self.energy_regulation_threshold = 0.9  # Start reducing at 90% of target
        
        self.swing_up_max_torque = 10.0
        self.kick_torque = 5.0
        
        # ============================================================
        # LQR PARAMETERS (STRONG)
        # ============================================================
        self.K_theta = 2.0
        self.K_alpha = 4000.0       
        self.K_theta_dot = 8.0
        self.K_alpha_dot = 100.0
        self.lqr_max_torque = 20.0
        
        # ============================================================
        # SWITCHING THRESHOLDS (RELAXED FOR HIGH-ENERGY CAPTURE)
        # ============================================================
        self.switch_angle = 0.15          # Wider: 17° (was 8.6°)
        self.switch_velocity = 3.0       # Higher: 3.0 rad/s (was 1.5)
        self.stabilization_time = 0.05   # Quick switch
        self.fall_angle = 0.7
        
        # Filtering
        self.filter_theta_dot = LowPassFilter(alpha=0.3)
        self.filter_alpha_dot = LowPassFilter(alpha=0.3)
        
        # State variables
        self.current_state = np.zeros(4)
        self.current_state_raw = np.zeros(4)
        self.state_received = False
        self.controller_mode = 'SWING_UP'
        
        # Kick mechanism
        self.kick_counter = 0
        self.kick_duration = int(0.2 * self.control_frequency)
        self.kick_applied = False
        
        self.K_lqr = np.array([
            self.K_theta, 
            self.K_alpha, 
            self.K_theta_dot, 
            self.K_alpha_dot
        ])
        
        # Desired energy
        self.E_desired = self.M1 * self.g * (2.0 * self.l1)
        
        self.stabilization_counter = 0
        self.stabilization_samples = int(self.stabilization_time * self.control_frequency)
        
        self.joint_names = ['revolute_joint', 'first_pendulum_joint']
        
        self.effort_pub = self.create_publisher(
            Float64MultiArray, '/effort_controller/commands', 10
        )
        self.mode_pub = self.create_publisher(String, '/controller_mode', 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        
        self._log_startup()
    
    def _log_startup(self):
        self.get_logger().info('='*70)
        self.get_logger().info('🚀 ENERGY REGULATED SWING-UP CONTROLLER')
        self.get_logger().info('='*70)
        self.get_logger().info('KEY FIX: Gradually stop pumping as E → E_desired')
        self.get_logger().info('')
        self.get_logger().info('SWING-UP:')
        self.get_logger().info(f'  energy_gain = {self.energy_gain}')
        self.get_logger().info(f'  regulation_threshold = {self.energy_regulation_threshold}')
        self.get_logger().info('')
        self.get_logger().info('LQR:')
        self.get_logger().info(f'  K_alpha = {self.K_alpha}')
        self.get_logger().info(f'  K_alpha_dot = {self.K_alpha_dot}')
        self.get_logger().info('')
        self.get_logger().info('SWITCHING (RELAXED):')
        self.get_logger().info(f'  switch_angle = {self.switch_angle} rad ({np.degrees(self.switch_angle):.1f}°)')
        self.get_logger().info(f'  switch_velocity = {self.switch_velocity} rad/s')
        self.get_logger().info('='*70)

    def joint_state_callback(self, msg):
        try:
            state_filtered = np.zeros(4)
            state_raw = np.zeros(4)
            name_map = {name: i for i, name in enumerate(msg.name)}
            
            raw_theta_dot = 0.0
            raw_alpha_dot = 0.0
            
            if self.joint_names[0] in name_map:
                idx = name_map[self.joint_names[0]]
                state_filtered[0] = self._wrap_to_pi(msg.position[idx])
                state_raw[0] = state_filtered[0]
                raw_theta_dot = msg.velocity[idx] if len(msg.velocity) > idx else 0.0
            
            if self.joint_names[1] in name_map:
                idx = name_map[self.joint_names[1]]
                state_filtered[1] = self._wrap_to_pi(msg.position[idx])
                state_raw[1] = state_filtered[1]
                raw_alpha_dot = msg.velocity[idx] if len(msg.velocity) > idx else 0.0
            
            state_raw[2] = raw_theta_dot
            state_raw[3] = raw_alpha_dot
            
            state_filtered[2] = self.filter_theta_dot.update(raw_theta_dot)
            state_filtered[3] = self.filter_alpha_dot.update(raw_alpha_dot)
            
            self.current_state = state_filtered
            self.current_state_raw = state_raw
            self.state_received = True
                
        except Exception as e:
            self.get_logger().error(f'Callback error: {e}')
    
    def _wrap_to_pi(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def _compute_energy(self, state):
        """Compute total energy"""
        _, alpha, theta_dot, alpha_dot = state
        
        # PE relative to hanging
        pe = self.M1 * self.g * self.l1 * (1.0 - np.cos(alpha))
        
        # KE (pendulum swing)
        ke = 0.5 * self.M1 * (self.l1 * alpha_dot)**2
        
        return pe + ke
    
    def swing_up_control(self, state):
        """Energy-based swing-up with REGULATION"""
        theta, alpha, theta_dot, alpha_dot = state
        
        # PHASE 1: Kick
        if not self.kick_applied:
            if self.kick_counter < self.kick_duration:
                self.kick_counter += 1
                return self.kick_torque
            else:
                self.kick_applied = True
                self.get_logger().info('✓ Kick done → Energy pumping')
        
        # PHASE 2: Energy pumping with regulation
        E_current = self._compute_energy(state)
        E_error = self.E_desired - E_current
        
        # NEW: Energy regulation factor
        # When E_current approaches E_desired, reduce pumping
        energy_ratio = E_current / self.E_desired
        
        if energy_ratio > self.energy_regulation_threshold:
            # Reduce pumping gain as we approach target
            # At 90% energy: factor = 1.0
            # At 100% energy: factor = 0.0
            regulation_factor = max(0.0, (1.0 - energy_ratio) / (1.0 - self.energy_regulation_threshold))
        else:
            regulation_factor = 1.0
        
        # Control law with regulation
        control_term = alpha_dot * np.cos(alpha)
        u_energy = self.pump_sign * self.energy_gain * E_error * control_term * regulation_factor
        u_damping = -self.damping_gain * theta_dot
        
        # Log regulation (throttled)
        if int(time.time() * 2) % 2 == 0:  # Every 0.5s
            self.get_logger().info(
                f'E: {E_current:.4f}/{self.E_desired:.4f} ({energy_ratio*100:.1f}%), '
                f'Reg: {regulation_factor:.2f}, α̇: {alpha_dot:.1f}',
                throttle_duration_sec=0.5
            )
        
        return u_energy + u_damping
    
    def lqr_control(self, state):
        """LQR stabilization"""
        theta, alpha, theta_dot, alpha_dot = state
        
        alpha_error = self._wrap_to_pi(alpha - np.pi)
        x_error = np.array([theta, alpha_error, theta_dot, alpha_dot])
        
        u = -self.K_lqr @ x_error
        
        return float(u)
    
    def check_switch_to_lqr(self, state):
        """Check if ready for LQR (RELAXED criteria)"""
        _, alpha, _, alpha_dot = state
        
        alpha_dist = abs(self._wrap_to_pi(alpha - np.pi))
        
        angle_ok = alpha_dist < self.switch_angle  # 0.3 rad (17°)
        velocity_ok = abs(alpha_dot) < self.switch_velocity  # 3.0 rad/s
        
        return angle_ok and velocity_ok
    
    def check_switch_to_swingup(self, state):
        """Check if fallen"""
        _, alpha, _, _ = state
        alpha_dist = abs(self._wrap_to_pi(alpha - np.pi))
        return alpha_dist > self.fall_angle
    
    def control_step(self):
        """Single control iteration"""
        
        rclpy.spin_once(self, timeout_sec=0.0)
        
        if not self.state_received:
            return
        
        # Use filtered for swing-up, raw for LQR
        if self.controller_mode == 'SWING_UP':
            x = self.current_state  # Filtered
        else:
            x = self.current_state_raw  # Raw
        
        # State machine
        if self.controller_mode == 'SWING_UP':
            if self.check_switch_to_lqr(self.current_state):
                self.stabilization_counter += 1
                if self.stabilization_counter >= self.stabilization_samples:
                    self.controller_mode = 'STABILIZE'
                    self.stabilization_counter = 0
                    self.get_logger().info('='*70)
                    self.get_logger().info('✓✓✓ SWITCHED TO LQR!')
                    self.get_logger().info('='*70)
            else:
                self.stabilization_counter = 0
            
            tau = self.swing_up_control(x)
            max_t = self.swing_up_max_torque
            
        else:  # STABILIZE
            if self.check_switch_to_swingup(x):
                self.controller_mode = 'SWING_UP'
                self.kick_applied = False
                self.kick_counter = 0
                self.get_logger().warn('⚠ FALLEN! Restarting')
            
            tau = self.lqr_control(x)
            max_t = self.lqr_max_torque
        
        # Saturate
        tau = np.clip(tau, -max_t, max_t)
        
        # Publish
        msg = Float64MultiArray()
        msg.data = [float(tau), 0.0]
        self.effort_pub.publish(msg)
        
        mode_msg = String()
        mode_msg.data = self.controller_mode
        self.mode_pub.publish(mode_msg)

    def run(self):
        """Main control loop"""
        self.get_logger().info('Starting control loop...')
        self.get_logger().info('='*70)
        
        try:
            while rclpy.ok():
                start = time.time()
                self.control_step()
                elapsed = time.time() - start
                if elapsed < self.control_period:
                    time.sleep(self.control_period - elapsed)
                    
        except KeyboardInterrupt:
            self.get_logger().info('\n' + '='*70)
            self.get_logger().info('Shutting down...')
            self.get_logger().info('='*70)
        finally:
            msg = Float64MultiArray()
            msg.data = [0.0, 0.0]
            self.effort_pub.publish(msg)
            time.sleep(0.1)

def main(args=None):
    print("="*70)
    print("🚀 ENERGY REGULATED CONTROLLER")
    print("="*70)
    
    rclpy.init(args=args)
    controller = SinglePendulumLoopController()
    controller.run()
    controller.destroy_node()
    rclpy.shutdown()
    
    print("\n" + "="*70)
    print("✓ Controller stopped")
    print("="*70)

if __name__ == '__main__':
    main()