#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Vector3Stamped
import scipy.linalg


class LQRControllerNode(Node):
    """
    LQR Controller for Underactuated Double Inverted Pendulum
    Single actuator: horizontal force on cart (revolute_joint)
    Passive joints: first_pendulum_joint, second_pendulum_joint
    
    Inputs:
        - /joint_states (sensor_msgs/JointState): Joint positions and velocities
        
    Outputs:
        - /effort_controller/commands (std_msgs/Float64MultiArray): Control effort [F, 0, 0]
        - /lqr_debug (geometry_msgs/Vector3Stamped): Debug information (optional)
    
    Frequency: 100 Hz (configurable)
    """
    
    def __init__(self):
        super().__init__('lqr_controller_node')
        
        # Declare and get parameters
        self._declare_parameters()
        self._get_parameters()
        
        # Initialize state
        self.current_state = np.zeros(6)  # [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        self.state_received = False
        
        # Compute LQR gain matrix
        self.K = self._compute_lqr_gain()
        self.get_logger().info(f'LQR Gain Matrix K computed with shape: {self.K.shape}')
        
        # Create subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            10
        )
        
        # Create publishers
        self.effort_pub = self.create_publisher(
            Float64MultiArray,
            self.effort_command_topic,
            10
        )
        
        if self.publish_debug_info:
            self.debug_pub = self.create_publisher(
                Vector3Stamped,
                self.debug_topic,
                10
            )
        
        # Create timer for control loop
        self.control_period = 1.0 / self.control_frequency
        self.control_timer = self.create_timer(
            self.control_period,
            self.control_loop
        )
        
        self.get_logger().info(f'LQR Controller initialized at {self.control_frequency} Hz')
        self.get_logger().info(f'Underactuated system: Single actuator (cart force only)')
        self.get_logger().info(f'Control saturation: {self.enable_saturation}')
    
    def _declare_parameters(self):
        """Declare all ROS parameters"""
        # Control parameters
        self.declare_parameter('control_frequency', 100.0)
        
        # Physical parameters
        self.declare_parameter('m0', 1.0)
        self.declare_parameter('m1', 0.5)
        self.declare_parameter('m2', 0.3)
        self.declare_parameter('l1', 0.25)
        self.declare_parameter('l2', 0.20)
        self.declare_parameter('L1', 0.5)
        self.declare_parameter('L2', 0.4)
        self.declare_parameter('I1_tensor', [0.01, 0.01, 0.001, 0.0, 0.0, 0.0])
        self.declare_parameter('I2_tensor', [0.005, 0.005, 0.0005, 0.0, 0.0, 0.0])
        self.declare_parameter('g', 9.81)
        
        # LQR parameters
        self.declare_parameter('Q_diagonal', [10.0, 100.0, 100.0, 1.0, 1.0, 1.0])
        self.declare_parameter('R_diagonal', [1.0])
        
        # Control limits
        self.declare_parameter('max_force', 50.0)
        
        # Flags
        self.declare_parameter('enable_gravity_compensation', True)
        self.declare_parameter('enable_saturation', True)
        
        # Topics
        self.declare_parameter('joint_names', ['revolute_joint', 'first_pendulum_joint', 'second_pendulum_joint'])
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('effort_command_topic', '/effort_controller/commands')
        self.declare_parameter('publish_debug_info', True)
        self.declare_parameter('debug_topic', '/lqr_debug')
    
    def _get_parameters(self):
        """Get all ROS parameters"""
        # Control parameters
        self.control_frequency = self.get_parameter('control_frequency').value
        
        # Physical parameters
        self.m0 = self.get_parameter('m0').value
        self.m1 = self.get_parameter('m1').value
        self.m2 = self.get_parameter('m2').value
        self.l1 = self.get_parameter('l1').value
        self.l2 = self.get_parameter('l2').value
        self.L1 = self.get_parameter('L1').value
        self.L2 = self.get_parameter('L2').value
        self.g = self.get_parameter('g').value
        
        # Inertia tensors
        I1_list = self.get_parameter('I1_tensor').value
        I2_list = self.get_parameter('I2_tensor').value
        
        # Build full inertia tensors [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        self.I1 = self._build_inertia_tensor(I1_list)
        self.I2 = self._build_inertia_tensor(I2_list)
        
        # For 2D planar motion, we primarily use Iyy component
        self.I1_yy = self.I1[1, 1]
        self.I2_yy = self.I2[1, 1]
        
        # LQR matrices
        Q_diag = self.get_parameter('Q_diagonal').value
        R_diag = self.get_parameter('R_diagonal').value
        self.Q = np.diag(Q_diag)
        self.R = np.array([[R_diag[0]]])  # Scalar as 1x1 matrix
        
        # Control limits
        self.max_force = self.get_parameter('max_force').value
        
        # Flags
        self.enable_gravity_compensation = self.get_parameter('enable_gravity_compensation').value
        self.enable_saturation = self.get_parameter('enable_saturation').value
        
        # Topics
        self.joint_names = self.get_parameter('joint_names').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.effort_command_topic = self.get_parameter('effort_command_topic').value
        self.publish_debug_info = self.get_parameter('publish_debug_info').value
        self.debug_topic = self.get_parameter('debug_topic').value
    
    def _build_inertia_tensor(self, I_list):
        """
        Build 3x3 inertia tensor from list [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        Returns symmetric inertia matrix
        """
        I = np.array([
            [I_list[0], I_list[3], I_list[4]],
            [I_list[3], I_list[1], I_list[5]],
            [I_list[4], I_list[5], I_list[2]]
        ])
        return I
    
    def _compute_lqr_gain(self):
        """
        Compute LQR gain matrix K by solving the continuous-time algebraic Riccati equation
        
        State: x = [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        Control: u = F (single actuator - cart force)
        
        System: x_dot = A*x + B*u (linearized around upright equilibrium)
        Control: u = -K*x
        
        Returns:
            K: 1x6 gain matrix
        """
        # Linearized state-space matrices around upright equilibrium
        A, B = self._get_linearized_system()
        
        # Solve continuous-time algebraic Riccati equation
        try:
            P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R) @ B.T @ P
            return K
        except Exception as e:
            self.get_logger().error(f'Failed to compute LQR gain: {e}')
            # Return zero gain as fallback (1x6 for single actuator)
            return np.zeros((1, 6))
    
    def _get_linearized_system(self):
        """
        Get linearized state-space matrices A and B around upright equilibrium
        
        For underactuated double inverted pendulum:
        State: [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        Control: F (cart force only - single actuator)
        
        Returns:
            A: 6x6 state matrix
            B: 6x1 input matrix
        """
        # Simplified linearization (assumes small angles, theta1≈0, theta2≈0)
        # This is a placeholder - you should derive exact linearization from your dynamics
        
        m0, m1, m2 = self.m0, self.m1, self.m2
        l1, l2 = self.l1, self.l2
        L1, L2 = self.L1, self.L2
        g = self.g
        I1, I2 = self.I1_yy, self.I2_yy
        
        # Total mass
        M = m0 + m1 + m2
        
        # Inertia terms with parallel axis theorem
        J1 = I1 + m1 * l1**2
        J2 = I2 + m2 * l2**2
        
        # Mass matrix elements (evaluated at equilibrium)
        # For underactuated system: M * [x_ddot, theta1_ddot, theta2_ddot]' = [F, 0, 0]' + gravity_terms
        M11 = M
        M12 = m1*l1 + m2*L1
        M13 = m2*l2
        M22 = J1 + m2*L1**2
        M23 = m2*L1*l2
        M33 = J2
        
        # Mass matrix
        M_mat = np.array([
            [M11, M12, M13],
            [M12, M22, M23],
            [M13, M23, M33]
        ])
        
        # Gravity gradient matrix (linearized around upright)
        G_mat = np.array([
            [0, -(m1*l1 + m2*L1)*g, -m2*l2*g],
            [0, 0, 0],
            [0, 0, 0]
        ])
        
        # Compute M^(-1)
        M_inv = np.linalg.inv(M_mat)
        
        # State matrix A (6x6)
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)  # Position derivatives
        A[3:6, 0:3] = -M_inv @ G_mat  # Acceleration from gravity
        
        # Input matrix B (6x1) - only cart force
        # Control input is [F, 0, 0]' in generalized coordinates
        B = np.zeros((6, 1))
        B[3:6, 0:1] = M_inv[:, 0:1]  # First column of M_inv (effect of cart force)
        
        return A, B
    
    def joint_state_callback(self, msg):
        """
        Callback for joint state messages
        
        INPUT:
            msg (sensor_msgs/JointState): Joint positions and velocities
        """
        try:
            # Extract states in correct order based on joint names
            state = np.zeros(6)
            
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    state[i] = msg.position[idx]
                    state[i+3] = msg.velocity[idx]
            
            self.current_state = state
            self.state_received = True
            
        except Exception as e:
            self.get_logger().error(f'Error in joint_state_callback: {e}')
    
    def control_loop(self):
        """
        Main control loop executed at control_frequency Hz
        
        OUTPUT:
            Publishes effort commands to /effort_controller/commands
            Format: [F, 0, 0] where F is cart force, other joints are passive
        """
        if not self.state_received:
            self.get_logger().warn('No joint states received yet', throttle_duration_sec=2.0)
            return
        
        try:
            # Extract state
            x = self.current_state
            
            # LQR control law: u = -K*x (scalar output)
            u_lqr = -self.K @ x  # Results in (1,) array
            u_lqr_scalar = float(u_lqr[0])
            
            # Note: Gravity compensation not applicable for underactuated system
            # Cannot directly apply torques to passive pendulum joints
            # The controller must balance the system through cart motion alone
            u_total_scalar = u_lqr_scalar
            
            # Apply saturation limits
            if self.enable_saturation:
                u_total_scalar = np.clip(u_total_scalar, -self.max_force, self.max_force)
            
            # Publish control command [F, 0, 0] - only cart is actuated
            self._publish_effort_command(u_total_scalar)
            
            # Publish debug info
            if self.publish_debug_info:
                self._publish_debug_info(x, u_lqr_scalar, u_total_scalar)
                
        except Exception as e:
            self.get_logger().error(f'Error in control_loop: {e}')
    
    # Note: Gravity compensation removed
    # For underactuated systems, we cannot directly apply torques to passive joints
    # The LQR controller must stabilize through cart motion alone
    
    def _publish_effort_command(self, F):
        """
        Publish effort command message for single actuator system
        
        Args:
            F: Cart force (scalar)
        
        OUTPUT:
            Float64MultiArray on /effort_controller/commands: [F, 0, 0]
            (Cart force, passive joint 1, passive joint 2)
        """
        msg = Float64MultiArray()
        msg.data = [float(F), 0.0, 0.0]  # Only cart is actuated
        self.effort_pub.publish(msg)
    
    def _publish_debug_info(self, state, u_lqr, u_total):
        """
        Publish debug information for single actuator system
        
        Args:
            state: State vector [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
            u_lqr: LQR control (scalar)
            u_total: Total control after saturation (scalar)
        
        OUTPUT:
            Vector3Stamped on /lqr_debug
            x: Cart force
            y: theta1 angle
            z: theta2 angle
        """
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'lqr_debug'
        msg.vector.x = float(u_total)  # Cart force
        msg.vector.y = float(state[1])  # theta1
        msg.vector.z = float(state[2])  # theta2
        self.debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LQRControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
