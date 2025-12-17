---
sidebar_position: 3
---

# Nodes, Topics, Services, Actions

## Understanding ROS 2 Communication Primitives

ROS 2 provides four fundamental communication primitives that form the backbone of robotic applications: Nodes, Topics, Services, and Actions. Understanding these primitives is essential for building effective humanoid robot systems that can handle the complexity of real-world interaction.

### The Communication Hierarchy

The four communication primitives serve different purposes in the ROS 2 ecosystem:

- **Nodes**: Execution units that perform specific functions
- **Topics**: Asynchronous, one-way communication for streaming data
- **Services**: Synchronous, request-response communication for queries
- **Actions**: Asynchronous, goal-oriented communication with feedback for long-running tasks

## Nodes: The Execution Foundation

Nodes are the fundamental execution units in ROS 2. Each node represents a single process that performs a specific function within the robot system.

### Node Creation and Management

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import threading
import time

class HumanoidNode(Node):
    def __init__(self):
        # Initialize the node with a unique name
        super().__init__('humanoid_node')

        # Create a publisher for joint states
        self.joint_publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        # Create a publisher for status messages
        self.status_publisher = self.create_publisher(
            String,
            'robot_status',
            10
        )

        # Create a subscription to commands
        self.command_subscription = self.create_subscription(
            String,
            'robot_commands',
            self.command_callback,
            10
        )

        # Create a timer for periodic updates
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz

        # Initialize node-specific data
        self.joint_names = ['hip_left', 'hip_right', 'knee_left', 'knee_right']
        self.joint_positions = [0.0, 0.0, 0.0, 0.0]
        self.status = 'idle'

        self.get_logger().info('Humanoid Node initialized')

    def command_callback(self, msg):
        """Handle incoming commands"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        if command == 'stand':
            self.status = 'standing'
            self.stand_up()
        elif command == 'walk':
            self.status = 'walking'
            self.start_walking()
        elif command == 'stop':
            self.status = 'idle'
            self.stop_motion()

    def timer_callback(self):
        """Publish periodic updates"""
        # Update joint states
        joint_msg = JointState()
        joint_msg.name = self.joint_names
        joint_msg.position = self.joint_positions
        joint_msg.header.stamp = self.get_clock().now().to_msg()

        self.joint_publisher.publish(joint_msg)

        # Publish status
        status_msg = String()
        status_msg.data = self.status
        self.status_publisher.publish(status_msg)

    def stand_up(self):
        """Execute stand up motion"""
        self.get_logger().info('Executing stand up motion')
        # In a real implementation, this would send commands to joints
        self.joint_positions = [0.0, 0.0, 0.0, 0.0]  # Reset to standing position

    def start_walking(self):
        """Start walking motion"""
        self.get_logger().info('Starting walking motion')
        # This would implement walking gait pattern
        pass

    def stop_motion(self):
        """Stop all motion"""
        self.get_logger().info('Stopping all motion')
        # This would send zero velocity commands to all joints
        pass

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Humanoid Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle Management

Nodes in ROS 2 can have lifecycle management for better control over initialization and cleanup:

```python
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
from sensor_msgs.msg import JointState

class LifecycleHumanoidNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_humanoid_node')
        self.get_logger().info('Lifecycle Humanoid Node created')

    def on_configure(self, state: LifecycleState):
        """Configure the node"""
        self.get_logger().info('Configuring node')

        # Create publishers/subscribers but don't activate them yet
        self.joint_publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        # Initialize hardware interfaces
        self.hardware_initialized = self.initialize_hardware()

        return TransitionCallbackReturn.SUCCESS if self.hardware_initialized else TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState):
        """Activate the node"""
        self.get_logger().info('Activating node')

        # Activate publishers/subscribers
        self.joint_publisher.on_activate()

        # Start timers
        self.control_timer = self.create_timer(0.01, self.control_loop)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState):
        """Deactivate the node"""
        self.get_logger().info('Deactivating node')

        # Stop timers
        self.control_timer.destroy()

        # Deactivate publishers/subscribers
        self.joint_publisher.on_deactivate()

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState):
        """Clean up the node"""
        self.get_logger().info('Cleaning up node')

        # Clean up resources
        self.cleanup_hardware()

        # Destroy publishers/subscribers
        self.destroy_publisher(self.joint_publisher)

        return TransitionCallbackReturn.SUCCESS

    def control_loop(self):
        """Main control loop"""
        self.get_logger().info('Executing control loop')

    def initialize_hardware(self):
        """Initialize hardware interfaces"""
        # Simulate hardware initialization
        self.get_logger().info('Initializing hardware interfaces')
        return True

    def cleanup_hardware(self):
        """Clean up hardware interfaces"""
        self.get_logger().info('Cleaning up hardware interfaces')

def main(args=None):
    rclpy.init(args=args)
    node = LifecycleHumanoidNode()

    try:
        # Manually trigger transitions for demonstration
        node.trigger_configure()
        node.trigger_activate()

        rclpy.spin(node)
    except KeyboardInterrupt:
        node.trigger_deactivate()
        node.trigger_cleanup()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics: Asynchronous Data Streaming

Topics provide a publish-subscribe communication model ideal for streaming data like sensor readings, joint states, and robot status.

### Publisher Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import math
import time

class SensorPublisherNode(Node):
    def __init__(self):
        super().__init__('sensor_publisher_node')

        # Publishers for different sensor types
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.odom_pub = self.create_publisher(Float64MultiArray, 'odom', 10)

        # Timer for sensor data publishing
        self.sensor_timer = self.create_timer(0.01, self.publish_sensor_data)  # 100Hz

        # Initialize sensor data
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow'
        ]
        self.joint_positions = [0.0] * len(self.joint_names)
        self.time_offset = time.time()

        self.get_logger().info('Sensor Publisher Node initialized')

    def publish_sensor_data(self):
        """Publish sensor data at high frequency"""
        current_time = time.time() - self.time_offset

        # Simulate joint position updates (could come from actual encoders)
        for i in range(len(self.joint_positions)):
            self.joint_positions[i] = math.sin(current_time + i * 0.1) * 0.5

        # Publish joint states
        joint_msg = JointState()
        joint_msg.name = self.joint_names
        joint_msg.position = self.joint_positions
        joint_msg.velocity = [0.0] * len(self.joint_names)  # Simplified
        joint_msg.effort = [0.0] * len(self.joint_names)    # Simplified
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = 'base_link'

        self.joint_pub.publish(joint_msg)

        # Publish IMU data
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        # Simulate IMU readings
        imu_msg.orientation.x = math.sin(current_time * 0.1) * 0.1
        imu_msg.orientation.y = math.cos(current_time * 0.1) * 0.1
        imu_msg.orientation.z = 0.0
        imu_msg.orientation.w = math.sqrt(1 - (imu_msg.orientation.x**2 + imu_msg.orientation.y**2))

        imu_msg.angular_velocity.x = math.cos(current_time * 10) * 0.1
        imu_msg.angular_velocity.y = math.sin(current_time * 10) * 0.1
        imu_msg.angular_velocity.z = 0.05

        imu_msg.linear_acceleration.x = math.sin(current_time * 5) * 9.81
        imu_msg.linear_acceleration.y = math.cos(current_time * 5) * 9.81
        imu_msg.linear_acceleration.z = 9.81

        self.imu_pub.publish(imu_msg)

        # Publish odometry (simplified)
        odom_msg = Float64MultiArray()
        odom_msg.data = [
            math.sin(current_time) * 0.1,  # x position
            math.cos(current_time) * 0.1,  # y position
            current_time * 0.01,           # theta orientation
            math.cos(current_time) * 0.1,  # x velocity
            -math.sin(current_time) * 0.1, # y velocity
            0.01                           # angular velocity
        ]

        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorPublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
import numpy as np

class SensorSubscriberNode(Node):
    def __init__(self):
        super().__init__('sensor_subscriber_node')

        # Subscribers for different sensor types
        self.joint_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            'imu/data',
            self.imu_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Store latest sensor data
        self.latest_joint_state = None
        self.latest_imu_data = None
        self.balance_state = {'stable': True, 'com_offset': 0.0}

        self.get_logger().info('Sensor Subscriber Node initialized')

    def joint_callback(self, msg):
        """Process joint state messages"""
        self.latest_joint_state = msg
        self.get_logger().debug(f'Received joint state with {len(msg.name)} joints')

        # Analyze joint positions for balance
        if 'left_ankle' in msg.name and 'right_ankle' in msg.name:
            left_idx = msg.name.index('left_ankle')
            right_idx = msg.name.index('right_ankle')

            # Simple balance check based on ankle positions
            ankle_diff = abs(msg.position[left_idx] - msg.position[right_idx])
            self.balance_state['stable'] = ankle_diff < 0.1  # 10 degrees threshold

        self.get_logger().info(f'Balance state: stable={self.balance_state["stable"]}')

    def imu_callback(self, msg):
        """Process IMU messages for balance control"""
        self.latest_imu_data = msg

        # Extract orientation from quaternion
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        roll, pitch, yaw = self.quaternion_to_euler(quat)

        # Check if robot is within safe orientation limits
        max_tilt = 0.3  # 17 degrees
        tilt_safe = abs(roll) < max_tilt and abs(pitch) < max_tilt

        if not tilt_safe:
            self.get_logger().warn(f'Robot tilt unsafe: roll={roll:.3f}, pitch={pitch:.3f}')
            self.emergency_stop()

        self.balance_state['com_offset'] = pitch  # Simplified CoM estimation

    def cmd_vel_callback(self, msg):
        """Process velocity commands"""
        self.get_logger().info(f'Received velocity command: linear={msg.linear.x}, angular={msg.angular.z}')

        # Validate command against safety limits
        max_linear_vel = 0.5  # m/s
        max_angular_vel = 0.5  # rad/s

        if abs(msg.linear.x) > max_linear_vel or abs(msg.angular.z) > max_angular_vel:
            self.get_logger().warn('Velocity command exceeds safety limits')
            return

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def emergency_stop(self):
        """Execute emergency stop procedure"""
        self.get_logger().error('EMERGENCY STOP: Robot orientation unsafe')
        # In a real system, this would send zero commands to all joints
        # and potentially engage safety mechanisms

def main(args=None):
    rclpy.init(args=args)
    node = SensorSubscriberNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services: Synchronous Request-Response

Services provide synchronous communication ideal for queries and operations that return a result immediately.

### Service Server Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger, SetBool
from std_srvs.srv import Empty
from humanoid_robot_msgs.srv import SetJointPosition  # Custom service

class RobotServiceServer(Node):
    def __init__(self):
        super().__init__('robot_service_server')

        # Create various service servers
        self.calibrate_service = self.create_service(
            Trigger,
            'calibrate_robot',
            self.calibrate_callback
        )

        self.emergency_stop_service = self.create_service(
            SetBool,
            'emergency_stop',
            self.emergency_stop_callback
        )

        self.reset_service = self.create_service(
            Empty,
            'reset_robot',
            self.reset_callback
        )

        # Custom service for setting joint positions
        self.joint_position_service = self.create_service(
            SetJointPosition,
            'set_joint_position',
            self.set_joint_position_callback
        )

        self.is_calibrated = False
        self.is_emergency_stopped = False

        self.get_logger().info('Robot Service Server initialized')

    def calibrate_callback(self, request, response):
        """Calibrate the robot"""
        self.get_logger().info('Calibration service called')

        try:
            # Simulate calibration process
            self.get_logger().info('Starting calibration sequence...')

            # Calibrate each joint
            for joint_name in ['hip_left', 'hip_right', 'knee_left', 'knee_right']:
                self.get_logger().info(f'Calibrating {joint_name}...')
                # Simulate calibration time
                time.sleep(0.1)

            self.is_calibrated = True
            response.success = True
            response.message = 'Robot calibration completed successfully'

            self.get_logger().info('Calibration completed')

        except Exception as e:
            response.success = False
            response.message = f'Calibration failed: {str(e)}'
            self.get_logger().error(f'Calibration error: {str(e)}')

        return response

    def emergency_stop_callback(self, request, response):
        """Handle emergency stop"""
        self.get_logger().warn(f'Emergency stop requested: {request.data}')

        self.is_emergency_stopped = request.data

        if request.data:
            self.get_logger().warn('EMERGENCY STOP ACTIVATED - All motion stopped')
            # In a real system, this would immediately stop all actuators
        else:
            self.get_logger().info('Emergency stop released')
            # Resume normal operation

        response.success = True
        response.message = f'Emergency stop set to {request.data}'

        return response

    def reset_callback(self, request, response):
        """Reset the robot to initial state"""
        self.get_logger().info('Reset service called')

        try:
            # Reset all joints to home position
            self.reset_joints()

            # Clear any error states
            self.is_emergency_stopped = False
            self.is_calibrated = False  # Need to recalibrate after reset

            # Reinitialize systems
            self.initialize_systems()

            self.get_logger().info('Reset completed successfully')

        except Exception as e:
            self.get_logger().error(f'Reset failed: {str(e)}')
            return response  # Return with default success=False

        return response

    def set_joint_position_callback(self, request, response):
        """Set specific joint position"""
        self.get_logger().info(f'Setting {request.joint_name} to {request.position} rad')

        try:
            # Validate joint name
            valid_joints = [
                'hip_left', 'hip_right', 'knee_left', 'knee_right',
                'ankle_left', 'ankle_right', 'shoulder_left', 'shoulder_right'
            ]

            if request.joint_name not in valid_joints:
                response.success = False
                response.message = f'Invalid joint name: {request.joint_name}'
                return response

            # Validate position limits
            if abs(request.position) > 3.14:  # 180 degrees
                response.success = False
                response.message = f'Position out of range: {request.position}'
                return response

            # In a real system, this would send the command to the joint controller
            self.send_joint_command(request.joint_name, request.position)

            response.success = True
            response.message = f'Successfully set {request.joint_name} to {request.position} rad'

        except Exception as e:
            response.success = False
            response.message = f'Failed to set joint position: {str(e)}'
            self.get_logger().error(f'Set joint position error: {str(e)}')

        return response

    def reset_joints(self):
        """Reset all joints to home position"""
        # Simulate joint reset
        self.get_logger().info('Resetting all joints to home position')

    def initialize_systems(self):
        """Reinitialize robot systems after reset"""
        self.get_logger().info('Reinitializing robot systems')

    def send_joint_command(self, joint_name, position):
        """Send command to joint controller"""
        self.get_logger().info(f'Sending command: {joint_name} = {position} rad')

def main(args=None):
    rclpy.init(args=args)
    server = RobotServiceServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger, SetBool
from std_srvs.srv import Empty
from humanoid_robot_msgs.srv import SetJointPosition
import time

class ServiceClientNode(Node):
    def __init__(self):
        super().__init__('service_client_node')

        # Create clients for various services
        self.calibrate_client = self.create_client(Trigger, 'calibrate_robot')
        self.emergency_stop_client = self.create_client(SetBool, 'emergency_stop')
        self.reset_client = self.create_client(Empty, 'reset_robot')
        self.joint_position_client = self.create_client(SetJointPosition, 'set_joint_position')

        # Wait for services to be available
        self.wait_for_services()

        # Timer to test services periodically
        self.test_timer = self.create_timer(5.0, self.test_services)

        self.get_logger().info('Service Client Node initialized')

    def wait_for_services(self):
        """Wait for all services to be available"""
        services = [
            ('calibrate_robot', self.calibrate_client),
            ('emergency_stop', self.emergency_stop_client),
            ('reset_robot', self.reset_client),
            ('set_joint_position', self.joint_position_client)
        ]

        for name, client in services:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Service {name} not available, waiting...')

    def test_services(self):
        """Test all services periodically"""
        self.get_logger().info('Testing robot services...')

        # Test calibration
        self.test_calibration()

        # Test joint position setting
        self.test_joint_position()

        # Test emergency stop (set to False to clear)
        self.test_emergency_stop(False)

    def test_calibration(self):
        """Test calibration service"""
        try:
            request = Trigger.Request()
            future = self.calibrate_client.call_async(request)

            # Wait for response with timeout
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

            response = future.result()
            if response:
                self.get_logger().info(f'Calibration result: {response.success}, {response.message}')
            else:
                self.get_logger().error('Calibration service call failed')

        except Exception as e:
            self.get_logger().error(f'Calibration test error: {str(e)}')

    def test_joint_position(self):
        """Test joint position service"""
        try:
            request = SetJointPosition.Request()
            request.joint_name = 'hip_left'
            request.position = 0.1  # 0.1 radians

            future = self.joint_position_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

            response = future.result()
            if response:
                self.get_logger().info(f'Joint position result: {response.success}, {response.message}')
            else:
                self.get_logger().error('Joint position service call failed')

        except Exception as e:
            self.get_logger().error(f'Joint position test error: {str(e)}')

    def test_emergency_stop(self, activate):
        """Test emergency stop service"""
        try:
            request = SetBool.Request()
            request.data = activate

            future = self.emergency_stop_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

            response = future.result()
            if response:
                self.get_logger().info(f'Emergency stop result: {response.success}, {response.message}')
            else:
                self.get_logger().error('Emergency stop service call failed')

        except Exception as e:
            self.get_logger().error(f'Emergency stop test error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    client = ServiceClientNode()

    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions: Long-Running Tasks with Feedback

Actions are perfect for long-running operations that provide feedback, such as walking, manipulation, or navigation tasks.

### Action Server Implementation

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ActionServer
from humanoid_robot_msgs.action import WalkToGoal  # Custom action
from geometry_msgs.msg import Pose
import time
import math

class WalkActionServer(Node):
    def __init__(self):
        super().__init__('walk_action_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            WalkToGoal,
            'walk_to_goal',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.current_position = [0.0, 0.0, 0.0]  # x, y, theta
        self.is_walking = False

        self.get_logger().info('Walk Action Server initialized')

    def goal_callback(self, goal_request):
        """Accept or reject goal"""
        self.get_logger().info(f'Received walk goal: x={goal_request.target_pose.pose.position.x}, y={goal_request.target_pose.pose.position.y}')

        # Check if goal is valid (not too far, not in obstacle, etc.)
        distance = math.sqrt(
            (goal_request.target_pose.pose.position.x - self.current_position[0])**2 +
            (goal_request.target_pose.pose.position.y - self.current_position[1])**2
        )

        if distance > 10.0:  # Max distance of 10m
            self.get_logger().warn('Goal too far, rejecting')
            return GoalResponse.REJECT

        # Check if already walking
        if self.is_walking:
            self.get_logger().warn('Already walking, rejecting new goal')
            return GoalResponse.REJECT

        self.get_logger().info('Goal accepted')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the walk goal"""
        self.get_logger().info('Executing walk goal...')
        self.is_walking = True

        # Get target position
        target_x = goal_handle.request.target_pose.pose.position.x
        target_y = goal_handle.request.target_pose.pose.position.y
        target_theta = self.quaternion_to_yaw(goal_handle.request.target_pose.pose.orientation)

        # Calculate path (simplified as straight line)
        start_x, start_y, start_theta = self.current_position
        distance = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)

        # Initialize feedback
        feedback_msg = WalkToGoal.Feedback()
        feedback_msg.current_pose.pose.position.x = start_x
        feedback_msg.current_pose.pose.position.y = start_y
        feedback_msg.distance_remaining = distance

        # Walk simulation
        steps = int(distance / 0.01)  # 1cm per step
        for i in range(steps):
            # Check if goal was canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.is_walking = False
                result = WalkToGoal.Result()
                result.success = False
                result.message = 'Goal canceled'
                return result

            # Calculate current position (simplified linear interpolation)
            progress = i / steps
            current_x = start_x + (target_x - start_x) * progress
            current_y = start_y + (target_y - start_y) * progress
            current_theta = start_theta + (target_theta - start_theta) * progress

            # Update current position
            self.current_position = [current_x, current_y, current_theta]

            # Update feedback
            feedback_msg.current_pose.pose.position.x = current_x
            feedback_msg.current_pose.pose.position.y = current_y
            feedback_msg.distance_remaining = distance * (1 - progress)
            feedback_msg.progress_percentage = progress * 100.0

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)

            # Log progress
            self.get_logger().info(f'Walking: {feedback_msg.progress_percentage:.1f}% complete')

            # Simulate walking time
            await rclpy.asyncio.sleep(0.02)  # 50Hz walking simulation

        # Check if we reached the goal
        final_distance = math.sqrt(
            (target_x - self.current_position[0])**2 +
            (target_y - self.current_position[1])**2
        )

        # Set goal succeeded
        goal_handle.succeed()
        self.is_walking = False

        # Create result
        result = WalkToGoal.Result()
        result.success = final_distance < 0.1  # Within 10cm of target
        result.message = f'Reached goal with final distance: {final_distance:.3f}m'
        result.final_pose.pose.position.x = self.current_position[0]
        result.final_pose.pose.position.y = self.current_position[1]

        self.get_logger().info(f'Walk completed: {result.message}')

        return result

    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    server = WalkActionServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Client Implementation

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from humanoid_robot_msgs.action import WalkToGoal
from geometry_msgs.msg import Pose, Point, Quaternion
import math

class WalkActionClient(Node):
    def __init__(self):
        super().__init__('walk_action_client')

        # Create action client
        self._action_client = ActionClient(self, WalkToGoal, 'walk_to_goal')

        # Timer to send walk goals periodically
        self.goal_timer = self.create_timer(10.0, self.send_walk_goal)

        self.goal_count = 0
        self.get_logger().info('Walk Action Client initialized')

    def send_walk_goal(self):
        """Send a walk goal to the action server"""
        self.get_logger().info('Waiting for action server...')

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return

        # Create goal
        goal_msg = WalkToGoal.Goal()

        # Set target pose (spiral pattern for demonstration)
        angle = self.goal_count * 0.5
        radius = 1.0 + (self.goal_count * 0.2)  # Spiral outward

        goal_msg.target_pose.pose.position.x = radius * math.cos(angle)
        goal_msg.target_pose.pose.position.y = radius * math.sin(angle)
        goal_msg.target_pose.pose.position.z = 0.0

        # Set orientation to face toward center
        target_angle = math.atan2(-goal_msg.target_pose.pose.position.y, -goal_msg.target_pose.pose.position.x)
        goal_msg.target_pose.pose.orientation = self.yaw_to_quaternion(target_angle)

        goal_msg.target_pose.header.frame_id = 'map'

        self.get_logger().info(f'Sending walk goal {self.goal_count + 1}: x={goal_msg.target_pose.pose.position.x:.2f}, y={goal_msg.target_pose.pose.position.y:.2f}')

        # Send goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

        self.goal_count += 1

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Get result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback during execution"""
        current_pos = feedback_msg.current_pose.pose.position
        self.get_logger().info(
            f'Walking: x={current_pos.x:.2f}, y={current_pos.y:.2f}, '
            f'distance_remaining={feedback_msg.distance_remaining:.2f}m, '
            f'progress={feedback_msg.progress_percentage:.1f}%'
        )

    def get_result_callback(self, future):
        """Handle result when goal completes"""
        result = future.result().result
        self.get_logger().info(f'Walk result: success={result.success}, {result.message}')

    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return Quaternion(x=0.0, y=0.0, z=sy, w=cy)

def main(args=None):
    rclpy.init(args=args)
    client = WalkActionClient()

    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration Example: Complete Humanoid System

Here's an example showing how all communication primitives work together in a humanoid robot system:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rclpy.action import ActionClient
from humanoid_robot_msgs.action import WalkToGoal
from example_interfaces.srv import Trigger

class HumanoidIntegratedSystem(Node):
    def __init__(self):
        super().__init__('humanoid_integrated_system')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'robot_status', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10
        )

        # Service client for calibration
        self.calibrate_client = self.create_client(Trigger, 'calibrate_robot')

        # Action client for walking
        self.walk_client = ActionClient(self, WalkToGoal, 'walk_to_goal')

        # Timer for main control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # State variables
        self.joint_states = None
        self.imu_data = None
        self.robot_status = 'idle'
        self.balance_ok = True

        self.get_logger().info('Humanoid Integrated System initialized')

    def joint_callback(self, msg):
        """Handle joint state updates"""
        self.joint_states = msg
        self.check_joint_limits()

    def imu_callback(self, msg):
        """Handle IMU updates for balance"""
        self.imu_data = msg
        self.check_balance()

    def check_joint_limits(self):
        """Check if joints are within safe limits"""
        if self.joint_states:
            for i, pos in enumerate(self.joint_states.position):
                if abs(pos) > 2.0:  # 114 degrees limit
                    self.get_logger().warn(f'Joint {self.joint_states.name[i]} at limit: {pos}')

    def check_balance(self):
        """Check robot balance using IMU data"""
        if self.imu_data:
            # Extract pitch from orientation
            quat = self.imu_data.orientation
            pitch = math.asin(2.0 * (quat.w * quat.y - quat.z * quat.x))

            # Check balance threshold
            if abs(pitch) > 0.3:  # 17 degrees
                self.balance_ok = False
                self.get_logger().error(f'Balance compromised: pitch={pitch:.3f}')
                self.emergency_stop()
            else:
                self.balance_ok = True

    def control_loop(self):
        """Main control loop"""
        # Publish robot status
        status_msg = String()
        status_msg.data = f'balance_ok={self.balance_ok}, status={self.robot_status}'
        self.status_pub.publish(status_msg)

        # Example: Send velocity commands based on some logic
        if self.balance_ok and self.robot_status == 'walking':
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.3  # Walk forward at 0.3 m/s
            self.cmd_vel_pub.publish(cmd_vel)

    def emergency_stop(self):
        """Emergency stop procedure"""
        self.robot_status = 'emergency_stop'
        # Send zero velocity command
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        self.get_logger().error('EMERGENCY STOP ACTIVATED')

def main(args=None):
    rclpy.init(args=args)
    system = HumanoidIntegratedSystem()

    try:
        rclpy.spin(system)
    except KeyboardInterrupt:
        system.get_logger().info('Shutting down humanoid system')
    finally:
        system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Common Mistakes and Best Practices

### Common Mistakes

#### Using Wrong Communication Pattern
- **Using Services for Streaming Data**: Services are synchronous and blocking, not suitable for high-frequency sensor data
- **Using Topics for Critical Commands**: Important commands might be lost with best-effort QoS
- **Using Actions for Simple Queries**: Actions have overhead for simple, quick operations

#### Poor Error Handling
- **Not Checking Service Availability**: Clients not waiting for services to be available
- **Ignoring Action Cancel Requests**: Not checking for cancellation during execution
- **No Timeout Handling**: Operations that hang indefinitely

#### Resource Management
- **Not Properly Destroying Objects**: Memory leaks from undestroyed publishers/subscribers
- **Too Many Callbacks**: Overloading the main thread with callbacks
- **Not Managing QoS Properly**: Using inappropriate QoS settings for the use case

### Best Practices

#### Communication Pattern Selection
- **Topics**: For streaming data like sensor readings, joint states, or status updates
- **Services**: For queries and operations that return results immediately
- **Actions**: For long-running tasks that need feedback or can be canceled

#### Error Handling
- Always check if services are available before calling them
- Implement proper timeout handling
- Check for action cancellation during execution
- Use try-catch blocks for robust error handling

#### Performance Optimization
- Use appropriate QoS settings for different data types
- Implement proper threading for CPU-intensive operations
- Use efficient message types and avoid oversized messages
- Implement data filtering and decimation where appropriate

## Why These Communication Primitives Matter for Humanoid Robotics

### Real-time Requirements
Humanoid robots have strict timing requirements:
- **High-frequency control**: Joint position updates at 100Hz+
- **Sensor fusion**: IMU, vision, and force sensor data integration
- **Balance control**: Continuous monitoring and adjustment
- **Safety systems**: Immediate response to dangerous situations

### Distributed Architecture
Humanoid robots typically have multiple computational units:
- **Real-time controllers**: For balance and joint control
- **Perception systems**: For vision, audio, and environmental understanding
- **Planning systems**: For motion and task planning
- **High-level decision making**: For behavior and interaction

### Safety and Reliability
The communication primitives ensure safe operation:
- **Redundant communication**: Multiple ways to send critical commands
- **Feedback mechanisms**: Continuous monitoring of system state
- **Emergency procedures**: Immediate stop capabilities
- **Graceful degradation**: System continues operation with partial failures

The combination of nodes, topics, services, and actions provides the flexible, robust communication infrastructure needed for complex humanoid robot systems that must operate safely in real-world environments.