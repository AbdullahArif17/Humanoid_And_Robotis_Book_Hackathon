---
sidebar_position: 4
---

# rclpy and Python Agents

## Introduction to rclpy

rclpy is the Python client library for ROS 2, providing a Pythonic interface to the ROS 2 ecosystem. It allows Python developers to create ROS 2 nodes, publish and subscribe to topics, provide and use services, and create and interact with actions. For humanoid robotics applications, rclpy is particularly valuable due to Python's strengths in rapid prototyping, AI/ML integration, and scientific computing.

### Why rclpy for Humanoid Robotics

Python's ecosystem makes rclpy ideal for humanoid robotics development:

- **AI/ML Integration**: Easy integration with TensorFlow, PyTorch, scikit-learn
- **Scientific Computing**: NumPy, SciPy for mathematical operations
- **Computer Vision**: OpenCV, PIL for image processing
- **Rapid Prototyping**: Quick development and testing of algorithms
- **Data Analysis**: Pandas, matplotlib for data processing and visualization

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import numpy as np
import time

class PythonHumanoidAgent(Node):
    def __init__(self):
        super().__init__('python_humanoid_agent')

        # Publishers
        self.joint_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        self.status_publisher = self.create_publisher(String, 'agent_status', 10)

        # Subscribers
        self.joint_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_callback,
            10
        )

        # Timer for main control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        # Initialize state
        self.joint_positions = {}
        self.target_positions = {}
        self.control_gains = {'kp': 10.0, 'ki': 0.1, 'kd': 1.0}
        self.integral_error = {}
        self.previous_error = {}

        self.get_logger().info('Python Humanoid Agent initialized')

    def joint_callback(self, msg):
        """Update joint position data"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def control_loop(self):
        """Main control loop using PID control"""
        if not self.joint_positions:
            return

        # Initialize target positions if not set
        if not self.target_positions:
            for joint_name in self.joint_positions.keys():
                self.target_positions[joint_name] = 0.0
                self.integral_error[joint_name] = 0.0
                self.previous_error[joint_name] = 0.0

        # Calculate PID control for each joint
        command_msg = JointState()
        command_msg.name = list(self.joint_positions.keys())
        command_msg.position = []

        for joint_name in self.joint_positions.keys():
            current_pos = self.joint_positions[joint_name]
            target_pos = self.target_positions[joint_name]

            # Calculate error
            error = target_pos - current_pos

            # Update integral and derivative terms
            self.integral_error[joint_name] += error * 0.01  # dt = 0.01s
            derivative = (error - self.previous_error[joint_name]) / 0.01
            self.previous_error[joint_name] = error

            # Calculate PID output
            output = (
                self.control_gains['kp'] * error +
                self.control_gains['ki'] * self.integral_error[joint_name] +
                self.control_gains['kd'] * derivative
            )

            # Apply control output to current position
            new_position = current_pos + output * 0.01  # Scale by time step
            command_msg.position.append(new_position)

        command_msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_publisher.publish(command_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f'Controlling {len(command_msg.position)} joints'
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    agent = PythonHumanoidAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Shutting down Python Humanoid Agent')
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced rclpy Features

### Async/Await Support

rclpy provides async/await support for non-blocking operations, which is crucial for humanoid robots that need to handle multiple tasks concurrently:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from std_msgs.msg import String
import asyncio
import aiohttp
from humanoid_robot_msgs.action import WalkToGoal

class AsyncHumanoidAgent(Node):
    def __init__(self):
        super().__init__('async_humanoid_agent')

        # Publishers and subscribers
        self.status_pub = self.create_publisher(String, 'async_status', 10)

        # Action client for walking
        self.walk_client = ActionClient(self, WalkToGoal, 'walk_to_goal')

        # Timer for async operations
        self.async_timer = self.create_timer(1.0, self.async_operations)

        self.get_logger().info('Async Humanoid Agent initialized')

    def async_operations(self):
        """Initiate async operations"""
        # Run async tasks in the background
        future = asyncio.run_coroutine_threadsafe(
            self.background_tasks(),
            rclpy.asyncio.get_event_loop()
        )
        future.add_done_callback(self.async_task_complete)

    async def background_tasks(self):
        """Perform background tasks asynchronously"""
        # Example: Fetch external data while robot operates
        try:
            # Simulate fetching environmental data
            env_data = await self.fetch_environmental_data()
            self.get_logger().info(f'Fetched environmental data: {env_data}')

            # Perform other async operations
            await self.monitor_system_health()
            await self.update_behavior_model()

        except Exception as e:
            self.get_logger().error(f'Async task error: {str(e)}')

    async def fetch_environmental_data(self):
        """Simulate fetching data from external sources"""
        # In a real system, this might fetch weather data, map updates, etc.
        await asyncio.sleep(0.5)  # Simulate network delay
        return {'temperature': 22.5, 'humidity': 45, 'pressure': 1013.25}

    async def monitor_system_health(self):
        """Monitor system health asynchronously"""
        await asyncio.sleep(0.1)  # Simulate monitoring delay
        # In a real system, this would check CPU, memory, battery, etc.
        self.get_logger().info('System health check completed')

    async def update_behavior_model(self):
        """Update behavior model asynchronously"""
        await asyncio.sleep(0.2)  # Simulate model update time
        # In a real system, this might update ML models
        self.get_logger().info('Behavior model updated')

    def async_task_complete(self, future):
        """Handle completion of async tasks"""
        try:
            result = future.result()
            self.get_logger().info('Async task completed successfully')
        except Exception as e:
            self.get_logger().error(f'Async task failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    agent = AsyncHumanoidAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Parameter Management

Effective parameter management is crucial for humanoid robots that need to adapt to different situations:

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64MultiArray
import numpy as np

class ParameterizedHumanoidAgent(Node):
    def __init__(self):
        super().__init__('parameterized_humanoid_agent')

        # Declare parameters with default values and descriptions
        self.declare_parameter('control_loop_rate', 100,
                              'Rate of control loop in Hz')
        self.declare_parameter('safety_limits.max_velocity', 1.0,
                              'Maximum joint velocity limit')
        self.declare_parameter('safety_limits.max_torque', 50.0,
                              'Maximum joint torque limit')
        self.declare_parameter('walking_params.step_height', 0.05,
                              'Height of walking steps in meters')
        self.declare_parameter('walking_params.step_length', 0.3,
                              'Length of walking steps in meters')
        self.declare_parameter('balance_params.com_threshold', 0.05,
                              'Center of mass threshold for balance')
        self.declare_parameter('behavior.active', True,
                              'Whether behavior system is active')
        self.declare_parameter('debug.enabled', False,
                              'Enable debug output')

        # Create publishers
        self.control_pub = self.create_publisher(Float64MultiArray, 'control_commands', 10)

        # Timer for parameter-dependent operations
        self.param_timer = self.create_timer(0.1, self.param_dependent_operations)

        # Initialize with current parameter values
        self.update_from_parameters()

        # Add parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info('Parameterized Humanoid Agent initialized')

    def parameter_callback(self, params):
        """Handle parameter changes"""
        successful = True
        reason = ''

        for param in params:
            if param.name == 'control_loop_rate':
                if param.value <= 0:
                    successful = False
                    reason = 'Control loop rate must be positive'
                    break
            elif param.name.startswith('safety_limits.'):
                if param.value < 0:
                    successful = False
                    reason = 'Safety limits must be non-negative'
                    break

        if successful:
            self.update_from_parameters()
            self.get_logger().info(f'Parameters updated: {[p.name for p in params]}')
        else:
            self.get_logger().error(f'Parameter update rejected: {reason}')

        return successful

    def update_from_parameters(self):
        """Update internal state from current parameters"""
        self.control_rate = self.get_parameter('control_loop_rate').value
        self.max_velocity = self.get_parameter('safety_limits.max_velocity').value
        self.max_torque = self.get_parameter('safety_limits.max_torque').value
        self.step_height = self.get_parameter('walking_params.step_height').value
        self.step_length = self.get_parameter('walking_params.step_length').value
        self.com_threshold = self.get_parameter('balance_params.com_threshold').value
        self.behavior_active = self.get_parameter('behavior.active').value
        self.debug_enabled = self.get_parameter('debug.enabled').value

        # Update timer rate if needed
        self.param_timer.timer_period_ns = int(1e9 / self.control_rate)

        if self.debug_enabled:
            self.get_logger().info('Debug mode enabled')

    def param_dependent_operations(self):
        """Operations that depend on parameters"""
        # Generate control commands based on current parameters
        control_msg = Float64MultiArray()

        if self.behavior_active:
            # Generate walking pattern based on parameters
            commands = self.generate_walking_pattern()
            control_msg.data = commands
            self.control_pub.publish(control_msg)

        # Log parameter-dependent information
        if self.debug_enabled:
            self.get_logger().info(
                f'Params - vel: {self.max_velocity}, '
                f'torque: {self.max_torque}, '
                f'step: {self.step_length}x{self.step_height}'
            )

    def generate_walking_pattern(self):
        """Generate walking pattern based on parameters"""
        # Simplified walking pattern generation
        phase = rclpy.clock.Clock().now().nanoseconds / 1e9  # Time in seconds
        commands = []

        # Generate commands for different joints based on walking parameters
        for i in range(6):  # 6 joints for example
            # Create walking gait pattern
            joint_command = (
                self.step_length * 0.5 * np.sin(phase * 2 * np.pi * 0.5 + i * np.pi / 3) +
                self.step_height * 0.2 * np.sin(phase * 4 * np.pi * 0.5 + i * np.pi / 3)
            )
            commands.append(joint_command)

        return commands

def main(args=None):
    rclpy.init(args=args)
    agent = ParameterizedHumanoidAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Python AI/ML Integration

### Machine Learning with TensorFlow/PyTorch

Integrating AI/ML models into humanoid robots using Python:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

class MLHumanoidAgent(Node):
    def __init__(self):
        super().__init__('ml_humanoid_agent')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers and subscribers
        self.action_pub = self.create_publisher(Float64MultiArray, 'ml_actions', 10)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)

        # Timer for ML inference
        self.ml_timer = self.create_timer(0.1, self.ml_inference_loop)

        # Initialize state
        self.latest_joint_state = None
        self.latest_image = None
        self.ml_model = None

        # Load ML model
        self.load_ml_model()

        self.get_logger().info('ML Humanoid Agent initialized')

    def load_ml_model(self):
        """Load the machine learning model"""
        try:
            # Create a simple neural network for demonstration
            # In practice, you would load a pre-trained model
            self.ml_model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(10,)),  # 10 joint positions
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(6, activation='tanh')  # 6 output actions
            ])

            # Generate dummy data to initialize the model
            dummy_input = np.random.random((1, 10))
            _ = self.ml_model(dummy_input)

            self.get_logger().info('ML model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load ML model: {str(e)}')

    def joint_callback(self, msg):
        """Update with latest joint state"""
        self.latest_joint_state = msg

    def image_callback(self, msg):
        """Process incoming image"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')

    def ml_inference_loop(self):
        """Perform ML inference and generate actions"""
        if self.latest_joint_state is None or self.ml_model is None:
            return

        # Prepare input data for the model
        joint_positions = list(self.latest_joint_state.position)

        # Pad or truncate to fixed size (10 joints for this example)
        if len(joint_positions) < 10:
            joint_positions.extend([0.0] * (10 - len(joint_positions)))
        elif len(joint_positions) > 10:
            joint_positions = joint_positions[:10]

        # Add some contextual information (time-based for demonstration)
        time_context = rclpy.clock.Clock().now().nanoseconds / 1e9
        input_data = np.array([joint_positions + [time_context]])  # Add time as extra feature

        try:
            # Run inference
            actions = self.ml_model(input_data).numpy()[0]

            # Publish actions
            action_msg = Float64MultiArray()
            action_msg.data = actions.tolist()
            self.action_pub.publish(action_msg)

            if len(actions) > 0:
                self.get_logger().info(f'ML actions: {[f"{a:.3f}" for a in actions[:3]]}...')

        except Exception as e:
            self.get_logger().error(f'ML inference error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    agent = MLHumanoidAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Reinforcement Learning Agent

Implementing a reinforcement learning agent for humanoid control:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import numpy as np
import random
from collections import deque

class RLHumanoidAgent(Node):
    def __init__(self):
        super().__init__('rl_humanoid_agent')

        # Publishers and subscribers
        self.action_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)

        # Timer for RL loop
        self.rl_timer = self.create_timer(0.1, self.rl_loop)

        # RL components
        self.q_table = {}  # State-action value table
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.experience_buffer = deque(maxlen=10000)

        # State variables
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.joint_state = None
        self.imu_data = None

        self.get_logger().info('RL Humanoid Agent initialized')

    def joint_callback(self, msg):
        """Update joint state"""
        self.joint_state = msg

    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_data = msg

    def get_state(self):
        """Extract state representation from sensor data"""
        if self.joint_state is None or self.imu_data is None:
            return None

        # Create a simple state representation
        # This could include joint positions, velocities, IMU readings, etc.
        joint_positions = list(self.joint_state.position[:5])  # Use first 5 joints
        imu_orient = [
            self.imu_data.orientation.x,
            self.imu_data.orientation.y,
            self.imu_data.orientation.z,
            self.imu_data.orientation.w
        ]

        # Simplified state: joint positions + orientation
        state = tuple(
            [round(pos, 2) for pos in joint_positions] +  # Discretize joint positions
            [round(orient, 2) for orient in imu_orient]    # Discretize orientation
        )

        return state

    def get_reward(self):
        """Calculate reward based on current state"""
        if self.imu_data is None:
            return 0.0

        # Reward for maintaining upright position
        # The quaternion represents orientation, where (0,0,0,1) is upright
        orient = self.imu_data.orientation
        upright_reward = 1.0 - abs(orient.z)  # Higher reward for staying upright

        # Penalty for excessive tilt
        tilt_penalty = -min(abs(orient.x), 0.5) - min(abs(orient.y), 0.5)

        # Small time penalty to encourage efficiency
        time_penalty = -0.01

        total_reward = upright_reward + tilt_penalty + time_penalty
        return max(min(total_reward, 1.0), -1.0)  # Clamp between -1 and 1

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore: choose random action
            action = random.randint(0, 3)  # 4 possible actions: forward, backward, left, right
        else:
            # Exploit: choose best known action
            if state not in self.q_table:
                self.q_table[state] = [0.0, 0.0, 0.0, 0.0]  # Initialize Q-values

            action = np.argmax(self.q_table[state])

        return action

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning algorithm"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]

        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0, 0.0, 0.0]

        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])

        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state][action] = new_q

    def rl_loop(self):
        """Main reinforcement learning loop"""
        current_state = self.get_state()
        if current_state is None:
            return

        # Choose action
        action = self.choose_action(current_state)

        # Execute action (publish command)
        cmd_vel = Twist()
        if action == 0:  # Move forward
            cmd_vel.linear.x = 0.2
        elif action == 1:  # Move backward
            cmd_vel.linear.x = -0.2
        elif action == 2:  # Turn left
            cmd_vel.angular.z = 0.3
        elif action == 3:  # Turn right
            cmd_vel.angular.z = -0.3

        self.action_pub.publish(cmd_vel)

        # Calculate reward
        reward = self.get_reward()

        # Update Q-table if we have a previous state-action pair
        if self.previous_state is not None and self.previous_action is not None:
            self.update_q_value(self.previous_state, self.previous_action, reward, current_state)

        # Store current state and action for next iteration
        self.previous_state = current_state
        self.previous_action = action

        # Log learning progress periodically
        if len(self.q_table) % 100 == 0:
            self.get_logger().info(f'Q-table size: {len(self.q_table)}, epsilon: {self.epsilon:.3f}')

def main(args=None):
    rclpy.init(args=args)
    agent = RLHumanoidAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Saving Q-table...')
        # In a real implementation, you would save the Q-table to a file
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Multi-threading and Concurrency

Handling multiple tasks concurrently in Python agents:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

class ConcurrentHumanoidAgent(Node):
    def __init__(self):
        super().__init__('concurrent_humanoid_agent')

        # Publishers
        self.status_pub = self.create_publisher(String, 'concurrent_status', 10)
        self.joint_pub = self.create_publisher(JointState, 'processed_joints', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10
        )

        # Thread-safe queues for inter-thread communication
        self.joint_queue = queue.Queue(maxsize=10)
        self.processed_queue = queue.Queue(maxsize=10)

        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Start processing threads
        self.processing_thread = threading.Thread(target=self.process_joint_data, daemon=True)
        self.processing_thread.start()

        self.publishing_thread = threading.Thread(target=self.publish_processed_data, daemon=True)
        self.publishing_thread.start()

        # Timer for status updates
        self.status_timer = self.create_timer(1.0, self.update_status)

        self.get_logger().info('Concurrent Humanoid Agent initialized')

    def joint_callback(self, msg):
        """Thread-safe callback to add data to queue"""
        try:
            self.joint_queue.put_nowait(msg)  # Non-blocking put
        except queue.Full:
            self.get_logger().warn('Joint queue full, dropping message')

    def process_joint_data(self):
        """Background thread for processing joint data"""
        while rclpy.ok():
            try:
                # Get joint data from queue (blocking with timeout)
                joint_msg = self.joint_queue.get(timeout=1.0)

                # Simulate processing (in real system, this might be filtering, transformation, etc.)
                processed_msg = self.apply_signal_processing(joint_msg)

                # Put processed data in output queue
                try:
                    self.processed_queue.put_nowait(processed_msg)
                except queue.Full:
                    self.get_logger().warn('Processed queue full, dropping message')

                self.joint_queue.task_done()

            except queue.Empty:
                continue  # Timeout occurred, continue loop
            except Exception as e:
                self.get_logger().error(f'Processing error: {str(e)}')

    def apply_signal_processing(self, joint_msg):
        """Apply signal processing to joint data"""
        processed_msg = JointState()
        processed_msg.header = joint_msg.header
        processed_msg.name = joint_msg.name

        # Apply filtering to joint positions (simple example)
        processed_msg.position = [
            pos * 0.9 + prev_pos * 0.1 if hasattr(self, 'prev_positions') and i < len(self.prev_positions)
            else pos
            for i, pos in enumerate(joint_msg.position)
        ]

        # Store for next iteration
        if not hasattr(self, 'prev_positions'):
            self.prev_positions = joint_msg.position
        else:
            self.prev_positions = processed_msg.position

        processed_msg.velocity = joint_msg.velocity
        processed_msg.effort = joint_msg.effort

        return processed_msg

    def publish_processed_data(self):
        """Background thread for publishing processed data"""
        while rclpy.ok():
            try:
                # Get processed data from queue
                processed_msg = self.processed_queue.get(timeout=1.0)

                # Publish in the main thread context
                self.get_logger().debug('Publishing processed joint data')
                self.joint_pub.publish(processed_msg)

                self.processed_queue.task_done()

            except queue.Empty:
                continue  # Timeout occurred, continue loop
            except Exception as e:
                self.get_logger().error(f'Publishing error: {str(e)}')

    def update_status(self):
        """Update status information"""
        status_msg = String()
        status_msg.data = f'Queue sizes: joint={self.joint_queue.qsize()}, processed={self.processed_queue.qsize()}'
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    agent = ConcurrentHumanoidAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Shutting down concurrent agent')
    finally:
        agent.executor.shutdown(wait=True)
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Common Mistakes and Best Practices

### Common Mistakes

#### Threading Issues
- **Race Conditions**: Multiple threads accessing shared data without proper synchronization
- **Blocking Operations**: Performing blocking operations in the main thread
- **Improper Resource Cleanup**: Not properly shutting down threads or executors

#### Memory Management
- **Memory Leaks**: Not properly destroying publishers/subscribers
- **Large Message Handling**: Processing oversized messages without proper management
- **Model Loading**: Loading large ML models repeatedly instead of once

#### Performance Issues
- **Inefficient Callbacks**: Performing heavy computations in callbacks
- **Wrong QoS Settings**: Using inappropriate QoS for the use case
- **Poor Parameter Validation**: Not validating parameters properly

### Best Practices

#### Threading Best Practices
- Use thread-safe queues for inter-thread communication
- Keep main thread lightweight with minimal processing
- Use ThreadPoolExecutor for managing worker threads
- Always implement proper cleanup in shutdown

#### Memory Management
- Use generators for large data processing
- Implement proper message buffering
- Load models once during initialization
- Use weak references where appropriate

#### Performance Optimization
- Use appropriate QoS settings for different data types
- Implement data decimation for high-frequency streams
- Use efficient data structures (NumPy arrays instead of Python lists)
- Profile and optimize critical paths

## Python Agent Design Patterns

### Behavior Tree Agent

Implementing a behavior tree for complex humanoid behaviors:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import time

class BehaviorTreeAgent(Node):
    def __init__(self):
        super().__init__('behavior_tree_agent')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'behavior_status', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)

        # Timer for behavior execution
        self.behavior_timer = self.create_timer(0.1, self.execute_behavior)

        # Initialize behavior tree
        self.root_behavior = self.create_behavior_tree()
        self.joint_state = None

        self.get_logger().info('Behavior Tree Agent initialized')

    def create_behavior_tree(self):
        """Create the behavior tree structure"""
        return Sequence([
            CheckBalanceCondition(self),
            Selector([
                Fallback([
                    ApproachTargetAction(self),
                    WanderAction(self)
                ]),
                EmergencyStopCondition(self)
            ])
        ])

    def joint_callback(self, msg):
        """Update joint state"""
        self.joint_state = msg

    def execute_behavior(self):
        """Execute the behavior tree"""
        if self.root_behavior:
            status = self.root_behavior.tick()

            status_msg = String()
            status_msg.data = f'Behavior status: {status}'
            self.status_pub.publish(status_msg)

class BehaviorNode:
    """Base class for behavior tree nodes"""
    def __init__(self, agent):
        self.agent = agent

    def tick(self):
        """Execute the behavior and return status"""
        raise NotImplementedError

class Sequence(BehaviorNode):
    """Sequence node - executes children in order until one fails"""
    def __init__(self, children):
        self.children = children
        self.current_child_idx = 0

    def tick(self):
        for i in range(self.current_child_idx, len(self.children)):
            child_status = self.children[i].tick()

            if child_status == 'failure':
                self.current_child_idx = 0
                return 'failure'
            elif child_status == 'running':
                self.current_child_idx = i
                return 'running'
            # If success, continue to next child

        self.current_child_idx = 0
        return 'success'

class Selector(BehaviorNode):
    """Selector node - executes children until one succeeds"""
    def __init__(self, children):
        self.children = children
        self.current_child_idx = 0

    def tick(self):
        for i in range(self.current_child_idx, len(self.children)):
            child_status = self.children[i].tick()

            if child_status == 'success':
                self.current_child_idx = 0
                return 'success'
            elif child_status == 'running':
                self.current_child_idx = i
                return 'running'
            # If failure, try next child

        self.current_child_idx = 0
        return 'failure'

class Fallback(BehaviorNode):
    """Fallback node - similar to selector but called differently"""
    def __init__(self, children):
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status != 'failure':
                return status
        return 'failure'

class CheckBalanceCondition(BehaviorNode):
    """Check if robot is balanced"""
    def tick(self):
        if self.agent.joint_state is None:
            return 'failure'

        # Simple balance check (in real system, use IMU data)
        joint_positions = self.agent.joint_state.position
        if len(joint_positions) > 0 and abs(joint_positions[0]) < 1.0:
            return 'success'
        return 'failure'

class ApproachTargetAction(BehaviorNode):
    """Approach a target location"""
    def tick(self):
        # Publish command to approach target
        cmd = Twist()
        cmd.linear.x = 0.2  # Move forward
        cmd.angular.z = 0.0
        self.agent.cmd_pub.publish(cmd)

        # For demo, return running for a while then success
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()

        if time.time() - self.start_time < 2.0:  # Run for 2 seconds
            return 'running'
        else:
            delattr(self, 'start_time')
            return 'success'

class WanderAction(BehaviorNode):
    """Wander behavior"""
    def tick(self):
        cmd = Twist()
        cmd.linear.x = 0.1
        cmd.angular.z = 0.1  # Gentle turn
        self.agent.cmd_pub.publish(cmd)
        return 'running'  # Always running

class EmergencyStopCondition(BehaviorNode):
    """Emergency stop if needed"""
    def tick(self):
        # Check for emergency conditions
        if self.agent.joint_state and len(self.agent.joint_state.position) > 0:
            if abs(self.agent.joint_state.position[0]) > 2.0:  # Dangerous position
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.agent.cmd_pub.publish(cmd)
                return 'success'  # Trigger emergency stop

        return 'failure'  # Don't trigger unless needed

def main(args=None):
    rclpy.init(args=args)
    agent = BehaviorTreeAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Why rclpy Matters for Humanoid Robotics

### Rapid Prototyping and Development
Python's simplicity and rich ecosystem enable rapid development of humanoid robot behaviors:
- Quick algorithm prototyping and testing
- Easy integration with AI/ML frameworks
- Rich debugging and visualization tools
- Extensive community support and libraries

### AI/ML Integration
Python's dominance in AI/ML makes rclpy ideal for intelligent humanoid systems:
- Seamless integration with TensorFlow, PyTorch, scikit-learn
- Easy deployment of neural networks for perception and control
- Access to pre-trained models and transfer learning
- Advanced data processing and analysis capabilities

### Scientific Computing
Python's scientific computing stack is essential for robotics:
- NumPy and SciPy for mathematical operations
- Pandas for data analysis and processing
- Matplotlib for visualization and debugging
- Jupyter notebooks for experimentation and documentation

rclpy provides the bridge between Python's powerful ecosystem and ROS 2's robust robotics framework, making it an essential tool for developing sophisticated humanoid robot systems.