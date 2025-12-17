---
sidebar_position: 2
---

# ROS 2 Architecture

## Overview of ROS 2 Architecture

ROS 2 employs a distributed architecture based on the Data Distribution Service (DDS) standard, which provides a middleware layer for communication between nodes. This architecture enables robust, scalable, and real-time communication across different platforms and programming languages.

### DDS-Based Communication Layer

The core of ROS 2's architecture is built on DDS (Data Distribution Service), which provides:

- **Decentralized Communication**: No single master node as in ROS 1
- **Publisher-Subscriber Model**: Asynchronous message passing
- **Service-Client Model**: Synchronous request-response communication
- **Quality of Service (QoS)**: Configurable communication policies

```python
# Example: Understanding ROS 2 architecture components
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class ArchitectureDemoNode(Node):
    def __init__(self):
        super().__init__('architecture_demo_node')

        # DDS-based publisher with specific QoS settings
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.publisher = self.create_publisher(String, 'architecture_demo', qos_profile)
        self.subscription = self.create_subscription(
            String,
            'architecture_demo',
            self.listener_callback,
            qos_profile
        )

        self.get_logger().info('ROS 2 Architecture Demo Node Initialized')

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = ArchitectureDemoNode()

    # Example message to demonstrate DDS communication
    msg = String()
    msg.data = 'DDS Communication Test'
    node.publisher.publish(msg)

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

## Client Library Architecture

ROS 2 supports multiple client libraries (rcl) that provide language-specific interfaces while sharing the same underlying architecture:

### rclcpp (C++)
- **Performance**: Optimized for high-performance applications
- **Real-time**: Suitable for time-critical control systems
- **Integration**: Direct integration with C++ robotics libraries

### rclpy (Python)
- **Rapid Development**: Quick prototyping and development
- **Scientific Computing**: Integration with NumPy, SciPy, etc.
- **AI/ML**: Easy integration with machine learning frameworks

### Architecture Layer Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                        Application Layer                    │
├─────────────────────────────────────────────────────────────┤
│              Language-Specific Client Library (rcl)         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │  rclcpp  │ │   rclpy  │ │  rcljava │ │  rclnode │      │
│  │   (C++)  │ │  (Python)│ │  (Java)  │ │ (Node.js)│      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    Middleware Layer (DDS)                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │    Fast DDS │ Cyclone DDS │ RTI Connext │ Eclipse Zenoh││
│  │     (Default)    │     │           │              │    ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                      OS Abstraction                         │
└─────────────────────────────────────────────────────────────┘
```

## Distributed Node Architecture

In ROS 2, nodes can be distributed across multiple machines, connected through the network:

```python
# Publisher node (could run on Robot Computer A)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Publish sensor data processed by perception system
        self.sensor_pub = self.create_publisher(JointState, '/processed_sensors', 10)

        # Simulate processing sensor data
        self.timer = self.create_timer(0.033, self.process_sensors)  # ~30Hz

    def process_sensors(self):
        msg = JointState()
        msg.name = ['hip_joint', 'knee_joint', 'ankle_joint']
        msg.position = [0.1, 0.2, 0.3]
        msg.header.stamp = self.get_clock().now().to_msg()

        self.sensor_pub.publish(msg)
        self.get_logger().info('Published processed sensor data')

# Subscriber node (could run on Robot Computer B)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # Subscribe to processed sensor data
        self.sensor_sub = self.create_subscription(
            JointState,
            '/processed_sensors',
            self.control_callback,
            10
        )

    def control_callback(self, msg):
        self.get_logger().info(f'Received sensor data for control: {msg.position}')
        # Process sensor data for control decisions
        self.compute_control_commands(msg)

    def compute_control_commands(self, sensor_data):
        # Compute control commands based on sensor data
        commands = [pos * 0.95 for pos in sensor_data.position]  # Example control law
        self.get_logger().info(f'Computed control commands: {commands}')

def main(args=None):
    rclpy.init(args=args)

    # Could run on different machines/computers
    perception_node = PerceptionNode()
    control_node = ControlNode()

    try:
        rclpy.spin_multi_threaded([perception_node, control_node])
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        control_node.destroy_node()
        rclpy.shutdown()
```

## Communication Patterns

### Publisher-Subscriber Pattern
The most common communication pattern in ROS 2, suitable for streaming data:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

class PublisherSubscriberDemo(Node):
    def __init__(self):
        super().__init__('pubsub_demo')

        # Publisher for joint angles
        self.joint_pub = self.create_publisher(Float64MultiArray, '/joint_angles', 10)

        # Publisher for velocity commands
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.imu_sub = self.create_subscription(
            Float64MultiArray,
            '/imu_data',
            self.imu_callback,
            10
        )

        self.vision_sub = self.create_subscription(
            Float64MultiArray,
            '/vision_targets',
            self.vision_callback,
            10
        )

        # Timer for publishing
        self.timer = self.create_timer(0.1, self.publish_data)

    def publish_data(self):
        # Publish joint angles
        joint_msg = Float64MultiArray()
        joint_msg.data = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example joint angles
        self.joint_pub.publish(joint_msg)

        # Publish velocity command
        vel_msg = Twist()
        vel_msg.linear.x = 0.5
        vel_msg.angular.z = 0.1
        self.vel_pub.publish(vel_msg)

    def imu_callback(self, msg):
        self.get_logger().info(f'IMU data received: {msg.data}')

    def vision_callback(self, msg):
        self.get_logger().info(f'Vision target received: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = PublisherSubscriberDemo()

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

### Service-Client Pattern
Used for request-response communication:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceServer(Node):
    def __init__(self):
        super().__init__('service_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Request: {request.a} + {request.b} = {response.sum}')
        return response

class ServiceClient(Node):
    def __init__(self):
        super().__init__('service_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)

    server = ServiceServer()
    client = ServiceClient()

    # Send a request
    response = client.send_request(1, 2)
    print(f'Result: {response.sum}')

    server.destroy_node()
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Server-Client Pattern
For long-running tasks with feedback:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.action.server import GoalResponse, CancelResponse
from rclpy.action.client import ActionClient
from example_interfaces.action import Fibonacci

class WalkActionServer(Node):
    def __init__(self):
        super().__init__('walk_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'walk_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

    def goal_callback(self, goal_request):
        self.get_logger().info('Received walk goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received walk cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing walk action...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, 10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Walk action canceled')
                return Fibonacci.Result()

            # Update feedback
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1]
            )
            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().info(f'Walk progress: {feedback_msg.sequence[-1]}')

            # Simulate walking delay
            await rclpy.asyncio.sleep(0.5)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence

        self.get_logger().info('Walk action succeeded')
        return result

class WalkActionClient(Node):
    def __init__(self):
        super().__init__('walk_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'walk_action')

    def send_goal(self):
        goal_msg = Fibonacci.Goal()

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info('Sending walk goal...')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Received feedback: {feedback_msg.feedback.sequence[-1]}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    # Server and client in same process for demonstration
    server = WalkActionServer()
    client = WalkActionClient()

    # Allow server to initialize
    import threading
    import time
    time.sleep(1)

    # Send goal from client
    client.send_goal()

    # Spin both nodes
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Parameter Server Architecture

ROS 2 includes a distributed parameter system that allows nodes to share configuration:

```python
import rclpy
from rclpy.node import Node

class ParameterDemoNode(Node):
    def __init__(self):
        super().__init__('parameter_demo_node')

        # Declare parameters with default values
        self.declare_parameter('control_loop_rate', 100)  # Hz
        self.declare_parameter('max_joint_velocity', 1.0)  # rad/s
        self.declare_parameter('robot_name', 'humanoid_robot')
        self.declare_parameter('safety_limits', [0.5, 0.5, 0.5])  # [vel, acc, jerk]

        # Get parameter values
        self.control_rate = self.get_parameter('control_loop_rate').value
        self.max_vel = self.get_parameter('max_joint_velocity').value
        self.robot_name = self.get_parameter('robot_name').value
        self.safety_limits = self.get_parameter('safety_limits').value

        self.get_logger().info(f'Parameters initialized for {self.robot_name}')
        self.get_logger().info(f'Control rate: {self.control_rate}Hz, Max vel: {self.max_vel} rad/s')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterDemoNode()

    # Example: Update parameters programmatically
    node.set_parameters([
        node.create_parameter('max_joint_velocity', 1.5),
        node.create_parameter('robot_name', 'advanced_humanoid')
    ])

    node.get_logger().info('Parameters updated')

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

## Launch System Architecture

ROS 2's launch system provides a declarative way to bring up complex systems:

```python
# launch/humanoid_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_name = LaunchConfiguration('robot_name', default='humanoid_robot')

    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value='humanoid_robot',
            description='Name of the robot'
        ),

        # Perception node
        Node(
            package='humanoid_robot',
            executable='perception_node',
            name='perception_node',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'robot_name': robot_name}
            ],
            remappings=[
                ('/camera/image_raw', '/front_camera/image_raw'),
                ('/imu/data', '/base_imu/data')
            ]
        ),

        # Control node
        Node(
            package='humanoid_robot',
            executable='control_node',
            name='control_node',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'robot_name': robot_name},
                {'control_loop_rate': 100}
            ]
        ),

        # State publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        )
    ])
```

## Lifecycle Nodes

ROS 2 supports lifecycle nodes that have explicit state management:

```python
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSProfile

class LifecycleHumanoidController(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_humanoid_controller')
        self.get_logger().info('Lifecycle Humanoid Controller Created')

    def on_configure(self, state: LifecycleState):
        self.get_logger().info('Configuring lifecycle controller')

        # Create publishers/subscribers in unconfigured state
        self.publisher = self.create_publisher(
            QoSProfile(depth=10),
            'controller_status',
            10
        )

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState):
        self.get_logger().info('Activating lifecycle controller')

        # Activate the publisher
        self.publisher.on_activate()

        # Start control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState):
        self.get_logger().info('Deactivating lifecycle controller')

        # Stop control loop
        self.control_timer.destroy()

        # Deactivate publisher
        self.publisher.on_deactivate()

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState):
        self.get_logger().info('Cleaning up lifecycle controller')

        # Destroy publishers/subscribers
        self.destroy_publisher(self.publisher)

        return TransitionCallbackReturn.SUCCESS

    def control_loop(self):
        # Implement control logic here
        self.get_logger().info('Executing control loop')

def main(args=None):
    rclpy.init(args=args)
    node = LifecycleHumanoidController()

    # Manually trigger transitions for demonstration
    node.trigger_configure()
    node.trigger_activate()

    try:
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

## Common Mistakes in Architecture Design

### Communication Pattern Misuse

#### Using Wrong Communication Pattern
- **Using Services for Streaming Data**: Services are synchronous and blocking, not suitable for high-frequency sensor data
- **Using Publishers for Critical Commands**: Without proper QoS, important commands might be lost
- **Ignoring Feedback Needs**: Using services instead of actions for long-running operations

#### Inappropriate QoS Configuration
- **Overly Conservative QoS**: Using reliable communication for sensor data that can tolerate some loss
- **Insufficient Reliability**: Using best-effort for critical control commands
- **Poor History Policy**: Not considering how much historical data is needed

### Node Architecture Issues

#### Poor Node Decomposition
- **Monolithic Nodes**: Creating nodes that handle too many responsibilities
- **Tight Coupling**: Nodes that are too dependent on each other
- **Inconsistent Interfaces**: Different nodes using different message types for similar data

#### Resource Management
- **Memory Leaks**: Not properly destroying publishers, subscribers, or timers
- **CPU Overload**: Not considering computational requirements per node
- **Network Congestion**: Not optimizing message sizes or frequencies

## Why ROS 2 Architecture Matters for Humanoid Robotics

### Real-time Requirements

Humanoid robots have strict real-time requirements for stability and safety:

- **Control Loop Timing**: Balance control requires 100Hz+ updates
- **Sensor Processing**: Perception systems need consistent processing rates
- **Communication Latency**: Low-latency communication between subsystems
- **Predictable Performance**: Deterministic behavior for safety-critical operations

### Distributed Processing

Humanoid robots often have multiple computational units:

- **Onboard Computers**: For real-time control and local processing
- **Edge Computing**: For AI inference and complex perception
- **Cloud Integration**: For high-level planning and learning
- **Heterogeneous Platforms**: Different hardware with different capabilities

### Safety and Reliability

The architecture must ensure safe operation:

- **Fault Tolerance**: Ability to continue operation with component failures
- **Graceful Degradation**: Reduced functionality rather than complete failure
- **Safety Isolation**: Critical safety functions separate from non-critical ones
- **Recovery Mechanisms**: Automatic recovery from common failure modes

The ROS 2 architecture provides the foundation for building robust, scalable, and maintainable humanoid robot systems. Its distributed nature, real-time capabilities, and comprehensive communication patterns make it well-suited for the complex requirements of Physical AI systems.