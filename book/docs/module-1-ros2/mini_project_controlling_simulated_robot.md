---
sidebar_position: 7
---

# Mini Project: Controlling a Simulated Robot

## Project Overview

In this mini-project, we'll create a complete humanoid robot control system using ROS 2. We'll simulate a humanoid robot in Gazebo, implement a controller node that can make the robot walk and balance, and create a user interface to control the robot's movements. This project will integrate all the concepts learned in Module 1: nodes, topics, services, actions, URDF, launch files, and parameters.

### Learning Objectives

By completing this project, you will:

1. Create a complete humanoid robot URDF model
2. Set up a Gazebo simulation environment
3. Implement a ROS 2 controller node for humanoid locomotion
4. Create launch files to start the complete system
5. Implement parameter management for robot control
6. Use actions for high-level robot commands
7. Integrate perception and control systems

## Project Structure

First, let's set up the project structure:

```
humanoid_robot_project/
├── CMakeLists.txt
├── package.xml
├── config/
│   ├── humanoid_control.yaml
│   └── robot_params.yaml
├── launch/
│   ├── simulation.launch.py
│   └── robot_control.launch.py
├── urdf/
│   └── humanoid.urdf.xacro
├── src/
│   ├── robot_controller.cpp
│   ├── walking_controller.cpp
│   └── perception_node.cpp
└── scripts/
    └── teleop_keyboard.py
```

## Step 1: Creating the Humanoid Robot URDF

Let's create a complete humanoid robot URDF using Xacro:

```xml
<!-- urdf/humanoid.urdf.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="mass_torso" value="8.0" />
  <xacro:property name="mass_head" value="2.0" />
  <xacro:property name="mass_arm" value="1.5" />
  <xacro:property name="mass_leg" value="2.0" />
  <xacro:property name="mass_foot" value="0.8" />
  <xacro:property name="gear_ratio" value="100.0" />

  <!-- Inertial macros -->
  <xacro:macro name="inertial_sphere" params="mass x y z radius">
    <inertial>
      <mass value="${mass}" />
      <origin xyz="${x} ${y} ${z}" />
      <inertia ixx="${0.4 * mass * radius * radius}" ixy="0.0" ixz="0.0"
               iyy="${0.4 * mass * radius * radius}" iyz="0.0"
               izz="${0.4 * mass * radius * radius}" />
    </inertial>
  </xacro:macro>

  <xacro:macro name="inertial_box" params="mass x y z x_size y_size z_size">
    <inertial>
      <mass value="${mass}" />
      <origin xyz="${x} ${y} ${z}" />
      <inertia ixx="${mass / 12.0 * (y_size * y_size + z_size * z_size)}" ixy="0.0" ixz="0.0"
               iyy="${mass / 12.0 * (x_size * x_size + z_size * z_size)}" iyz="0.0"
               izz="${mass / 12.0 * (x_size * x_size + y_size * y_size)}" />
    </inertial>
  </xacro:macro>

  <xacro:macro name="inertial_cylinder" params="mass x y z radius length">
    <inertial>
      <mass value="${mass}" />
      <origin xyz="${x} ${y} ${z}" />
      <inertia ixx="${mass * (3 * radius * radius + length * length) / 12.0}" ixy="0.0" ixz="0.0"
               iyy="${mass * (3 * radius * radius + length * length) / 12.0}" iyz="0.0"
               izz="${mass * radius * radius / 2.0}" />
    </inertial>
  </xacro:macro>

  <!-- Material definitions -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.4235 0.0392 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 0.4"/>
      </geometry>
      <material name="grey"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <inertial>
      <mass value="${mass_torso}"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <inertia ixx="0.15" ixy="0" ixz="0" iyy="0.15" iyz="0" izz="0.15"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="10.0" velocity="2.0"/>
  </joint>

  <link name="head">
    <inertial>
      <mass value="${mass_head}"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Arm Chain -->
  <xacro:macro name="arm_chain" params="side shoulder_pos">
    <!-- Shoulder joint -->
    <joint name="torso_to_${side}_shoulder" type="revolute">
      <parent link="torso"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="${shoulder_pos}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="-2.0" upper="2.0" effort="20.0" velocity="2.0"/>
    </joint>

    <link name="${side}_upper_arm">
      <inertial>
        <mass value="${mass_arm}"/>
        <origin xyz="0 0 0.15" rpy="0 0 0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
      </inertial>

      <visual>
        <origin xyz="0 0 0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
        <material name="blue"/>
      </visual>

      <collision>
        <origin xyz="0 0 0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
      </collision>
    </link>

    <!-- Elbow joint -->
    <joint name="${side}_shoulder_to_elbow" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-2.0" upper="0.5" effort="15.0" velocity="2.0"/>
    </joint>

    <link name="${side}_lower_arm">
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 0.12" rpy="0 0 0"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
      </inertial>

      <visual>
        <origin xyz="0 0 0.12" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.04" length="0.24"/>
        </geometry>
        <material name="blue"/>
      </visual>

      <collision>
        <origin xyz="0 0 0.12" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.04" length="0.24"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Instantiate arms -->
  <xacro:arm_chain side="left" shoulder_pos="0.2 0 0.4"/>
  <xacro:arm_chain side="right" shoulder_pos="-0.2 0 0.4"/>

  <!-- Left Leg Chain -->
  <xacro:macro name="leg_chain" params="side hip_pos">
    <!-- Hip joint -->
    <joint name="torso_to_${side}_hip" type="revolute">
      <parent link="torso"/>
      <child link="${side}_upper_leg"/>
      <origin xyz="${hip_pos}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="-1.57" upper="1.57" effort="50.0" velocity="2.0"/>
    </joint>

    <link name="${side}_upper_leg">
      <inertial>
        <mass value="${mass_leg}"/>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.03"/>
      </inertial>

      <visual>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.06" length="0.4"/>
        </geometry>
        <material name="red"/>
      </visual>

      <collision>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.06" length="0.4"/>
        </geometry>
      </collision>
    </link>

    <!-- Knee joint -->
    <joint name="${side}_hip_to_knee" type="revolute">
      <parent link="${side}_upper_leg"/>
      <child link="${side}_lower_leg"/>
      <origin xyz="0 0 -0.4" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-2.0" upper="0.5" effort="40.0" velocity="2.0"/>
    </joint>

    <link name="${side}_lower_leg">
      <inertial>
        <mass value="1.5"/>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
      </inertial>

      <visual>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
        <material name="red"/>
      </visual>

      <collision>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
      </collision>
    </link>

    <!-- Ankle joint -->
    <joint name="${side}_knee_to_ankle" type="revolute">
      <parent link="${side}_lower_leg"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 -0.3" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="-0.5" upper="0.5" effort="30.0" velocity="2.0"/>
    </joint>

    <link name="${side}_foot">
      <inertial>
        <mass value="${mass_foot}"/>
        <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
        <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/>
      </inertial>

      <visual>
        <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
        <geometry>
          <box size="0.15 0.1 0.05"/>
        </geometry>
        <material name="black"/>
      </visual>

      <collision>
        <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
        <geometry>
          <box size="0.15 0.1 0.05"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Instantiate legs -->
  <xacro:leg_chain side="left" hip_pos="0.1 0 0"/>
  <xacro:leg_chain side="right" hip_pos="-0.1 0 0"/>

  <!-- Gazebo plugins -->
  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
  </gazebo>

  <gazebo reference="torso">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="left_upper_arm">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="right_upper_arm">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_upper_leg">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="right_upper_leg">
    <material>Gazebo/Red</material>
  </gazebo>

  <!-- ROS2 Control plugin -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find humanoid_robot_description)/config/humanoid_control.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- Camera sensor on head -->
  <joint name="head_to_camera" type="fixed">
    <parent link="head"/>
    <child link="camera_link"/>
    <origin xyz="0.1 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <gazebo reference="camera_link">
    <sensor type="camera" name="camera_sensor">
      <update_rate>30</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
        <min_depth>0.1</min_depth>
        <max_depth>100</max_depth>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU sensor in torso -->
  <gazebo reference="torso">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>true</visualize>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
        <topicName>imu/data</topicName>
        <bodyName>torso</bodyName>
        <updateRateHZ>100.0</updateRateHZ>
        <gaussianNoise>0.001</gaussianNoise>
        <xyzOffset>0 0 0</xyzOffset>
        <rpyOffset>0 0 0</rpyOffset>
        <frameName>torso</frameName>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

## Step 2: Creating Control Configuration

Now let's create the control configuration file:

```yaml
# config/humanoid_control.yaml
controller_manager:
  ros__parameters:
    update_rate: 1000  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    torso_to_head_position_controller:
      type: position_controllers/JointGroupPositionController

    left_arm_controller:
      type: position_controllers/JointGroupPositionController

    right_arm_controller:
      type: position_controllers/JointGroupPositionController

    left_leg_controller:
      type: position_controllers/JointGroupPositionController

    right_leg_controller:
      type: position_controllers/JointGroupPositionController

    balance_controller:
      type: effort_controllers/JointGroupEffortController

# Head controller
torso_to_head_position_controller:
  ros__parameters:
    joints:
      - torso_to_head

# Left arm controller
left_arm_controller:
  ros__parameters:
    joints:
      - torso_to_left_shoulder
      - left_shoulder_to_elbow

# Right arm controller
right_arm_controller:
  ros__parameters:
    joints:
      - torso_to_right_shoulder
      - right_shoulder_to_elbow

# Left leg controller
left_leg_controller:
  ros__parameters:
    joints:
      - torso_to_left_hip
      - left_hip_to_knee
      - left_knee_to_ankle

# Right leg controller
right_leg_controller:
  ros__parameters:
    joints:
      - torso_to_right_hip
      - right_hip_to_knee
      - right_knee_to_ankle

# Balance controller (for real-time balance adjustments)
balance_controller:
  ros__parameters:
    joints:
      - torso_to_left_hip
      - torso_to_right_hip
      - left_hip_to_knee
      - right_hip_to_knee
      - left_knee_to_ankle
      - right_knee_to_ankle
```

## Step 3: Implementing the Robot Controller Node

Let's create the main controller node in Python:

```python
# src/humanoid_controller.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from humanoid_robot_msgs.action import WalkToGoal
import numpy as np
import math
import time
from collections import deque

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Initialize parameters
        self.declare_parameter('control_loop_rate', 100)
        self.declare_parameter('balance_kp', 100.0)
        self.declare_parameter('balance_kd', 10.0)
        self.declare_parameter('walking_step_height', 0.05)
        self.declare_parameter('walking_step_length', 0.3)
        self.declare_parameter('walking_step_duration', 1.0)

        self.control_rate = self.get_parameter('control_loop_rate').value
        self.balance_kp = self.get_parameter('balance_kp').value
        self.balance_kd = self.get_parameter('balance_kd').value
        self.step_height = self.get_parameter('walking_step_height').value
        self.step_length = self.get_parameter('walking_step_length').value
        self.step_duration = self.get_parameter('walking_step_duration').value

        # Robot state
        self.joint_states = {}
        self.imu_data = None
        self.target_positions = {}
        self.current_trajectory = None
        self.trajectory_start_time = None
        self.trajectory_index = 0
        self.is_walking = False
        self.walk_target = None
        self.robot_position = [0.0, 0.0, 0.0]  # x, y, theta

        # Control variables
        self.balance_error_history = deque(maxlen=10)
        self.previous_balance_error = 0.0

        # Create publishers
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Create subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        self.imu_sub = self.create_subscription(
            Imu,
            'imu/data',
            self.imu_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        # Create action servers
        self.walk_action_server = ActionServer(
            self,
            WalkToGoal,
            'walk_to_goal',
            execute_callback=self.walk_execute_callback,
            goal_callback=self.walk_goal_callback,
            cancel_callback=self.walk_cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        # Create service servers
        self.balance_service = self.create_service(
            Empty,
            'enable_balance_control',
            self.enable_balance_control
        )

        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop
        )

        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        """Update joint state information"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_states[name] = {
                    'position': msg.position[i],
                    'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                    'effort': msg.effort[i] if i < len(msg.effort) else 0.0
                }

    def imu_callback(self, msg):
        """Update IMU data for balance control"""
        self.imu_data = msg

    def walk_goal_callback(self, goal_request):
        """Accept or reject walk goals"""
        self.get_logger().info(f'Received walk goal: {goal_request.target_pose}')
        return GoalResponse.ACCEPT

    def walk_cancel_callback(self, goal_handle):
        """Handle walk goal cancellation"""
        self.get_logger().info('Walk goal canceled')
        return CancelResponse.ACCEPT

    async def walk_execute_callback(self, goal_handle):
        """Execute walk to goal action"""
        self.get_logger().info('Executing walk to goal')

        target_x = goal_handle.request.target_pose.pose.position.x
        target_y = goal_handle.request.target_pose.pose.position.y

        # Calculate distance to target
        current_x, current_y, _ = self.robot_position
        distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)

        # Generate walking trajectory
        steps = int(distance / self.step_length) + 1
        step_size = distance / steps if steps > 0 else 0

        feedback_msg = WalkToGoal.Feedback()
        result = WalkToGoal.Result()

        for step in range(steps):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = 'Goal canceled'
                return result

            # Calculate intermediate target
            progress = (step + 1) / steps
            intermediate_x = current_x + (target_x - current_x) * progress
            intermediate_y = current_y + (target_y - current_y) * progress

            # Execute step (simplified)
            self.execute_step(intermediate_x, intermediate_y)

            # Update feedback
            feedback_msg.distance_remaining = distance * (1 - progress)
            feedback_msg.progress_percentage = progress * 100.0
            goal_handle.publish_feedback(feedback_msg)

            # Wait for step completion
            await rclpy.asyncio.sleep(0.5)

        # Check if reached target
        final_distance = math.sqrt(
            (target_x - self.robot_position[0])**2 +
            (target_y - self.robot_position[1])**2
        )

        result.success = final_distance < 0.2  # Within 20cm
        result.message = f'Reached target with final distance: {final_distance:.3f}m'
        result.final_pose.pose.position.x = self.robot_position[0]
        result.final_pose.pose.position.y = self.robot_position[1]

        goal_handle.succeed()
        return result

    def execute_step(self, target_x, target_y):
        """Execute a single walking step"""
        # Simplified walking step execution
        # In a real implementation, this would generate proper walking patterns
        self.get_logger().info(f'Executing step toward ({target_x:.2f}, {target_y:.2f})')

        # Update robot position (simplified)
        self.robot_position[0] = target_x
        self.robot_position[1] = target_y

    def enable_balance_control(self, request, response):
        """Enable or disable balance control"""
        # In a real implementation, this would toggle balance control
        self.get_logger().info('Balance control service called')
        return response

    def control_loop(self):
        """Main control loop running at specified rate"""
        if not self.joint_states or not self.imu_data:
            return

        # Update robot position based on IMU and joint data
        self.update_robot_position()

        # Apply balance control if enabled
        self.apply_balance_control()

        # Execute any active trajectories
        self.execute_active_trajectory()

        # Check for stability
        self.check_stability()

    def update_robot_position(self):
        """Update robot position estimate"""
        # Simplified position update based on joint integration
        # In a real system, this would use odometry or visual-inertial fusion
        pass

    def apply_balance_control(self):
        """Apply balance control based on IMU data"""
        if not self.imu_data:
            return

        # Extract roll and pitch from IMU quaternion
        quat = self.imu_data.orientation
        roll, pitch = self.quaternion_to_rpy([quat.x, quat.y, quat.z, quat.w])[0:2]

        # Calculate balance error (simplified - would use ZMP in real system)
        balance_error = pitch  # Using pitch as simple balance measure

        # Store error for derivative calculation
        self.balance_error_history.append(balance_error)

        # Calculate PID control (simplified)
        p_term = self.balance_kp * balance_error

        if len(self.balance_error_history) > 1:
            d_term = self.balance_kd * (balance_error - self.previous_balance_error)
        else:
            d_term = 0.0

        self.previous_balance_error = balance_error

        control_output = p_term + d_term

        # Apply control to joints (simplified)
        self.apply_balance_torques(control_output)

    def quaternion_to_rpy(self, quat):
        """Convert quaternion to roll-pitch-yaw"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def apply_balance_torques(self, control_output):
        """Apply balance control torques to joints"""
        # Simplified balance control application
        # In a real system, this would calculate appropriate joint torques
        # based on the control output and robot dynamics
        pass

    def execute_active_trajectory(self):
        """Execute any active joint trajectories"""
        if self.current_trajectory and self.trajectory_start_time:
            # Calculate progress in trajectory
            elapsed_time = self.get_clock().now().nanoseconds / 1e9 - self.trajectory_start_time
            total_duration = self.current_trajectory.points[-1].time_from_start.sec + \
                           self.current_trajectory.points[-1].time_from_start.nanosec / 1e9

            if elapsed_time >= total_duration:
                # Trajectory completed
                self.current_trajectory = None
                self.trajectory_start_time = None
                self.trajectory_index = 0
                return

            # Interpolate between trajectory points
            self.interpolate_trajectory(elapsed_time)

    def interpolate_trajectory(self, elapsed_time):
        """Interpolate current position from trajectory"""
        # Find current trajectory segment
        for i, point in enumerate(self.current_trajectory.points):
            point_time = point.time_from_start.sec + point.time_from_start.nanosec / 1e9

            if point_time > elapsed_time:
                if i == 0:
                    # Use first point
                    self.send_joint_commands(self.current_trajectory.points[0])
                else:
                    # Interpolate between previous and current point
                    prev_point = self.current_trajectory.points[i-1]
                    prev_time = prev_point.time_from_start.sec + prev_point.time_from_start.nanosec / 1e9

                    if point_time > prev_time:
                        alpha = (elapsed_time - prev_time) / (point_time - prev_time)
                        interpolated_point = self.interpolate_points(prev_point, point, alpha)
                        self.send_joint_commands(interpolated_point)
                break

    def interpolate_points(self, point1, point2, alpha):
        """Linearly interpolate between two trajectory points"""
        result = JointTrajectoryPoint()
        result.positions = [
            p1 + alpha * (p2 - p1)
            for p1, p2 in zip(point1.positions, point2.positions)
        ]
        result.velocities = [
            v1 + alpha * (v2 - v1)
            for v1, v2 in zip(point1.velocities, point2.velocities)
        ] if point1.velocities and point2.velocities else []
        result.accelerations = [
            a1 + alpha * (a2 - a1)
            for a1, a2 in zip(point1.accelerations, point2.accelerations)
        ] if point1.accelerations and point2.accelerations else []

        # Set time to match elapsed time
        total_time = Duration()
        total_time.sec = int(elapsed_time)
        total_time.nanosec = int((elapsed_time - int(elapsed_time)) * 1e9)
        result.time_from_start = total_time

        return result

    def send_joint_commands(self, point):
        """Send joint position commands"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.current_trajectory.joint_names
        trajectory.points = [point]

        self.joint_trajectory_pub.publish(trajectory)

    def check_stability(self):
        """Check robot stability and trigger safety measures if needed"""
        if not self.imu_data:
            return

        # Check if robot is falling (simplified)
        quat = self.imu_data.orientation
        roll, pitch, _ = self.quaternion_to_rpy([quat.x, quat.y, quat.z, quat.w])

        # Emergency stop if tilt is too large
        if abs(roll) > 1.0 or abs(pitch) > 1.0:  # ~57 degrees
            self.emergency_stop()
            self.get_logger().error(f'EMERGENCY STOP: Roll={roll:.3f}, Pitch={pitch:.3f}')

    def emergency_stop(self):
        """Execute emergency stop"""
        self.get_logger().error('Emergency stop activated!')
        # Send zero commands to all joints
        # In a real system, this would engage safety mechanisms

def main(args=None):
    rclpy.init(args=args)

    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Humanoid Controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 4: Creating the Launch Files

Now let's create the launch files to bring up the complete system:

```python
# launch/simulation.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default='empty')
    headless = LaunchConfiguration('headless', default='false')
    gui = LaunchConfiguration('gui', default='true')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world,
            'gui': gui,
            'headless': headless,
            'verbose': 'false'
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'publish_frequency': 50.0}
        ],
        arguments=[PathJoinSubstitution([
            FindPackageShare('humanoid_robot_description'),
            'urdf',
            'humanoid.urdf.xacro'
        ])]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'
        ],
        output='screen'
    )

    # Joint state publisher (for testing)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'rate': 50}
        ]
    )

    # Controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='ros2_control_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('humanoid_robot_description'),
                'config',
                'humanoid_control.yaml'
            ])
        ],
        output='both'
    )

    # Load controllers
    def load_controller(controller_name):
        return Node(
            package='controller_manager',
            executable='spawner.py',
            arguments=[controller_name],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )

    # Load all controllers after spawning
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner.py',
        arguments=['joint_state_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    robot_controller_spawner = Node(
        package='controller_manager',
        executable='spawner.py',
        arguments=['torso_to_head_position_controller',
                  'left_arm_controller',
                  'right_arm_controller',
                  'left_leg_controller',
                  'right_leg_controller',
                  'balance_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'world',
            default_value='empty',
            description='Choose one of the world files from `/gazebo_ros/worlds`'
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Enable headless mode'
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Enable GUI'
        ),

        # Launch Gazebo
        gazebo,

        # Launch robot state publisher
        robot_state_publisher,
        joint_state_publisher,

        # Spawn robot in Gazebo
        spawn_entity,

        # Launch controller manager
        controller_manager,

        # Load controllers
        joint_state_broadcaster_spawner,
        robot_controller_spawner,
    ])
```

```python
# launch/robot_control.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_name = LaunchConfiguration('robot_name', default='humanoid_robot')
    config_file = LaunchConfiguration('config_file', default='')
    enable_vision = LaunchConfiguration('enable_vision', default='false')
    enable_audio = LaunchConfiguration('enable_audio', default='false')

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('humanoid_robot_description'),
                'config',
                'robot_params.yaml'
            ])
        ],
        arguments=[PathJoinSubstitution([
            FindPackageShare('humanoid_robot_description'),
            'urdf',
            'humanoid.urdf.xacro'
        ])]
    )

    # Main humanoid controller
    humanoid_controller = Node(
        package='humanoid_robot_control',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_name': robot_name},
            {'control_loop_rate': 1000},
            {'balance_kp': 150.0},
            {'balance_kd': 15.0},
            {'walking_step_height': 0.05},
            {'walking_step_length': 0.3},
            {'walking_step_duration': 1.0},
        ],
        remappings=[
            ('/joint_states', '/joint_states'),
            ('/imu/data', '/imu/data'),
            ('/cmd_vel', '/cmd_vel'),
        ],
        respawn=True,
        respawn_delay=2.0,
        output='screen'
    )

    # Perception node (conditional)
    perception_node = Node(
        package='humanoid_robot_perception',
        executable='perception_node',
        name='perception_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'camera_topic': '/camera/image_raw'},
            {'detection_model': 'yolov8n.pt'},
        ],
        condition=IfCondition(enable_vision),
        respawn=True
    )

    # Behavior manager
    behavior_manager = Node(
        package='humanoid_robot_behavior',
        executable='behavior_manager',
        name='behavior_manager',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'default_behavior': 'idle'},
            {'behavior_priority': ['emergency', 'balance', 'navigation', 'interaction']},
        ],
        respawn=True
    )

    # RViz2 for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d',
            PathJoinSubstitution([
                FindPackageShare('humanoid_robot_viz'),
                'rviz',
                'humanoid.rviz'
            ])
        ],
        condition=IfCondition(LaunchConfiguration('use_rviz', default='true'))
    )

    # Include simulation launch if needed
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_robot_description'),
                'launch',
                'simulation.launch.py'
            ])
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
        condition=IfCondition(LaunchConfiguration('use_simulation', default='false'))
    )

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value='humanoid_robot',
            description='Name of the robot'
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value='',
            description='Path to custom configuration file'
        ),
        DeclareLaunchArgument(
            'enable_vision',
            default_value='false',
            description='Enable vision processing nodes'
        ),
        DeclareLaunchArgument(
            'enable_audio',
            default_value='false',
            description='Enable audio processing nodes'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz2'
        ),
        DeclareLaunchArgument(
            'use_simulation',
            default_value='false',
            description='Launch simulation environment'
        ),

        # Launch nodes
        robot_state_publisher,
        humanoid_controller,
        perception_node,
        behavior_manager,
        rviz,

        # Simulation launch (if enabled)
        simulation_launch,
    ])
```

## Step 5: Creating a Simple Teleoperation Script

Let's create a keyboard teleoperation script to control the robot:

```python
#!/usr/bin/env python3
# scripts/teleop_keyboard.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import sys
import select
import termios
import tty

msg = """
Control Your Humanoid Robot!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
space key, k : force stop
anything else : stop smoothly

CTRL-C to quit
"""

moveBindings = {
    'i': (1, 0, 0, 0),
    'o': (1, 0, 0, -1),
    'j': (0, 0, 0, 1),
    'l': (0, 0, 0, -1),
    'u': (1, 0, 0, 1),
    ',': (-1, 0, 0, 0),
    '.': (-1, 0, 0, 1),
    'm': (-1, 0, 0, -1),
}

speedBindings = {
    'q': (1.1, 1.1),
    'z': (.9, .9),
    'w': (1.1, 1),
    'x': (.9, 1),
    'e': (1, 1.1),
    'c': (1, .9),
}

class TeleopHumanoid(Node):
    def __init__(self):
        super().__init__('teleop_humanoid')

        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.speed = 0.5
        self.turn = 1.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.th = 0.0
        self.status = 0

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def vels(self, speed, turn):
        return f"currently:\tspeed {speed}\tturn {turn}"

    def run(self):
        self.settings = termios.tcgetattr(sys.stdin)

        self.get_logger().info(msg)
        self.get_logger().info(self.vels(self.speed, self.turn))

        try:
            while True:
                key = self.getKey()

                if key in moveBindings.keys():
                    self.x = moveBindings[key][0]
                    self.y = moveBindings[key][1]
                    self.z = moveBindings[key][2]
                    self.th = moveBindings[key][3]
                elif key in speedBindings.keys():
                    self.speed = self.speed * speedBindings[key][0]
                    self.turn = self.turn * speedBindings[key][1]

                    self.get_logger().info(self.vels(self.speed, self.turn))
                    if (self.status == 14):
                        self.get_logger().info(msg)
                    self.status = (self.status + 1) % 15
                elif key == ' ' or key == 'k':
                    self.x = 0.0
                    self.y = 0.0
                    self.z = 0.0
                    self.th = 0.0
                else:
                    self.x = 0.0
                    self.y = 0.0
                    self.z = 0.0
                    self.th = 0.0
                    if (key == '\x03'):
                        break

                twist = Twist()
                twist.linear.x = self.x * self.speed
                twist.linear.y = self.y * self.speed
                twist.linear.z = self.z * self.speed
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = self.th * self.turn
                self.pub.publish(twist)

        except Exception as e:
            self.get_logger().error(f'Exception: {str(e)}')
        finally:
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.pub.publish(twist)

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

def main(args=None):
    rclpy.init(args=args)
    teleop_node = TeleopHumanoid()

    try:
        teleop_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        teleop_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 6: Creating a Walking Pattern Generator

Let's create a node that generates walking patterns:

```python
# src/walking_pattern_generator.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import math

class WalkingPatternGenerator(Node):
    def __init__(self):
        super().__init__('walking_pattern_generator')

        # Publisher for walking trajectories
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/walking_trajectory',
            10
        )

        # Timer for generating walking patterns
        self.pattern_timer = self.create_timer(0.1, self.generate_walking_pattern)

        # Walking parameters
        self.step_height = 0.05  # meters
        self.step_length = 0.3   # meters
        self.step_duration = 1.0 # seconds
        self.time_step = 0.05    # seconds per step in pattern

        # Robot joint names for legs
        self.left_leg_joints = [
            'torso_to_left_hip',
            'left_hip_to_knee',
            'left_knee_to_ankle'
        ]

        self.right_leg_joints = [
            'torso_to_right_hip',
            'right_hip_to_knee',
            'right_knee_to_ankle'
        ]

        self.get_logger().info('Walking Pattern Generator initialized')

    def generate_walking_pattern(self):
        """Generate a walking trajectory pattern"""
        # Create a simple walking gait pattern
        trajectory = JointTrajectory()
        trajectory.joint_names = self.left_leg_joints + self.right_leg_joints

        # Generate trajectory points for one step cycle
        num_points = int(self.step_duration / self.time_step)

        for i in range(num_points):
            time_from_start = i * self.time_step

            point = JointTrajectoryPoint()
            point.time_from_start = Duration(
                sec=int(time_from_start),
                nanosec=int((time_from_start - int(time_from_start)) * 1e9)
            )

            # Generate walking pattern using sinusoidal functions
            phase = (2 * math.pi * i) / num_points

            # Left leg pattern (supports in first half, swings in second half)
            left_positions = self.generate_leg_pattern(
                phase,
                is_support_phase=(phase < math.pi),
                step_height=self.step_height,
                step_length=self.step_length
            )

            # Right leg pattern (opposite of left leg)
            right_phase = (phase + math.pi) % (2 * math.pi)
            right_positions = self.generate_leg_pattern(
                right_phase,
                is_support_phase=(right_phase < math.pi),
                step_height=self.step_height,
                step_length=self.step_length
            )

            # Combine left and right leg positions
            point.positions = left_positions + right_positions
            point.velocities = [0.0] * len(point.positions)  # Simplified
            point.accelerations = [0.0] * len(point.positions)  # Simplified

            trajectory.points.append(point)

        # Publish the trajectory
        self.trajectory_pub.publish(trajectory)

    def generate_leg_pattern(self, phase, is_support_phase, step_height, step_length):
        """Generate joint positions for a single leg based on phase"""
        # Simplified 3-DOF leg pattern
        # In a real implementation, this would use inverse kinematics

        if is_support_phase:
            # Support phase: leg straight down, ready to support weight
            hip_angle = 0.0
            knee_angle = 0.0
            ankle_angle = 0.0
        else:
            # Swing phase: leg lifts and moves forward
            # Use sinusoidal functions to create smooth motion
            lift_factor = math.sin(phase) if phase < math.pi else 0
            forward_factor = math.sin(phase / 2) if phase < math.pi else math.sin(math.pi / 2)

            hip_angle = forward_factor * step_length / 2  # Move forward
            knee_angle = lift_factor * step_height * 2    # Lift leg
            ankle_angle = -lift_factor * step_height      # Adjust ankle for ground contact

        return [hip_angle, knee_angle, ankle_angle]

def main(args=None):
    rclpy.init(args=args)
    walker = WalkingPatternGenerator()

    try:
        rclpy.spin(walker)
    except KeyboardInterrupt:
        walker.get_logger().info('Shutting down Walking Pattern Generator')
    finally:
        walker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 7: Testing and Integration

Now let's create a simple test script to verify our system works:

```python
# scripts/test_humanoid_system.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import JointState, Imu
from humanoid_robot_msgs.action import WalkToGoal
import time

class HumanoidSystemTester(Node):
    def __init__(self):
        super().__init__('humanoid_system_tester')

        # Subscribers to verify system is running
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10
        )

        # Publisher for basic movement
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Action client for walking
        self.walk_client = ActionClient(self, WalkToGoal, 'walk_to_goal')

        # State variables
        self.joint_received = False
        self.imu_received = False

        self.get_logger().info('Humanoid System Tester initialized')

    def joint_callback(self, msg):
        self.joint_received = True
        if len(msg.name) > 0:
            self.get_logger().info(f'Received joint states: {len(msg.name)} joints')

    def imu_callback(self, msg):
        self.imu_received = True
        # Extract orientation to check if IMU is working
        self.get_logger().info('Received IMU data')

    def test_system(self):
        """Test the humanoid robot system"""
        self.get_logger().info('Starting system test...')

        # Wait for data to confirm system is running
        timeout = time.time() + 10.0  # 10 second timeout
        while not (self.joint_received and self.imu_received) and time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not (self.joint_received and self.imu_received):
            self.get_logger().error('System test failed: Did not receive joint states or IMU data')
            return False

        self.get_logger().info('✓ Joint states and IMU data received successfully')

        # Test basic movement
        self.test_basic_movement()

        # Test walking action (if available)
        self.test_walking_action()

        self.get_logger().info('System test completed successfully!')
        return True

    def test_basic_movement(self):
        """Test basic movement commands"""
        self.get_logger().info('Testing basic movement...')

        # Send forward command
        cmd = Twist()
        cmd.linear.x = 0.2  # Move forward at 0.2 m/s
        cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Sent forward movement command')

        time.sleep(2.0)  # Move for 2 seconds

        # Stop
        cmd.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Stopped movement')

    def test_walking_action(self):
        """Test walking action"""
        self.get_logger().info('Testing walking action...')

        if not self.walk_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Walk action server not available')
            return

        # Create a simple walk goal
        goal_msg = WalkToGoal.Goal()
        goal_msg.target_pose.pose.position.x = 1.0  # Move 1 meter forward
        goal_msg.target_pose.pose.position.y = 0.0
        goal_msg.target_pose.pose.position.z = 0.0

        # Send goal
        future = self.walk_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Walk goal was rejected')
            return

        self.get_logger().info('Walk goal accepted, waiting for result...')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(f'Walk result: {result.success}, {result.message}')

def main(args=None):
    rclpy.init(args=args)
    tester = HumanoidSystemTester()

    try:
        success = tester.test_system()
        if success:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed!")
    except KeyboardInterrupt:
        tester.get_logger().info('Test interrupted by user')
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Running the Complete System

To run the complete system, you would use the following commands:

```bash
# Terminal 1: Start the simulation
ros2 launch humanoid_robot_description simulation.launch.py

# Terminal 2: Start the robot controller
ros2 launch humanoid_robot_control robot_control.launch.py

# Terminal 3: Test the system
ros2 run humanoid_robot_control test_humanoid_system.py

# Terminal 4: Control with keyboard (optional)
python3 scripts/teleop_keyboard.py
```

## Project Summary

This mini-project demonstrates a complete humanoid robot control system using ROS 2. Key components include:

1. **URDF Model**: A complete humanoid robot with proper inertial properties and sensor integration
2. **Control Configuration**: ROS2 Control setup for precise joint control
3. **Controller Node**: Main controller implementing balance and walking control
4. **Launch Files**: Comprehensive launch files for system startup
5. **Teleoperation**: Keyboard control for manual operation
6. **Walking Pattern Generation**: Automatic walking pattern creation
7. **Testing Framework**: System verification tools

The project integrates all Module 1 concepts: nodes for different functions, topics for communication, actions for complex tasks, URDF for robot description, launch files for system management, and parameters for configuration. This provides a solid foundation for more advanced humanoid robotics applications.