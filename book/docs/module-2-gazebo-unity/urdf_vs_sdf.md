---
sidebar_position: 5
---

# URDF vs SDF

## Introduction to Robot Description Formats

In the robotics ecosystem, two primary XML-based formats are used to describe robots: URDF (Unified Robot Description Format) and SDF (Simulation Description Format). Both formats serve essential roles in the development and simulation of robotic systems, particularly for humanoid robots. Understanding the differences, similarities, and appropriate use cases for each format is crucial for effective robot development.

### Overview of URDF and SDF

**URDF (Unified Robot Description Format)**:
- Developed for ROS (Robot Operating System)
- Primarily used for robot structure description
- Focus on kinematic and geometric properties
- Extensible through Xacro macros

**SDF (Simulation Description Format)**:
- Developed for Gazebo simulation environment
- Comprehensive simulation-specific features
- Supports multiple robot models in one file
- Designed for physics simulation and sensor modeling

```xml
<!-- Example URDF: Simple robot description -->
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <link name="arm_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint -->
  <joint name="base_to_arm" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>
```

```xml
<!-- Example SDF: Equivalent robot with simulation features -->
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="simple_robot">
    <pose>0 0 0.1 0 0 0</pose>

    <!-- Links -->
    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>
      <visual name="base_visual">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>

      <collision name="base_collision">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
      </collision>

      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.1</iyy>
          <iyz>0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
    </link>

    <link name="arm_link">
      <pose>0 0 0.3 0 0 0</pose>
      <visual name="arm_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>

      <collision name="arm_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
      </collision>

      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.01</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Joint -->
    <joint name="base_to_arm" type="revolute">
      <parent>base_link</parent>
      <child>arm_link</child>
      <pose>0 0 0.2 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>10</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>

    <!-- Gazebo-specific plugin -->
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/simple_robot</robotNamespace>
    </plugin>
  </model>
</sdf>
```

## URDF: The ROS Standard

### Structure and Components

URDF is designed to describe the physical structure of a robot, focusing on kinematic chains and geometric properties:

```xml
<!-- Detailed URDF example with all components -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="detailed_robot">

  <!-- Materials -->
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
  <material name="brown">
    <color rgba="0.8706 0.8118 0.7647 1.0"/>
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
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="grey"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Sensor link -->
  <joint name="base_to_laser" type="fixed">
    <parent link="base_link"/>
    <child link="laser_link"/>
    <origin xyz="0.2 0 0.2" rpy="0 0 0"/>
  </joint>

  <link name="laser_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.02" length="0.04"/>
      </geometry>
      <material name="red"/>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.02" length="0.04"/>
      </geometry>
    </collision>
  </link>

  <!-- Arm link -->
  <joint name="base_to_arm" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0.3 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
    <safety_controller k_position="100.0" k_velocity="10.0"
                      soft_lower_limit="-1.5" soft_upper_limit="1.5"/>
  </joint>

  <link name="arm_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
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

  <!-- Gazebo-specific tags for URDF -->
  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
    <turnGravityOff>false</turnGravityOff>
  </gazebo>

  <gazebo reference="laser_link">
    <material>Gazebo/Red</material>
    <sensor type="ray" name="laser_sensor">
      <pose>0 0 0 0 0 0</pose>
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-1.570796</min_angle>
            <max_angle>1.570796</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.10</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
        <topicName>/scan</topicName>
        <frameName>laser_link</frameName>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="arm_link">
    <material>Gazebo/Blue</material>
  </gazebo>

</robot>
```

### URDF Strengths and Limitations

**Strengths:**
- Simple and intuitive structure
- Excellent integration with ROS ecosystem
- Well-established tooling and community support
- Focus on kinematic description
- Extensible through Xacro

**Limitations:**
- Limited simulation-specific features
- No native multi-robot support
- Less detailed physics control
- Not designed for complex simulation scenarios

## SDF: The Simulation Standard

### Structure and Components

SDF is specifically designed for simulation environments, providing comprehensive support for physics, sensors, and simulation-specific features:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <!-- World definition -->
  <world name="default">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Models -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Robot model -->
    <model name="humanoid_robot">
      <pose>0 0 1 0 0 0</pose>

      <!-- Links -->
      <link name="base_link">
        <pose>0 0 0 0 0 0</pose>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.1</izz>
          </inertia>
        </inertial>

        <visual name="base_visual">
          <geometry>
            <box>
              <size>0.5 0.5 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>

        <collision name="base_collision">
          <geometry>
            <box>
              <size>0.5 0.5 0.2</size>
            </box>
          </geometry>
        </collision>
      </link>

      <!-- Sensors -->
      <sensor name="camera" type="camera">
        <pose>0.2 0 0.1 0 0 0</pose>
        <camera name="head_camera">
          <horizontal_fov>1.047</horizontal_fov>
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
          <frame_name>camera_frame</frame_name>
          <min_depth>0.1</min_depth>
          <max_depth>100</max_depth>
        </plugin>
      </sensor>

      <sensor name="imu" type="imu">
        <pose>0 0 0.1 0 0 0</pose>
        <always_on>true</always_on>
        <update_rate>100</update_rate>
        <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
          <topicName>imu/data</topicName>
          <bodyName>base_link</bodyName>
          <updateRateHZ>100.0</updateRateHZ>
          <gaussianNoise>0.001</gaussianNoise>
          <xyzOffset>0 0 0</xyzOffset>
          <rpyOffset>0 0 0</rpyOffset>
          <frameName>base_link</frameName>
        </plugin>
      </sensor>

      <!-- Plugins -->
      <plugin name="ros_control_plugin" filename="libgazebo_ros_control.so">
        <robotNamespace>/humanoid_robot</robotNamespace>
      </plugin>
    </model>

    <!-- Include other models -->
    <include>
      <uri>model://table</uri>
      <pose>2 0 0 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### SDF Strengths and Limitations

**Strengths:**
- Comprehensive simulation features
- Multi-robot support
- Advanced physics control
- Built-in sensor simulation
- World description capabilities
- Version control and evolution

**Limitations:**
- More complex than URDF
- Simulation-focused (less general robot description)
- Less ROS integration in its pure form
- Steeper learning curve

## Xacro: Extending URDF Capabilities

Xacro (XML Macros) extends URDF with powerful features for creating complex robot descriptions:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_humanoid">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="mass_torso" value="8.0" />
  <xacro:property name="mass_head" value="2.0" />
  <xacro:property name="mass_arm" value="1.5" />
  <xacro:property name="mass_leg" value="2.0" />
  <xacro:property name="mass_foot" value="0.8" />

  <!-- Inertial macros -->
  <xacro:macro name="inertial_sphere" params="mass x y z radius">
    <inertial>
      <mass value="${mass}" />
      <origin xyz="${x} ${y} ${z}" />
      <inertia ixx="${0.4 * mass * radius * radius}" ixy="0" ixz="0"
               iyy="${0.4 * mass * radius * radius}" iyz="0"
               izz="${0.4 * mass * radius * radius}" />
    </inertial>
  </xacro:macro>

  <xacro:macro name="inertial_box" params="mass x y z x_size y_size z_size">
    <inertial>
      <mass value="${mass}" />
      <origin xyz="${x} ${y} ${z}" />
      <inertia ixx="${mass / 12.0 * (y_size * y_size + z_size * z_size)}" ixy="0" ixz="0"
               iyy="${mass / 12.0 * (x_size * x_size + z_size * z_size)}" iyz="0"
               izz="${mass / 12.0 * (x_size * x_size + y_size * y_size)}" />
    </inertial>
  </xacro:macro>

  <xacro:macro name="inertial_cylinder" params="mass x y z radius length">
    <inertial>
      <mass value="${mass}" />
      <origin xyz="${x} ${y} ${z}" />
      <inertia ixx="${mass * (3 * radius * radius + length * length) / 12.0}" ixy="0" ixz="0"
               iyy="${mass * (3 * radius * radius + length * length) / 12.0}" iyz="0"
               izz="${mass * radius * radius / 2.0}" />
    </inertial>
  </xacro:macro>

  <!-- Link macro -->
  <xacro:macro name="simple_link" params="name mass xyz size type *visual *collision">
    <link name="${name}">
      <xacro:if value="${type == 'sphere'}">
        <xacro:inertial_sphere mass="${mass}" x="${xyz.split()[0]}" y="${xyz.split()[1]}" z="${xyz.split()[2]}" radius="${size.split()[0]}" />
      </xacro:if>
      <xacro:if value="${type == 'box'}">
        <xacro:inertial_box mass="${mass}" x="${xyz.split()[0]}" y="${xyz.split()[1]}" z="${xyz.split()[2]}"
                            x_size="${size.split()[0]}" y_size="${size.split()[1]}" z_size="${size.split()[2]}" />
      </xacro:if>
      <xacro:if value="${type == 'cylinder'}">
        <xacro:inertial_cylinder mass="${mass}" x="${xyz.split()[0]}" y="${xyz.split()[1]}" z="${xyz.split()[2]}"
                                 radius="${size.split()[0]}" length="${size.split()[1]}" />
      </xacro:if>

      <visual>
        <origin xyz="${xyz}" />
        <xacro:insert_block name="visual" />
      </visual>

      <collision>
        <origin xyz="${xyz}" />
        <xacro:insert_block name="collision" />
      </collision>
    </link>
  </xacro:macro>

  <!-- Joint macro -->
  <xacro:macro name="simple_joint" params="name type parent child xyz rpy axis lower upper effort velocity">
    <joint name="${name}" type="${type}">
      <parent link="${parent}"/>
      <child link="${child}"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <xacro:if value="${type != 'fixed'}">
        <axis xyz="${axis}"/>
        <limit lower="${lower}" upper="${upper}" effort="${effort}" velocity="${velocity}"/>
      </xacro:if>
    </joint>
  </xacro:macro>

  <!-- Base link -->
  <xacro:simple_link name="base_link" mass="10.0" xyz="0 0 0.2" size="0.3 0.3 0.4" type="box">
    <geometry>
      <box size="0.3 0.3 0.4"/>
    </geometry>
    <material name="grey">
      <color rgba="0.5 0.5 0.5 1"/>
    </material>
  </xacro:simple_link>

  <!-- Torso -->
  <xacro:simple_joint name="base_to_torso" type="fixed" parent="base_link" child="torso"
                      xyz="0 0 0.4" rpy="0 0 0" axis="0 0 0" lower="0" upper="0" effort="0" velocity="0"/>

  <xacro:simple_link name="torso" mass="${mass_torso}" xyz="0 0 0.3" size="0.15 0.6" type="cylinder">
    <geometry>
      <cylinder radius="0.15" length="0.6"/>
    </geometry>
    <material name="white">
      <color rgba="1 1 1 1"/>
    </material>
  </xacro:simple_link>

  <!-- Head -->
  <xacro:simple_joint name="torso_to_head" type="revolute" parent="torso" child="head"
                      xyz="0 0 0.6" rpy="0 0 0" axis="0 1 0"
                      lower="${-M_PI/2}" upper="${M_PI/2}" effort="10.0" velocity="1.0"/>

  <xacro:simple_link name="head" mass="${mass_head}" xyz="0 0 0.1" size="0.15" type="sphere">
    <geometry>
      <sphere radius="0.15"/>
    </geometry>
    <material name="white">
      <color rgba="1 1 1 1"/>
    </material>
  </xacro:simple_link>

  <!-- Arms -->
  <xacro:macro name="arm_chain" params="side shoulder_pos">
    <!-- Shoulder joint -->
    <xacro:simple_joint name="torso_to_${side}_shoulder" type="revolute" parent="torso" child="${side}_upper_arm"
                        xyz="${shoulder_pos}" rpy="0 0 0" axis="0 0 1"
                        lower="${-M_PI/2}" upper="${M_PI/2}" effort="20.0" velocity="1.0"/>

    <xacro:simple_link name="${side}_upper_arm" mass="${mass_arm}" xyz="0 0 0.15" size="0.05 0.3" type="cylinder">
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </xacro:simple_link>

    <!-- Elbow joint -->
    <xacro:simple_joint name="${side}_shoulder_to_elbow" type="revolute" parent="${side}_upper_arm" child="${side}_lower_arm"
                        xyz="0 0 0.3" rpy="0 0 0" axis="0 1 0"
                        lower="${-M_PI/2}" upper="${M_PI/2}" effort="15.0" velocity="1.0"/>

    <xacro:simple_link name="${side}_lower_arm" mass="1.0" xyz="0 0 0.12" size="0.04 0.24" type="cylinder">
      <geometry>
        <cylinder radius="0.04" length="0.24"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </xacro:simple_link>
  </xacro:macro>

  <!-- Instantiate arms -->
  <xacro:arm_chain side="left" shoulder_pos="0.2 0 0.4"/>
  <xacro:arm_chain side="right" shoulder_pos="-0.2 0 0.4"/>

  <!-- Legs -->
  <xacro:macro name="leg_chain" params="side hip_pos">
    <!-- Hip joint -->
    <xacro:simple_joint name="torso_to_${side}_hip" type="revolute" parent="torso" child="${side}_upper_leg"
                        xyz="${hip_pos}" rpy="0 0 0" axis="0 0 1"
                        lower="${-M_PI/2}" upper="${M_PI/2}" effort="30.0" velocity="1.0"/>

    <xacro:simple_link name="${side}_upper_leg" mass="${mass_leg}" xyz="0 0 -0.2" size="0.06 0.4" type="cylinder">
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </xacro:simple_link>

    <!-- Knee joint -->
    <xacro:simple_joint name="${side}_hip_to_knee" type="revolute" parent="${side}_upper_leg" child="${side}_lower_leg"
                        xyz="0 0 -0.4" rpy="0 0 0" axis="0 1 0"
                        lower="${-M_PI/2}" upper="${M_PI/2}" effort="25.0" velocity="1.0"/>

    <xacro:simple_link name="${side}_lower_leg" mass="1.5" xyz="0 0 -0.15" size="0.05 0.3" type="cylinder">
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </xacro:simple_link>

    <!-- Ankle joint -->
    <xacro:simple_joint name="${side}_knee_to_foot" type="revolute" parent="${side}_lower_leg" child="${side}_foot"
                        xyz="0 0 -0.3" rpy="0 0 0" axis="1 0 0"
                        lower="-0.5" upper="0.5" effort="20.0" velocity="1.0"/>

    <xacro:simple_link name="${side}_foot" mass="${mass_foot}" xyz="0.05 0 -0.05" size="0.2 0.1 0.1" type="box">
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </xacro:simple_link>
  </xacro:macro>

  <!-- Instantiate legs -->
  <xacro:leg_chain side="left" hip_pos="0.1 0 0"/>
  <xacro:leg_chain side="right" hip_pos="-0.1 0 0"/>

  <!-- Gazebo-specific tags -->
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

</robot>
```

## Conversion Between URDF and SDF

### URDF to SDF Conversion Process

The conversion from URDF to SDF is handled by Gazebo's parser, but understanding the process helps in creating compatible models:

```python
# Python example of understanding the conversion process
import xml.etree.ElementTree as ET
import numpy as np

class URDFToSDFConverter:
    """Simulates the conversion process from URDF to SDF"""

    def __init__(self):
        self.urdf_tree = None
        self.sdf_structure = None

    def parse_urdf(self, urdf_content):
        """Parse URDF content and extract robot structure"""
        self.urdf_tree = ET.fromstring(urdf_content)
        robot_name = self.urdf_tree.get('name', 'converted_robot')

        # Extract links
        links = {}
        for link_elem in self.urdf_tree.findall('link'):
            link_name = link_elem.get('name')
            link_data = self.parse_link(link_elem)
            links[link_name] = link_data

        # Extract joints
        joints = {}
        for joint_elem in self.urdf_tree.findall('joint'):
            joint_name = joint_elem.get('name')
            joint_data = self.parse_joint(joint_elem)
            joints[joint_name] = joint_data

        # Extract materials
        materials = {}
        for material_elem in self.urdf_tree.findall('material'):
            material_name = material_elem.get('name')
            material_data = self.parse_material(material_elem)
            materials[material_name] = material_data

        self.urdf_structure = {
            'name': robot_name,
            'links': links,
            'joints': joints,
            'materials': materials
        }

    def parse_link(self, link_elem):
        """Parse a link element from URDF"""
        link_name = link_elem.get('name')
        link_data = {'name': link_name}

        # Parse inertial
        inertial_elem = link_elem.find('inertial')
        if inertial_elem is not None:
            link_data['inertial'] = self.parse_inertial(inertial_elem)

        # Parse visual
        visual_elem = link_elem.find('visual')
        if visual_elem is not None:
            link_data['visual'] = self.parse_visual(visual_elem)

        # Parse collision
        collision_elem = link_elem.find('collision')
        if collision_elem is not None:
            link_data['collision'] = self.parse_collision(collision_elem)

        # Check for Gazebo-specific tags
        gazebo_elem = link_elem.find(f"../gazebo[@reference='{link_name}']")
        if gazebo_elem is not None:
            link_data['gazebo'] = self.parse_gazebo_tags(gazebo_elem)

        return link_data

    def parse_inertial(self, inertial_elem):
        """Parse inertial element"""
        inertial_data = {}

        mass_elem = inertial_elem.find('mass')
        if mass_elem is not None:
            inertial_data['mass'] = float(mass_elem.get('value', 0))

        origin_elem = inertial_elem.find('origin')
        if origin_elem is not None:
            xyz = origin_elem.get('xyz', '0 0 0').split()
            rpy = origin_elem.get('rpy', '0 0 0').split()
            inertial_data['origin'] = {
                'xyz': [float(x) for x in xyz],
                'rpy': [float(r) for r in rpy]
            }

        inertia_elem = inertial_elem.find('inertia')
        if inertia_elem is not None:
            inertial_data['inertia'] = {
                'ixx': float(inertia_elem.get('ixx', 0)),
                'ixy': float(inertia_elem.get('ixy', 0)),
                'ixz': float(inertia_elem.get('ixz', 0)),
                'iyy': float(inertia_elem.get('iyy', 0)),
                'iyz': float(inertia_elem.get('iyz', 0)),
                'izz': float(inertia_elem.get('izz', 0))
            }

        return inertial_data

    def parse_visual(self, visual_elem):
        """Parse visual element"""
        visual_data = {}

        origin_elem = visual_elem.find('origin')
        if origin_elem is not None:
            xyz = origin_elem.get('xyz', '0 0 0').split()
            rpy = origin_elem.get('rpy', '0 0 0').split()
            visual_data['origin'] = {
                'xyz': [float(x) for x in xyz],
                'rpy': [float(r) for r in rpy]
            }

        geometry_elem = visual_elem.find('geometry')
        if geometry_elem is not None:
            visual_data['geometry'] = self.parse_geometry(geometry_elem)

        material_elem = visual_elem.find('material')
        if material_elem is not None:
            visual_data['material'] = self.parse_material(material_elem)

        return visual_data

    def parse_collision(self, collision_elem):
        """Parse collision element"""
        collision_data = {}

        origin_elem = collision_elem.find('origin')
        if origin_elem is not None:
            xyz = origin_elem.get('xyz', '0 0 0').split()
            rpy = origin_elem.get('rpy', '0 0 0').split()
            collision_data['origin'] = {
                'xyz': [float(x) for x in xyz],
                'rpy': [float(r) for r in rpy]
            }

        geometry_elem = collision_elem.find('geometry')
        if geometry_elem is not None:
            collision_data['geometry'] = self.parse_geometry(geometry_elem)

        return collision_data

    def parse_geometry(self, geometry_elem):
        """Parse geometry element"""
        geometry_data = {}

        box_elem = geometry_elem.find('box')
        if box_elem is not None:
            size = box_elem.get('size', '1 1 1').split()
            geometry_data = {
                'type': 'box',
                'size': [float(s) for s in size]
            }
            return geometry_data

        cylinder_elem = geometry_elem.find('cylinder')
        if cylinder_elem is not None:
            geometry_data = {
                'type': 'cylinder',
                'radius': float(cylinder_elem.get('radius', 0)),
                'length': float(cylinder_elem.get('length', 0))
            }
            return geometry_data

        sphere_elem = geometry_elem.find('sphere')
        if sphere_elem is not None:
            geometry_data = {
                'type': 'sphere',
                'radius': float(sphere_elem.get('radius', 0))
            }
            return geometry_data

        return geometry_data

    def parse_material(self, material_elem):
        """Parse material element"""
        material_data = {'name': material_elem.get('name')}

        color_elem = material_elem.find('color')
        if color_elem is not None:
            rgba = color_elem.get('rgba', '1 1 1 1').split()
            material_data['color'] = [float(c) for c in rgba]

        return material_data

    def parse_joint(self, joint_elem):
        """Parse joint element"""
        joint_data = {
            'name': joint_elem.get('name'),
            'type': joint_elem.get('type'),
            'parent': joint_elem.find('parent').get('link'),
            'child': joint_elem.find('child').get('link')
        }

        origin_elem = joint_elem.find('origin')
        if origin_elem is not None:
            xyz = origin_elem.get('xyz', '0 0 0').split()
            rpy = origin_elem.get('rpy', '0 0 0').split()
            joint_data['origin'] = {
                'xyz': [float(x) for x in xyz],
                'rpy': [float(r) for r in rpy]
            }

        axis_elem = joint_elem.find('axis')
        if axis_elem is not None:
            xyz = axis_elem.get('xyz', '1 0 0').split()
            joint_data['axis'] = [float(x) for x in xyz]

        limit_elem = joint_elem.find('limit')
        if limit_elem is not None:
            joint_data['limit'] = {
                'lower': float(limit_elem.get('lower', 0)),
                'upper': float(limit_elem.get('upper', 0)),
                'effort': float(limit_elem.get('effort', 0)),
                'velocity': float(limit_elem.get('velocity', 0))
            }

        return joint_data

    def parse_gazebo_tags(self, gazebo_elem):
        """Parse Gazebo-specific tags within URDF"""
        gazebo_data = {}

        # Extract material
        material_elem = gazebo_elem.find('material')
        if material_elem is not None:
            gazebo_data['material'] = material_elem.text

        # Extract sensors
        sensor_elems = gazebo_elem.findall('sensor')
        if sensor_elems:
            gazebo_data['sensors'] = []
            for sensor_elem in sensor_elems:
                sensor_data = {
                    'name': sensor_elem.get('name'),
                    'type': sensor_elem.get('type'),
                    'pose': sensor_elem.find('pose').text if sensor_elem.find('pose') is not None else '0 0 0 0 0 0'
                }
                gazebo_data['sensors'].append(sensor_data)

        # Extract plugins
        plugin_elems = gazebo_elem.findall('plugin')
        if plugin_elems:
            gazebo_data['plugins'] = []
            for plugin_elem in plugin_elems:
                plugin_data = {
                    'name': plugin_elem.get('name'),
                    'filename': plugin_elem.get('filename'),
                }
                # Add plugin parameters
                plugin_data['params'] = {}
                for param_elem in plugin_elem:
                    if param_elem.text:
                        plugin_data['params'][param_elem.tag] = param_elem.text
                gazebo_data['plugins'].append(plugin_data)

        return gazebo_data

    def convert_to_sdf_structure(self):
        """Convert parsed URDF structure to SDF-like structure"""
        if not hasattr(self, 'urdf_structure'):
            raise ValueError("URDF must be parsed first")

        sdf_model = {
            'name': self.urdf_structure['name'],
            'links': self.urdf_structure['links'],
            'joints': self.urdf_structure['joints'],
            'materials': self.urdf_structure['materials']
        }

        # Add default SDF elements that are simulation-specific
        sdf_model['physics'] = {
            'type': 'ode',
            'max_step_size': 0.001,
            'real_time_factor': 1,
            'real_time_update_rate': 1000
        }

        sdf_model['plugins'] = []
        # Add ROS control plugin by default
        sdf_model['plugins'].append({
            'name': 'ros_control_plugin',
            'filename': 'libgazebo_ros_control.so',
            'params': {'robotNamespace': f'/{self.urdf_structure["name"]}'}
        })

        return sdf_model

# Example usage
urdf_content = """
<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
  </link>
  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
  </gazebo>
</robot>
"""

converter = URDFToSDFConverter()
converter.parse_urdf(urdf_content)
sdf_structure = converter.convert_to_sdf_structure()

print(f"Converted robot: {sdf_structure['name']}")
print(f"Links: {len(sdf_structure['links'])}")
print(f"Joints: {len(sdf_structure['joints'])}")
print(f"Plugins: {len(sdf_structure['plugins'])}")
```

## Practical Considerations for Humanoid Robots

### Best Practices for Format Selection

When developing humanoid robots, the choice between URDF and SDF depends on the specific use case:

```python
# Python code demonstrating practical considerations
class RobotDescriptionManager:
    """Manages robot descriptions for different use cases"""

    def __init__(self):
        self.robots = {}
        self.description_types = {}  # Maps robot name to description type (URDF/SDF)

    def add_urdf_robot(self, name: str, urdf_path: str):
        """Add a robot described in URDF format"""
        self.robots[name] = {
            'type': 'urdf',
            'path': urdf_path,
            'format': 'URDF'
        }
        self.description_types[name] = 'URDF'

    def add_sdf_robot(self, name: str, sdf_path: str):
        """Add a robot described in SDF format"""
        self.robots[name] = {
            'type': 'sdf',
            'path': sdf_path,
            'format': 'SDF'
        }
        self.description_types[name] = 'SDF'

    def get_description_for_purpose(self, robot_name: str, purpose: str):
        """
        Get appropriate description based on purpose:
        - 'simulation': SDF if available, otherwise URDF converted
        - 'control': URDF
        - 'visualization': URDF
        - 'analysis': URDF
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot {robot_name} not found")

        robot_info = self.robots[robot_name]
        description_type = self.description_types[robot_name]

        if purpose == 'simulation':
            # Prefer SDF for simulation, but can use URDF
            if description_type == 'SDF':
                return self.load_sdf(robot_info['path'])
            else:
                # Convert URDF to SDF for simulation
                urdf_content = self.load_urdf(robot_info['path'])
                return self.convert_urdf_to_sdf(urdf_content)
        elif purpose in ['control', 'visualization', 'analysis']:
            # Use URDF for these purposes
            if description_type == 'URDF':
                return self.load_urdf(robot_info['path'])
            else:
                # Extract URDF-like structure from SDF
                sdf_content = self.load_sdf(robot_info['path'])
                return self.extract_urdf_structure_from_sdf(sdf_content)

        return None

    def load_urdf(self, path: str) -> str:
        """Load URDF content from file"""
        with open(path, 'r') as f:
            return f.read()

    def load_sdf(self, path: str) -> str:
        """Load SDF content from file"""
        with open(path, 'r') as f:
            return f.read()

    def convert_urdf_to_sdf(self, urdf_content: str) -> str:
        """Convert URDF to SDF (simplified simulation)"""
        # In reality, this would use Gazebo's parser
        # This is a simplified representation
        sdf_template = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="converted_model">
    <!-- Converted from URDF -->
    {urdf_content.replace('<robot', '<link').replace('</robot>', '</link>')}
    <plugin name="ros_control" filename="libgazebo_ros_control.so"/>
  </model>
</sdf>"""
        return sdf_template

    def extract_urdf_structure_from_sdf(self, sdf_content: str) -> str:
        """Extract URDF-like structure from SDF (simplified simulation)"""
        # This would extract the model part from SDF and convert to URDF format
        # Simplified implementation
        urdf_template = f"""<?xml version="1.0"?>
<robot name="extracted_from_sdf">
  <!-- Extracted from SDF model -->
  {sdf_content.replace('<sdf', '<robot').replace('</sdf>', '</robot>')}
</robot>"""
        return urdf_template

# Example usage for humanoid robot development
manager = RobotDescriptionManager()

# Add different robot descriptions
manager.add_urdf_robot("nao_robot", "robots/nao.urdf")
manager.add_sdf_robot("atlas_robot", "robots/atlas.sdf")
manager.add_urdf_robot("custom_humanoid", "robots/custom_humanoid.urdf")

# Get appropriate descriptions for different purposes
simulation_desc = manager.get_description_for_purpose("nao_robot", "simulation")
control_desc = manager.get_description_for_purpose("nao_robot", "control")

print("Humanoid robot description management:")
print(f"  NAO robot - Simulation format: {manager.description_types['nao_robot']} (converted if needed)")
print(f"  Atlas robot - Simulation format: {manager.description_types['atlas_robot']}")
print(f"  Custom humanoid - Control format: {manager.description_types['custom_humanoid']}")
```

## Integration Patterns

### Using Both Formats Together

Many humanoid robot projects use both formats in different parts of the development pipeline:

```xml
<!-- Example: Using URDF for ROS, SDF for simulation -->
<!-- The typical workflow involves: -->
<!-- 1. Develop robot in URDF for ROS integration -->
<!-- 2. Use Gazebo to convert URDF to SDF for simulation -->
<!-- 3. Add simulation-specific elements to the URDF using <gazebo> tags -->

<!-- In URDF file (robot.urdf): -->
<!--
<robot name="humanoid">
  <link name="base_link">
    <inertial>...</inertial>
    <visual>...</visual>
    <collision>...</collision>
  </link>

  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
    <turnGravityOff>false</turnGravityOff>
  </gazebo>

  <gazebo>
    <plugin name="control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
    </plugin>
  </gazebo>
</robot>
-->
```

```python
# Python example of integration workflow
class IntegrationWorkflow:
    """Manages the integration workflow between URDF and SDF"""

    def __init__(self):
        self.urdf_models = {}
        self.sdf_worlds = {}
        self.conversion_cache = {}

    def setup_development_workflow(self):
        """Set up the typical development workflow"""
        workflow_steps = [
            "1. Design robot kinematics in URDF",
            "2. Add simulation elements with <gazebo> tags in URDF",
            "3. Load URDF into Gazebo (auto-converted to SDF)",
            "4. Develop and test in simulation",
            "5. Deploy to real robot using URDF",
            "6. Iterate based on real-world performance"
        ]
        return workflow_steps

    def validate_urdf_for_simulation(self, urdf_content: str) -> list:
        """Validate URDF content for simulation compatibility"""
        issues = []

        # Check for common issues
        if '<gazebo>' not in urdf_content:
            issues.append("No Gazebo-specific tags found - simulation features may be limited")

        if 'libgazebo_ros_control.so' not in urdf_content:
            issues.append("No ROS control plugin found - robot control may not work in simulation")

        if '<material>' not in urdf_content:
            issues.append("No materials defined - visual appearance may be poor")

        # Check for proper inertial properties
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(urdf_content)
            links_without_inertial = []
            for link in root.findall('link'):
                if link.find('inertial') is None:
                    links_without_inertial.append(link.get('name'))
            if links_without_inertial:
                issues.append(f"Links without inertial properties: {links_without_inertial}")
        except ET.ParseError:
            issues.append("Invalid URDF XML")

        return issues

    def suggest_improvements(self, urdf_content: str) -> list:
        """Suggest improvements for better simulation"""
        suggestions = []

        if 'safety_controller' not in urdf_content:
            suggestions.append("Add safety controllers to joint limits for simulation stability")

        if 'mesh' in urdf_content and not urdf_content.count('package://'):
            suggestions.append("Use package:// URIs for mesh files to ensure portability")

        if 'damping' not in urdf_content:
            suggestions.append("Add damping to joints to prevent simulation instability")

        if 'friction' not in urdf_content:
            suggestions.append("Add friction values to joints for more realistic simulation")

        return suggestions

# Example validation
workflow = IntegrationWorkflow()
sample_urdf = """
<?xml version="1.0"?>
<robot name="test_humanoid">
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
  </link>
</robot>
"""

issues = workflow.validate_urdf_for_simulation(sample_urdf)
suggestions = workflow.suggest_improvements(sample_urdf)

print("URDF Validation Results:")
if issues:
    print("Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("  No issues found")

print("\nSuggestions for improvement:")
if suggestions:
    for suggestion in suggestions:
        print(f"  - {suggestion}")
else:
    print("  No suggestions")
```

## Performance and Optimization

### Format-Specific Optimizations

Each format has specific optimization strategies:

**URDF Optimizations:**
- Use Xacro for complex models to reduce file size
- Minimize the number of visual and collision elements
- Use simple geometric shapes instead of complex meshes when possible

**SDF Optimizations:**
- Use appropriate physics engine settings
- Optimize sensor update rates
- Use level-of-detail (LOD) models for distant objects

The choice between URDF and SDF depends on the specific requirements of your humanoid robot project. URDF excels in ROS integration and kinematic description, while SDF provides comprehensive simulation capabilities. Understanding both formats and their appropriate use cases is essential for successful humanoid robot development.