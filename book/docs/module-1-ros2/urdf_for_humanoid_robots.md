---
sidebar_position: 5
---

# URDF for Humanoid Robots

## Understanding URDF in Robotics

URDF (Unified Robot Description Format) is an XML-based format used to describe robot models in ROS. For humanoid robots, URDF provides a comprehensive way to define the robot's physical structure, including links, joints, inertial properties, visual representations, and collision geometries. Understanding URDF is crucial for humanoid robotics as it enables simulation, visualization, and control of complex multi-link systems.

### URDF Components Overview

A URDF file contains several key components:

- **Links**: Rigid bodies that make up the robot structure
- **Joints**: Connections between links with specific degrees of freedom
- **Visual**: How the robot appears in simulation and visualization
- **Collision**: Collision detection geometry
- **Inertial**: Mass, center of mass, and inertia tensor for physics simulation

## URDF Structure for Humanoid Robots

### Basic URDF Template

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link definition -->
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Example joint connecting to another link -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.4"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.4"/>
      </geometry>
    </collision>
  </link>
</robot>
```

### Complete Humanoid Robot URDF

Here's a more comprehensive URDF for a humanoid robot with multiple limbs:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- MATERIALS -->
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

  <!-- BASE LINK -->
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

  <!-- TORSO -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <inertial>
      <mass value="8.0"/>
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

  <!-- HEAD -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="head">
    <inertial>
      <mass value="2.0"/>
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

  <!-- LEFT ARM -->
  <joint name="torso_to_left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0 0.4" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="left_upper_arm">
    <inertial>
      <mass value="1.5"/>
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

  <joint name="left_shoulder_to_elbow" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="15.0" velocity="1.0"/>
  </joint>

  <link name="left_lower_arm">
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

  <!-- RIGHT ARM -->
  <joint name="torso_to_right_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.2 0 0.4" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="right_upper_arm">
    <inertial>
      <mass value="1.5"/>
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

  <joint name="right_shoulder_to_elbow" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="15.0" velocity="1.0"/>
  </joint>

  <link name="right_lower_arm">
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

  <!-- LEFT LEG -->
  <joint name="torso_to_left_hip" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="30.0" velocity="1.0"/>
  </joint>

  <link name="left_upper_leg">
    <inertial>
      <mass value="2.0"/>
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

  <joint name="left_hip_to_knee" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="25.0" velocity="1.0"/>
  </joint>

  <link name="left_lower_leg">
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

  <joint name="left_knee_to_foot" type="revolute">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/>
    </inertial>

    <visual>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="black"/>
    </visual>

    <collision>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- RIGHT LEG -->
  <joint name="torso_to_right_hip" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="30.0" velocity="1.0"/>
  </joint>

  <link name="right_upper_leg">
    <inertial>
      <mass value="2.0"/>
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

  <joint name="right_hip_to_knee" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="25.0" velocity="1.0"/>
  </joint>

  <link name="right_lower_leg">
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

  <joint name="right_knee_to_foot" type="revolute">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/>
    </inertial>

    <visual>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="black"/>
    </visual>

    <collision>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
  </link>
</robot>
```

## Xacro for Complex Humanoid URDFs

Xacro (XML Macros) extends URDF by allowing parameterization, macros, and includes, making it easier to create complex humanoid robot descriptions:

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

  <!-- Inertial macro -->
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

  <!-- Left Arm -->
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

  <!-- Left Leg -->
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

## Advanced URDF Concepts for Humanoid Robots

### Transmission Elements

Transmission elements define how actuators connect to joints:

```xml
<!-- Example transmission for a humanoid joint -->
<transmission name="left_hip_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="torso_to_left_hip">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_hip_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Safety Controllers

Safety limits for humanoid robot joints:

```xml
<!-- Joint with safety limits -->
<joint name="safe_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="30.0" velocity="2.0"/>
  <safety_controller k_position="100.0" k_velocity="10.0"
                    soft_lower_limit="-1.5" soft_upper_limit="1.5"/>
</joint>
```

## URDF Validation and Testing

### Python Tools for URDF Validation

```python
import xml.etree.ElementTree as ET
import rclpy
from rclpy.node import Node
from urdf_parser_py.urdf import URDF
import os

class URDFValidator(Node):
    def __init__(self):
        super().__init__('urdf_validator')

        # Load and validate URDF
        self.urdf_path = os.path.join(os.getcwd(), 'simple_humanoid.urdf')
        self.validate_urdf()

    def validate_urdf(self):
        """Validate the URDF file"""
        try:
            # Parse URDF using urdf_parser_py
            robot = URDF.from_xml_string(open(self.urdf_path).read())

            self.get_logger().info(f'URDF validation successful!')
            self.get_logger().info(f'Robot name: {robot.name}')
            self.get_logger().info(f'Number of links: {len(robot.links)}')
            self.get_logger().info(f'Number of joints: {len(robot.joints)}')

            # Check for common issues
            self.check_urdf_issues(robot)

        except Exception as e:
            self.get_logger().error(f'URDF validation failed: {str(e)}')

    def check_urdf_issues(self, robot):
        """Check for common URDF issues"""
        issues = []

        # Check for links without visual/collision elements
        for link in robot.links:
            if not link.visual and not link.collision:
                issues.append(f'Link {link.name} has no visual or collision geometry')

        # Check for joints without proper parent-child relationships
        link_names = {link.name for link in robot.links}
        for joint in robot.joints:
            if joint.parent not in link_names:
                issues.append(f'Joint {joint.name} has invalid parent link: {joint.parent}')
            if joint.child not in link_names:
                issues.append(f'Joint {joint.name} has invalid child link: {joint.child}')

        # Report issues
        if issues:
            for issue in issues:
                self.get_logger().warn(f'URDF Issue: {issue}')
        else:
            self.get_logger().info('No URDF issues found')

def main(args=None):
    rclpy.init(args=args)
    validator = URDFValidator()

    # Run validation once
    rclpy.spin_once(validator)
    validator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### URDF Processing with Python

```python
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

class URDFProcessor:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()

    def get_joint_info(self):
        """Extract joint information from URDF"""
        joints = []

        for joint in self.root.findall('joint'):
            joint_info = {
                'name': joint.get('name'),
                'type': joint.get('type'),
                'parent': joint.find('parent').get('link'),
                'child': joint.find('child').get('link'),
            }

            # Get origin (position and orientation)
            origin = joint.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0').split()
                rpy = origin.get('rpy', '0 0 0').split()
                joint_info['origin_xyz'] = [float(x) for x in xyz]
                joint_info['origin_rpy'] = [float(r) for r in rpy]

            # Get axis
            axis = joint.find('axis')
            if axis is not None:
                joint_info['axis'] = [float(x) for x in axis.get('xyz').split()]

            # Get limits
            limit = joint.find('limit')
            if limit is not None:
                joint_info['limit'] = {
                    'lower': float(limit.get('lower', -np.pi)),
                    'upper': float(limit.get('upper', np.pi)),
                    'effort': float(limit.get('effort', 0)),
                    'velocity': float(limit.get('velocity', 0))
                }

            joints.append(joint_info)

        return joints

    def get_link_info(self):
        """Extract link information from URDF"""
        links = []

        for link in self.root.findall('link'):
            link_info = {
                'name': link.get('name'),
            }

            # Get inertial properties
            inertial = link.find('inertial')
            if inertial is not None:
                mass = inertial.find('mass')
                if mass is not None:
                    link_info['mass'] = float(mass.get('value'))

                inertia = inertial.find('inertia')
                if inertia is not None:
                    link_info['inertia'] = {
                        'ixx': float(inertia.get('ixx', 0)),
                        'ixy': float(inertia.get('ixy', 0)),
                        'ixz': float(inertia.get('ixz', 0)),
                        'iyy': float(inertia.get('iyy', 0)),
                        'iyz': float(inertia.get('iyz', 0)),
                        'izz': float(inertia.get('izz', 0))
                    }

            links.append(link_info)

        return links

    def get_chain_from_root(self, end_link):
        """Get kinematic chain from root to specified link"""
        # Build parent-child relationships
        child_to_parent = {}
        for joint in self.root.findall('joint'):
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            child_to_parent[child] = (parent, joint.get('name'))

        # Trace back from end link to root
        chain = []
        current = end_link
        while current in child_to_parent:
            parent, joint_name = child_to_parent[current]
            chain.append({
                'joint': joint_name,
                'child': current,
                'parent': parent
            })
            current = parent

        return list(reversed(chain))

# Example usage
def analyze_urdf(urdf_path):
    processor = URDFProcessor(urdf_path)

    print("=== Joint Information ===")
    joints = processor.get_joint_info()
    for joint in joints[:5]:  # Show first 5 joints
        print(f"Joint: {joint['name']}, Type: {joint['type']}")
        if 'limit' in joint:
            print(f"  Limits: {joint['limit']['lower']:.3f} to {joint['limit']['upper']:.3f}")

    print("\n=== Link Information ===")
    links = processor.get_link_info()
    for link in links[:5]:  # Show first 5 links
        print(f"Link: {link['name']}")
        if 'mass' in link:
            print(f"  Mass: {link['mass']:.3f} kg")

    print("\n=== Kinematic Chain to Left Foot ===")
    chain = processor.get_chain_from_root('left_foot')
    for link in chain:
        print(f"  {link['parent']} --({link['joint']})--> {link['child']}")

# This would be called with a URDF file path
# analyze_urdf('simple_humanoid.urdf')
```

## Common URDF Mistakes for Humanoid Robots

### Structural Mistakes

#### Inconsistent Units
- Mixing different units (meters vs millimeters) for link dimensions
- Using degrees instead of radians for joint limits
- Inconsistent mass units

#### Improper Inertial Properties
- Setting mass to zero or negative values
- Inertia tensors that don't represent the actual geometry
- Center of mass not matching the link geometry

#### Invalid Joint Limits
- Joint limits that are too restrictive for humanoid movement
- Limits that don't match the physical capabilities of actuators
- Missing joint limits for revolute joints

### Performance Issues

#### Complex Geometries
- Using high-resolution meshes for collision detection
- Too many small links instead of compound geometries
- Overly complex visual representations

#### Poor Link Hierarchy
- Creating disconnected link chains
- Incorrect parent-child relationships
- Missing base link or improper root structure

## Best Practices for Humanoid URDF Design

### Design Principles

#### Modular Structure
- Organize URDF into logical components (torso, limbs)
- Use Xacro includes for reusable components
- Separate visual and collision geometries appropriately

#### Realistic Physical Properties
- Use actual robot measurements and masses
- Verify inertial tensors match physical properties
- Include proper safety margins in joint limits

#### Simulation Considerations
- Use simplified collision geometries for performance
- Balance visual quality with simulation speed
- Consider the physics engine's capabilities

### Validation Checklist

1. **Kinematic Structure**: Verify all links are connected in a proper tree structure
2. **Physical Properties**: Check masses, inertias, and centers of mass
3. **Joint Limits**: Ensure limits are realistic for the robot's capabilities
4. **Collision Detection**: Verify collision geometries properly represent the robot
5. **Visual Representation**: Check that visual elements match the physical robot
6. **Transmission Setup**: Validate actuator connections and interfaces

## URDF in the ROS 2 Ecosystem

### Integration with Robot State Publisher

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import math
import numpy as np

class HumanoidStatePublisher(Node):
    def __init__(self):
        super().__init__('humanoid_state_publisher')

        # Create joint state publisher
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Create transform broadcaster for TF
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing state
        self.timer = self.create_timer(0.05, self.publish_states)  # 20Hz

        # Initialize joint positions
        self.joint_names = [
            'torso_to_head', 'torso_to_left_shoulder', 'left_shoulder_to_elbow',
            'torso_to_right_shoulder', 'right_shoulder_to_elbow',
            'torso_to_left_hip', 'left_hip_to_knee', 'left_knee_to_foot',
            'torso_to_right_hip', 'right_hip_to_knee', 'right_knee_to_foot'
        ]
        self.joint_positions = [0.0] * len(self.joint_names)

        self.get_logger().info('Humanoid State Publisher initialized')

    def publish_states(self):
        """Publish joint states and transforms"""
        # Update joint positions (in a real system, these would come from encoders)
        current_time = self.get_clock().now()

        # Generate example joint movements
        time_sec = current_time.nanoseconds / 1e9
        for i in range(len(self.joint_positions)):
            # Create different movement patterns for different joints
            self.joint_positions[i] = 0.2 * math.sin(time_sec * 0.5 + i * 0.3)

        # Publish joint states
        joint_state = JointState()
        joint_state.header.stamp = current_time.to_msg()
        joint_state.header.frame_id = 'base_link'
        joint_state.name = self.joint_names
        joint_state.position = self.joint_positions
        joint_state.velocity = [0.0] * len(self.joint_positions)  # Simplified
        joint_state.effort = [0.0] * len(self.joint_positions)    # Simplified

        self.joint_pub.publish(joint_state)

def main(args=None):
    rclpy.init(args=args)
    publisher = HumanoidStatePublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info('Shutting down Humanoid State Publisher')
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

URDF is fundamental to humanoid robotics in ROS 2, providing the essential description of the robot's physical structure. Properly designed URDF files enable accurate simulation, visualization, and control of humanoid robots. Understanding URDF concepts, best practices, and validation techniques is crucial for developing effective humanoid robot systems that can operate safely and efficiently in real-world environments.