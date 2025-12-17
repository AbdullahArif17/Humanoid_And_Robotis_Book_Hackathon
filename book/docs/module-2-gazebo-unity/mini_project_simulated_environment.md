---
sidebar_position: 7
---

# Mini Project: Simulated Environment

## Project Overview

In this mini-project, we'll create a complete simulated environment for humanoid robotics using Gazebo and Unity. This project will demonstrate the integration of physics simulation, sensor modeling, and visualization to create a realistic digital twin of a humanoid robot system. The environment will include a humanoid robot, a structured world with obstacles, sensor systems, and a control interface.

### Learning Objectives

By completing this project, you will:

1. Create a complete Gazebo simulation environment
2. Implement realistic physics and sensor models
3. Set up ROS integration for robot control
4. Create Unity visualization for the simulation
5. Implement a complete control pipeline from simulation to visualization
6. Understand the workflow for creating digital twins

## Step 1: Creating the Gazebo World

First, let's create a comprehensive Gazebo world file that includes our humanoid robot and environment:

```xml
<!-- worlds/humanoid_world.sdf -->
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Physics engine -->
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

    <!-- Ground plane -->
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

    <!-- Environment objects -->
    <model name="table">
      <pose>3 0 0 0 0 0</pose>
      <link name="table_base">
        <inertial>
          <mass>20.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1.0</iyy>
            <iyz>0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
        <visual name="table_visual">
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
        <collision name="table_collision">
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
      </link>
      <link name="table_top">
        <pose>0 0 0.75 0 0 0</pose>
        <inertial>
          <mass>5.0</mass>
          <inertia>
            <ixx>0.5</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.5</iyy>
            <iyz>0</iyz>
            <izz>0.5</izz>
          </inertia>
        </inertial>
        <visual name="top_visual">
          <geometry>
            <box>
              <size>1.6 0.9 0.05</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <collision name="top_collision">
          <geometry>
            <box>
              <size>1.6 0.9 0.05</size>
            </box>
          </geometry>
        </collision>
      </link>
      <joint name="table_support" type="fixed">
        <parent>table_base</parent>
        <child>table_top</child>
      </joint>
    </model>

    <!-- Obstacles -->
    <model name="obstacle_1">
      <pose>-2 1 0 0 0 0</pose>
      <link name="link">
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
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
      </link>
    </model>

    <model name="obstacle_2">
      <pose>-2 -1 0 0 0 0</pose>
      <link name="link">
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
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.4 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.8 1</ambient>
            <diffuse>0.2 0.2 0.8 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.4 0.4</size>
            </box>
          </geometry>
        </collision>
      </link>
    </model>

    <!-- Humanoid robot -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 0.8 0 0 0</pose>
    </include>

    <!-- Sensors -->
    <model name="environment_sensors">
      <pose>0 0 2 0 0 0</pose>
      <link name="sensor_mount">
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.001</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.001</iyy>
            <iyz>0</iyz>
            <izz>0.001</izz>
          </inertia>
        </inertial>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>0.05</radius>
            </sphere>
          </geometry>
        </visual>
      </link>

      <!-- Overhead camera -->
      <sensor name="overhead_camera" type="camera">
        <pose>0 0 0 0 1.57 0</pose>
        <camera name="overhead">
          <horizontal_fov>1.57</horizontal_fov>
          <image>
            <width>800</width>
            <height>600</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>10</far>
          </clip>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          <frame_name>overhead_camera_frame</frame_name>
          <topic_name>/overhead_camera/image_raw</topic_name>
        </plugin>
      </sensor>
    </model>

    <!-- Plugins -->
    <plugin name="gazebo_ros_api_plugin" filename="libgazebo_ros_api_plugin.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
    </plugin>
  </world>
</sdf>
```

## Step 2: Creating the Humanoid Robot Model

Now let's create a complete humanoid robot model in SDF format:

```xml
<!-- models/humanoid_robot/model.sdf -->
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="humanoid_robot">
    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.2</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.2</iyy>
          <iyz>0</iyz>
          <izz>0.2</izz>
        </inertia>
      </inertial>

      <visual name="base_visual">
        <geometry>
          <box>
            <size>0.3 0.3 0.4</size>
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
            <size>0.3 0.3 0.4</size>
          </box>
        </geometry>
      </collision>
    </link>

    <link name="torso">
      <pose>0 0 0.3 0 0 0</pose>
      <inertial>
        <mass>8.0</mass>
        <inertia>
          <ixx>0.15</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.15</iyy>
          <iyz>0</iyz>
          <izz>0.15</izz>
        </inertia>
      </inertial>

      <visual name="torso_visual">
        <geometry>
          <cylinder>
            <radius>0.15</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
        </material>
      </visual>

      <collision name="torso_collision">
        <geometry>
          <cylinder>
            <radius>0.15</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <link name="head">
      <pose>0 0 0.6 0 0 0</pose>
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.02</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.02</iyy>
          <iyz>0</iyz>
          <izz>0.02</izz>
        </inertia>
      </inertial>

      <visual name="head_visual">
        <geometry>
          <sphere>
            <radius>0.15</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
        </material>
      </visual>

      <collision name="head_collision">
        <geometry>
          <sphere>
            <radius>0.15</radius>
          </sphere>
        </geometry>
      </collision>
    </link>

    <!-- Arms -->
    <link name="left_upper_arm">
      <pose>0.2 0 0.4 0 0 0</pose>
      <inertial>
        <mass>1.5</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.01</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>

      <visual name="left_upper_arm_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>

      <collision name="left_upper_arm_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <link name="left_lower_arm">
      <pose>0.2 0 0.1 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.005</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.005</iyy>
          <iyz>0</iyz>
          <izz>0.005</izz>
        </inertia>
      </inertial>

      <visual name="left_lower_arm_visual">
        <geometry>
          <cylinder>
            <radius>0.04</radius>
            <length>0.25</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>

      <collision name="left_lower_arm_collision">
        <geometry>
          <cylinder>
            <radius>0.04</radius>
            <length>0.25</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <link name="right_upper_arm">
      <pose>-0.2 0 0.4 0 0 0</pose>
      <inertial>
        <mass>1.5</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.01</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>

      <visual name="right_upper_arm_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>

      <collision name="right_upper_arm_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <link name="right_lower_arm">
      <pose>-0.2 0 0.1 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.005</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.005</iyy>
          <iyz>0</iyz>
          <izz>0.005</izz>
        </inertia>
      </inertial>

      <visual name="right_lower_arm_visual">
        <geometry>
          <cylinder>
            <radius>0.04</radius>
            <length>0.25</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>

      <collision name="right_lower_arm_collision">
        <geometry>
          <cylinder>
            <radius>0.04</radius>
            <length>0.25</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <!-- Legs -->
    <link name="left_upper_leg">
      <pose>0.1 0 -0.2 0 0 0</pose>
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.03</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.03</iyy>
          <iyz>0</iyz>
          <izz>0.03</izz>
        </inertia>
      </inertial>

      <visual name="left_upper_leg_visual">
        <geometry>
          <cylinder>
            <radius>0.06</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>

      <collision name="left_upper_leg_collision">
        <geometry>
          <cylinder>
            <radius>0.06</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <link name="left_lower_leg">
      <pose>0.1 0 -0.6 0 0 0</pose>
      <inertial>
        <mass>1.5</mass>
        <inertia>
          <ixx>0.02</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.02</iyy>
          <iyz>0</iyz>
          <izz>0.02</izz>
        </inertia>
      </inertial>

      <visual name="left_lower_leg_visual">
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

      <collision name="left_lower_leg_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <link name="left_foot">
      <pose>0.1 0 -0.8 0 0 0</pose>
      <inertial>
        <mass>0.8</mass>
        <inertia>
          <ixx>0.003</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.003</iyy>
          <iyz>0</iyz>
          <izz>0.003</izz>
        </inertia>
      </inertial>

      <visual name="left_foot_visual">
        <geometry>
          <box>
            <size>0.15 0.08 0.05</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 0 1</ambient>
          <diffuse>0 0 0 1</diffuse>
        </material>
      </visual>

      <collision name="left_foot_collision">
        <geometry>
          <box>
            <size>0.15 0.08 0.05</size>
          </box>
        </geometry>
      </collision>
    </link>

    <link name="right_upper_leg">
      <pose>-0.1 0 -0.2 0 0 0</pose>
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.03</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.03</iyy>
          <iyz>0</iyz>
          <izz>0.03</izz>
        </inertia>
      </inertial>

      <visual name="right_upper_leg_visual">
        <geometry>
          <cylinder>
            <radius>0.06</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>

      <collision name="right_upper_leg_collision">
        <geometry>
          <cylinder>
            <radius>0.06</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <link name="right_lower_leg">
      <pose>-0.1 0 -0.6 0 0 0</pose>
      <inertial>
        <mass>1.5</mass>
        <inertia>
          <ixx>0.02</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.02</iyy>
          <iyz>0</iyz>
          <izz>0.02</izz>
        </inertia>
      </inertial>

      <visual name="right_lower_leg_visual">
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

      <collision name="right_lower_leg_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <link name="right_foot">
      <pose>-0.1 0 -0.8 0 0 0</pose>
      <inertial>
        <mass>0.8</mass>
        <inertia>
          <ixx>0.003</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.003</iyy>
          <iyz>0</iyz>
          <izz>0.003</izz>
        </inertia>
      </inertial>

      <visual name="right_foot_visual">
        <geometry>
          <box>
            <size>0.15 0.08 0.05</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 0 1</ambient>
          <diffuse>0 0 0 1</diffuse>
        </material>
      </visual>

      <collision name="right_foot_collision">
        <geometry>
          <box>
            <size>0.15 0.08 0.05</size>
          </box>
        </geometry>
      </collision>
    </link>

    <!-- Joints -->
    <joint name="base_to_torso" type="fixed">
      <parent>base_link</parent>
      <child>torso</child>
      <pose>0 0 0.2 0 0 0</pose>
    </joint>

    <joint name="torso_to_head" type="revolute">
      <parent>torso</parent>
      <child>head</child>
      <pose>0 0 0.6 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.0</lower>
          <upper>1.0</upper>
          <effort>10</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <!-- Left Arm Joints -->
    <joint name="torso_to_left_shoulder" type="revolute">
      <parent>torso</parent>
      <child>left_upper_arm</child>
      <pose>0.2 0 0.4 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>20</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <joint name="left_shoulder_to_elbow" type="revolute">
      <parent>left_upper_arm</parent>
      <child>left_lower_arm</child>
      <pose>0.2 0 0.1 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>15</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <!-- Right Arm Joints -->
    <joint name="torso_to_right_shoulder" type="revolute">
      <parent>torso</parent>
      <child>right_upper_arm</child>
      <pose>-0.2 0 0.4 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>20</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <joint name="right_shoulder_to_elbow" type="revolute">
      <parent>right_upper_arm</parent>
      <child>right_lower_arm</child>
      <pose>-0.2 0 0.1 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>15</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <!-- Left Leg Joints -->
    <joint name="torso_to_left_hip" type="revolute">
      <parent>base_link</parent>
      <child>left_upper_leg</child>
      <pose>0.1 0 0 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>30</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <joint name="left_hip_to_knee" type="revolute">
      <parent>left_upper_leg</parent>
      <child>left_lower_leg</child>
      <pose>0.1 0 -0.4 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>0.1</upper>
          <effort>25</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <joint name="left_knee_to_ankle" type="revolute">
      <parent>left_lower_leg</parent>
      <child>left_foot</child>
      <pose>0.1 0 -0.3 0 0 0</pose>
      <axis>
        <xyz>1 0 0</xyz>
        <limit>
          <lower>-0.5</lower>
          <upper>0.5</upper>
          <effort>20</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <!-- Right Leg Joints -->
    <joint name="torso_to_right_hip" type="revolute">
      <parent>base_link</parent>
      <child>right_upper_leg</child>
      <pose>-0.1 0 0 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>30</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <joint name="right_hip_to_knee" type="revolute">
      <parent>right_upper_leg</parent>
      <child>right_lower_leg</child>
      <pose>-0.1 0 -0.4 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>0.1</upper>
          <effort>25</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <joint name="right_knee_to_ankle" type="revolute">
      <parent>right_lower_leg</parent>
      <child>right_foot</child>
      <pose>-0.1 0 -0.3 0 0 0</pose>
      <axis>
        <xyz>1 0 0</xyz>
        <limit>
          <lower>-0.5</lower>
          <upper>0.5</upper>
          <effort>20</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <!-- Sensors -->
    <sensor name="head_camera" type="camera">
      <pose>0.1 0 0.6 0 0 0</pose>
      <camera name="head_cam">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>head_camera_frame</frame_name>
        <topic_name>/camera/image_raw</topic_name>
      </plugin>
    </sensor>

    <sensor name="imu_sensor" type="imu">
      <pose>0 0 0.1 0 0 0</pose>
      <always_on>true</always_on>
      <update_rate>100</update_rate>
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

    <sensor name="lidar_360" type="ray">
      <pose>0 0 0.7 0 0 0</pose>
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
        <topicName>/scan</topicName>
        <frameName>lidar_frame</frameName>
      </plugin>
    </sensor>

    <!-- ROS Control Plugin -->
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
    </plugin>
  </model>
</sdf>
```

## Step 3: Creating ROS Control Configuration

Now let's create the ROS control configuration file for our humanoid robot:

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
```

## Step 4: Creating the Launch File

Let's create a launch file to start our complete simulation:

```python
# launch/humanoid_simulation.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default='humanoid_world')

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
            'world': PathJoinSubstitution([
                FindPackageShare('humanoid_simulation'),
                'worlds',
                LaunchConfiguration('world')
            ]),
            'gui': 'true',
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
            FindPackageShare('humanoid_simulation'),
            'models',
            'humanoid_robot',
            'model.sdf'
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
            '-z', '0.8'
        ],
        output='screen'
    )

    # Controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='ros2_control_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('humanoid_simulation'),
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
                  'right_leg_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # RViz2 for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d',
            PathJoinSubstitution([
                FindPackageShare('humanoid_simulation'),
                'rviz',
                'humanoid.rviz'
            ])
        ],
        parameters=[{'use_sim_time': use_sim_time}]
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
            default_value='humanoid_world',
            description='Choose one of the world files from `/humanoid_simulation/worlds`'
        ),

        # Launch Gazebo
        gazebo,

        # Launch robot state publisher
        robot_state_publisher,

        # Spawn robot in Gazebo
        spawn_entity,

        # Launch controller manager
        controller_manager,

        # Load controllers
        joint_state_broadcaster_spawner,
        robot_controller_spawner,

        # Launch RViz2
        rviz,
    ])
```

## Step 5: Creating a Robot Controller Node

Let's create a controller node to manage the humanoid robot:

```python
# src/humanoid_controller.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from sensor_msgs.msg import JointState, Imu, Image, LaserScan
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from humanoid_robot_msgs.action import WalkToGoal
import numpy as np
import math
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

        self.control_rate = self.get_parameter('control_loop_rate').value
        self.balance_kp = self.get_parameter('balance_kp').value
        self.balance_kd = self.get_parameter('balance_kd').value
        self.step_height = self.get_parameter('walking_step_height').value
        self.step_length = self.get_parameter('walking_step_length').value

        # Robot state
        self.joint_states = JointState()
        self.imu_data = None
        self.laser_scan = None
        self.camera_image = None

        # Balance control
        self.balance_error_history = deque(maxlen=10)
        self.previous_balance_error = 0.0

        # Publishers
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            'imu/data',
            self.imu_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10
        )

        # Action servers
        self.walk_action_server = ActionServer(
            self,
            WalkToGoal,
            'walk_to_goal',
            execute_callback=self.walk_execute_callback
        )

        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop
        )

        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        """Update joint state information"""
        self.joint_states = msg

    def imu_callback(self, msg):
        """Update IMU data for balance control"""
        self.imu_data = msg

    def laser_callback(self, msg):
        """Update laser scan data"""
        self.laser_scan = msg

    def camera_callback(self, msg):
        """Update camera image data"""
        self.camera_image = msg

    def walk_execute_callback(self, goal_handle):
        """Execute walk to goal action"""
        self.get_logger().info('Executing walk to goal')

        # Extract target position
        target_x = goal_handle.request.target_pose.pose.position.x
        target_y = goal_handle.request.target_pose.pose.position.y

        # Calculate distance to target
        current_x = 0.0  # Would get from odometry in real system
        current_y = 0.0
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
            self.get_clock().sleep_for(Duration(nanosec=500000000))  # 0.5 seconds

        # Check if reached target
        final_distance = math.sqrt(
            (target_x - current_x)**2 +
            (target_y - current_y)**2
        )

        result.success = final_distance < 0.2  # Within 20cm
        result.message = f'Reached target with final distance: {final_distance:.3f}m'
        result.final_pose.pose.position.x = current_x
        result.final_pose.pose.position.y = current_y

        goal_handle.succeed()
        return result

    def execute_step(self, target_x, target_y):
        """Execute a single walking step"""
        self.get_logger().info(f'Executing step toward ({target_x:.2f}, {target_y:.2f})')

    def control_loop(self):
        """Main control loop"""
        if not self.imu_data:
            return

        # Apply balance control
        self.apply_balance_control()

        # Check for stability
        self.check_stability()

    def apply_balance_control(self):
        """Apply balance control based on IMU data"""
        if not self.imu_data:
            return

        # Extract roll and pitch from IMU quaternion
        quat = self.imu_data.orientation
        roll = math.atan2(2.0 * (quat.w * quat.x + quat.y * quat.z),
                         1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y))
        pitch = math.asin(2.0 * (quat.w * quat.y - quat.z * quat.x))

        # Calculate balance error
        balance_error = pitch  # Using pitch as simple balance measure

        # Store error for derivative calculation
        self.balance_error_history.append(balance_error)

        # Calculate PID control
        p_term = self.balance_kp * balance_error

        if len(self.balance_error_history) > 1:
            d_term = self.balance_kd * (balance_error - self.previous_balance_error)
        else:
            d_term = 0.0

        self.previous_balance_error = balance_error

        control_output = p_term + d_term

        # Apply control to joints (simplified)
        self.apply_balance_control_output(control_output)

    def apply_balance_control_output(self, control_output):
        """Apply balance control output to joints"""
        # This would generate appropriate joint commands to maintain balance
        # For this example, we'll just log the control output
        if abs(control_output) > 1.0:
            self.get_logger().warn(f'Large balance control: {control_output:.3f}')

    def check_stability(self):
        """Check robot stability and trigger safety measures if needed"""
        if not self.imu_data:
            return

        # Extract orientation
        quat = self.imu_data.orientation
        roll = math.atan2(2.0 * (quat.w * quat.x + quat.y * quat.z),
                         1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y))
        pitch = math.asin(2.0 * (quat.w * quat.y - quat.z * quat.x))

        # Emergency stop if tilt is too large
        if abs(roll) > 1.0 or abs(pitch) > 1.0:  # ~57 degrees
            self.emergency_stop()
            self.get_logger().error(f'EMERGENCY STOP: Roll={roll:.3f}, Pitch={pitch:.3f}')

    def emergency_stop(self):
        """Execute emergency stop"""
        self.get_logger().error('Emergency stop activated!')
        # Send zero commands to all joints

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

## Step 6: Creating Unity Visualization

Now let's create the Unity scripts for visualization. First, the main visualization manager:

```csharp
// Assets/Scripts/SimulationVisualizationManager.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using System.Collections.Generic;

public class SimulationVisualizationManager : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Robot Visualization")]
    public GameObject humanoidRobotPrefab;
    public Transform robotParent;

    [Header("Sensor Visualization")]
    public GameObject lidarPointPrefab;
    public GameObject cameraFeedQuad;

    [Header("UI Elements")]
    public UnityEngine.UI.Text statusText;
    public UnityEngine.UI.Text sensorDataText;

    private ROSConnection ros;
    private GameObject humanoidRobot;
    private RobotVisualizer robotVisualizer;
    private SensorDataVisualizer sensorVisualizer;

    private List<GameObject> lidarPoints = new List<GameObject>();
    private int maxLidarPoints = 1000;

    void Start()
    {
        InitializeROSConnection();
        SpawnRobotVisualization();
        InitializeSensorVisualization();
    }

    void InitializeROSConnection()
    {
        ros = ROSConnection.instance;
        if (ros == null)
        {
            GameObject rosGO = new GameObject("ROSConnection");
            ros = rosGO.AddComponent<ROSConnection>();
        }

        ros.HostAddress = rosIPAddress;
        ros.HostPort = rosPort;

        // Subscribe to robot topics
        ros.Subscribe<JointStateMsg>("/joint_states", JointStateCallback);
        ros.Subscribe<ImuMsg>("/imu/data", ImuCallback);
        ros.Subscribe<LaserScanMsg>("/scan", LaserScanCallback);
        ros.Subscribe<ImageMsg>("/camera/image_raw", CameraImageCallback);

        Debug.Log("ROS connection initialized for simulation visualization");
    }

    void SpawnRobotVisualization()
    {
        if (humanoidRobotPrefab != null)
        {
            humanoidRobot = Instantiate(humanoidRobotPrefab, robotParent);
            robotVisualizer = humanoidRobot.GetComponent<RobotVisualizer>();

            if (robotVisualizer == null)
            {
                robotVisualizer = humanoidRobot.AddComponent<RobotVisualizer>();
            }
        }
    }

    void InitializeSensorVisualization()
    {
        sensorVisualizer = GetComponent<SensorDataVisualizer>();
        if (sensorVisualizer == null)
        {
            sensorVisualizer = gameObject.AddComponent<SensorDataVisualizer>();
        }
    }

    void JointStateCallback(JointStateMsg jointState)
    {
        if (robotVisualizer != null)
        {
            Dictionary<string, float> jointPositions = new Dictionary<string, float>();

            for (int i = 0; i < jointState.name.Count; i++)
            {
                if (i < jointState.position.Count)
                {
                    jointPositions[jointState.name[i]] = (float)jointState.position[i];
                }
            }

            robotVisualizer.SetJointPositions(jointPositions);

            // Update UI
            if (statusText != null)
            {
                statusText.text = $"Robot Active - Joints: {jointPositions.Count}";
            }
        }
    }

    void ImuCallback(ImuMsg imuData)
    {
        if (sensorVisualizer != null)
        {
            Quaternion orientation = new Quaternion(
                (float)imuData.orientation.x,
                (float)imuData.orientation.y,
                (float)imuData.orientation.z,
                (float)imuData.orientation.w
            );

            sensorVisualizer.UpdateImuOrientation(orientation);
        }
    }

    void LaserScanCallback(LaserScanMsg laserScan)
    {
        if (sensorVisualizer != null)
        {
            List<Vector3> points = new List<Vector3>();

            float angle = (float)laserScan.angle_min;
            for (int i = 0; i < laserScan.ranges.Count; i++)
            {
                float range = (float)laserScan.ranges[i];

                if (range >= laserScan.range_min && range <= laserScan.range_max)
                {
                    float x = range * Mathf.Cos(angle);
                    float y = range * Mathf.Sin(angle);

                    // Convert to world coordinates relative to robot
                    Vector3 point = humanoidRobot.transform.position +
                                  humanoidRobot.transform.right * x +
                                  humanoidRobot.transform.forward * y +
                                  Vector3.up * 0.1f; // Slightly above ground

                    points.Add(point);
                }

                angle += (float)laserScan.angle_increment;
            }

            sensorVisualizer.UpdateLidarPointCloud(points);
        }
    }

    void CameraImageCallback(ImageMsg imageMsg)
    {
        if (cameraFeedQuad != null)
        {
            Texture2D texture = ConvertRosImageToTexture(imageMsg);
            if (texture != null)
            {
                Renderer renderer = cameraFeedQuad.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material.mainTexture = texture;
                }
            }
        }

        if (sensorDataText != null)
        {
            sensorDataText.text = $"Camera: {imageMsg.width}x{imageMsg.height}, " +
                                $"Encoding: {imageMsg.encoding}, " +
                                $"Timestamp: {imageMsg.header.stamp.sec}";
        }
    }

    Texture2D ConvertRosImageToTexture(ImageMsg imageMsg)
    {
        if (imageMsg.encoding != "rgb8" && imageMsg.encoding != "bgr8")
        {
            Debug.LogWarning($"Unsupported image encoding: {imageMsg.encoding}");
            return null;
        }

        int width = (int)imageMsg.width;
        int height = (int)imageMsg.height;

        if (width <= 0 || height <= 0)
        {
            Debug.LogWarning("Invalid image dimensions");
            return null;
        }

        Texture2D texture = new Texture2D(width, height);

        try
        {
            Color32[] colors = new Color32[width * height];

            for (int i = 0; i < colors.Length; i++)
            {
                int dataIndex = i * 3;
                if (dataIndex + 2 < imageMsg.data.Count)
                {
                    byte r = imageMsg.data[dataIndex];
                    byte g = imageMsg.data[dataIndex + 1];
                    byte b = imageMsg.data[dataIndex + 2];
                    byte a = 255;

                    // Swap R and B for BGR format
                    if (imageMsg.encoding == "bgr8")
                    {
                        byte temp = r;
                        r = b;
                        b = temp;
                    }

                    colors[i] = new Color32(r, g, b, a);
                }
            }

            texture.SetPixels32(colors);
            texture.Apply();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error converting ROS image to texture: {e.Message}");
            return null;
        }

        return texture;
    }

    void OnDestroy()
    {
        // Clean up ROS connection
        if (ros != null)
        {
            ros.OnApplicationQuit();
        }

        // Clean up lidar points
        ClearLidarPoints();
    }

    void ClearLidarPoints()
    {
        foreach (GameObject point in lidarPoints)
        {
            if (point != null)
            {
                DestroyImmediate(point);
            }
        }
        lidarPoints.Clear();
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        if (ros != null)
        {
            var cmdVel = new TwistMsg();
            cmdVel.linear = new Vector3Msg();
            cmdVel.angular = new Vector3Msg();

            cmdVel.linear.x = linearX;
            cmdVel.angular.z = angularZ;

            ros.Publish("/cmd_vel", cmdVel);
        }
    }
}
```

And the Robot Visualizer script:

```csharp
// Assets/Scripts/RobotVisualizer.cs
using UnityEngine;
using System.Collections.Generic;

public class RobotVisualizer : MonoBehaviour
{
    [Header("Joint Configuration")]
    public List<JointInfo> joints = new List<JointInfo>();

    [Header("Visualization Settings")]
    public float jointSpeed = 10f;
    public Color activeJointColor = Color.green;
    public Color passiveJointColor = Color.gray;

    private Dictionary<string, JointInfo> jointLookup = new Dictionary<string, JointInfo>();
    private Dictionary<string, Material> originalMaterials = new Dictionary<string, Material>();

    [System.Serializable]
    public class JointInfo
    {
        public string jointName;
        public Transform jointTransform;
        public ArticulationBody articulationBody;
        public float currentTarget = 0f;
        public float currentPosition = 0f;
    }

    void Start()
    {
        InitializeJointLookup();
        StoreOriginalMaterials();
    }

    void InitializeJointLookup()
    {
        jointLookup.Clear();
        foreach (JointInfo joint in joints)
        {
            if (!string.IsNullOrEmpty(joint.jointName) && joint.jointTransform != null)
            {
                jointLookup[joint.jointName] = joint;
            }
        }
    }

    void StoreOriginalMaterials()
    {
        originalMaterials.Clear();
        foreach (JointInfo joint in joints)
        {
            if (joint.jointTransform != null)
            {
                Renderer renderer = joint.jointTransform.GetComponent<Renderer>();
                if (renderer != null && renderer.material != null)
                {
                    originalMaterials[joint.jointName] = renderer.material;
                }
            }
        }
    }

    public void SetJointPositions(Dictionary<string, float> jointPositions)
    {
        foreach (var kvp in jointPositions)
        {
            if (jointLookup.ContainsKey(kvp.Key))
            {
                JointInfo jointInfo = jointLookup[kvp.Key];
                jointInfo.currentTarget = kvp.Value;

                // Update articulation body if available
                if (jointInfo.articulationBody != null)
                {
                    ArticulationDrive drive = jointInfo.articulationBody.jointDrive;
                    drive.target = kvp.Value;
                    jointInfo.articulationBody.jointDrive = drive;
                }
            }
        }
    }

    void Update()
    {
        UpdateJointPositions();
        UpdateJointVisualization();
    }

    void UpdateJointPositions()
    {
        foreach (JointInfo joint in joints)
        {
            // Smooth interpolation to target position
            joint.currentPosition = Mathf.Lerp(
                joint.currentPosition,
                joint.currentTarget,
                Time.deltaTime * jointSpeed
            );

            // Update transform rotation if not using articulation body
            if (joint.articulationBody == null && joint.jointTransform != null)
            {
                // Simple rotation around Z axis as example
                joint.jointTransform.localRotation = Quaternion.Euler(0, 0, joint.currentPosition * Mathf.Rad2Deg);
            }
        }
    }

    void UpdateJointVisualization()
    {
        // Update joint colors based on activity
        foreach (JointInfo joint in joints)
        {
            if (joint.jointTransform != null)
            {
                Renderer renderer = joint.jointTransform.GetComponent<Renderer>();
                if (renderer != null)
                {
                    float diff = Mathf.Abs(joint.currentTarget - joint.currentPosition);
                    Color color = diff > 0.01f ? activeJointColor : passiveJointColor;
                    renderer.material.color = color;
                }
            }
        }
    }

    public void HighlightJoint(string jointName, float duration = 1f)
    {
        if (jointLookup.ContainsKey(jointName))
        {
            JointInfo joint = jointLookup[jointName];
            if (joint.jointTransform != null)
            {
                Renderer renderer = joint.jointTransform.GetComponent<Renderer>();
                if (renderer != null && originalMaterials.ContainsKey(jointName))
                {
                    renderer.material.color = Color.yellow;
                    Invoke(nameof(ResetJointColor), duration, jointName);
                }
            }
        }
    }

    void ResetJointColor(string jointName)
    {
        if (originalMaterials.ContainsKey(jointName))
        {
            JointInfo joint = jointLookup[jointName];
            if (joint.jointTransform != null)
            {
                Renderer renderer = joint.jointTransform.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material = originalMaterials[jointName];
                }
            }
        }
    }
}
```

## Step 7: Creating the Unity Scene

Now let's create a simple script to set up the Unity scene:

```csharp
// Assets/Scripts/SimulationSceneSetup.cs
using UnityEngine;
using UnityEngine.Rendering;

public class SimulationSceneSetup : MonoBehaviour
{
    [Header("Scene Configuration")]
    public Light mainLight;
    public Camera mainCamera;
    public Material groundMaterial;

    [Header("Environment")]
    public GameObject[] environmentObjects;

    void Start()
    {
        SetupSceneLighting();
        SetupCamera();
        SetupEnvironment();
    }

    void SetupSceneLighting()
    {
        if (mainLight != null)
        {
            mainLight.type = LightType.Directional;
            mainLight.color = Color.white;
            mainLight.intensity = 1f;
            mainLight.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        }

        // Configure global lighting
        RenderSettings.ambientLight = new Color(0.4f, 0.4f, 0.4f, 1f);
        RenderSettings.fog = true;
        RenderSettings.fogColor = Color.gray;
        RenderSettings.fogDensity = 0.01f;
    }

    void SetupCamera()
    {
        if (mainCamera != null)
        {
            mainCamera.fieldOfView = 60f;
            mainCamera.nearClipPlane = 0.1f;
            mainCamera.farClipPlane = 100f;

            // Position camera to view the robot and environment
            mainCamera.transform.position = new Vector3(0f, 5f, 8f);
            mainCamera.transform.LookAt(Vector3.zero);
        }
    }

    void SetupEnvironment()
    {
        // Create a simple ground plane
        CreateGroundPlane();

        // Position environment objects
        if (environmentObjects != null)
        {
            foreach (GameObject obj in environmentObjects)
            {
                if (obj != null)
                {
                    // Position objects appropriately for the humanoid robot environment
                    PositionEnvironmentObject(obj);
                }
            }
        }
    }

    void CreateGroundPlane()
    {
        GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.localScale = Vector3.one * 10f; // 10x10 meter ground

        if (groundMaterial != null)
        {
            Renderer groundRenderer = ground.GetComponent<Renderer>();
            if (groundRenderer != null)
            {
                groundRenderer.material = groundMaterial;
            }
        }

        // Add a box collider for physics interaction
        DestroyImmediate(ground.GetComponent<BoxCollider>());
        ground.AddComponent<MeshCollider>();
    }

    void PositionEnvironmentObject(GameObject obj)
    {
        // Position objects in a way that makes sense for humanoid robot simulation
        // For example, place obstacles, tables, etc.
        if (obj.CompareTag("Obstacle"))
        {
            // Randomly position obstacles
            obj.transform.position = new Vector3(
                Random.Range(-4f, 4f),
                obj.transform.position.y,
                Random.Range(-3f, 3f)
            );
        }
    }

    void Update()
    {
        // Update scene elements as needed
        UpdateSceneDynamics();
    }

    void UpdateSceneDynamics()
    {
        // Handle any dynamic scene updates
        // For example, moving obstacles, changing lighting, etc.
    }
}
```

## Step 8: Creating a Simple Control Interface

Finally, let's create a simple control interface for the simulation:

```csharp
// Assets/Scripts/SimulationControlInterface.cs
using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class SimulationControlInterface : MonoBehaviour
{
    [Header("Control References")]
    public SimulationVisualizationManager simManager;
    public Slider linearVelocitySlider;
    public Slider angularVelocitySlider;
    public Button moveButton;
    public Button stopButton;
    public Button resetButton;
    public Text velocityDisplay;

    [Header("Control Settings")]
    public float maxLinearVelocity = 1.0f;
    public float maxAngularVelocity = 1.0f;

    void Start()
    {
        SetupControlInterface();
    }

    void SetupControlInterface()
    {
        if (linearVelocitySlider != null)
        {
            linearVelocitySlider.minValue = -maxLinearVelocity;
            linearVelocitySlider.maxValue = maxLinearVelocity;
            linearVelocitySlider.value = 0f;
        }

        if (angularVelocitySlider != null)
        {
            angularVelocitySlider.minValue = -maxAngularVelocity;
            angularVelocitySlider.maxValue = maxAngularVelocity;
            angularVelocitySlider.value = 0f;
        }

        if (moveButton != null)
        {
            moveButton.onClick.AddListener(SendMoveCommand);
        }

        if (stopButton != null)
        {
            stopButton.onClick.AddListener(SendStopCommand);
        }

        if (resetButton != null)
        {
            resetButton.onClick.AddListener(ResetSimulation);
        }

        UpdateVelocityDisplay();
    }

    void Update()
    {
        UpdateVelocityDisplay();
    }

    void UpdateVelocityDisplay()
    {
        if (velocityDisplay != null && linearVelocitySlider != null && angularVelocitySlider != null)
        {
            velocityDisplay.text = $"Linear: {linearVelocitySlider.value:F2}, " +
                                 $"Angular: {angularVelocitySlider.value:F2}";
        }
    }

    public void SendMoveCommand()
    {
        if (simManager != null && linearVelocitySlider != null && angularVelocitySlider != null)
        {
            float linearX = linearVelocitySlider.value;
            float angularZ = angularVelocitySlider.value;

            simManager.SendVelocityCommand(linearX, angularZ);
        }
    }

    public void SendStopCommand()
    {
        if (simManager != null)
        {
            simManager.SendVelocityCommand(0f, 0f);
        }
    }

    public void ResetSimulation()
    {
        // This would typically send a reset command to Gazebo
        // For this example, we'll just stop the robot
        SendStopCommand();

        // Reset sliders
        if (linearVelocitySlider != null) linearVelocitySlider.value = 0f;
        if (angularVelocitySlider != null) angularVelocitySlider.value = 0f;
    }

    public void SetLinearVelocity(float value)
    {
        if (linearVelocitySlider != null)
        {
            linearVelocitySlider.value = Mathf.Clamp(value, -maxLinearVelocity, maxLinearVelocity);
        }
    }

    public void SetAngularVelocity(float value)
    {
        if (angularVelocitySlider != null)
        {
            angularVelocitySlider.value = Mathf.Clamp(value, -maxAngularVelocity, maxAngularVelocity);
        }
    }
}
```

## Running the Complete Simulation

To run the complete simulation, you would:

1. Start the Gazebo simulation:
```bash
ros2 launch humanoid_simulation humanoid_simulation.launch.py
```

2. Start the Unity visualization (after building the Unity project)

3. Use the control interface to move the robot around the environment

4. Observe the sensor data visualization in real-time

This mini-project demonstrates a complete digital twin environment for humanoid robotics, integrating Gazebo for physics simulation, ROS for robot control, and Unity for advanced visualization. The system provides real-time feedback from multiple sensors, allowing for comprehensive testing and development of humanoid robot behaviors in a safe, controlled environment.