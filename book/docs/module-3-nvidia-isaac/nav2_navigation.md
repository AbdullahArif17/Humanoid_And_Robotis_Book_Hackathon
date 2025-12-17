# Navigation with Nav2 for Humanoid Robots

## Overview

Navigation is a critical capability for humanoid robots, enabling them to move autonomously through complex environments while avoiding obstacles and reaching specified goals. The Navigation2 (Nav2) framework provides a comprehensive, modular, and extensible system for robot navigation built on ROS 2. For humanoid robots, Nav2 offers the flexibility to handle the unique challenges of bipedal locomotion, dynamic stability, and complex kinematics.

This chapter explores the implementation of Nav2 for humanoid robotics, covering the architecture, configuration, and specialized components needed for effective navigation of humanoid platforms.

## Nav2 Architecture

Nav2 is built on a behavior tree architecture that provides a flexible and modular approach to navigation. The core components include:

- **Navigation Server**: Central coordinator for navigation tasks
- **Behavior Tree Engine**: Executes navigation behaviors
- **Planners**: Global and local path planning
- **Controllers**: Local trajectory generation and execution
- **Recovery Behaviors**: Strategies for handling navigation failures

```python
# Nav2 Node Example for Humanoid Robot
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan, PointCloud2
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
import numpy as np
from typing import List, Dict, Tuple
import tf2_ros
from geometry_msgs.msg import TransformStamped

class HumanoidNav2Node(Node):
    """Nav2 node specialized for humanoid robot navigation"""

    def __init__(self):
        super().__init__('humanoid_nav2_node')

        # Initialize navigation parameters
        self.linear_vel_limit = 0.5  # m/s - conservative for humanoid stability
        self.angular_vel_limit = 0.5  # rad/s
        self.min_obstacle_distance = 0.5  # m
        self.path_tolerance = 0.2  # m
        self.goal_tolerance = 0.1  # m

        # Robot state
        self.current_pose = None
        self.current_path = None
        self.target_goal = None
        self.obstacles = []

        # TF broadcaster for robot transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Create subscribers
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(
            String,  # Using String for humanoid-specific commands
            '/humanoid/nav_commands',
            10
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/nav2_markers',
            10
        )

        # Navigation state machine
        self.nav_state = 'IDLE'  # IDLE, PLANNING, EXECUTING, RECOVERY, STOPPED
        self.nav_timer = self.create_timer(0.1, self.navigation_loop)

        # Initialize navigation system
        self.initialize_navigation()

        self.get_logger().info('Humanoid Nav2 Node Initialized')

    def initialize_navigation(self):
        """Initialize navigation system parameters"""
        # Set up navigation parameters specific to humanoid robots
        self.nav_params = {
            'min_vel_x': 0.05,      # Minimum linear velocity (m/s)
            'max_vel_x': 0.3,       # Maximum linear velocity (m/s) - conservative for stability
            'min_vel_theta': 0.1,   # Minimum angular velocity (rad/s)
            'max_vel_theta': 0.3,   # Maximum angular velocity (rad/s)
            'acc_lim_x': 0.5,       # Linear acceleration limit (m/s^2)
            'acc_lim_theta': 0.5,   # Angular acceleration limit (rad/s^2)
            'xy_goal_tolerance': 0.2,  # Goal tolerance in XY plane (m)
            'yaw_goal_tolerance': 0.1, # Goal tolerance in yaw (rad)
            'oscillation_distance': 0.05,  # Oscillation detection threshold
        }

        self.get_logger().info('Navigation system initialized with humanoid-specific parameters')

    def pose_callback(self, msg):
        """Update robot pose from localization system"""
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        # Convert laser scan to obstacle points
        self.obstacles = []
        angle = msg.angle_min

        for i, range_val in enumerate(msg.ranges):
            if msg.range_min <= range_val <= msg.range_max:
                # Convert polar to Cartesian coordinates
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                self.obstacles.append((x, y))

            angle += msg.angle_increment

    def path_callback(self, msg):
        """Update current navigation path"""
        self.current_path = msg.poses

    def navigation_loop(self):
        """Main navigation control loop"""
        if self.current_pose is None:
            return

        if self.nav_state == 'IDLE':
            # Wait for navigation goal
            pass

        elif self.nav_state == 'EXECUTING':
            if self.current_path and self.target_goal:
                # Execute path following
                cmd = self.follow_path()
                if cmd:
                    self.cmd_vel_pub.publish(cmd)

                # Check if goal reached
                if self.is_goal_reached():
                    self.nav_state = 'IDLE'
                    self.get_logger().info('Goal reached successfully')

                # Check for obstacles
                if self.detect_obstacles():
                    self.nav_state = 'RECOVERY'
                    self.get_logger().info('Obstacle detected, entering recovery mode')

    def follow_path(self) -> String:
        """Follow the current navigation path"""
        if not self.current_path:
            return None

        # Get next waypoint
        next_waypoint = self.get_next_waypoint()
        if not next_waypoint:
            return None

        # Calculate direction to waypoint
        dx = next_waypoint.pose.position.x - self.current_pose.position.x
        dy = next_waypoint.pose.position.y - self.current_pose.position.y
        distance = np.sqrt(dx**2 + dy**2)

        # Calculate required heading
        target_angle = np.arctan2(dy, dx)
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        # Calculate angular difference
        angle_diff = target_angle - current_yaw
        # Normalize angle to [-π, π]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Create navigation command
        cmd = String()
        cmd.data = f"move:{min(distance, self.nav_params['max_vel_x']):.2f}:{angle_diff:.2f}"

        return cmd

    def get_next_waypoint(self):
        """Get the next waypoint in the path"""
        if not self.current_path:
            return None

        # Find closest point on path and return next point
        closest_idx = 0
        min_dist = float('inf')

        for i, pose in enumerate(self.current_path):
            dist = np.sqrt(
                (pose.pose.position.x - self.current_pose.position.x)**2 +
                (pose.pose.position.y - self.current_pose.position.y)**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Return next waypoint after closest point
        next_idx = min(closest_idx + 1, len(self.current_path) - 1)
        return self.current_path[next_idx]

    def is_goal_reached(self) -> bool:
        """Check if the robot has reached the goal"""
        if not self.target_goal or not self.current_pose:
            return False

        # Calculate distance to goal
        dx = self.target_goal.pose.position.x - self.current_pose.position.x
        dy = self.target_goal.pose.position.y - self.current_pose.position.y
        distance = np.sqrt(dx**2 + dy**2)

        # Check if within tolerance
        return distance <= self.nav_params['xy_goal_tolerance']

    def detect_obstacles(self) -> bool:
        """Detect obstacles in the robot's path"""
        if not self.current_path or not self.obstacles:
            return False

        # Check if obstacles are in the path corridor
        for obstacle in self.obstacles:
            # Convert obstacle to robot frame (simplified)
            obs_x, obs_y = obstacle

            # Check distance to current path
            for pose in self.current_path[:10]:  # Check first 10 waypoints
                path_x = pose.pose.position.x
                path_y = pose.pose.position.y

                dist = np.sqrt((obs_x - path_x)**2 + (obs_y - path_y)**2)
                if dist < self.min_obstacle_distance:
                    return True

        return False

    def get_yaw_from_quaternion(self, q) -> float:
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def set_navigation_goal(self, x: float, y: float, theta: float = 0.0):
        """Set a navigation goal for the robot"""
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        # Convert Euler angle to quaternion
        from math import sin, cos
        cy = cos(theta * 0.5)
        sy = sin(theta * 0.5)
        cp = cos(0.0 * 0.5)
        sp = sin(0.0 * 0.5)
        cr = cos(0.0 * 0.5)
        sr = sin(0.0 * 0.5)

        goal.pose.orientation.w = cr * cp * cy + sr * sp * sy
        goal.pose.orientation.x = sr * cp * cy - cr * sp * sy
        goal.pose.orientation.y = cr * sp * cy + sr * cp * sy
        goal.pose.orientation.z = cr * cp * sy - sr * sp * cy

        self.target_goal = goal
        self.nav_state = 'EXECUTING'

        self.goal_pub.publish(goal)
        self.get_logger().info(f'Navigation goal set: ({x}, {y}, {theta})')
```

## Humanoid-Specific Navigation Challenges

Humanoid robots face unique challenges in navigation that require specialized approaches:

### 1. Dynamic Stability

Humanoid robots must maintain balance while navigating, which affects their motion planning:

```python
# Dynamic stability-aware navigation for humanoid robots
class StabilityAwareNavigator(HumanoidNav2Node):
    """Navigation system considering dynamic stability of humanoid robot"""

    def __init__(self):
        super().__init__()

        # Stability parameters
        self.zmp_margin = 0.05  # Zero Moment Point safety margin (m)
        self.max_step_height = 0.1  # Maximum step height (m)
        self.max_step_length = 0.2  # Maximum step length (m)
        self.com_height = 0.8  # Center of mass height (m)

        # Stability-related subscribers
        self.imu_sub = self.create_subscription(
            String,  # Placeholder for IMU data
            '/humanoid/imu',
            self.imu_callback,
            10
        )

        self.joint_states_sub = self.create_subscription(
            String,  # Placeholder for joint states
            '/humanoid/joint_states',
            self.joint_states_callback,
            10
        )

        # Stability state
        self.current_com = np.array([0.0, 0.0, self.com_height])  # Center of mass
        self.support_polygon = []  # Current support polygon vertices
        self.stability_margin = 0.0  # Current stability margin

    def imu_callback(self, msg):
        """Process IMU data for stability assessment"""
        # In practice, would extract orientation, angular velocity, and linear acceleration
        # For this example, we'll simulate stability calculation
        pass

    def joint_states_callback(self, msg):
        """Process joint states for stability calculation"""
        # Calculate current support polygon based on foot positions
        # This is a simplified approach - in reality, would use forward kinematics
        pass

    def calculate_stability_margin(self) -> float:
        """Calculate current stability margin based on ZMP"""
        # Simplified ZMP calculation
        # In practice, would use more sophisticated dynamic models
        zmp_x = self.current_pose.position.x if self.current_pose else 0.0
        zmp_y = self.current_pose.position.y if self.current_pose else 0.0

        # Calculate distance to nearest support polygon edge
        if self.support_polygon:
            # Find minimum distance from ZMP to support polygon
            min_dist = float('inf')
            for i in range(len(self.support_polygon)):
                p1 = self.support_polygon[i]
                p2 = self.support_polygon[(i + 1) % len(self.support_polygon)]

                # Calculate distance from point to line segment
                dist = self.point_to_line_distance((zmp_x, zmp_y), p1, p2)
                min_dist = min(min_dist, dist)

            return min_dist - self.zmp_margin

        return 0.0  # No support polygon defined

    def point_to_line_distance(self, point: Tuple[float, float],
                              line_start: Tuple[float, float],
                              line_end: Tuple[float, float]) -> float:
        """Calculate distance from point to line segment"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Vector from line_start to line_end
        line_vec = np.array([x2 - x1, y2 - y1])
        point_vec = np.array([px - x1, py - y1])

        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)

        line_unitvec = line_vec / line_len
        proj_len = np.dot(point_vec, line_unitvec)

        # Clamp projection to line segment
        proj_len = max(0, min(line_len, proj_len))

        closest_point = np.array([x1, y1]) + proj_len * line_unitvec
        return np.linalg.norm(np.array([px, py]) - closest_point)

    def adjust_path_for_stability(self, path: List) -> List:
        """Adjust navigation path considering stability constraints"""
        if not path:
            return []

        adjusted_path = []

        for i, pose in enumerate(path):
            # Calculate stability for this pose
            stability = self.estimate_stability_at_pose(pose)

            if stability < 0:  # Unstable
                # Find alternative path that maintains stability
                adjusted_pose = self.find_stable_alternative(pose, path, i)
                adjusted_path.append(adjusted_pose)
            else:
                adjusted_path.append(pose)

        return adjusted_path

    def estimate_stability_at_pose(self, pose) -> float:
        """Estimate stability at a given pose"""
        # Simplified stability estimation
        # In practice, would use dynamic simulation or ZMP analysis
        return 0.1  # Placeholder

    def find_stable_alternative(self, original_pose, path: List, index: int):
        """Find a stable alternative to an unstable pose"""
        # Search for nearby stable poses
        search_radius = 0.5  # m
        resolution = 0.1  # m

        base_x = original_pose.pose.position.x
        base_y = original_pose.pose.position.y

        for dx in np.arange(-search_radius, search_radius + resolution, resolution):
            for dy in np.arange(-search_radius, search_radius + resolution, resolution):
                test_pose = original_pose
                test_pose.pose.position.x = base_x + dx
                test_pose.pose.position.y = base_y + dy

                stability = self.estimate_stability_at_pose(test_pose)
                if stability >= 0:  # Found stable alternative
                    return test_pose

        # If no stable alternative found, return original pose
        # (navigation system will need to handle this case)
        return original_pose
```

### 2. Footstep Planning

Humanoid robots require careful footstep planning for stable locomotion:

```python
# Footstep planning for humanoid navigation
class FootstepPlanner:
    """Footstep planning system for humanoid navigation"""

    def __init__(self):
        self.foot_separation = 0.2  # Distance between feet (m)
        self.step_duration = 1.0  # Time per step (s)
        self.max_step_yaw = 0.2  # Maximum yaw change per step (rad)

    def plan_footsteps(self, path: List, start_pose: PoseStamped) -> List:
        """Plan footstep sequence for navigation path"""
        footsteps = []

        if not path:
            return footsteps

        # Start with current pose
        current_pose = start_pose
        left_foot = self.calculate_left_foot_pose(current_pose)
        right_foot = self.calculate_right_foot_pose(current_pose)

        footsteps.append({
            'type': 'left',
            'pose': left_foot,
            'time': 0.0
        })
        footsteps.append({
            'type': 'right',
            'pose': right_foot,
            'time': 0.0
        })

        # Plan footsteps along the path
        for i in range(1, len(path)):
            target_pose = path[i]

            # Determine which foot to move next
            next_foot = self.determine_next_foot(footsteps)

            # Calculate foot placement
            foot_pose = self.calculate_foot_pose(
                current_pose, target_pose, next_foot
            )

            # Add to footsteps
            footsteps.append({
                'type': next_foot,
                'pose': foot_pose,
                'time': len(footsteps) * self.step_duration
            })

        return footsteps

    def calculate_left_foot_pose(self, body_pose):
        """Calculate left foot pose from body pose"""
        left_foot = PoseStamped()
        left_foot.header = body_pose.header

        # Offset left foot from body center
        left_foot.pose.position.x = body_pose.pose.position.x
        left_foot.pose.position.y = body_pose.pose.position.y + self.foot_separation / 2
        left_foot.pose.position.z = 0.0  # Ground level

        left_foot.pose.orientation = body_pose.pose.orientation

        return left_foot

    def calculate_right_foot_pose(self, body_pose):
        """Calculate right foot pose from body pose"""
        right_foot = PoseStamped()
        right_foot.header = body_pose.header

        # Offset right foot from body center
        right_foot.pose.position.x = body_pose.pose.position.x
        right_foot.pose.position.y = body_pose.pose.position.y - self.foot_separation / 2
        right_foot.pose.position.z = 0.0  # Ground level

        right_foot.pose.orientation = body_pose.pose.orientation

        return right_foot

    def determine_next_foot(self, footsteps: List) -> str:
        """Determine which foot should move next"""
        if not footsteps:
            return 'left'  # Start with left foot

        # Alternate feet
        last_foot = footsteps[-1]['type']
        return 'right' if last_foot == 'left' else 'left'

    def calculate_foot_pose(self, current_pose, target_pose, foot_type: str):
        """Calculate where to place the next foot"""
        foot_pose = PoseStamped()
        foot_pose.header = target_pose.header

        # Calculate target position based on body trajectory
        dx = target_pose.pose.position.x - current_pose.pose.position.x
        dy = target_pose.pose.position.y - current_pose.pose.position.y
        distance = np.sqrt(dx**2 + dy**2)

        # Adjust for foot placement
        if foot_type == 'left':
            # Place left foot with appropriate offset
            foot_pose.pose.position.x = target_pose.pose.position.x
            foot_pose.pose.position.y = target_pose.pose.position.y + self.foot_separation / 2
        else:  # right foot
            foot_pose.pose.position.x = target_pose.pose.position.x
            foot_pose.pose.position.y = target_pose.pose.position.y - self.foot_separation / 2

        foot_pose.pose.position.z = 0.0  # Ground level

        # Maintain orientation
        foot_pose.pose.orientation = target_pose.pose.orientation

        return foot_pose
```

## Nav2 Configuration for Humanoid Robots

### Costmap Configuration

Humanoid robots require specialized costmap configuration to account for their size and stability:

```yaml
# Costmap configuration for humanoid robot
global_costmap:
  global_frame: map
  robot_base_frame: base_link
  update_frequency: 5.0
  publish_frequency: 2.0
  transform_tolerance: 0.5
  resolution: 0.05  # Higher resolution for precise navigation
  inflation_radius: 0.8  # Account for robot size and stability margin
  plugins:
    - {name: static_layer, type: "nav2_costmap_2d::StaticLayer"}
    - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}

local_costmap:
  global_frame: odom
  robot_base_frame: base_link
  update_frequency: 10.0
  publish_frequency: 5.0
  transform_tolerance: 0.1
  resolution: 0.025  # Even higher resolution for local planning
  inflation_radius: 0.6
  robot_radius: 0.3  # Account for robot's physical size
  plugins:
    - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: voxel_layer, type: "nav2_costmap_2d::VoxelLayer"}
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}
  obstacle_layer:
    enabled: True
    observation_sources: scan
    scan:
      topic: /scan
      max_obstacle_height: 2.0
      clearing: True
      marking: True
      data_type: LaserScan
```

### Behavior Tree Configuration

```xml
<!-- Behavior Tree for Humanoid Navigation -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <RecoveryNode number_of_retries="6" name="NavigateRecovery">
            <PipelineSequence name="NavigateWithRecovery">
                <RateController hz="10">
                    <RecoveryNode number_of_retries="2" name="ComputePathToPoseRecovery">
                        <PipelineSequence name="ComputePathToPoseWithRecovery">
                            <PoseToPose2D input_port="goal" output_port="output_goal" name="PoseToPose2D"/>
                            <ComputePathToPose goal="{output_goal}" path="{path}" planner_id="GridBased"/>
                            <PoseToPose input_port="output_goal" output_port="goal" name="PoseToPose"/>
                        </PipelineSequence>
                        <ReactiveFallback name="ComputePathToPoseFailure">
                            <GoalUpdated/>
                            <ClearEntireCostmap name="ClearGlobalCostmap-Context" service_name="global_costmap/clear_entirely_global_costmap"/>
                        </ReactiveFallback>
                    </RecoveryNode>
                </RateController>
                <RecoveryNode number_of_retries="2" name="FollowPathRecovery">
                    <PipelineSequence name="FollowPathWithRecovery">
                        <FollowPath path="{path}" controller_id="FollowPath"/>
                    </PipelineSequence>
                    <ReactiveFallback name="FollowPathFailure">
                        <GoalUpdated/>
                        <ClearEntireCostmap name="ClearLocalCostmap-Context" service_name="local_costmap/clear_entirely_local_costmap"/>
                        <ClearEntireCostmap name="ClearGlobalCostmap-Context" service_name="global_costmap/clear_entirely_global_costmap"/>
                    </ReactiveFallback>
                </RecoveryNode>
            </PipelineSequence>
            <ReactiveFallback name="NavigateRecoveryFallback">
                <GoalUpdated/>
                <StuckOnUnstableGround/>  <!-- Humanoid-specific recovery -->
                <ClearEntireCostmap name="ClearLocalCostmap-Context" service_name="local_costmap/clear_entirely_local_costmap"/>
                <ClearEntireCostmap name="ClearGlobalCostmap-Context" service_name="global_costmap/clear_entirely_global_costmap"/>
            </ReactiveFallback>
        </RecoveryNode>
    </BehaviorTree>

    <BehaviorTree ID="StuckOnUnstableGround">
        <SequenceStar name="StuckOnUnstableGroundSequence">
            <IsStabilityCompromised/>  <!-- Check if robot is on unstable ground -->
            <BackUp distance="0.3" speed="0.1" name="BackUpStability"/>
            <Spin angle="1.57" name="TurnToCheck"/>  <!-- Turn 90 degrees -->
            <Wait wait_duration="2" name="WaitToStabilize"/>
        </SequenceStar>
    </BehaviorTree>
</root>
```

## Integration with VSLAM Maps

### Using VSLAM Maps for Navigation

```python
# Integration between VSLAM and Nav2
class VSLAMNav2Integrator:
    """Integrator between VSLAM and Nav2 systems"""

    def __init__(self, vslam_system, nav2_node):
        self.vslam = vslam_system
        self.nav2 = nav2_node

        # Map conversion parameters
        self.map_resolution = 0.1  # m/cell
        self.map_width = 100  # cells
        self.map_height = 100  # cells

    def convert_vslam_to_costmap(self):
        """Convert VSLAM map to Nav2 costmap format"""
        # Get map points from VSLAM
        map_points = self.vslam.map_points

        # Create costmap grid
        costmap = np.zeros((self.map_height, self.map_width), dtype=np.uint8)

        # Convert 3D map points to 2D occupancy grid
        for point_id, point in map_points.items():
            if not point.is_bad:
                # Convert 3D point to grid coordinates
                grid_x = int((point.position[0] + self.map_width * self.map_resolution / 2) / self.map_resolution)
                grid_y = int((point.position[1] + self.map_height * self.map_resolution / 2) / self.map_resolution)

                # Check bounds
                if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
                    # Mark as occupied (100 = definitely occupied)
                    costmap[grid_y, grid_x] = 100

        return costmap

    def update_nav2_costmap(self):
        """Update Nav2 costmap with VSLAM data"""
        costmap = self.convert_vslam_to_costmap()

        # Publish to Nav2 costmap
        # This would involve creating and publishing an OccupancyGrid message
        pass

    def refine_navigation_path(self, original_path):
        """Refine navigation path using VSLAM map information"""
        refined_path = []

        for pose in original_path:
            # Check VSLAM map for detailed obstacle information
            refined_pose = self.check_vslam_obstacles(pose)
            refined_path.append(refined_pose)

        return refined_path

    def check_vslam_obstacles(self, pose):
        """Check for obstacles using VSLAM map"""
        # Get nearby map points from VSLAM
        nearby_points = self.get_nearby_map_points(pose)

        # Check for potential collisions
        for point in nearby_points:
            distance = np.linalg.norm([
                pose.pose.position.x - point.position[0],
                pose.pose.position.y - point.position[1]
            ])

            if distance < 0.5:  # 50cm safety margin
                # Adjust pose to avoid obstacle
                pose = self.adjust_pose_for_obstacle(pose, point)
                break

        return pose

    def get_nearby_map_points(self, pose, radius=1.0):
        """Get map points within a certain radius of a pose"""
        nearby_points = []
        robot_pos = np.array([pose.pose.position.x, pose.pose.position.y])

        for point in self.vslam.map_points.values():
            if not point.is_bad:
                point_pos = np.array([point.position[0], point.position[1]])
                distance = np.linalg.norm(robot_pos - point_pos)

                if distance <= radius:
                    nearby_points.append(point)

        return nearby_points

    def adjust_pose_for_obstacle(self, original_pose, obstacle_point):
        """Adjust pose to avoid a detected obstacle"""
        # Calculate direction away from obstacle
        robot_pos = np.array([original_pose.pose.position.x, original_pose.pose.position.y])
        obstacle_pos = np.array([obstacle_point.position[0], obstacle_point.position[1]])

        direction = robot_pos - obstacle_pos
        direction = direction / np.linalg.norm(direction)  # Normalize

        # Move away from obstacle by safety margin
        safety_margin = 0.3  # meters
        new_pos = robot_pos + direction * safety_margin

        adjusted_pose = original_pose
        adjusted_pose.pose.position.x = new_pos[0]
        adjusted_pose.pose.position.y = new_pos[1]

        return adjusted_pose
```

## Recovery Behaviors for Humanoid Robots

### Humanoid-Specific Recovery Behaviors

```python
# Humanoid-specific recovery behaviors
class HumanoidRecoveryBehaviors:
    """Recovery behaviors tailored for humanoid robots"""

    def __init__(self, humanoid_nav_node):
        self.nav_node = humanoid_nav_node
        self.recovery_timeout = 30.0  # seconds

    def stability_recovery(self):
        """Recovery behavior for stability issues"""
        self.nav_node.get_logger().info('Executing stability recovery')

        # Stop current motion
        stop_cmd = String()
        stop_cmd.data = "stop"
        self.nav_node.cmd_vel_pub.publish(stop_cmd)

        # Wait for stabilization
        import time
        start_time = time.time()

        while time.time() - start_time < 5.0:  # Wait up to 5 seconds for stabilization
            current_stability = self.nav_node.calculate_stability_margin()
            if current_stability > 0:  # Stable
                break
            time.sleep(0.1)

        # If still unstable, try different recovery
        if current_stability <= 0:
            self.small_step_recovery()

        return current_stability > 0

    def small_step_recovery(self):
        """Recovery by taking smaller, more stable steps"""
        self.nav_node.get_logger().info('Executing small step recovery')

        # Reduce step size and speed
        self.nav_node.nav_params['max_vel_x'] *= 0.5  # Reduce speed by half
        self.nav_node.nav_params['max_vel_theta'] *= 0.5

        # Plan a safer path around the obstacle
        if self.nav_node.current_path:
            safe_path = self.find_safer_path(self.nav_node.current_path)
            if safe_path:
                self.nav_node.current_path = safe_path

        # Restore normal parameters after recovery
        import threading
        threading.Timer(10.0, self.restore_normal_params).start()

    def find_safer_path(self, original_path):
        """Find a safer path around obstacles"""
        # This would implement a local planner that considers stability
        # For this example, we'll return the original path with safety modifications
        return original_path

    def restore_normal_params(self):
        """Restore normal navigation parameters"""
        self.nav_node.nav_params['max_vel_x'] = 0.3  # Restore original value
        self.nav_node.nav_params['max_vel_theta'] = 0.3

    def balance_recovery(self):
        """Recovery behavior for balance issues"""
        self.nav_node.get_logger().info('Executing balance recovery')

        # Command robot to return to neutral stance
        balance_cmd = String()
        balance_cmd.data = "balance:neutral"
        self.nav_node.cmd_vel_pub.publish(balance_cmd)

        # Wait for balance to be restored
        import time
        start_time = time.time()

        while time.time() - start_time < self.recovery_timeout:
            if self.is_balanced():
                self.nav_node.get_logger().info('Balance restored')
                return True
            time.sleep(0.1)

        self.nav_node.get_logger().warn('Balance recovery failed')
        return False

    def is_balanced(self) -> bool:
        """Check if robot is in balanced state"""
        # This would check IMU data, joint positions, etc.
        # For this example, we'll return True
        return True
```

## Simulation and Testing

### Nav2 Simulation with Isaac Sim

```python
# Integration with Isaac Sim for navigation testing
class IsaacSimNav2Integration:
    """Integration between Nav2 and Isaac Sim for testing"""

    def __init__(self, nav2_node):
        self.nav2_node = nav2_node
        self.simulation_active = False

        # Isaac Sim interface
        self.isaac_sim_interface = None

    def setup_simulation_environment(self):
        """Set up navigation testing environment in Isaac Sim"""
        # This would involve:
        # 1. Loading the humanoid robot model in Isaac Sim
        # 2. Setting up sensors (cameras, LIDAR, IMU)
        # 3. Creating test environments
        # 4. Configuring physics properties

        self.get_logger().info('Isaac Sim navigation environment set up')

    def run_navigation_simulation(self, test_scenario: str):
        """Run navigation simulation in Isaac Sim"""
        if not self.simulation_active:
            self.setup_simulation_environment()

        # Execute specific test scenario
        if test_scenario == "obstacle_avoidance":
            self.test_obstacle_avoidance()
        elif test_scenario == "stair_navigation":
            self.test_stair_navigation()
        elif test_scenario == "dynamic_obstacles":
            self.test_dynamic_obstacles()
        else:
            self.get_logger().warn(f'Unknown test scenario: {test_scenario}')

    def test_obstacle_avoidance(self):
        """Test obstacle avoidance in simulation"""
        # Place obstacles in the environment
        obstacles = [
            {"position": [2.0, 0.0, 0.0], "size": [0.5, 0.5, 1.0]},
            {"position": [3.0, 1.0, 0.0], "size": [0.3, 0.3, 1.0]},
        ]

        # Add obstacles to simulation
        for obs in obstacles:
            self.add_obstacle_to_simulation(obs)

        # Set navigation goal
        self.nav2_node.set_navigation_goal(5.0, 0.0, 0.0)

        # Monitor navigation performance
        self.monitor_navigation_performance()

    def test_stair_navigation(self):
        """Test stair navigation in simulation"""
        # Create stair environment in Isaac Sim
        stairs_config = {
            "num_steps": 5,
            "step_height": 0.17,
            "step_depth": 0.3,
            "step_width": 1.0
        }

        self.create_stairs_in_simulation(stairs_config)

        # Test navigation up and down stairs
        self.nav2_node.set_navigation_goal(3.0, 0.0, 0.0)
        self.monitor_navigation_performance()

    def add_obstacle_to_simulation(self, obstacle_config):
        """Add obstacle to Isaac Sim environment"""
        # This would use Isaac Sim APIs to add obstacles
        pass

    def create_stairs_in_simulation(self, stairs_config):
        """Create stairs in Isaac Sim environment"""
        # This would use Isaac Sim APIs to create stairs
        pass

    def monitor_navigation_performance(self):
        """Monitor navigation performance metrics"""
        # Track metrics like:
        # - Time to reach goal
        # - Path efficiency
        # - Stability metrics
        # - Collision avoidance
        pass
```

## Performance Optimization

### Multi-Threaded Navigation

```python
# Multi-threaded navigation for better performance
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

class MultiThreadedNavigator:
    """Multi-threaded navigation system for improved performance"""

    def __init__(self, nav2_node):
        self.nav2_node = nav2_node
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Queues for inter-thread communication
        self.sensor_queue = queue.Queue(maxsize=10)
        self.path_queue = queue.Queue(maxsize=5)
        self.command_queue = queue.Queue(maxsize=5)

        # Threading events
        self.shutdown_event = threading.Event()

        # Start processing threads
        self.sensor_thread = threading.Thread(target=self.process_sensors)
        self.planning_thread = threading.Thread(target=self.process_planning)
        self.control_thread = threading.Thread(target=self.process_control)

        self.sensor_thread.start()
        self.planning_thread.start()
        self.control_thread.start()

    def process_sensors(self):
        """Process sensor data in separate thread"""
        while not self.shutdown_event.is_set():
            try:
                # Get sensor data (simplified)
                sensor_data = self.get_sensor_data()

                # Process sensor data
                processed_data = self.process_sensor_input(sensor_data)

                # Put on queue for planning thread
                try:
                    self.sensor_queue.put_nowait(processed_data)
                except queue.Full:
                    continue  # Drop frame if queue is full

            except Exception as e:
                self.nav2_node.get_logger().error(f'Sensor processing error: {e}')

            # Throttle sensor processing
            import time
            time.sleep(0.05)  # 20 Hz

    def process_planning(self):
        """Process path planning in separate thread"""
        while not self.shutdown_event.is_set():
            try:
                # Get sensor data from queue
                try:
                    sensor_data = self.sensor_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Update costmap with sensor data
                self.update_costmap(sensor_data)

                # Plan path if needed
                if self.nav2_node.nav_state == 'EXECUTING':
                    path = self.plan_path()
                    try:
                        self.path_queue.put_nowait(path)
                    except queue.Full:
                        continue

            except Exception as e:
                self.nav2_node.get_logger().error(f'Planning error: {e}')

    def process_control(self):
        """Process motion control in separate thread"""
        while not self.shutdown_event.is_set():
            try:
                # Get planned path
                try:
                    path = self.path_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Generate control commands
                command = self.generate_control_command(path)

                # Execute command
                self.execute_command(command)

            except Exception as e:
                self.nav2_node.get_logger().error(f'Control error: {e}')

    def get_sensor_data(self):
        """Get sensor data from robot"""
        # This would get data from actual sensors
        # For this example, we'll return placeholder data
        return {}

    def process_sensor_input(self, sensor_data):
        """Process raw sensor input"""
        # Process and fuse sensor data
        return sensor_data

    def update_costmap(self, sensor_data):
        """Update costmap with sensor data"""
        # Update local and global costmaps
        pass

    def plan_path(self):
        """Plan navigation path"""
        # Plan path using current maps and goal
        return []

    def generate_control_command(self, path):
        """Generate control command from path"""
        # Generate velocity commands or footstep plans
        return {}

    def execute_command(self, command):
        """Execute navigation command"""
        # Send command to robot
        pass

    def shutdown(self):
        """Shutdown multi-threaded navigation"""
        self.shutdown_event.set()
        self.executor.shutdown(wait=True)
        self.sensor_thread.join()
        self.planning_thread.join()
        self.control_thread.join()
```

## Summary

Navigation with Nav2 for humanoid robots requires careful consideration of the unique challenges posed by bipedal locomotion, dynamic stability, and complex kinematics. The system must account for stability margins, implement specialized recovery behaviors, and potentially integrate with perception systems like VSLAM for enhanced navigation capabilities.

Key takeaways:
- Humanoid navigation requires stability-aware path planning and execution
- Footstep planning is crucial for stable bipedal locomotion
- Specialized recovery behaviors are needed for humanoid-specific failures
- Integration with perception systems enhances navigation robustness
- Multi-threading can improve navigation performance

The next chapter will explore Reinforcement Learning for Control, focusing on how machine learning can be used to develop adaptive control strategies for humanoid robots.