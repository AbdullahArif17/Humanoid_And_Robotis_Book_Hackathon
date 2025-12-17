---
sidebar_position: 3
---

# Physics Simulation (gravity, collision)

## Introduction to Physics Simulation in Gazebo

Physics simulation forms the backbone of realistic robotic simulation in Gazebo. For humanoid robots, accurate physics simulation is crucial for developing stable locomotion, manipulation skills, and safe interaction behaviors. The physics engine must accurately model gravitational forces, collision detection and response, joint dynamics, and contact mechanics to create a believable digital twin of the physical robot.

### Core Physics Concepts

Physics simulation in Gazebo encompasses several fundamental concepts:

- **Gravitational Forces**: Constant downward acceleration affecting all objects
- **Collision Detection**: Identifying when objects intersect or come into contact
- **Collision Response**: Computing forces and motions resulting from collisions
- **Rigid Body Dynamics**: Motion of objects that maintain their shape
- **Joint Constraints**: Connections between bodies with specific degrees of freedom

```python
# Physics simulation framework for humanoid robots
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

class JointType(Enum):
    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2
    UNIVERSAL = 3
    BALL = 4

@dataclass
class RigidBody:
    """Represents a rigid body in the physics simulation"""
    name: str
    mass: float
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [w, x, y, z] quaternion
    velocity: np.ndarray  # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    inertia: np.ndarray  # [ixx, iyy, izz] diagonal inertia matrix
    linear_damping: float = 0.01
    angular_damping: float = 0.01
    collision_shapes: List[Dict] = None

    def __post_init__(self):
        if self.collision_shapes is None:
            self.collision_shapes = []

@dataclass
class Joint:
    """Represents a joint connecting two rigid bodies"""
    name: str
    type: JointType
    parent_body: str
    child_body: str
    position: np.ndarray  # Position in parent frame
    axis: np.ndarray  # Joint axis in parent frame
    limits: Tuple[float, float]  # Lower and upper limits
    effort_limit: float = 100.0
    velocity_limit: float = 10.0

class PhysicsEngine:
    """Core physics simulation engine"""

    def __init__(self, gravity: np.ndarray = None):
        self.gravity = gravity if gravity is not None else np.array([0, 0, -9.81])
        self.bodies: Dict[str, RigidBody] = {}
        self.joints: Dict[str, Joint] = {}
        self.contacts: List[Dict] = []
        self.time_step = 0.001  # 1ms default time step
        self.current_time = 0.0

    def add_body(self, body: RigidBody):
        """Add a rigid body to the simulation"""
        self.bodies[body.name] = body

    def add_joint(self, joint: Joint):
        """Add a joint to the simulation"""
        self.joints[joint.name] = joint

    def step_simulation(self, dt: float = None):
        """Step the physics simulation forward in time"""
        if dt is None:
            dt = self.time_step

        # Update current time
        self.current_time += dt

        # Apply forces (including gravity)
        self.apply_gravity()
        self.apply_damping()

        # Update positions and velocities using numerical integration
        self.integrate_motion(dt)

        # Detect and resolve collisions
        self.detect_collisions()
        self.resolve_collisions()

        # Process joints and constraints
        self.process_joints()

    def apply_gravity(self):
        """Apply gravitational force to all bodies"""
        for body in self.bodies.values():
            # F = m * g
            gravity_force = body.mass * self.gravity
            self.apply_force(body, gravity_force)

    def apply_damping(self):
        """Apply damping forces to reduce oscillations"""
        for body in self.bodies.values():
            # Linear damping
            damping_force = -body.linear_damping * body.velocity
            self.apply_force(body, damping_force)

            # Angular damping
            damping_torque = -body.angular_damping * body.angular_velocity
            self.apply_torque(body, damping_torque)

    def apply_force(self, body: RigidBody, force: np.ndarray):
        """Apply a force to a rigid body"""
        # F = m * a, so a = F / m
        acceleration = force / body.mass
        body.velocity += acceleration * self.time_step

    def apply_torque(self, body: RigidBody, torque: np.ndarray):
        """Apply a torque to a rigid body"""
        # τ = I * α, so α = I^(-1) * τ
        # For diagonal inertia matrix: α = τ / I
        angular_acceleration = torque / body.inertia
        body.angular_velocity += angular_acceleration * self.time_step

    def integrate_motion(self, dt: float):
        """Integrate motion using simple Euler integration"""
        for body in self.bodies.values():
            # Update position
            body.position += body.velocity * dt

            # Update orientation using quaternion integration
            # Convert angular velocity to quaternion derivative
            omega = body.angular_velocity
            q = body.orientation
            omega_quat = np.array([0, omega[0], omega[1], omega[2]])
            q_dot = 0.5 * self.quaternion_multiply(omega_quat, q)
            body.orientation += q_dot * dt

            # Normalize quaternion
            body.orientation /= np.linalg.norm(body.orientation)

    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])

    def detect_collisions(self):
        """Detect collisions between bodies"""
        self.contacts = []
        body_list = list(self.bodies.values())

        for i in range(len(body_list)):
            for j in range(i + 1, len(body_list)):
                body1, body2 = body_list[i], body_list[j]

                # Simple sphere collision detection (simplified)
                pos1, pos2 = body1.position, body2.position
                distance = np.linalg.norm(pos1 - pos2)

                # Assume unit spheres for simplicity
                if distance < 2.0:  # Collision occurred
                    # Calculate collision normal and penetration depth
                    normal = (pos2 - pos1) / distance if distance > 0 else np.array([1, 0, 0])
                    penetration_depth = 2.0 - distance

                    contact = {
                        'body1': body1.name,
                        'body2': body2.name,
                        'position': (pos1 + pos2) / 2,
                        'normal': normal,
                        'penetration_depth': penetration_depth
                    }
                    self.contacts.append(contact)

    def resolve_collisions(self):
        """Resolve detected collisions using impulse-based method"""
        for contact in self.contacts:
            body1 = self.bodies[contact['body1']]
            body2 = self.bodies[contact['body2']]
            normal = contact['normal']

            # Calculate relative velocity at contact point
            relative_velocity = body2.velocity - body1.velocity

            # Velocity along normal
            velocity_along_normal = np.dot(relative_velocity, normal)

            # Only resolve if objects are moving toward each other
            if velocity_along_normal < 0:
                # Calculate impulse magnitude
                # Simplified for equal masses and coefficient of restitution
                e = 0.8  # coefficient of restitution
                impulse_magnitude = -(1 + e) * velocity_along_normal
                impulse = impulse_magnitude * normal

                # Apply impulse to both bodies
                body1.velocity -= impulse / body1.mass
                body2.velocity += impulse / body2.mass

    def process_joints(self):
        """Process joint constraints and dynamics"""
        for joint in self.joints.values():
            if joint.type == JointType.REVOLUTE:
                self.process_revolute_joint(joint)

    def process_revolute_joint(self, joint: Joint):
        """Process revolute joint constraints"""
        parent_body = self.bodies[joint.parent_body]
        child_body = self.bodies[joint.child_body]

        # Apply joint limits (simplified)
        # In a real implementation, this would use constraint solving
        pass

# Example: Creating a simple humanoid model for physics simulation
def create_simple_humanoid():
    """Create a simplified humanoid robot model for physics simulation"""
    engine = PhysicsEngine()

    # Create torso (main body)
    torso = RigidBody(
        name="torso",
        mass=8.0,
        position=np.array([0, 0, 0.8]),
        orientation=np.array([1, 0, 0, 0]),
        velocity=np.array([0, 0, 0]),
        angular_velocity=np.array([0, 0, 0]),
        inertia=np.array([0.1, 0.1, 0.1])
    )
    engine.add_body(torso)

    # Create head
    head = RigidBody(
        name="head",
        mass=2.0,
        position=np.array([0, 0, 1.1]),
        orientation=np.array([1, 0, 0, 0]),
        velocity=np.array([0, 0, 0]),
        angular_velocity=np.array([0, 0, 0]),
        inertia=np.array([0.02, 0.02, 0.02])
    )
    engine.add_body(head)

    # Create left leg
    left_thigh = RigidBody(
        name="left_thigh",
        mass=3.0,
        position=np.array([0.1, 0, 0.4]),
        orientation=np.array([1, 0, 0, 0]),
        velocity=np.array([0, 0, 0]),
        angular_velocity=np.array([0, 0, 0]),
        inertia=np.array([0.05, 0.05, 0.01])
    )
    engine.add_body(left_thigh)

    left_shin = RigidBody(
        name="left_shin",
        mass=2.0,
        position=np.array([0.1, 0, 0.1]),
        orientation=np.array([1, 0, 0, 0]),
        velocity=np.array([0, 0, 0]),
        angular_velocity=np.array([0, 0, 0]),
        inertia=np.array([0.03, 0.03, 0.01])
    )
    engine.add_body(left_shin)

    # Add joints (simplified)
    hip_joint = Joint(
        name="left_hip",
        type=JointType.REVOLUTE,
        parent_body="torso",
        child_body="left_thigh",
        position=np.array([0.1, 0, 0.4]),
        axis=np.array([0, 1, 0]),
        limits=(-1.57, 1.57)
    )
    engine.add_joint(hip_joint)

    knee_joint = Joint(
        name="left_knee",
        type=JointType.REVOLUTE,
        parent_body="left_thigh",
        child_body="left_shin",
        position=np.array([0.1, 0, 0.1]),
        axis=np.array([0, 1, 0]),
        limits=(0, 1.57)
    )
    engine.add_joint(knee_joint)

    return engine

# Example usage
humanoid_sim = create_simple_humanoid()
print(f"Created humanoid simulation with {len(humanoid_sim.bodies)} bodies and {len(humanoid_sim.joints)} joints")

# Run a simple simulation
for step in range(100):
    humanoid_sim.step_simulation(0.01)  # 10ms time steps
    if step % 20 == 0:
        torso_pos = humanoid_sim.bodies["torso"].position
        print(f"Step {step}: Torso position = ({torso_pos[0]:.3f}, {torso_pos[1]:.3f}, {torso_pos[2]:.3f})")
```

## Gravity Simulation in Humanoid Robotics

### Understanding Gravitational Effects

Gravity is the dominant force affecting humanoid robots, making accurate gravity simulation critical for realistic behavior:

```python
class GravitySimulator:
    """Advanced gravity simulation for humanoid robots"""

    def __init__(self, base_gravity: np.ndarray = np.array([0, 0, -9.81])):
        self.base_gravity = base_gravity
        self.local_gravity_variations = {}  # For different environments
        self.gravity_noise = 0.0  # Noise in gravitational measurements

    def get_gravity_at_position(self, position: np.ndarray) -> np.ndarray:
        """Get gravitational acceleration at a specific position"""
        # For Earth, gravity is approximately constant near surface
        # Add small variations for realistic simulation
        variation = np.random.normal(0, self.gravity_noise, 3)
        return self.base_gravity + variation

    def simulate_gravity_effects(self, robot_state: Dict) -> Dict:
        """Simulate various gravity-related effects on a humanoid robot"""
        effects = {}

        # Center of Mass calculation
        com = self.calculate_center_of_mass(robot_state)
        effects['center_of_mass'] = com

        # Gravity-induced torques
        gravity_torques = self.calculate_gravity_torques(robot_state, com)
        effects['gravity_torques'] = gravity_torques

        # Stability analysis
        stability = self.analyze_stability(robot_state, com)
        effects['stability'] = stability

        # Balance control requirements
        balance_requirements = self.calculate_balance_requirements(robot_state, com)
        effects['balance_requirements'] = balance_requirements

        return effects

    def calculate_center_of_mass(self, robot_state: Dict) -> np.ndarray:
        """Calculate the center of mass of the humanoid robot"""
        total_mass = 0
        weighted_position_sum = np.zeros(3)

        # This would iterate through all links in the robot
        # For this example, we'll use a simplified approach
        for link_name, properties in robot_state.get('links', {}).items():
            mass = properties.get('mass', 1.0)
            position = np.array(properties.get('position', [0, 0, 0]))
            total_mass += mass
            weighted_position_sum += mass * position

        if total_mass > 0:
            return weighted_position_sum / total_mass
        else:
            return np.array([0, 0, 0])

    def calculate_gravity_torques(self, robot_state: Dict, com: np.ndarray) -> Dict[str, float]:
        """Calculate gravity-induced torques on joints"""
        torques = {}

        # Simplified calculation for each joint
        for joint_name, joint_state in robot_state.get('joints', {}).items():
            # Calculate moment arm from COM to joint
            joint_pos = np.array(joint_state.get('position', [0, 0, 0]))
            moment_arm = com - joint_pos

            # Gravity force at COM
            total_mass = sum(link.get('mass', 1.0) for link in robot_state.get('links', {}).values())
            gravity_force = total_mass * self.base_gravity

            # Torque = r × F
            torque = np.cross(moment_arm, gravity_force)
            torques[joint_name] = np.linalg.norm(torque)

        return torques

    def analyze_stability(self, robot_state: Dict, com: np.ndarray) -> Dict[str, any]:
        """Analyze the stability of the humanoid robot"""
        # Calculate support polygon (simplified as feet positions)
        support_points = self.get_support_polygon(robot_state)

        if len(support_points) < 3:
            return {'stable': False, 'reason': 'Insufficient support points'}

        # Check if COM is within support polygon (2D projection)
        com_2d = com[:2]
        stable = self.point_in_polygon(com_2d, support_points)

        # Calculate stability margin
        if stable:
            margin = self.calculate_stability_margin(com_2d, support_points)
        else:
            margin = -self.calculate_distance_to_polygon(com_2d, support_points)

        return {
            'stable': stable,
            'stability_margin': margin,
            'support_points': support_points.tolist(),
            'com_projected': com_2d.tolist()
        }

    def get_support_polygon(self, robot_state: Dict) -> np.ndarray:
        """Get the support polygon from foot contacts"""
        # Simplified: get positions of feet
        support_points = []

        # In a real implementation, this would check for ground contact
        # For now, we'll use fixed foot positions as an example
        left_foot_pos = robot_state.get('left_foot_position', [0.1, 0.05, 0])
        right_foot_pos = robot_state.get('right_foot_position', [-0.1, 0.05, 0])

        # Add some points around the feet for a larger support polygon
        support_points.extend([
            [left_foot_pos[0] + 0.05, left_foot_pos[1] + 0.05],
            [left_foot_pos[0] + 0.05, left_foot_pos[1] - 0.05],
            [left_foot_pos[0] - 0.05, left_foot_pos[1] - 0.05],
            [left_foot_pos[0] - 0.05, left_foot_pos[1] + 0.05],
            [right_foot_pos[0] + 0.05, right_foot_pos[1] + 0.05],
            [right_foot_pos[0] + 0.05, right_foot_pos[1] - 0.05],
            [right_foot_pos[0] - 0.05, right_foot_pos[1] - 0.05],
            [right_foot_pos[0] - 0.05, right_foot_pos[1] + 0.05],
        ])

        return np.array(support_points)

    def point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if a point is inside a polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def calculate_stability_margin(self, point: np.ndarray, polygon: np.ndarray) -> float:
        """Calculate the minimum distance from COM to support polygon edge"""
        min_distance = float('inf')

        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]

            # Distance from point to line segment
            distance = self.distance_point_to_segment(point, p1, p2)
            min_distance = min(min_distance, distance)

        return min_distance

    def distance_point_to_segment(self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        """Calculate distance from point to line segment"""
        # Vector from segment start to point
        ap = point - seg_start
        # Vector from segment start to end
        ab = seg_end - seg_start

        # Project ap onto ab
        ab_squared = np.dot(ab, ab)
        if ab_squared == 0:
            # Segment is a point
            return np.linalg.norm(ap)

        t = max(0, min(1, np.dot(ap, ab) / ab_squared))
        projection = seg_start + t * ab
        return np.linalg.norm(point - projection)

    def calculate_distance_to_polygon(self, point: np.ndarray, polygon: np.ndarray) -> float:
        """Calculate minimum distance from point to polygon"""
        min_distance = float('inf')

        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            distance = self.distance_point_to_segment(point, p1, p2)
            min_distance = min(min_distance, distance)

        return min_distance

    def calculate_balance_requirements(self, robot_state: Dict, com: np.ndarray) -> Dict[str, float]:
        """Calculate the balance control requirements"""
        stability_analysis = self.analyze_stability(robot_state, com)

        requirements = {
            'com_offset': np.linalg.norm(com[:2]),  # Horizontal offset from origin
            'stability_margin': stability_analysis['stability_margin'],
            'required_com_adjustment': 0.0,
            'balance_effort_estimate': 0.0
        }

        if not stability_analysis['stable']:
            # Calculate how much COM needs to be adjusted to achieve stability
            support_points = np.array(stability_analysis['support_points'])
            # Find the closest point in the support polygon to the COM
            closest_point = self.find_closest_point_in_polygon(com[:2], support_points)
            requirements['required_com_adjustment'] = np.linalg.norm(com[:2] - closest_point)

        # Estimate balance control effort (simplified)
        requirements['balance_effort_estimate'] = abs(requirements['stability_margin']) * 50  # Scaling factor

        return requirements

    def find_closest_point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Find the closest point to the polygon"""
        min_distance = float('inf')
        closest_point = point

        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            # Find closest point on segment
            t = max(0, min(1, np.dot(point - p1, p2 - p1) / np.dot(p2 - p1, p2 - p1)))
            segment_point = p1 + t * (p2 - p1)
            distance = np.linalg.norm(point - segment_point)

            if distance < min_distance:
                min_distance = distance
                closest_point = segment_point

        return closest_point

# Example usage of gravity simulator
gravity_sim = GravitySimulator()

# Simulate a robot state
robot_state = {
    'links': {
        'torso': {'mass': 8.0, 'position': [0, 0, 0.8]},
        'head': {'mass': 2.0, 'position': [0, 0, 1.1]},
        'left_thigh': {'mass': 3.0, 'position': [0.1, 0, 0.4]},
        'left_shin': {'mass': 2.0, 'position': [0.1, 0, 0.1]},
    },
    'joints': {
        'left_hip': {'position': [0.1, 0, 0.4]},
        'left_knee': {'position': [0.1, 0, 0.1]},
    },
    'left_foot_position': [0.1, 0.05, 0],
    'right_foot_position': [-0.1, 0.05, 0],
}

gravity_effects = gravity_sim.simulate_gravity_effects(robot_state)
print(f"Gravity simulation results:")
print(f"  Center of Mass: {gravity_effects['center_of_mass']}")
print(f"  Stability: {gravity_effects['stability']}")
print(f"  Balance Requirements: {gravity_effects['balance_requirements']}")
```

## Collision Detection and Response

### Advanced Collision Detection Systems

Collision detection in humanoid robots involves complex multi-body systems with many potential collision pairs:

```python
class CollisionDetector:
    """Advanced collision detection system for humanoid robots"""

    def __init__(self):
        self.broad_phase_pairs = set()
        self.narrow_phase_results = {}
        self.collision_filters = {}  # For ignoring certain collisions
        self.contact_callbacks = []

    def add_collision_filter(self, body1: str, body2: str, ignore: bool = True):
        """Add a collision filter to ignore certain body pairs"""
        pair = tuple(sorted([body1, body2]))
        self.collision_filters[pair] = ignore

    def broad_phase_collision(self, bodies: Dict[str, RigidBody]) -> List[Tuple[str, str]]:
        """Perform broad-phase collision detection using bounding volumes"""
        pairs = []
        body_names = list(bodies.keys())

        # Simple N^2 broad phase for demonstration
        # In practice, spatial partitioning (octrees, bounding volume hierarchies) would be used
        for i in range(len(body_names)):
            for j in range(i + 1, len(body_names)):
                body1_name, body2_name = body_names[i], body_names[j]

                # Check collision filter
                pair = tuple(sorted([body1_name, body2_name]))
                if self.collision_filters.get(pair, False):
                    continue

                body1 = bodies[body1_name]
                body2 = bodies[body2_name]

                # Use bounding spheres for broad phase
                pos1, pos2 = body1.position, body2.position
                distance = np.linalg.norm(pos1 - pos2)

                # Approximate bounding sphere radius based on body size
                radius1 = self.estimate_bounding_radius(body1)
                radius2 = self.estimate_bounding_radius(body2)

                if distance < (radius1 + radius2):
                    pairs.append((body1_name, body2_name))

        return pairs

    def estimate_bounding_radius(self, body: RigidBody) -> float:
        """Estimate bounding sphere radius for a body"""
        # Simplified estimation based on mass (assuming similar density)
        # In reality, this would use the actual geometry
        return max(0.1, body.mass * 0.1)

    def narrow_phase_collision(self, body1: RigidBody, body2: RigidBody) -> Optional[Dict]:
        """Perform narrow-phase collision detection"""
        # For this example, we'll use a simplified sphere-sphere collision
        # In a real implementation, this would use GJK, SAT, or other algorithms

        pos1, pos2 = body1.position, body2.position
        distance = np.linalg.norm(pos1 - pos2)

        # Assume unit spheres for simplicity
        radius1 = self.estimate_bounding_radius(body1)
        radius2 = self.estimate_bounding_radius(body2)

        if distance < (radius1 + radius2):
            # Collision detected
            normal = (pos2 - pos1) / distance if distance > 0 else np.array([1, 0, 0])
            penetration_depth = (radius1 + radius2) - distance

            return {
                'position': (pos1 + pos2) / 2,
                'normal': normal,
                'penetration_depth': penetration_depth,
                'distance': distance
            }

        return None

    def detect_all_collisions(self, bodies: Dict[str, RigidBody]) -> List[Dict]:
        """Detect all collisions in the simulation"""
        broad_phase_pairs = self.broad_phase_collision(bodies)
        collisions = []

        for body1_name, body2_name in broad_phase_pairs:
            body1 = bodies[body1_name]
            body2 = bodies[body2_name]

            collision_info = self.narrow_phase_collision(body1, body2)
            if collision_info:
                collision_info['body1'] = body1_name
                collision_info['body2'] = body2_name
                collisions.append(collision_info)

        return collisions

    def resolve_collision(self, body1: RigidBody, body2: RigidBody, collision_info: Dict):
        """Resolve a single collision using impulse-based method"""
        normal = collision_info['normal']
        penetration_depth = collision_info['penetration_depth']

        # Calculate relative velocity at contact point
        relative_velocity = body2.velocity - body1.velocity
        velocity_along_normal = np.dot(relative_velocity, normal)

        # Calculate impulse magnitude
        # For now, using a simplified approach
        e = 0.8  # coefficient of restitution
        total_inverse_mass = (1/body1.mass) + (1/body2.mass)

        # Calculate impulse to prevent interpenetration
        impulse_magnitude = -(1 + e) * velocity_along_normal
        impulse_magnitude += penetration_depth * 1000  # Position correction term

        impulse = impulse_magnitude * normal

        # Apply impulse to both bodies
        body1.velocity -= impulse / body1.mass
        body2.velocity += impulse / body2.mass

        # Apply position correction to prevent sinking
        correction_separation = 0.01  # Small separation to prevent sinking
        correction = min(penetration_depth, correction_separation) * normal
        body1.position -= correction * (1/body1.mass) / total_inverse_mass
        body2.position += correction * (1/body2.mass) / total_inverse_mass

    def resolve_all_collisions(self, bodies: Dict[str, RigidBody], collisions: List[Dict]):
        """Resolve all detected collisions"""
        for collision in collisions:
            body1 = bodies[collision['body1']]
            body2 = bodies[collision['body2']]
            self.resolve_collision(body1, body2, collision)

class ContactMaterial:
    """Defines material properties for contact handling"""

    def __init__(self,
                 friction_coefficient: float = 0.5,
                 restitution: float = 0.8,
                 stiffness: float = 100000,
                 damping: float = 1000):
        self.friction_coefficient = friction_coefficient
        self.restitution = restitution  # Coefficient of restitution
        self.stiffness = stiffness
        self.damping = damping

class AdvancedCollisionSystem:
    """Advanced collision system with material properties and friction"""

    def __init__(self):
        self.collision_detector = CollisionDetector()
        self.contact_materials = {}  # Material properties for different body pairs
        self.friction_solver = FrictionSolver()

    def set_contact_material(self, body1: str, body2: str, material: ContactMaterial):
        """Set contact material properties for a body pair"""
        pair = tuple(sorted([body1, body2]))
        self.contact_materials[pair] = material

    def get_contact_material(self, body1: str, body2: str) -> ContactMaterial:
        """Get contact material for a body pair"""
        pair = tuple(sorted([body1, body2]))
        return self.contact_materials.get(pair, ContactMaterial())

    def detect_and_resolve_collisions(self, bodies: Dict[str, RigidBody]):
        """Complete collision detection and resolution cycle"""
        # Detect collisions
        collisions = self.collision_detector.detect_all_collisions(bodies)

        # Resolve collisions with advanced physics
        for collision in collisions:
            body1 = bodies[collision['body1']]
            body2 = bodies[collision['body2']]

            # Get material properties
            material = self.get_contact_material(body1.name, body2.name)

            # Resolve with material properties
            self.resolve_collision_with_materials(body1, body2, collision, material)

    def resolve_collision_with_materials(self, body1: RigidBody, body2: RigidBody,
                                       collision_info: Dict, material: ContactMaterial):
        """Resolve collision using material properties"""
        normal = collision_info['normal']
        penetration_depth = collision_info['penetration_depth']

        # Relative velocity at contact point
        relative_velocity = body2.velocity - body1.velocity
        velocity_normal = np.dot(relative_velocity, normal)

        # Calculate normal impulse
        e = material.restitution
        total_inverse_mass = (1/body1.mass) + (1/body2.mass)

        # Normal impulse to reverse separation velocity
        impulse_normal = -(1 + e) * velocity_normal
        impulse_normal += penetration_depth * material.stiffness * self.collision_detector.time_step**2

        # Apply normal impulse
        j_normal = impulse_normal / total_inverse_mass
        impulse_vec = j_normal * normal

        body1.velocity -= impulse_vec / body1.mass
        body2.velocity += impulse_vec / body2.mass

        # Calculate and apply friction impulse
        self.friction_solver.apply_friction(body1, body2, normal, j_normal, material.friction_coefficient)

class FrictionSolver:
    """Solves friction forces in collisions"""

    def apply_friction(self, body1: RigidBody, body2: RigidBody,
                      contact_normal: np.ndarray, normal_impulse: float,
                      friction_coeff: float):
        """Apply friction forces based on normal impulse"""
        # Calculate relative tangential velocity
        relative_velocity = body2.velocity - body1.velocity
        normal_velocity = np.dot(relative_velocity, contact_normal) * contact_normal
        tangential_velocity = relative_velocity - normal_velocity

        tangential_speed = np.linalg.norm(tangential_velocity)
        if tangential_speed < 1e-6:  # Very small, no friction needed
            return

        # Calculate friction direction
        friction_direction = -tangential_velocity / tangential_speed

        # Calculate maximum friction impulse (Coulomb friction)
        max_friction_impulse = friction_coeff * abs(normal_impulse)

        # Calculate required friction impulse to stop tangential motion
        total_inverse_mass = (1/body1.mass) + (1/body2.mass)
        required_friction_impulse = tangential_speed / total_inverse_mass

        # Apply friction impulse (with friction limit)
        friction_impulse = min(required_friction_impulse, max_friction_impulse)
        friction_vec = friction_impulse * friction_direction

        body1.velocity -= friction_vec / body1.mass
        body2.velocity += friction_vec / body2.mass

# Example: Setting up collision detection for a humanoid robot
def setup_humanoid_collision_system():
    """Set up collision detection for a humanoid robot"""
    collision_system = AdvancedCollisionSystem()

    # Define contact materials for different body part interactions
    ground_material = ContactMaterial(friction_coefficient=0.7, restitution=0.1)
    self_collision_material = ContactMaterial(friction_coefficient=0.3, restitution=0.5)

    # Set up collision filters to ignore self-collisions between adjacent links
    # (In a real robot, you'd want to be more specific about which pairs to ignore)
    collision_system.collision_detector.add_collision_filter("torso", "head", False)
    collision_system.collision_detector.add_collision_filter("left_thigh", "torso", False)
    collision_system.collision_detector.add_collision_filter("left_shin", "left_thigh", False)

    # Set contact materials
    collision_system.set_contact_material("left_foot", "ground", ground_material)
    collision_system.set_contact_material("right_foot", "ground", ground_material)

    return collision_system

collision_system = setup_humanoid_collision_system()
print("Humanoid collision system initialized with material properties")
```

## Joint Simulation and Constraints

### Realistic Joint Dynamics

Humanoid robots have complex joint systems that require accurate simulation:

```python
class JointSimulator:
    """Simulates realistic joint dynamics with constraints"""

    def __init__(self):
        self.joint_states = {}
        self.joint_limits = {}
        self.joint_dynamics = {}  # Inertia, friction, etc.

    def add_joint(self, name: str, joint_type: JointType, limits: Tuple[float, float],
                  dynamics_params: Dict = None):
        """Add a joint to the simulation"""
        self.joint_states[name] = {
            'position': 0.0,
            'velocity': 0.0,
            'acceleration': 0.0,
            'effort': 0.0
        }
        self.joint_limits[name] = limits

        # Default dynamics parameters if not provided
        self.joint_dynamics[name] = dynamics_params or {
            'inertia': 0.01,
            'friction': 0.1,
            'damping': 0.05,
            'stiction': 0.05
        }

    def update_joint(self, name: str, command_torque: float, dt: float):
        """Update joint state based on applied torque"""
        if name not in self.joint_states:
            return

        state = self.joint_states[name]
        dynamics = self.joint_dynamics[name]
        limits = self.joint_limits[name]

        # Calculate net torque (command - friction - damping)
        friction_torque = self.calculate_friction_torque(state['velocity'], dynamics['friction'])
        damping_torque = -dynamics['damping'] * state['velocity']
        stiction_torque = self.calculate_stiction_torque(state['velocity'], dynamics['stiction'])

        net_torque = command_torque - friction_torque - damping_torque - stiction_torque

        # Apply joint limits using constraint forces
        limit_torque = self.apply_joint_limits(name, state['position'])
        net_torque += limit_torque

        # Calculate acceleration: τ = I * α, so α = τ / I
        acceleration = net_torque / dynamics['inertia']

        # Update state using numerical integration
        state['acceleration'] = acceleration
        state['velocity'] += acceleration * dt
        state['position'] += state['velocity'] * dt
        state['effort'] = net_torque  # Store net effort for reporting

        # Ensure position stays within limits (backup check)
        state['position'] = max(limits[0], min(limits[1], state['position']))

    def calculate_friction_torque(self, velocity: float, friction_coeff: float) -> float:
        """Calculate friction torque based on velocity"""
        # Simple Coulomb friction model
        return friction_coeff * np.sign(velocity) if abs(velocity) > 1e-6 else 0.0

    def calculate_stiction_torque(self, velocity: float, stiction_coeff: float) -> float:
        """Calculate stiction (static friction) torque"""
        # Simplified stiction model
        if abs(velocity) < 1e-3:  # Very low velocity
            return stiction_coeff * np.sign(velocity) if stiction_coeff > abs(velocity) else 0.0
        return 0.0

    def apply_joint_limits(self, name: str, position: float) -> float:
        """Apply soft joint limits using spring-damper model"""
        limits = self.joint_limits[name]
        if position <= limits[0]:  # At or below lower limit
            # Apply restoring torque
            return 100 * (limits[0] - position) - 10 * self.joint_states[name]['velocity']
        elif position >= limits[1]:  # At or above upper limit
            # Apply restoring torque
            return 100 * (limits[1] - position) - 10 * self.joint_states[name]['velocity']
        return 0.0  # Within limits

    def get_joint_state(self, name: str) -> Dict:
        """Get the current state of a joint"""
        return self.joint_states.get(name, {})

class HumanoidJointController:
    """Advanced joint controller for humanoid robots"""

    def __init__(self):
        self.joint_simulator = JointSimulator()
        self.impedance_controllers = {}
        self.trajectory_generators = {}

    def setup_humanoid_joints(self):
        """Set up all joints for a humanoid robot"""
        # Torso to head (neck joint)
        self.joint_simulator.add_joint(
            "neck_pitch", JointType.REVOLUTE, (-0.5, 0.5),
            {'inertia': 0.005, 'friction': 0.05, 'damping': 0.02}
        )

        # Left arm joints
        self.joint_simulator.add_joint(
            "left_shoulder_pitch", JointType.REVOLUTE, (-1.57, 1.57),
            {'inertia': 0.02, 'friction': 0.1, 'damping': 0.05}
        )
        self.joint_simulator.add_joint(
            "left_shoulder_roll", JointType.REVOLUTE, (-1.57, 0.5),
            {'inertia': 0.02, 'friction': 0.1, 'damping': 0.05}
        )
        self.joint_simulator.add_joint(
            "left_elbow", JointType.REVOLUTE, (-1.57, 0.1),
            {'inertia': 0.01, 'friction': 0.08, 'damping': 0.03}
        )

        # Right arm joints
        self.joint_simulator.add_joint(
            "right_shoulder_pitch", JointType.REVOLUTE, (-1.57, 1.57),
            {'inertia': 0.02, 'friction': 0.1, 'damping': 0.05}
        )
        self.joint_simulator.add_joint(
            "right_shoulder_roll", JointType.REVOLUTE, (-0.5, 1.57),
            {'inertia': 0.02, 'friction': 0.1, 'damping': 0.05}
        )
        self.joint_simulator.add_joint(
            "right_elbow", JointType.REVOLUTE, (-0.1, 1.57),
            {'inertia': 0.01, 'friction': 0.08, 'damping': 0.03}
        )

        # Left leg joints
        self.joint_simulator.add_joint(
            "left_hip_pitch", JointType.REVOLUTE, (-1.57, 1.57),
            {'inertia': 0.05, 'friction': 0.2, 'damping': 0.1}
        )
        self.joint_simulator.add_joint(
            "left_hip_roll", JointType.REVOLUTE, (-0.5, 0.5),
            {'inertia': 0.04, 'friction': 0.15, 'damping': 0.08}
        )
        self.joint_simulator.add_joint(
            "left_knee", JointType.REVOLUTE, (-1.57, 0.1),
            {'inertia': 0.04, 'friction': 0.15, 'damping': 0.08}
        )
        self.joint_simulator.add_joint(
            "left_ankle_pitch", JointType.REVOLUTE, (-0.5, 0.5),
            {'inertia': 0.01, 'friction': 0.05, 'damping': 0.02}
        )
        self.joint_simulator.add_joint(
            "left_ankle_roll", JointType.REVOLUTE, (-0.3, 0.3),
            {'inertia': 0.01, 'friction': 0.05, 'damping': 0.02}
        )

        # Right leg joints
        self.joint_simulator.add_joint(
            "right_hip_pitch", JointType.REVOLUTE, (-1.57, 1.57),
            {'inertia': 0.05, 'friction': 0.2, 'damping': 0.1}
        )
        self.joint_simulator.add_joint(
            "right_hip_roll", JointType.REVOLUTE, (-0.5, 0.5),
            {'inertia': 0.04, 'friction': 0.15, 'damping': 0.08}
        )
        self.joint_simulator.add_joint(
            "right_knee", JointType.REVOLUTE, (-1.57, 0.1),
            {'inertia': 0.04, 'friction': 0.15, 'damping': 0.08}
        )
        self.joint_simulator.add_joint(
            "right_ankle_pitch", JointType.REVOLUTE, (-0.5, 0.5),
            {'inertia': 0.01, 'friction': 0.05, 'damping': 0.02}
        )
        self.joint_simulator.add_joint(
            "right_ankle_roll", JointType.REVOLUTE, (-0.3, 0.3),
            {'inertia': 0.01, 'friction': 0.05, 'damping': 0.02}
        )

    def update_all_joints(self, joint_commands: Dict[str, float], dt: float):
        """Update all joints with new commands"""
        for joint_name, command_torque in joint_commands.items():
            self.joint_simulator.update_joint(joint_name, command_torque, dt)

    def get_all_joint_states(self) -> Dict[str, Dict]:
        """Get states of all joints"""
        states = {}
        for joint_name in self.joint_simulator.joint_states.keys():
            states[joint_name] = self.joint_simulator.get_joint_state(joint_name)
        return states

# Example: Setting up and using the humanoid joint controller
joint_controller = HumanoidJointController()
joint_controller.setup_humanoid_joints()

print(f"Humanoid joint controller initialized with {len(joint_controller.joint_simulator.joint_states)} joints")

# Example of updating joints with commands
commands = {
    "left_hip_pitch": 50.0,   # 50 Nm torque
    "right_hip_pitch": 45.0,  # 45 Nm torque
    "left_knee": -30.0,       # -30 Nm torque (flexing)
    "right_knee": -25.0,      # -25 Nm torque (flexing)
}

# Update the joints (in a simulation loop, this would be called every time step)
for step in range(100):
    joint_controller.update_all_joints(commands, 0.001)  # 1ms time step

    if step % 20 == 0:  # Print every 20 steps
        left_knee_state = joint_controller.joint_simulator.get_joint_state("left_knee")
        right_knee_state = joint_controller.joint_simulator.get_joint_state("right_knee")
        print(f"Step {step}: Left knee pos={left_knee_state['position']:.3f}, "
              f"Right knee pos={right_knee_state['position']:.3f}")
```

## Performance Optimization

### Efficient Physics Simulation

Physics simulation performance is crucial for real-time humanoid robot simulation:

```python
class OptimizedPhysicsEngine:
    """Performance-optimized physics engine for humanoid simulation"""

    def __init__(self, max_bodies: int = 100):
        self.max_bodies = max_bodies
        self.bodies = np.zeros((max_bodies, 13))  # Flattened array for performance
        # Each row: [mass, inv_mass, x, y, z, vx, vy, vz, ax, ay, az, unused, unused]
        self.active_bodies = set()
        self.gravity = np.array([0, 0, -9.81])
        self.time_step = 0.001
        self.collision_pairs = []

        # Spatial partitioning for efficient collision detection
        self.spatial_grid = SpatialGrid(cell_size=1.0)

    def add_body_vectorized(self, body_id: int, mass: float, position: np.ndarray,
                           velocity: np.ndarray):
        """Add body using vectorized operations for better performance"""
        if body_id >= self.max_bodies:
            raise ValueError("Body ID exceeds maximum capacity")

        self.bodies[body_id, 0] = mass  # mass
        self.bodies[body_id, 1] = 1.0 / mass if mass > 0 else 0  # inverse mass
        self.bodies[body_id, 2:5] = position  # position (x, y, z)
        self.bodies[body_id, 5:8] = velocity  # velocity (vx, vy, vz)
        self.active_bodies.add(body_id)

        # Add to spatial grid
        self.spatial_grid.add_object(body_id, position)

    def step_simulation_vectorized(self):
        """Vectorized simulation step for better performance"""
        if not self.active_bodies:
            return

        # Get active body indices
        active_indices = list(self.active_bodies)

        # Apply gravity to all active bodies at once
        active_bodies_data = self.bodies[active_indices]

        # Apply gravitational acceleration (a = g for all bodies)
        gravity_acc = np.tile(self.gravity, (len(active_indices), 1))
        active_bodies_data[:, 8:11] = gravity_acc  # Set acceleration

        # Apply damping forces
        velocities = active_bodies_data[:, 5:8]
        damping_force = -0.01 * velocities  # Simple linear damping
        accelerations = damping_force * active_bodies_data[:, 1:2]  # Apply inverse mass
        active_bodies_data[:, 8:11] += accelerations

        # Update positions and velocities
        dt = self.time_step
        active_bodies_data[:, 5:8] += active_bodies_data[:, 8:11] * dt  # Update velocities
        active_bodies_data[:, 2:5] += active_bodies_data[:, 5:8] * dt   # Update positions

        # Update spatial grid with new positions
        for i, body_idx in enumerate(active_indices):
            self.spatial_grid.update_object_position(body_idx, active_bodies_data[i, 2:5])

        # Detect and resolve collisions using spatial grid
        self.detect_and_resolve_collisions_optimized(active_indices)

    def detect_and_resolve_collisions_optimized(self, active_indices):
        """Optimized collision detection using spatial partitioning"""
        # Get nearby pairs from spatial grid
        potential_pairs = self.spatial_grid.get_nearby_pairs()

        for body1_idx, body2_idx in potential_pairs:
            if body1_idx not in self.active_bodies or body2_idx not in self.active_bodies:
                continue

            # Simple sphere collision detection
            pos1 = self.bodies[body1_idx, 2:5]
            pos2 = self.bodies[body2_idx, 2:5]
            distance = np.linalg.norm(pos1 - pos2)

            # Assume unit spheres
            if distance < 2.0:
                self.resolve_collision_optimized(body1_idx, body2_idx, pos1, pos2, distance)

    def resolve_collision_optimized(self, idx1: int, idx2: int, pos1: np.ndarray,
                                  pos2: np.ndarray, distance: float):
        """Optimized collision resolution"""
        if distance < 1e-6:  # Avoid division by zero
            return

        normal = (pos2 - pos1) / distance
        relative_velocity = (self.bodies[idx2, 5:8] - self.bodies[idx1, 5:8])
        velocity_along_normal = np.dot(relative_velocity, normal)

        if velocity_along_normal >= 0:  # Objects separating
            return

        # Calculate impulse (simplified for equal masses)
        e = 0.8  # coefficient of restitution
        inv_mass1 = self.bodies[idx1, 1]
        inv_mass2 = self.bodies[idx2, 1]

        impulse_magnitude = -(1 + e) * velocity_along_normal / (inv_mass1 + inv_mass2)
        impulse = impulse_magnitude * normal

        # Apply impulse
        self.bodies[idx1, 5:8] -= impulse * inv_mass1
        self.bodies[idx2, 5:8] += impulse * inv_mass2

class SpatialGrid:
    """Spatial partitioning grid for efficient collision detection"""

    def __init__(self, cell_size: float = 1.0):
        self.cell_size = cell_size
        self.grid = {}
        self.object_cells = {}  # object_id -> set of cell_keys

    def get_cell_key(self, position: np.ndarray) -> tuple:
        """Get the grid cell key for a position"""
        x, y, z = position
        return (int(x // self.cell_size),
                int(y // self.cell_size),
                int(z // self.cell_size))

    def add_object(self, obj_id: int, position: np.ndarray):
        """Add an object to the spatial grid"""
        cell_key = self.get_cell_key(position)
        if cell_key not in self.grid:
            self.grid[cell_key] = set()
        self.grid[cell_key].add(obj_id)
        self.object_cells[obj_id] = {cell_key}

    def update_object_position(self, obj_id: int, new_position: np.ndarray):
        """Update an object's position in the grid"""
        # Remove from old cells
        old_cells = self.object_cells.get(obj_id, set())
        for cell_key in old_cells:
            if cell_key in self.grid and obj_id in self.grid[cell_key]:
                self.grid[cell_key].remove(obj_id)
                if not self.grid[cell_key]:  # Clean up empty cells
                    del self.grid[cell_key]

        # Add to new cell
        new_cell_key = self.get_cell_key(new_position)
        if new_cell_key not in self.grid:
            self.grid[new_cell_key] = set()
        self.grid[new_cell_key].add(obj_id)
        self.object_cells[obj_id] = {new_cell_key}

    def get_nearby_pairs(self) -> List[Tuple[int, int]]:
        """Get all potentially colliding object pairs"""
        pairs = set()

        for cell_objects in self.grid.values():
            cell_list = list(cell_objects)
            for i in range(len(cell_list)):
                for j in range(i + 1, len(cell_list)):
                    pairs.add(tuple(sorted([cell_list[i], cell_list[j]])))

        return list(pairs)

# Example usage of optimized physics engine
optimized_engine = OptimizedPhysicsEngine(max_bodies=50)

# Add some bodies to the optimized engine
for i in range(10):
    mass = 1.0
    position = np.array([i * 0.5, 0, 2.0 + i * 0.1])
    velocity = np.array([0, 0, 0])
    optimized_engine.add_body_vectorized(i, mass, position, velocity)

print(f"Optimized physics engine initialized with {len(optimized_engine.active_bodies)} bodies")

# Run optimized simulation
import time
start_time = time.time()
for step in range(1000):  # 1000 steps
    optimized_engine.step_simulation_vectorized()

    if step % 200 == 0:
        pos = optimized_engine.bodies[0, 2:5]  # Position of first body
        print(f"Optimized sim - Step {step}: Position = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

end_time = time.time()
print(f"Completed 1000 simulation steps in {end_time - start_time:.4f} seconds")
```

Physics simulation in Gazebo provides the foundation for realistic humanoid robot behavior. Accurate gravity modeling, collision detection, and joint dynamics are essential for creating believable digital twins that can effectively bridge the sim-to-real gap. The combination of efficient algorithms and realistic physical models enables developers to test and validate complex humanoid behaviors in a safe, repeatable virtual environment.