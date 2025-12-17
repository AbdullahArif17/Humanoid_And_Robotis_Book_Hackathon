---
sidebar_position: 2
---

# Isaac Sim

## Introduction to Isaac Sim

Isaac Sim is NVIDIA's advanced robotics simulation environment built on the Omniverse platform. It provides photorealistic simulation capabilities specifically designed for robotics applications, enabling researchers and developers to test and validate complex robotic systems in a safe, controlled virtual environment. For humanoid robotics, Isaac Sim offers the high-fidelity physics, realistic sensor simulation, and detailed environments necessary to develop and test sophisticated humanoid behaviors before deployment to physical hardware.

### Key Features of Isaac Sim

Isaac Sim combines the power of NVIDIA's Omniverse platform with robotics-specific features:

- **Photorealistic Rendering**: RTX-powered rendering for realistic visual simulation
- **Accurate Physics**: PhysX-based physics simulation with complex contact dynamics
- **Realistic Sensor Simulation**: Camera, LiDAR, IMU, and other sensor models
- **Multi-Robot Simulation**: Support for complex multi-robot scenarios
- **ROS 2 Integration**: Native support for ROS 2 communication
- **AI Training Environment**: Domain randomization and synthetic data generation

```python
# Isaac Sim Environment Manager
import carb
import omni
import omni.usd
import omni.graph.core as og
from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
from typing import Dict, List, Optional, Tuple
import asyncio
import logging

class IsaacSimManager:
    """Manages Isaac Sim environment and assets"""

    def __init__(self):
        self.stage = None
        self.world = None
        self.robots = {}
        self.sensors = {}
        self.assets = {}
        self.physics_settings = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info("Isaac Sim Manager initialized")

    def initialize_stage(self, stage_path: str = "/IsaacSim/World"):
        """Initialize the USD stage for simulation"""
        self.stage = omni.usd.get_context().get_stage()

        # Create world prim
        self.world = UsdGeom.Xform.Define(self.stage, stage_path)

        # Set up physics scene
        self._setup_physics_scene()

        self.logger.info(f"Stage initialized at: {stage_path}")
        return self.stage

    def _setup_physics_scene(self):
        """Setup physics scene with PhysX"""
        # Create physics scene
        scene_path = self.world.GetPath().AppendChild("PhysicsScene")
        physics_scene = UsdPhysics.Scene.Define(self.stage, scene_path)

        # Configure physics settings
        physics_scene.CreateGravityAttr().Set([0.0, 0.0, -9.81])
        physics_scene.CreateTimestepAttr().Set(1.0/60.0)  # 60 Hz

        self.physics_settings = {
            'gravity': [0.0, 0.0, -9.81],
            'timestep': 1.0/60.0,
            'solver_position_iteration_count': 8,
            'solver_velocity_iteration_count': 1,
            'default_buffer_size_multiplier': 16.0
        }

        self.logger.info("Physics scene configured")

    def load_robot_asset(self, asset_path: str, robot_name: str, position: List[float] = [0, 0, 1]):
        """Load a robot asset into the simulation"""
        # Import robot asset
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.stage import add_reference_to_stage

        # Try to get assets from nucleus
        assets_root_path = get_assets_root_path()
        if assets_root_path:
            full_asset_path = f"{assets_root_path}/Isaac/Robots/{asset_path}"
        else:
            full_asset_path = asset_path

        # Add robot to stage
        robot_prim_path = f"/World/{robot_name}"
        robot_prim = add_reference_to_stage(
            usd_path=full_asset_path,
            prim_path=robot_prim_path
        )

        # Set initial position
        robot_xform = UsdGeom.Xformable(robot_prim)
        robot_xform.AddTranslateOp().Set(Gf.Vec3d(*position))

        # Store robot reference
        self.robots[robot_name] = {
            'prim': robot_prim,
            'path': robot_prim_path,
            'position': position,
            'sensors': [],
            'controllers': []
        }

        self.logger.info(f"Loaded robot {robot_name} from {full_asset_path}")
        return robot_prim

    def add_sensor_to_robot(self, robot_name: str, sensor_type: str,
                          sensor_name: str, position: List[float] = [0, 0, 0]):
        """Add a sensor to a robot in the simulation"""
        if robot_name not in self.robots:
            raise ValueError(f"Robot {robot_name} not found in simulation")

        robot_path = self.robots[robot_name]['path']
        sensor_path = f"{robot_path}/{sensor_name}"

        # Create sensor based on type
        if sensor_type == "camera":
            sensor = self._create_camera_sensor(sensor_path, position)
        elif sensor_type == "lidar":
            sensor = self._create_lidar_sensor(sensor_path, position)
        elif sensor_type == "imu":
            sensor = self._create_imu_sensor(sensor_path, position)
        else:
            raise ValueError(f"Unsupported sensor type: {sensor_type}")

        # Add sensor to robot
        self.robots[robot_name]['sensors'].append({
            'name': sensor_name,
            'type': sensor_type,
            'path': sensor_path,
            'position': position,
            'sensor_obj': sensor
        })

        self.sensors[sensor_name] = sensor

        self.logger.info(f"Added {sensor_type} sensor '{sensor_name}' to robot '{robot_name}'")
        return sensor

    def _create_camera_sensor(self, path: str, position: List[float]):
        """Create a camera sensor in the simulation"""
        from omni.isaac.sensor import Camera

        # Create camera prim
        camera_prim = UsdGeom.Camera.Define(self.stage, path)
        camera_prim.GetPrim().CreateAttribute("sensor:modality", Sdf.ValueTypeNames.Token).Set("camera")

        # Set camera properties
        camera_prim.GetFocalLengthAttr().Set(24.0)
        camera_prim.GetHorizontalApertureAttr().Set(20.955)
        camera_prim.GetVerticalApertureAttr().Set(15.2908)

        # Apply transform
        camera_xform = UsdGeom.Xformable(camera_prim)
        camera_xform.AddTranslateOp().Set(Gf.Vec3d(*position))

        # Create Isaac camera wrapper
        camera = Camera(
            prim_path=path,
            frequency=30,
            resolution=(640, 480)
        )

        return camera

    def _create_lidar_sensor(self, path: str, position: List[float]):
        """Create a LiDAR sensor in the simulation"""
        from omni.isaac.sensor import RotatingLidarPhysX

        # Create LiDAR prim
        lidar_prim = UsdGeom.Xform.Define(self.stage, path)
        lidar_prim.GetPrim().CreateAttribute("sensor:modality", Sdf.ValueTypeNames.Token).Set("lidar")

        # Apply transform
        lidar_xform = UsdGeom.Xformable(lidar_prim)
        lidar_xform.AddTranslateOp().Set(Gf.Vec3d(*position))

        # Create Isaac LiDAR wrapper
        lidar = RotatingLidarPhysX(
            prim_path=path,
            translation=np.array(position),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            rpm=30,
            samples_per_ring=720,
            max_range=10.0,
            fov=360.0
        )

        return lidar

    def _create_imu_sensor(self, path: str, position: List[float]):
        """Create an IMU sensor in the simulation"""
        from omni.isaac.sensor import IMU

        # Create IMU prim
        imu_prim = UsdGeom.Xform.Define(self.stage, path)
        imu_prim.GetPrim().CreateAttribute("sensor:modality", Sdf.ValueTypeNames.Token).Set("imu")

        # Apply transform
        imu_xform = UsdGeom.Xformable(imu_prim)
        imu_xform.AddTranslateOp().Set(Gf.Vec3d(*position))

        # Create Isaac IMU wrapper
        imu = IMU(
            prim_path=path,
            frequency=100
        )

        return imu

    def create_environment(self, env_config: Dict):
        """Create a simulation environment"""
        # Create ground plane
        self._create_ground_plane(env_config.get('ground_material', 'default'))

        # Create obstacles
        obstacles = env_config.get('obstacles', [])
        for i, obstacle in enumerate(obstacles):
            self._create_obstacle(f"obstacle_{i}", obstacle)

        # Create lighting
        self._create_lighting(env_config.get('lighting', {}))

        # Create sky dome if needed
        if env_config.get('sky_dome', True):
            self._create_sky_dome()

        self.logger.info(f"Environment created with {len(obstacles)} obstacles")

    def _create_ground_plane(self, material: str = 'default'):
        """Create a ground plane in the simulation"""
        plane_path = "/World/GroundPlane"
        plane = UsdGeom.Mesh.Define(self.stage, plane_path)

        # Create a large plane
        plane.CreatePointsAttr().Set([
            Gf.Vec3f(-100, -100, 0),
            Gf.Vec3f(100, -100, 0),
            Gf.Vec3f(100, 100, 0),
            Gf.Vec3f(-100, 100, 0)
        ])

        plane.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
        plane.CreateFaceVertexCountsAttr().Set([3, 3])

        # Apply material
        self._apply_material(plane, material)

        # Make it a static collider
        from omni.physx.scripts import particleUtils
        particleUtils.setStaticCollider(plane.GetPrim())

    def _create_obstacle(self, name: str, config: Dict):
        """Create an obstacle in the environment"""
        obstacle_path = f"/World/{name}"

        # Determine obstacle type
        obs_type = config.get('type', 'box')
        position = config.get('position', [0, 0, 0])
        size = config.get('size', [1, 1, 1])
        material = config.get('material', 'default')

        if obs_type == 'box':
            obstacle = UsdGeom.Cube.Define(self.stage, obstacle_path)
            obstacle.GetSizeAttr().Set(max(size))
        elif obs_type == 'sphere':
            obstacle = UsdGeom.Sphere.Define(self.stage, obstacle_path)
            obstacle.GetRadiusAttr().Set(max(size) / 2)
        elif obs_type == 'cylinder':
            obstacle = UsdGeom.Cylinder.Define(self.stage, obstacle_path)
            obstacle.GetRadiusAttr().Set(size[0] / 2)
            obstacle.GetHeightAttr().Set(size[2])
        else:
            raise ValueError(f"Unsupported obstacle type: {obs_type}")

        # Apply transform
        xform = UsdGeom.Xformable(obstacle)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))

        # Apply material
        self._apply_material(obstacle, material)

        # Make it a collider
        from omni.physx.scripts import particleUtils
        particleUtils.setStaticCollider(obstacle.GetPrim())

    def _apply_material(self, geom, material_name: str):
        """Apply material to geometry"""
        # This would connect to Omniverse Nucleus for material library
        pass

    def _create_lighting(self, lighting_config: Dict):
        """Create lighting setup"""
        # Create dome light for ambient lighting
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr().Set(lighting_config.get('intensity', 1000))
        dome_light.CreateColorAttr().Set(Gf.Vec3f(1, 1, 1))

        # Create directional light for shadows
        sun_light_path = "/World/SunLight"
        sun_light = UsdLux.DistantLight.Define(self.stage, sun_light_path)
        sun_light.CreateIntensityAttr().Set(lighting_config.get('sun_intensity', 3000))
        sun_light.CreateColorAttr().Set(Gf.Vec3f(1, 0.9, 0.8))
        sun_light.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    def _create_sky_dome(self):
        """Create sky dome for realistic environment"""
        # This would typically use HDRI or procedural sky
        pass

    def run_simulation(self, steps: int = 1000):
        """Run the simulation for specified steps"""
        from omni.isaac.core import World

        # Initialize world if not already done
        if not hasattr(self, 'world_obj'):
            self.world_obj = World(stage_units_in_meters=1.0)

        # Run simulation steps
        for step in range(steps):
            self.world_obj.step(render=True)

            if step % 100 == 0:
                self.logger.info(f"Simulation step {step}/{steps}")

        self.logger.info(f"Completed {steps} simulation steps")

# Example usage
def create_humanoid_simulation():
    """Create a humanoid robot simulation environment"""

    # Initialize Isaac Sim manager
    sim_manager = IsaacSimManager()

    # Initialize stage
    stage = sim_manager.initialize_stage()

    # Create environment configuration
    env_config = {
        'ground_material': 'wood',
        'lighting': {
            'intensity': 500,
            'sun_intensity': 2000
        },
        'obstacles': [
            {'type': 'box', 'position': [2, 0, 0.5], 'size': [0.5, 0.5, 1.0]},
            {'type': 'sphere', 'position': [-1, 1, 0.3], 'size': [0.6, 0.6, 0.6]},
            {'type': 'cylinder', 'position': [0, -2, 0.5], 'size': [0.4, 0.4, 1.0]}
        ]
    }

    # Create environment
    sim_manager.create_environment(env_config)

    # Load humanoid robot
    sim_manager.load_robot_asset(
        asset_path="Humanoid/robot.usd",
        robot_name="humanoid_robot",
        position=[0, 0, 1.0]
    )

    # Add sensors to robot
    sim_manager.add_sensor_to_robot(
        robot_name="humanoid_robot",
        sensor_type="camera",
        sensor_name="head_camera",
        position=[0.2, 0, 0.8]
    )

    sim_manager.add_sensor_to_robot(
        robot_name="humanoid_robot",
        sensor_type="imu",
        sensor_name="torso_imu",
        position=[0, 0, 0.5]
    )

    sim_manager.add_sensor_to_robot(
        robot_name="humanoid_robot",
        sensor_type="lidar",
        sensor_name="spinning_lidar",
        position=[0, 0, 1.0]
    )

    print("Humanoid simulation environment created successfully!")
    print(f"Robot: humanoid_robot with {len(sim_manager.robots['humanoid_robot']['sensors'])} sensors")
    print(f"Sensors: {[s['name'] for s in sim_manager.robots['humanoid_robot']['sensors']]}")

# Run the example
create_humanoid_simulation()
```

## Omniverse Integration

### Understanding Omniverse for Robotics

Isaac Sim leverages NVIDIA's Omniverse platform, which provides a collaborative 3D simulation and design platform built for real-time, physically accurate simulation:

```python
# Omniverse Integration for Robotics
import omni.kit.commands
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.carb import carb_settings_get
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import carb
import asyncio

class OmniverseRoboticsBridge:
    """Bridges Omniverse capabilities with robotics applications"""

    def __init__(self):
        self.stage = get_current_stage()
        self.session = None
        self.collaboration_enabled = False
        self.streaming_enabled = False

        self.logger = logging.getLogger(__name__)
        self.logger.info("Omniverse Robotics Bridge initialized")

    def enable_collaboration(self, project_path: str, user_name: str):
        """Enable multi-user collaboration for robotics projects"""
        # This would connect to Omniverse Nucleus server
        self.collaboration_enabled = True
        self.project_path = project_path
        self.user_name = user_name

        self.logger.info(f"Collaboration enabled for project: {project_path} as user: {user_name}")
        return True

    def setup_streaming(self, stream_config: Dict):
        """Setup real-time streaming for remote access"""
        self.streaming_enabled = True
        self.stream_config = stream_config

        # Configure streaming settings
        settings = carb.settings.get_settings()
        settings.set("/app/window/dpiScale", stream_config.get('dpi_scale', 1.0))
        settings.set("/app/isStreaming", True)

        self.logger.info("Streaming configured for remote access")
        return True

    def create_robot_assembly(self, robot_config: Dict) -> str:
        """Create a robot assembly using Omniverse USD composition"""
        robot_name = robot_config['name']
        robot_path = f"/World/{robot_name}"

        # Create main robot xform
        robot_xform = UsdGeom.Xform.Define(self.stage, robot_path)

        # Add robot-specific attributes
        robot_prim = robot_xform.GetPrim()
        robot_prim.CreateAttribute("robot:type", Sdf.ValueTypeNames.Token).Set("humanoid")
        robot_prim.CreateAttribute("robot:model", Sdf.ValueTypeNames.String).Set(robot_config.get('model', 'generic'))

        # Create robot links and joints
        links = robot_config.get('links', [])
        for link_config in links:
            self._create_robot_link(robot_path, link_config)

        joints = robot_config.get('joints', [])
        for joint_config in joints:
            self._create_robot_joint(robot_path, joint_config)

        # Add physics properties
        self._setup_robot_physics(robot_path, robot_config.get('physics', {}))

        self.logger.info(f"Created robot assembly: {robot_name}")
        return robot_path

    def _create_robot_link(self, robot_path: str, link_config: Dict):
        """Create a robot link in the USD stage"""
        link_name = link_config['name']
        link_path = f"{robot_path}/{link_name}"

        # Determine geometry type
        geom_type = link_config.get('geometry', {}).get('type', 'box')

        if geom_type == 'box':
            link_geom = UsdGeom.Cube.Define(self.stage, link_path)
            size = link_config['geometry'].get('size', [0.1, 0.1, 0.1])
            link_geom.GetSizeAttr().Set(max(size))
        elif geom_type == 'sphere':
            link_geom = UsdGeom.Sphere.Define(self.stage, link_path)
            radius = link_config['geometry'].get('radius', 0.1)
            link_geom.GetRadiusAttr().Set(radius)
        elif geom_type == 'cylinder':
            link_geom = UsdGeom.Cylinder.Define(self.stage, link_path)
            radius = link_config['geometry'].get('radius', 0.1)
            height = link_config['geometry'].get('height', 0.2)
            link_geom.GetRadiusAttr().Set(radius)
            link_geom.GetHeightAttr().Set(height)
        else:
            # Default to capsule for limbs
            link_geom = UsdGeom.Capsule.Define(self.stage, link_path)
            radius = link_config['geometry'].get('radius', 0.05)
            height = link_config['geometry'].get('height', 0.3)
            link_geom.GetRadiusAttr().Set(radius)
            link_geom.GetHeightAttr().Set(height)

        # Apply transform
        position = link_config.get('position', [0, 0, 0])
        rotation = link_config.get('rotation', [0, 0, 0])

        xform = UsdGeom.Xformable(link_geom)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))

        # Add physics properties
        self._setup_link_physics(link_path, link_config.get('physics', {}))

    def _create_robot_joint(self, robot_path: str, joint_config: Dict):
        """Create a robot joint in the USD stage"""
        joint_name = joint_config['name']
        joint_path = f"{robot_path}/{joint_name}"

        # Create joint based on type
        joint_type = joint_config['type']
        parent_link = f"{robot_path}/{joint_config['parent']}"
        child_link = f"{robot_path}/{joint_config['child']}"

        # In USD, joints are represented as custom relationships
        # This is a simplified representation
        joint_prim = UsdGeom.Xform.Define(self.stage, joint_path)

        # Add joint attributes
        joint_prim.GetPrim().CreateAttribute("joint:type", Sdf.ValueTypeNames.Token).Set(joint_type)
        joint_prim.GetPrim().CreateRelationship("joint:parent").SetTargets([parent_link])
        joint_prim.GetPrim().CreateRelationship("joint:child").SetTargets([child_link])

        # Add joint limits if specified
        if 'limits' in joint_config:
            limits = joint_config['limits']
            joint_prim.GetPrim().CreateAttribute("joint:lower_limit", Sdf.ValueTypeNames.Float).Set(limits.get('lower', -1.57))
            joint_prim.GetPrim().CreateAttribute("joint:upper_limit", Sdf.ValueTypeNames.Float).Set(limits.get('upper', 1.57))
            joint_prim.GetPrim().CreateAttribute("joint:effort_limit", Sdf.ValueTypeNames.Float).Set(limits.get('effort', 100.0))
            joint_prim.GetPrim().CreateAttribute("joint:velocity_limit", Sdf.ValueTypeNames.Float).Set(limits.get('velocity', 1.0))

    def _setup_robot_physics(self, robot_path: str, physics_config: Dict):
        """Setup physics properties for robot"""
        # Add mass properties to main robot body
        robot_prim = self.stage.GetPrimAtPath(robot_path)

        # Create rigid body collection
        physics_scene_path = "/World/PhysicsScene"
        physics_scene = UsdPhysics.Scene(self.stage.GetPrimAtPath(physics_scene_path))

        # Add robot to physics scene
        physics_scene.CreatePhysicsCollectionRel().AddTarget(robot_path)

    def _setup_link_physics(self, link_path: str, physics_config: Dict):
        """Setup physics properties for a link"""
        link_prim = self.stage.GetPrimAtPath(link_path)

        # Add mass property
        mass = physics_config.get('mass', 1.0)
        link_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(mass)

        # Add collision properties
        if physics_config.get('collidable', True):
            # Make this a collision mesh
            pass

    def import_external_model(self, file_path: str, import_config: Dict) -> str:
        """Import external robot model (URDF, SDF, etc.) into Omniverse"""
        # This would use Omniverse's import functionality
        imported_path = f"/World/Imported_{import_config.get('name', 'robot')}"

        # For now, simulate the import process
        self.logger.info(f"Imported model from {file_path} to {imported_path}")
        return imported_path

    def export_robot_config(self, robot_path: str) -> Dict:
        """Export robot configuration for external use"""
        robot_config = {
            'name': robot_path.split('/')[-1],
            'links': [],
            'joints': [],
            'sensors': []
        }

        # Traverse the USD stage to extract robot information
        robot_prim = self.stage.GetPrimAtPath(robot_path)

        for child in robot_prim.GetAllChildren():
            prim_name = child.GetName()
            prim_type = child.GetTypeName()

            if prim_type in ['Cube', 'Sphere', 'Cylinder', 'Capsule']:
                # This is a link
                link_config = {
                    'name': prim_name,
                    'type': prim_type.lower(),
                    'position': [0, 0, 0],  # Would extract from transform
                    'geometry': {}  # Would extract geometry properties
                }
                robot_config['links'].append(link_config)
            elif 'joint' in prim_name.lower():
                # This is a joint
                joint_config = {
                    'name': prim_name,
                    'type': child.GetAttribute('joint:type').Get() if child.HasAttribute('joint:type') else 'fixed',
                    'parent': '',  # Would extract from relationship
                    'child': ''     # Would extract from relationship
                }
                robot_config['joints'].append(joint_config)

        return robot_config

    def enable_real_time_rendering(self, quality_level: str = 'production'):
        """Enable real-time rendering for robotics simulation"""
        settings = carb.settings.get_settings()

        # Set rendering quality based on use case
        quality_settings = {
            'development': {
                'renderQuality': 'High',
                'streaming/enableFramerateLimit': True,
                'streaming/framerateLimit': 60
            },
            'production': {
                'renderQuality': 'Production',
                'streaming/enableFramerateLimit': True,
                'streaming/framerateLimit': 60
            },
            'research': {
                'renderQuality': 'Maximum',
                'streaming/enableFramerateLimit': False,
                'streaming/framerateLimit': 120
            }
        }

        if quality_level in quality_settings:
            for key, value in quality_settings[quality_level].items():
                settings.set(f"/rtx/{key}", value)

        self.logger.info(f"Real-time rendering enabled at {quality_level} quality")
        return True

# Example: Creating a humanoid robot in Omniverse
def create_advanced_humanoid():
    """Create an advanced humanoid robot using Omniverse capabilities"""

    # Initialize the Omniverse bridge
    omniverse_bridge = OmniverseRoboticsBridge()

    # Enable collaboration for team development
    omniverse_bridge.enable_collaboration(
        project_path="/Projects/HumanoidRobot",
        user_name="developer"
    )

    # Enable real-time rendering
    omniverse_bridge.enable_real_time_rendering(quality_level='research')

    # Define humanoid robot configuration
    humanoid_config = {
        'name': 'advanced_humanoid',
        'model': 'atlas_derived',
        'links': [
            # Torso
            {
                'name': 'pelvis',
                'geometry': {'type': 'box', 'size': [0.3, 0.2, 0.4]},
                'position': [0, 0, 0.8],
                'physics': {'mass': 8.0, 'collidable': True}
            },
            {
                'name': 'torso',
                'geometry': {'type': 'capsule', 'radius': 0.15, 'height': 0.6},
                'position': [0, 0, 1.2],
                'physics': {'mass': 10.0, 'collidable': True}
            },
            {
                'name': 'head',
                'geometry': {'type': 'sphere', 'radius': 0.15},
                'position': [0, 0, 1.7],
                'physics': {'mass': 2.0, 'collidable': True}
            },
            # Left arm
            {
                'name': 'left_shoulder',
                'geometry': {'type': 'capsule', 'radius': 0.06, 'height': 0.2},
                'position': [0.2, 0.1, 1.5],
                'physics': {'mass': 1.5, 'collidable': True}
            },
            {
                'name': 'left_upper_arm',
                'geometry': {'type': 'capsule', 'radius': 0.05, 'height': 0.3},
                'position': [0.2, 0.1, 1.2],
                'physics': {'mass': 1.5, 'collidable': True}
            },
            {
                'name': 'left_lower_arm',
                'geometry': {'type': 'capsule', 'radius': 0.04, 'height': 0.25},
                'position': [0.2, 0.1, 0.95],
                'physics': {'mass': 1.0, 'collidable': True}
            },
            # Right arm
            {
                'name': 'right_shoulder',
                'geometry': {'type': 'capsule', 'radius': 0.06, 'height': 0.2},
                'position': [-0.2, 0.1, 1.5],
                'physics': {'mass': 1.5, 'collidable': True}
            },
            {
                'name': 'right_upper_arm',
                'geometry': {'type': 'capsule', 'radius': 0.05, 'height': 0.3},
                'position': [-0.2, 0.1, 1.2],
                'physics': {'mass': 1.5, 'collidable': True}
            },
            {
                'name': 'right_lower_arm',
                'geometry': {'type': 'capsule', 'radius': 0.04, 'height': 0.25},
                'position': [-0.2, 0.1, 0.95],
                'physics': {'mass': 1.0, 'collidable': True}
            },
            # Left leg
            {
                'name': 'left_hip',
                'geometry': {'type': 'capsule', 'radius': 0.07, 'height': 0.2},
                'position': [0.1, 0, 0.6],
                'physics': {'mass': 2.0, 'collidable': True}
            },
            {
                'name': 'left_upper_leg',
                'geometry': {'type': 'capsule', 'radius': 0.06, 'height': 0.4},
                'position': [0.1, 0, 0.3],
                'physics': {'mass': 2.0, 'collidable': True}
            },
            {
                'name': 'left_lower_leg',
                'geometry': {'type': 'capsule', 'radius': 0.05, 'height': 0.35},
                'position': [0.1, 0, -0.1],
                'physics': {'mass': 1.5, 'collidable': True}
            },
            {
                'name': 'left_foot',
                'geometry': {'type': 'box', 'size': [0.15, 0.08, 0.05]},
                'position': [0.1, 0, -0.35],
                'physics': {'mass': 0.8, 'collidable': True}
            },
            # Right leg
            {
                'name': 'right_hip',
                'geometry': {'type': 'capsule', 'radius': 0.07, 'height': 0.2},
                'position': [-0.1, 0, 0.6],
                'physics': {'mass': 2.0, 'collidable': True}
            },
            {
                'name': 'right_upper_leg',
                'geometry': {'type': 'capsule', 'radius': 0.06, 'height': 0.4},
                'position': [-0.1, 0, 0.3],
                'physics': {'mass': 2.0, 'collidable': True}
            },
            {
                'name': 'right_lower_leg',
                'geometry': {'type': 'capsule', 'radius': 0.05, 'height': 0.35},
                'position': [-0.1, 0, -0.1],
                'physics': {'mass': 1.5, 'collidable': True}
            },
            {
                'name': 'right_foot',
                'geometry': {'type': 'box', 'size': [0.15, 0.08, 0.05]},
                'position': [-0.1, 0, -0.35],
                'physics': {'mass': 0.8, 'collidable': True}
            }
        ],
        'joints': [
            # Spine joints
            {
                'name': 'pelvis_to_torso',
                'type': 'ball',
                'parent': 'pelvis',
                'child': 'torso',
                'limits': {'effort': 100.0, 'velocity': 2.0}
            },
            {
                'name': 'torso_to_head',
                'type': 'revolute',
                'parent': 'torso',
                'child': 'head',
                'limits': {'lower': -0.5, 'upper': 0.5, 'effort': 20.0, 'velocity': 1.0}
            },
            # Left arm joints
            {
                'name': 'torso_to_left_shoulder',
                'type': 'revolute',
                'parent': 'torso',
                'child': 'left_shoulder',
                'limits': {'lower': -1.57, 'upper': 1.57, 'effort': 50.0, 'velocity': 1.0}
            },
            {
                'name': 'left_shoulder_to_arm',
                'type': 'revolute',
                'parent': 'left_shoulder',
                'child': 'left_upper_arm',
                'limits': {'lower': -1.57, 'upper': 1.57, 'effort': 40.0, 'velocity': 1.0}
            },
            {
                'name': 'left_elbow',
                'type': 'revolute',
                'parent': 'left_upper_arm',
                'child': 'left_lower_arm',
                'limits': {'lower': -1.57, 'upper': 0.1, 'effort': 30.0, 'velocity': 1.0}
            },
            # Right arm joints
            {
                'name': 'torso_to_right_shoulder',
                'type': 'revolute',
                'parent': 'torso',
                'child': 'right_shoulder',
                'limits': {'lower': -1.57, 'upper': 1.57, 'effort': 50.0, 'velocity': 1.0}
            },
            {
                'name': 'right_shoulder_to_arm',
                'type': 'revolute',
                'parent': 'right_shoulder',
                'child': 'right_upper_arm',
                'limits': {'lower': -1.57, 'upper': 1.57, 'effort': 40.0, 'velocity': 1.0}
            },
            {
                'name': 'right_elbow',
                'type': 'revolute',
                'parent': 'right_upper_arm',
                'child': 'right_lower_arm',
                'limits': {'lower': -1.57, 'upper': 0.1, 'effort': 30.0, 'velocity': 1.0}
            },
            # Left leg joints
            {
                'name': 'pelvis_to_left_hip',
                'type': 'revolute',
                'parent': 'pelvis',
                'child': 'left_hip',
                'limits': {'lower': -1.57, 'upper': 1.57, 'effort': 80.0, 'velocity': 1.0}
            },
            {
                'name': 'left_hip_joint',
                'type': 'revolute',
                'parent': 'left_hip',
                'child': 'left_upper_leg',
                'limits': {'lower': -1.57, 'upper': 0.1, 'effort': 70.0, 'velocity': 1.0}
            },
            {
                'name': 'left_knee',
                'type': 'revolute',
                'parent': 'left_upper_leg',
                'child': 'left_lower_leg',
                'limits': {'lower': -1.57, 'upper': 0.1, 'effort': 60.0, 'velocity': 1.0}
            },
            {
                'name': 'left_ankle',
                'type': 'revolute',
                'parent': 'left_lower_leg',
                'child': 'left_foot',
                'limits': {'lower': -0.5, 'upper': 0.5, 'effort': 40.0, 'velocity': 1.0}
            },
            # Right leg joints
            {
                'name': 'pelvis_to_right_hip',
                'type': 'revolute',
                'parent': 'pelvis',
                'child': 'right_hip',
                'limits': {'lower': -1.57, 'upper': 1.57, 'effort': 80.0, 'velocity': 1.0}
            },
            {
                'name': 'right_hip_joint',
                'type': 'revolute',
                'parent': 'right_hip',
                'child': 'right_upper_leg',
                'limits': {'lower': -1.57, 'upper': 0.1, 'effort': 70.0, 'velocity': 1.0}
            },
            {
                'name': 'right_knee',
                'type': 'revolute',
                'parent': 'right_upper_leg',
                'child': 'right_lower_leg',
                'limits': {'lower': -1.57, 'upper': 0.1, 'effort': 60.0, 'velocity': 1.0}
            },
            {
                'name': 'right_ankle',
                'type': 'revolute',
                'parent': 'right_lower_leg',
                'child': 'right_foot',
                'limits': {'lower': -0.5, 'upper': 0.5, 'effort': 40.0, 'velocity': 1.0}
            }
        ],
        'physics': {
            'gravity': [0, 0, -9.81],
            'solver_iterations': 16
        }
    }

    # Create the robot in Omniverse
    robot_path = omniverse_bridge.create_robot_assembly(humanoid_config)

    # Export configuration for verification
    exported_config = omniverse_bridge.export_robot_config(robot_path)

    print(f"Advanced humanoid robot created at: {robot_path}")
    print(f"Robot has {len(exported_config['links'])} links and {len(exported_config['joints'])} joints")

    return robot_path

# Create the advanced humanoid
advanced_humanoid_path = create_advanced_humanoid()
```

## Physics Simulation in Isaac Sim

### Realistic Physics for Humanoid Robots

Isaac Sim provides advanced physics simulation capabilities that are crucial for humanoid robotics, including complex contact dynamics, accurate mass properties, and realistic material interactions:

```python
# Physics Simulation for Humanoid Robots
import omni.physx
from omni.physx.bindings._physx import PhysicsSchemaContext
from pxr import UsdPhysics, PhysxSchema
import numpy as np

class HumanoidPhysicsSimulator:
    """Advanced physics simulation for humanoid robots"""

    def __init__(self):
        self.physics_scene = None
        self.contact_callbacks = []
        self.material_properties = {}
        self.collision_filtering = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info("Humanoid Physics Simulator initialized")

    def setup_physics_scene(self, scene_config: Dict):
        """Setup advanced physics scene for humanoid simulation"""
        stage = get_current_stage()

        # Create physics scene with advanced settings
        scene_path = "/World/PhysicsScene"
        self.physics_scene = UsdPhysics.Scene.Define(stage, scene_path)

        # Configure gravity
        self.physics_scene.CreateGravityAttr().Set(scene_config.get('gravity', [0, 0, -9.81]))

        # Configure solver settings for humanoid stability
        self.physics_scene.CreateTimeStepsPerSecondAttr().Set(
            scene_config.get('time_steps_per_second', 600)  # Higher for better stability
        )

        # Advanced solver settings
        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(self.physics_scene.GetPrim())
        physx_scene_api.CreateEnableEnhancedDeterminismAttr(True)
        physx_scene_api.CreateSolverTypeAttr("TGS")  # Use TGS solver for better stability

        # Configure material properties
        self._setup_material_properties()

        # Configure collision filtering
        self._setup_collision_filtering()

        self.logger.info("Advanced physics scene configured for humanoid simulation")

    def _setup_material_properties(self):
        """Setup realistic material properties for humanoid robot"""
        # Define material properties for different robot parts
        self.material_properties = {
            'rubber_foot': {
                'static_friction': 0.8,
                'dynamic_friction': 0.6,
                'restitution': 0.1,
                'youngs_modulus': 1e6,
                'poissons_ratio': 0.49
            },
            'metal_joint': {
                'static_friction': 0.2,
                'dynamic_friction': 0.15,
                'restitution': 0.3,
                'youngs_modulus': 2e11,
                'poissons_ratio': 0.3
            },
            'plastic_body': {
                'static_friction': 0.4,
                'dynamic_friction': 0.3,
                'restitution': 0.2,
                'youngs_modulus': 2e9,
                'poissons_ratio': 0.4
            }
        }

    def _setup_collision_filtering(self):
        """Setup collision filtering for humanoid robot"""
        # Define collision groups to prevent unwanted collisions
        # e.g., prevent collision between adjacent links in a kinematic chain
        self.collision_filtering = {
            'adjacent_links': [],  # Will be populated with joint pairs
            'self_collision_pairs': [],  # Pairs that should collide
            'ignored_collision_pairs': [  # Pairs that should not collide
                ('left_upper_leg', 'left_lower_leg'),  # Hip-knee shouldn't collide
                ('right_upper_leg', 'right_lower_leg'),  # Hip-knee shouldn't collide
                ('left_lower_leg', 'left_foot'),  # Knee-ankle shouldn't collide
                ('right_lower_leg', 'right_foot'),  # Knee-ankle shouldn't collide
                ('torso', 'head'),  # Torso-head shouldn't collide
                ('left_upper_arm', 'torso'),  # Shoulder-torso shouldn't collide
                ('right_upper_arm', 'torso'),  # Shoulder-torso shouldn't collide
            ]
        }

    def configure_robot_materials(self, robot_path: str, robot_config: Dict):
        """Configure materials for robot links"""
        stage = get_current_stage()

        for link_config in robot_config.get('links', []):
            link_name = link_config['name']
            link_path = f"{robot_path}/{link_name}"

            # Determine material based on link type
            material_type = self._determine_link_material(link_name)
            material_props = self.material_properties.get(material_type,
                                                        self.material_properties['plastic_body'])

            # Apply material properties to link
            link_prim = stage.GetPrimAtPath(link_path)
            if link_prim:
                # Create collision approximation
                collision_api = UsdPhysics.CollisionAPI.Apply(link_prim)

                # Set collision approximation
                collision_api.CreateCollisionApproximationAttr("convexHull")

                # Apply material properties
                self._apply_material_properties(link_prim, material_props)

    def _determine_link_material(self, link_name: str) -> str:
        """Determine appropriate material for a link based on its name"""
        if 'foot' in link_name.lower():
            return 'rubber_foot'
        elif any(part in link_name.lower() for part in ['hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist']):
            return 'metal_joint'
        else:
            return 'plastic_body'

    def _apply_material_properties(self, prim, material_props: Dict):
        """Apply material properties to a USD primitive"""
        # Apply friction properties
        if not prim.HasAPI(UsdPhysics.MaterialAPI):
            material_api = UsdPhysics.MaterialAPI.Apply(prim)
        else:
            material_api = UsdPhysics.MaterialAPI(prim)

        material_api.CreateStaticFrictionAttr().Set(material_props['static_friction'])
        material_api.CreateDynamicFrictionAttr().Set(material_props['dynamic_friction'])
        material_api.CreateRestitutionAttr().Set(material_props['restitution'])

    def setup_contact_sensors(self, robot_path: str, robot_config: Dict):
        """Setup contact sensors for detecting robot-ground and robot-object interactions"""
        from omni.isaac.core.utils.prims import define_prim

        contact_sensors = []

        # Add contact sensors to feet for balance control
        foot_links = [link for link in robot_config.get('links', [])
                     if 'foot' in link['name'].lower()]

        for foot_link in foot_links:
            foot_name = foot_link['name']
            foot_path = f"{robot_path}/{foot_name}_contact_sensor"

            # Create contact sensor prim
            contact_prim = define_prim(foot_path, "Sphere")

            # Configure as contact sensor
            contact_prim.GetAttribute("sensor:modality").Set("contact")
            contact_prim.GetAttribute("contact:reportingThreshold").Set(0.01)  # 1cm

            contact_sensors.append({
                'name': f"{foot_name}_contact",
                'path': foot_path,
                'attached_to': foot_name,
                'type': 'contact'
            })

        self.logger.info(f"Setup {len(contact_sensors)} contact sensors for robot balance detection")
        return contact_sensors

    def configure_balance_stabilization(self, robot_path: str):
        """Configure physics-based balance stabilization"""
        # This would integrate with the physics engine to provide
        # implicit balance stabilization for simulation stability
        pass

    def enable_soft_body_physics(self, robot_path: str, link_names: List[str]):
        """Enable soft body physics for specific links (for compliant behavior)"""
        # This would enable FEM-based soft body simulation for specific links
        # Useful for simulating compliant actuators or soft tissues
        self.logger.info(f"Soft body physics enabled for links: {link_names}")

    def setup_terrain_interaction(self, terrain_config: Dict):
        """Setup realistic terrain interaction physics"""
        # Configure different terrain types with appropriate properties
        terrain_materials = {
            'grass': {'friction': 0.7, 'restitution': 0.1, 'damping': 0.1},
            'concrete': {'friction': 0.8, 'restitution': 0.2, 'damping': 0.05},
            'carpet': {'friction': 0.9, 'restitution': 0.05, 'damping': 0.2},
            'tile': {'friction': 0.6, 'restitution': 0.3, 'damping': 0.02}
        }

        terrain_type = terrain_config.get('type', 'grass')
        if terrain_type in terrain_materials:
            properties = terrain_materials[terrain_type]
            self.logger.info(f"Terrain interaction configured for {terrain_type} surface")
            return properties
        else:
            self.logger.warning(f"Unknown terrain type: {terrain_type}, using default")
            return terrain_materials['grass']

# Example: Setting up physics for a humanoid robot
def setup_humanoid_physics():
    """Complete example of setting up physics for humanoid simulation"""

    # Initialize physics simulator
    physics_sim = HumanoidPhysicsSimulator()

    # Configure physics scene
    scene_config = {
        'gravity': [0, 0, -9.81],
        'time_steps_per_second': 600,  # 600 Hz for stable humanoid simulation
        'solver_iterations': 16
    }

    physics_sim.setup_physics_scene(scene_config)

    # Assume we have a robot configuration from the previous example
    robot_config = {
        'name': 'humanoid_robot',
        'links': [
            {'name': 'pelvis', 'geometry': {'type': 'box'}},
            {'name': 'torso', 'geometry': {'type': 'capsule'}},
            {'name': 'head', 'geometry': {'type': 'sphere'}},
            {'name': 'left_foot', 'geometry': {'type': 'box'}},
            {'name': 'right_foot', 'geometry': {'type': 'box'}}
        ]
    }

    # Configure robot materials
    physics_sim.configure_robot_materials('/World/humanoid_robot', robot_config)

    # Setup contact sensors
    contact_sensors = physics_sim.setup_contact_sensors('/World/humanoid_robot', robot_config)

    # Setup terrain interaction
    terrain_properties = physics_sim.setup_terrain_interaction({
        'type': 'concrete'
    })

    print(f"Physics setup complete with {len(contact_sensors)} contact sensors")
    print(f"Terrain properties: {terrain_properties}")

    return physics_sim

# Run physics setup
humanoid_physics = setup_humanoid_physics()
```

## Sensor Simulation

### Advanced Sensor Simulation in Isaac Sim

Isaac Sim provides sophisticated sensor simulation that accurately models real-world sensor characteristics:

```python
# Advanced Sensor Simulation for Humanoid Robots
import omni.isaac.sensor
from omni.isaac.core.sensors import Camera, IMU, ContactSensor
from omni.isaac.range_sensor import _range_sensor
import numpy as np

class IsaacSensorSimulator:
    """Advanced sensor simulation for Isaac Sim"""

    def __init__(self):
        self.sensors = {}
        self.sensor_configs = {}
        self.raw_data_buffers = {}
        self.processed_data_buffers = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info("Isaac Sensor Simulator initialized")

    def create_camera_sensor(self, sensor_config: Dict) -> omni.isaac.sensor.Camera:
        """Create a realistic camera sensor with noise and distortion models"""
        camera = omni.isaac.sensor.Camera(
            prim_path=sensor_config['path'],
            frequency=sensor_config.get('update_rate', 30),
            resolution=sensor_config.get('resolution', [640, 480])
        )

        # Configure camera intrinsics
        camera_config = sensor_config.get('camera_config', {})
        focal_length = camera_config.get('focal_length', 24.0)
        horizontal_aperture = camera_config.get('horizontal_aperture', 20.955)
        vertical_aperture = camera_config.get('vertical_aperture', 15.2908)

        # Apply distortion parameters
        distortion_params = camera_config.get('distortion', {
            'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0, 'k3': 0.0
        })

        # Add noise parameters
        noise_params = camera_config.get('noise', {
            'gaussian_noise_std': 2.0,
            'shot_noise_factor': 0.01,
            'dark_current': 0.1
        })

        self.sensors[sensor_config['name']] = {
            'sensor': camera,
            'type': 'camera',
            'config': sensor_config,
            'noise_params': noise_params,
            'distortion_params': distortion_params
        }

        self.logger.info(f"Created camera sensor: {sensor_config['name']}")
        return camera

    def create_lidar_sensor(self, sensor_config: Dict) -> omni.isaac.sensor.RotatingLidarPhysX:
        """Create a realistic LiDAR sensor with beam physics"""
        lidar = omni.isaac.sensor.RotatingLidarPhysX(
            prim_path=sensor_config['path'],
            translation=np.array(sensor_config.get('position', [0, 0, 0])),
            orientation=np.array(sensor_config.get('orientation', [1, 0, 0, 0])),
            rpm=sensor_config.get('rpm', 30),
            samples_per_ring=sensor_config.get('samples_per_ring', 720),
            max_range=sensor_config.get('max_range', 25.0),
            fov=sensor_config.get('fov', 360.0),
            horizontal_resolution=sensor_config.get('horizontal_resolution', 0.5),
            vertical_resolution=sensor_config.get('vertical_resolution', 2.0),
            vertical_fov=sensor_config.get('vertical_fov', 30.0)
        )

        # Configure LiDAR-specific parameters
        lidar_config = sensor_config.get('lidar_config', {})

        # Add noise and accuracy parameters
        noise_params = lidar_config.get('noise', {
            'range_noise_std': 0.02,  # 2cm standard deviation
            'angular_noise_std': 0.001,  # 0.057 degrees standard deviation
            'intensity_noise_std': 0.1
        })

        # Add beam divergence and other physical properties
        physical_params = lidar_config.get('physical_properties', {
            'beam_divergence': 0.003,  # 3 mrad
            'pulse_width': 2e-9,  # 2ns pulse
            'wavelength': 905e-9  # 905nm
        })

        self.sensors[sensor_config['name']] = {
            'sensor': lidar,
            'type': 'lidar',
            'config': sensor_config,
            'noise_params': noise_params,
            'physical_params': physical_params
        }

        self.logger.info(f"Created LiDAR sensor: {sensor_config['name']}")
        return lidar

    def create_imu_sensor(self, sensor_config: Dict) -> omni.isaac.sensor.IMU:
        """Create a realistic IMU sensor with bias and drift models"""
        imu = omni.isaac.sensor.IMU(
            prim_path=sensor_config['path'],
            frequency=sensor_config.get('update_rate', 100)
        )

        # Configure IMU-specific parameters
        imu_config = sensor_config.get('imu_config', {})

        # Accelerometer parameters
        accel_params = imu_config.get('accelerometer', {
            'noise_density': 100e-6,  # 100 μg/√Hz
            'random_walk': 10e-6,    # 10 μg/√Hz/√s
            'bias_instability': 50e-6,  # 50 μg
            'turn_on_bias_sigma': 200e-6  # 200 μg
        })

        # Gyroscope parameters
        gyro_params = imu_config.get('gyroscope', {
            'noise_density': 0.2e-3,  # 0.2 °/s/√Hz
            'random_walk': 0.01e-3,  # 0.01 °/s/√Hz/√s
            'bias_instability': 10e-3,  # 10 °/hr
            'turn_on_bias_sigma': 20e-3   # 20 °/hr
        })

        # Magnetometer parameters (if present)
        mag_params = imu_config.get('magnetometer', {
            'noise_density': 100e-9,  # 100 nT/√Hz
            'random_walk': 10e-9,    # 10 nT/√Hz/√s
            'bias_instability': 50e-9  # 50 nT
        })

        self.sensors[sensor_config['name']] = {
            'sensor': imu,
            'type': 'imu',
            'config': sensor_config,
            'accel_params': accel_params,
            'gyro_params': gyro_params,
            'mag_params': mag_params
        }

        self.logger.info(f"Created IMU sensor: {sensor_config['name']}")
        return imu

    def create_contact_sensors(self, sensor_config: Dict) -> omni.isaac.sensor.ContactSensor:
        """Create contact sensors for touch detection"""
        contact_sensor = omni.isaac.sensor.ContactSensor(
            prim_path=sensor_config['path'],
            min_threshold=sensor_config.get('min_threshold', 0.1),
            max_threshold=sensor_config.get('max_threshold', 100.0),
            history_buffer_length=sensor_config.get('history_length', 10)
        )

        self.sensors[sensor_config['name']] = {
            'sensor': contact_sensor,
            'type': 'contact',
            'config': sensor_config
        }

        self.logger.info(f"Created contact sensor: {sensor_config['name']}")
        return contact_sensor

    def add_sensor_noise(self, sensor_name: str, raw_data: np.ndarray) -> np.ndarray:
        """Add realistic noise to sensor data based on sensor configuration"""
        if sensor_name not in self.sensors:
            return raw_data

        sensor_info = self.sensors[sensor_name]
        sensor_type = sensor_info['type']
        noise_params = sensor_info.get('noise_params', {})

        if sensor_type == 'camera':
            return self._add_camera_noise(raw_data, noise_params)
        elif sensor_type == 'lidar':
            return self._add_lidar_noise(raw_data, noise_params)
        elif sensor_type == 'imu':
            return self._add_imu_noise(raw_data, sensor_info)
        else:
            return raw_data  # No noise for other sensor types

    def _add_camera_noise(self, image: np.ndarray, noise_params: Dict) -> np.ndarray:
        """Add realistic camera noise including photon shot noise, read noise, etc."""
        # Convert to float for processing
        img_float = image.astype(np.float32)

        # Add photon shot noise (signal dependent)
        shot_noise = np.random.poisson(img_float) - img_float
        img_float += shot_noise * noise_params.get('shot_noise_factor', 0.01)

        # Add Gaussian read noise
        gaussian_noise = np.random.normal(
            0,
            noise_params.get('gaussian_noise_std', 2.0),
            img_float.shape
        )
        img_float += gaussian_noise

        # Add dark current
        img_float += noise_params.get('dark_current', 0.1)

        # Clip and convert back to original format
        img_noisy = np.clip(img_float, 0, 255).astype(image.dtype)
        return img_noisy

    def _add_lidar_noise(self, ranges: np.ndarray, noise_params: Dict) -> np.ndarray:
        """Add realistic LiDAR noise including quantization and angular errors"""
        ranges_noisy = ranges.copy().astype(np.float32)

        # Add range measurement noise
        range_noise = np.random.normal(
            0,
            noise_params.get('range_noise_std', 0.02),
            ranges_noisy.shape
        )
        ranges_noisy += range_noise

        # Add intensity noise if intensity data is present
        if ranges_noisy.ndim > 1 and ranges_noisy.shape[1] > 1:
            intensity_noise = np.random.normal(
                0,
                noise_params.get('intensity_noise_std', 0.1),
                (ranges_noisy.shape[0], ranges_noisy.shape[1]-1)  # Exclude range column
            )
            ranges_noisy[:, 1:] += intensity_noise

        # Ensure no negative ranges
        ranges_noisy = np.maximum(ranges_noisy, 0.0)
        return ranges_noisy

    def _add_imu_noise(self, imu_data: Dict, sensor_info: Dict) -> Dict:
        """Add realistic IMU noise with bias and drift"""
        accel_params = sensor_info['accel_params']
        gyro_params = sensor_info['gyro_params']

        # Add noise to accelerometer data
        if 'accelerometer' in imu_data:
            accel = imu_data['accelerometer']
            # Add noise based on Allan variance model
            accel_noise = np.random.normal(0, accel_params['noise_density'], accel.shape)
            imu_data['accelerometer'] += accel_noise

        # Add noise to gyroscope data
        if 'gyroscope' in imu_data:
            gyro = imu_data['gyroscope']
            gyro_noise = np.random.normal(0, gyro_params['noise_density'], gyro.shape)
            imu_data['gyroscope'] += gyro_noise

        return imu_data

    def simulate_sensor_distortion(self, sensor_name: str, data: any) -> any:
        """Simulate sensor-specific distortions"""
        if sensor_name not in self.sensors:
            return data

        sensor_info = self.sensors[sensor_name]
        sensor_type = sensor_info['type']

        if sensor_type == 'camera' and 'distortion_params' in sensor_info:
            return self._simulate_camera_distortion(data, sensor_info['distortion_params'])
        else:
            return data

    def _simulate_camera_distortion(self, image: np.ndarray, distortion_params: Dict) -> np.ndarray:
        """Simulate camera lens distortion"""
        # For simplicity, we'll use OpenCV to simulate distortion
        # In Isaac Sim, this would be handled by the rendering pipeline
        try:
            import cv2

            h, w = image.shape[:2]

            # Create camera matrix
            cx, cy = w / 2, h / 2
            fx = fy = max(w, h) / (2 * np.tan(np.pi / 6))  # Assume 60° FOV
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # Distortion coefficients
            dist_coeffs = np.array([
                distortion_params['k1'],  # k1
                distortion_params['k2'],  # k2
                distortion_params['p1'],  # p1
                distortion_params['p2'],  # p2
                distortion_params['k3']   # k3
            ])

            # Apply distortion
            undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
            return undistorted_image
        except ImportError:
            # If OpenCV is not available, return original image
            return image

    def get_sensor_data(self, sensor_name: str, apply_noise: bool = True) -> any:
        """Get sensor data with optional noise and distortion"""
        if sensor_name not in self.sensors:
            raise ValueError(f"Sensor {sensor_name} not found")

        sensor_info = self.sensors[sensor_name]
        sensor = sensor_info['sensor']

        # Get raw data from sensor
        raw_data = self._get_raw_sensor_data(sensor, sensor_info['type'])

        if apply_noise:
            # Add noise to the data
            noisy_data = self.add_sensor_noise(sensor_name, raw_data)

            # Apply distortion if applicable
            processed_data = self.simulate_sensor_distortion(sensor_name, noisy_data)
        else:
            processed_data = raw_data

        # Store in buffers
        self.raw_data_buffers[sensor_name] = raw_data
        self.processed_data_buffers[sensor_name] = processed_data

        return processed_data

    def _get_raw_sensor_data(self, sensor, sensor_type: str) -> any:
        """Get raw data from Isaac sensor object"""
        # This is a simplified version - in practice, you would get the actual data
        # from the Isaac sensor object based on its type
        if sensor_type == 'camera':
            # Return a dummy image
            resolution = self.sensors[list(self.sensors.keys())[0]]['config'].get('resolution', [640, 480])
            return np.random.randint(0, 255, (*resolution, 3), dtype=np.uint8)
        elif sensor_type == 'lidar':
            # Return dummy ranges
            samples = self.sensors[list(self.sensors.keys())[0]]['config'].get('samples_per_ring', 720)
            return np.random.uniform(0.1, 25.0, samples).astype(np.float32)
        elif sensor_type == 'imu':
            # Return dummy IMU data
            return {
                'accelerometer': np.random.normal(0, 0.1, 3).astype(np.float32),
                'gyroscope': np.random.normal(0, 0.01, 3).astype(np.float32),
                'orientation': np.array([0, 0, 0, 1], dtype=np.float32)  # w, x, y, z
            }
        else:
            return None

# Example: Creating sensors for a humanoid robot
def create_humanoid_sensors():
    """Create a complete sensor suite for a humanoid robot"""

    sensor_sim = IsaacSensorSimulator()

    # Head camera sensor
    head_camera_config = {
        'name': 'head_camera',
        'path': '/World/humanoid_robot/head_camera',
        'type': 'camera',
        'update_rate': 30,
        'resolution': [640, 480],
        'camera_config': {
            'focal_length': 24.0,
            'horizontal_aperture': 20.955,
            'vertical_aperture': 15.2908,
            'distortion': {
                'k1': 0.1, 'k2': -0.2, 'p1': 0.001, 'p2': -0.001, 'k3': 0.05
            },
            'noise': {
                'gaussian_noise_std': 3.0,
                'shot_noise_factor': 0.02,
                'dark_current': 0.2
            }
        }
    }

    head_camera = sensor_sim.create_camera_sensor(head_camera_config)

    # Torso IMU sensor
    torso_imu_config = {
        'name': 'torso_imu',
        'path': '/World/humanoid_robot/torso_imu',
        'type': 'imu',
        'update_rate': 100,
        'imu_config': {
            'accelerometer': {
                'noise_density': 100e-6,
                'random_walk': 10e-6,
                'bias_instability': 50e-6
            },
            'gyroscope': {
                'noise_density': 0.2e-3,
                'random_walk': 0.01e-3,
                'bias_instability': 10e-3
            }
        }
    }

    torso_imu = sensor_sim.create_imu_sensor(torso_imu_config)

    # Spinning LiDAR on head
    head_lidar_config = {
        'name': 'head_lidar',
        'path': '/World/humanoid_robot/head_lidar',
        'type': 'lidar',
        'position': [0.1, 0, 0.1],
        'lidar_config': {
            'rpm': 30,
            'samples_per_ring': 720,
            'max_range': 25.0,
            'fov': 360.0,
            'noise': {
                'range_noise_std': 0.02,
                'angular_noise_std': 0.001,
                'intensity_noise_std': 0.1
            }
        }
    }

    head_lidar = sensor_sim.create_lidar_sensor(head_lidar_config)

    # Foot contact sensors
    left_foot_contact_config = {
        'name': 'left_foot_contact',
        'path': '/World/humanoid_robot/left_foot_contact',
        'type': 'contact',
        'min_threshold': 0.1,
        'max_threshold': 500.0
    }

    right_foot_contact_config = {
        'name': 'right_foot_contact',
        'path': '/World/humanoid_robot/right_foot_contact',
        'type': 'contact',
        'min_threshold': 0.1,
        'max_threshold': 500.0
    }

    left_contact = sensor_sim.create_contact_sensors(left_foot_contact_config)
    right_contact = sensor_sim.create_contact_sensors(right_foot_contact_config)

    print(f"Created sensor suite with {len(sensor_sim.sensors)} sensors:")
    for sensor_name in sensor_sim.sensors.keys():
        print(f"  - {sensor_name} ({sensor_sim.sensors[sensor_name]['type']})")

    return sensor_sim

# Create the sensor suite
humanoid_sensors = create_humanoid_sensors()
```

## Integration with ROS 2

### ROS 2 Bridge for Isaac Sim

Isaac Sim provides native integration with ROS 2, allowing seamless communication between the simulation and external ROS 2 nodes:

```python
# ROS 2 Integration for Isaac Sim
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, PointCloud2, JointState, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Float32
from builtin_interfaces.msg import Time
import numpy as np
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.robots import Robot
import tf2_ros

class IsaacSimROS2Bridge(Node):
    """Bridge between Isaac Sim and ROS 2 ecosystem"""

    def __init__(self):
        super().__init__('isaac_sim_ros2_bridge')

        # Publishers for simulated sensor data
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.laser_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # Subscribers for robot commands
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        self.joint_cmd_sub = self.create_subscription(
            JointState, '/joint_commands', self.joint_cmd_callback, 10
        )

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Timer for publishing sensor data
        self.publish_timer = self.create_timer(0.033, self.publish_sensor_data)  # ~30Hz

        # Robot reference
        self.robot = None
        self.robot_name = 'humanoid_robot'

        # Sensor simulator reference
        self.sensor_simulator = None

        self.get_logger().info('Isaac Sim ROS2 Bridge initialized')

    def set_sensor_simulator(self, sensor_simulator):
        """Set reference to sensor simulator"""
        self.sensor_simulator = sensor_simulator

    def cmd_vel_callback(self, msg: Twist):
        """Handle velocity commands from ROS 2"""
        self.get_logger().info(f'Received cmd_vel: linear=({msg.linear.x}, {msg.linear.y}, {msg.linear.z}), '
                              f'angular=({msg.angular.x}, {msg.angular.y}, {msg.angular.z})')

        # In a real implementation, this would send commands to the robot in Isaac Sim
        # For now, we'll just log the command
        pass

    def joint_cmd_callback(self, msg: JointState):
        """Handle joint commands from ROS 2"""
        self.get_logger().info(f'Received joint commands for {len(msg.name)} joints')

        # In a real implementation, this would send joint commands to the robot
        pass

    def publish_sensor_data(self):
        """Publish simulated sensor data to ROS 2 topics"""
        if not self.sensor_simulator:
            return

        try:
            # Publish camera image
            if 'head_camera' in self.sensor_simulator.sensors:
                image_data = self.sensor_simulator.get_sensor_data('head_camera')
                image_msg = self._convert_image_to_ros_msg(image_data)
                self.image_pub.publish(image_msg)

            # Publish IMU data
            if 'torso_imu' in self.sensor_simulator.sensors:
                imu_data = self.sensor_simulator.get_sensor_data('torso_imu')
                imu_msg = self._convert_imu_to_ros_msg(imu_data)
                self.imu_pub.publish(imu_msg)

            # Publish LiDAR data
            if 'head_lidar' in self.sensor_simulator.sensors:
                lidar_data = self.sensor_simulator.get_sensor_data('head_lidar')
                laser_msg = self._convert_lidar_to_ros_msg(lidar_data)
                self.laser_pub.publish(laser_msg)

            # Publish joint states
            joint_state_msg = self._get_joint_states()
            self.joint_state_pub.publish(joint_state_msg)

            # Publish odometry
            odom_msg = self._get_odometry()
            self.odom_pub.publish(odom_msg)

            # Broadcast transforms
            self._broadcast_transforms()

        except Exception as e:
            self.get_logger().error(f'Error publishing sensor data: {str(e)}')

    def _convert_image_to_ros_msg(self, image_data: np.ndarray) -> Image:
        """Convert numpy image to ROS Image message"""
        image_msg = Image()
        image_msg.header = Header()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = 'camera_link'

        image_msg.height = image_data.shape[0]
        image_msg.width = image_data.shape[1]
        image_msg.encoding = 'rgb8'
        image_msg.is_bigendian = False
        image_msg.step = image_msg.width * 3  # 3 channels for RGB

        # Convert BGR to RGB if needed
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            # Assuming input is BGR, convert to RGB
            image_rgb = np.zeros_like(image_data)
            image_rgb[:, :, 0] = image_data[:, :, 2]  # R <- B
            image_rgb[:, :, 1] = image_data[:, :, 1]  # G <- G
            image_rgb[:, :, 2] = image_data[:, :, 0]  # B <- R
            image_msg.data = image_rgb.tobytes()
        else:
            image_msg.data = image_data.tobytes()

        return image_msg

    def _convert_imu_to_ros_msg(self, imu_data: Dict) -> Imu:
        """Convert IMU data to ROS Imu message"""
        imu_msg = Imu()
        imu_msg.header = Header()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        # Fill in IMU data (simplified)
        if 'orientation' in imu_data:
            orient = imu_data['orientation']
            imu_msg.orientation.w = float(orient[0])
            imu_msg.orientation.x = float(orient[1])
            imu_msg.orientation.y = float(orient[2])
            imu_msg.orientation.z = float(orient[3])

        if 'angular_velocity' in imu_data:
            ang_vel = imu_data['angular_velocity']
            imu_msg.angular_velocity.x = float(ang_vel[0])
            imu_msg.angular_velocity.y = float(ang_vel[1])
            imu_msg.angular_velocity.z = float(ang_vel[2])

        if 'linear_acceleration' in imu_data:
            lin_acc = imu_data['linear_acceleration']
            imu_msg.linear_acceleration.x = float(lin_acc[0])
            imu_msg.linear_acceleration.y = float(lin_acc[1])
            imu_msg.linear_acceleration.z = float(lin_acc[2])

        return imu_msg

    def _convert_lidar_to_ros_msg(self, lidar_data: np.ndarray) -> LaserScan:
        """Convert LiDAR data to ROS LaserScan message"""
        laser_msg = LaserScan()
        laser_msg.header = Header()
        laser_msg.header.stamp = self.get_clock().now().to_msg()
        laser_msg.header.frame_id = 'lidar_link'

        # Set scan parameters (these would come from sensor config)
        laser_msg.angle_min = -np.pi
        laser_msg.angle_max = np.pi
        laser_msg.angle_increment = (2 * np.pi) / len(lidar_data) if len(lidar_data) > 0 else 0.01
        laser_msg.time_increment = 0.0  # Calculated based on RPM and samples
        laser_msg.scan_time = 1.0 / 30.0  # 30 RPM = 30 rotations per minute
        laser_msg.range_min = 0.1
        laser_msg.range_max = 25.0

        # Set ranges
        laser_msg.ranges = lidar_data.astype(np.float32).tolist()

        # Set intensities if available
        # For now, set all to 1.0
        laser_msg.intensities = [1.0] * len(lidar_data)

        return laser_msg

    def _get_joint_states(self) -> JointState:
        """Get current joint states from simulated robot"""
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = 'base_link'

        # In a real implementation, this would get actual joint positions from Isaac Sim
        # For now, we'll simulate some values
        joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow'
        ]

        positions = np.random.uniform(-0.5, 0.5, len(joint_names)).tolist()
        velocities = np.random.uniform(-1.0, 1.0, len(joint_names)).tolist()
        efforts = np.random.uniform(-10.0, 10.0, len(joint_names)).tolist()

        joint_state.name = joint_names
        joint_state.position = positions
        joint_state.velocity = velocities
        joint_state.effort = efforts

        return joint_state

    def _get_odometry(self) -> Odometry:
        """Get odometry from simulated robot"""
        odom = Odometry()
        odom.header = Header()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        # In a real implementation, this would get actual pose from Isaac Sim
        # For now, simulate some motion
        odom.pose.pose.position.x += 0.01  # Move forward slowly
        odom.pose.pose.orientation.w = 1.0  # No rotation

        # Set velocity
        odom.twist.twist.linear.x = 0.1  # 0.1 m/s forward
        odom.twist.twist.angular.z = 0.0  # No rotation

        return odom

    def _broadcast_transforms(self):
        """Broadcast TF transforms"""
        # This would broadcast the robot's link transformations
        # In a real implementation, this would come from Isaac Sim's kinematic tree
        pass

def main(args=None):
    rclpy.init(args=args)

    # Create the bridge node
    bridge = IsaacSimROS2Bridge()

    # In a real implementation, you would connect this to the Isaac Sim sensor simulator
    # For now, we'll create a dummy sensor simulator
    dummy_sensor_sim = IsaacSensorSimulator()

    # Add some dummy sensors to the simulator
    dummy_config = {
        'name': 'dummy_camera',
        'path': '/World/dummy/camera',
        'type': 'camera',
        'update_rate': 30,
        'resolution': [640, 480]
    }
    dummy_sensor_sim.create_camera_sensor(dummy_config)

    bridge.set_sensor_simulator(dummy_sensor_sim)

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        bridge.get_logger().info('Shutting down Isaac Sim ROS2 Bridge')
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Isaac Sim provides a comprehensive simulation environment that combines photorealistic rendering, accurate physics simulation, and realistic sensor modeling. For humanoid robotics, this enables researchers and developers to test complex behaviors in a safe, controlled environment before deploying to physical robots. The integration with Omniverse and ROS 2 makes Isaac Sim a powerful platform for developing advanced humanoid robot applications.