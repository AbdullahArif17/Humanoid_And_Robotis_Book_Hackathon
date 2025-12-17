---
sidebar_position: 4
---

# Sensor Simulation (LiDAR, Camera, IMU)

## Introduction to Sensor Simulation in Gazebo

Sensor simulation is a critical component of realistic humanoid robot simulation in Gazebo. Accurate sensor models enable robots to perceive their environment just as they would with real hardware, making the transition from simulation to reality more feasible. Gazebo provides sophisticated sensor simulation capabilities that model the physical properties, noise characteristics, and limitations of real sensors.

### Sensor Simulation Fundamentals

Sensor simulation in Gazebo involves:

- **Physics-based Rendering**: Accurate modeling of light transport, reflections, and refractions
- **Noise Modeling**: Realistic noise patterns that match real sensor characteristics
- **Distortion Simulation**: Modeling of lens distortions and sensor imperfections
- **Temporal Effects**: Frame rates, exposure times, and motion blur
- **Environmental Factors**: Lighting conditions, weather effects, and atmospheric conditions

```python
# Sensor simulation framework
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import cv2
from scipy.spatial.transform import Rotation as R

class SensorType(Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    IMU = "imu"
    GPS = "gps"
    FORCE_TORQUE = "force_torque"

@dataclass
class SensorSpecs:
    """Base specifications for sensor simulation"""
    name: str
    sensor_type: SensorType
    update_rate: float
    position: np.ndarray  # Position relative to parent link
    orientation: np.ndarray  # Orientation as quaternion [w, x, y, z]
    noise_mean: float
    noise_std: float
    bias: float = 0.0
    drift_rate: float = 0.0

class SensorSimulator:
    """Base class for sensor simulation"""

    def __init__(self, specs: SensorSpecs):
        self.specs = specs
        self.last_update_time = 0.0
        self.bias = specs.bias
        self.drift = 0.0
        self.is_active = True

    def update(self, current_time: float, world_state: Dict) -> Optional[any]:
        """Update sensor and return data if it's time"""
        if not self.is_active:
            return None

        if current_time - self.last_update_time >= 1.0 / self.specs.update_rate:
            self.last_update_time = current_time
            raw_data = self.generate_raw_data(world_state)
            return self.add_noise_and_bias(raw_data, current_time)
        return None

    def generate_raw_data(self, world_state: Dict) -> any:
        """Generate raw sensor data - to be implemented by subclasses"""
        raise NotImplementedError

    def add_noise_and_bias(self, data: any, current_time: float) -> any:
        """Add noise, bias, and drift to sensor data"""
        # Update drift over time
        self.drift += self.specs.drift_rate * (current_time - self.last_update_time)

        if isinstance(data, (int, float)):
            noise = np.random.normal(self.specs.noise_mean, self.specs.noise_std)
            return data + noise + self.bias + self.drift
        elif isinstance(data, np.ndarray):
            noise = np.random.normal(self.specs.noise_mean, self.specs.noise_std, data.shape)
            bias_drift = self.bias + self.drift
            return data + noise + bias_drift
        else:
            return data

class CameraSimulator(SensorSimulator):
    """Camera sensor simulation with realistic effects"""

    def __init__(self, specs: SensorSpecs, width: int = 640, height: int = 480,
                 fov_horizontal: float = 1.047, fov_vertical: float = 0.785):  # 60°, 45°
        super().__init__(specs)
        self.width = width
        self.height = height
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.intrinsic_matrix = self.calculate_intrinsic_matrix()
        self.distortion_coeffs = np.array([0.1, -0.2, 0, 0, 0.1])  # Simplified distortion

    def calculate_intrinsic_matrix(self) -> np.ndarray:
        """Calculate camera intrinsic matrix"""
        fx = self.width / (2 * math.tan(self.fov_horizontal / 2))
        fy = self.height / (2 * math.tan(self.fov_vertical / 2))
        cx = self.width / 2
        cy = self.height / 2

        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    def generate_raw_data(self, world_state: Dict) -> np.ndarray:
        """Generate camera image data"""
        # Create a base image
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Simulate scene elements
        self.render_scene_elements(image, world_state)

        # Apply camera effects
        image = self.apply_camera_effects(image)

        # Apply noise
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image

    def render_scene_elements(self, image: np.ndarray, world_state: Dict):
        """Render scene elements into the image"""
        # Draw a simple scene with objects
        height, width = image.shape[:2]

        # Draw a floor grid
        for i in range(0, width, 50):
            cv2.line(image, (i, height//2), (i, height), (100, 100, 100), 1)
        for j in range(height//2, height, 30):
            cv2.line(image, (0, j), (width, j), (100, 100, 100), 1)

        # Draw some objects (simplified)
        # Red cube
        cv2.rectangle(image, (width//2 - 30, height//2 - 60), (width//2 + 30, height//2 - 30), (0, 0, 255), -1)
        # Blue cylinder
        cv2.circle(image, (width//2 + 100, height//2 - 40), 25, (255, 0, 0), -1)

    def apply_camera_effects(self, image: np.ndarray) -> np.ndarray:
        """Apply camera-specific effects like distortion"""
        height, width = image.shape[:2]

        # Generate coordinate grids
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        # Convert to normalized coordinates
        x_norm = (xx - self.width/2) / (self.width/2)
        y_norm = (yy - self.height/2) / (self.height/2)

        # Apply radial distortion
        r_squared = x_norm**2 + y_norm**2
        distortion_factor = (1 + self.distortion_coeffs[0] * r_squared +
                           self.distortion_coeffs[1] * r_squared**2 +
                           self.distortion_coeffs[4] * r_squared**3)

        x_distorted = x_norm * distortion_factor + \
                     2*self.distortion_coeffs[2]*x_norm*y_norm + \
                     self.distortion_coeffs[3]*(r_squared + 2*x_norm**2)

        y_distorted = y_norm * distortion_factor + \
                     self.distortion_coeffs[2]*(r_squared + 2*y_norm**2) + \
                     2*self.distortion_coeffs[3]*x_norm*y_norm

        # Convert back to pixel coordinates
        x_new = (x_distorted * self.width/2 + self.width/2).astype(np.float32)
        y_new = (y_distorted * self.height/2 + self.height/2).astype(np.float32)

        # Apply distortion using remapping
        distorted_image = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR)

        return distorted_image

class LidarSimulator(SensorSimulator):
    """LiDAR sensor simulation with realistic beam modeling"""

    def __init__(self, specs: SensorSpecs, num_beams: int = 720,
                 min_range: float = 0.1, max_range: float = 10.0,
                 fov_horizontal: float = 2*math.pi,  # 360 degrees
                 fov_vertical: float = 0.174):       # 10 degrees
        super().__init__(specs)
        self.num_beams = num_beams
        self.min_range = min_range
        self.max_range = max_range
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.angles = np.linspace(0, fov_horizontal, num_beams, endpoint=False)

    def generate_raw_data(self, world_state: Dict) -> np.ndarray:
        """Generate LiDAR scan data"""
        ranges = np.full(self.num_beams, self.max_range, dtype=np.float32)

        # Simulate ray casting for each beam
        for i, angle in enumerate(self.angles):
            # In a real implementation, this would cast rays into the 3D scene
            # For simulation, we'll create a simplified environment
            distance = self.simulate_lidar_beam(angle, world_state)
            ranges[i] = min(distance, self.max_range)

        return ranges

    def simulate_lidar_beam(self, angle: float, world_state: Dict) -> float:
        """Simulate a single LiDAR beam"""
        # Simplified beam simulation
        # In reality, this would use ray-scene intersection

        # Simulate some objects in the environment
        objects = [
            {'position': np.array([2.0, 0.0, 0.0]), 'size': 0.5},  # Object at 2m
            {'position': np.array([-1.5, 1.0, 0.0]), 'size': 0.3},  # Object at 1.5m
            {'position': np.array([0.0, -3.0, 0.0]), 'size': 0.4},  # Object at 3m
        ]

        # Calculate ray direction
        ray_dir = np.array([math.cos(angle), math.sin(angle), 0])

        min_distance = self.max_range
        for obj in objects:
            obj_pos = obj['position']
            # Calculate distance to object center
            distance_to_center = np.linalg.norm(obj_pos[:2])  # 2D distance
            # Calculate minimum distance considering object size
            min_obj_distance = distance_to_center - obj['size']

            if min_obj_distance < min_distance:
                min_distance = min_obj_distance

        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.02)  # 2cm noise
        return max(self.min_range, min_distance + noise)

class IMUSimulator(SensorSimulator):
    """IMU sensor simulation with accelerometer, gyroscope, and magnetometer"""

    def __init__(self, specs: SensorSpecs,
                 accelerometer_noise_density: float = 0.017,  # (m/s^2)/sqrt(Hz)
                 gyroscope_noise_density: float = 0.001,      # (rad/s)/sqrt(Hz)
                 magnetometer_noise: float = 0.1):           # uT
        super().__init__(specs)
        self.accelerometer_noise_density = accelerometer_noise_density
        self.gyroscope_noise_density = gyroscope_noise_density
        self.magnetometer_noise = magnetometer_noise

        # IMU bias and drift parameters
        self.accel_bias = np.random.normal(0, 0.01, 3)  # Initial bias
        self.gyro_bias = np.random.normal(0, 0.001, 3)  # Initial bias
        self.accel_bias_walk = np.random.normal(0, 0.0001, 3)  # Bias walk
        self.gyro_bias_walk = np.random.normal(0, 0.00001, 3)  # Bias walk

    def generate_raw_data(self, world_state: Dict) -> Dict[str, np.ndarray]:
        """Generate IMU data: accelerometer, gyroscope, magnetometer"""
        # Get true values from world state
        true_accel = world_state.get('linear_acceleration', np.array([0, 0, 9.81]))
        true_gyro = world_state.get('angular_velocity', np.array([0, 0, 0]))
        true_mag = world_state.get('magnetic_field', np.array([25000, 0, 45000]))  # Approx Earth's field

        # Apply biases and noise
        accel_measurement = self.simulate_accelerometer(true_accel)
        gyro_measurement = self.simulate_gyroscope(true_gyro)
        mag_measurement = self.simulate_magnetometer(true_mag)

        return {
            'accelerometer': accel_measurement,
            'gyroscope': gyro_measurement,
            'magnetometer': mag_measurement
        }

    def simulate_accelerometer(self, true_accel: np.ndarray) -> np.ndarray:
        """Simulate accelerometer with realistic noise model"""
        # Add bias and bias walk
        self.accel_bias += self.accel_bias_walk * (1.0 / self.specs.update_rate)
        biased_accel = true_accel + self.accel_bias

        # Add noise (simplified model)
        # In reality, this would consider noise density and bandwidth
        noise_std = self.accelerometer_noise_density * np.sqrt(self.specs.update_rate / 2)
        noise = np.random.normal(0, noise_std, 3)

        return biased_accel + noise

    def simulate_gyroscope(self, true_gyro: np.ndarray) -> np.ndarray:
        """Simulate gyroscope with realistic noise model"""
        # Add bias and bias walk
        self.gyro_bias += self.gyro_bias_walk * (1.0 / self.specs.update_rate)
        biased_gyro = true_gyro + self.gyro_bias

        # Add noise
        noise_std = self.gyroscope_noise_density * np.sqrt(self.specs.update_rate / 2)
        noise = np.random.normal(0, noise_std, 3)

        return biased_gyro + noise

    def simulate_magnetometer(self, true_mag: np.ndarray) -> np.ndarray:
        """Simulate magnetometer with noise"""
        noise = np.random.normal(0, self.magnetometer_noise, 3)
        return true_mag + noise

# Example: Creating a sensor suite for a humanoid robot
def create_humanoid_sensor_suite():
    """Create a complete sensor suite for a humanoid robot"""

    # Head camera
    head_camera_specs = SensorSpecs(
        name="head_camera",
        sensor_type=SensorType.CAMERA,
        update_rate=30.0,  # 30 Hz
        position=np.array([0.0, 0.0, 0.05]),  # 5cm above head center
        orientation=np.array([0, 0, 0, 1]),   # Looking forward
        noise_mean=0.0,
        noise_std=2.0,  # Pixel noise
        bias=0.0
    )

    # Torso IMU
    torso_imu_specs = SensorSpecs(
        name="torso_imu",
        sensor_type=SensorType.IMU,
        update_rate=100.0,  # 100 Hz
        position=np.array([0.0, 0.0, 0.0]),  # Center of torso
        orientation=np.array([0, 0, 0, 1]),  # Aligned with body
        noise_mean=0.0,
        noise_std=0.001,
        bias=0.0,
        drift_rate=0.00001  # Slow drift
    )

    # 360-degree LiDAR on head
    lidar_specs = SensorSpecs(
        name="head_lidar",
        sensor_type=SensorType.LIDAR,
        update_rate=10.0,  # 10 Hz
        position=np.array([0.0, 0.0, 0.1]),  # 10cm above head
        orientation=np.array([0, 0, 0, 1]),  # Horizontal plane
        noise_mean=0.0,
        noise_std=0.02,  # 2cm range noise
        bias=0.0
    )

    # Create sensors
    head_camera = CameraSimulator(
        head_camera_specs,
        width=640, height=480,
        fov_horizontal=math.pi/3,  # 60 degrees
        fov_vertical=math.pi/4     # 45 degrees
    )

    torso_imu = IMUSimulator(torso_imu_specs)

    head_lidar = LidarSimulator(
        lidar_specs,
        num_beams=720,  # 0.5 degree resolution
        min_range=0.1,
        max_range=10.0,
        fov_horizontal=2*math.pi,  # 360 degrees
        fov_vertical=0.174         # 10 degrees
    )

    return {
        'head_camera': head_camera,
        'torso_imu': torso_imu,
        'head_lidar': head_lidar
    }

# Example usage
sensor_suite = create_humanoid_sensor_suite()
print("Humanoid sensor suite created with:")
for name, sensor in sensor_suite.items():
    print(f"  - {name} ({sensor.specs.sensor_type.value})")
```

## Camera Simulation in Detail

### Realistic Camera Modeling

Camera simulation in Gazebo models the complete imaging pipeline, from light transport to digital conversion:

```python
class AdvancedCameraSimulator:
    """Advanced camera simulator with realistic optical and electronic effects"""

    def __init__(self, width: int = 640, height: int = 480,
                 fov_horizontal: float = 1.047,  # 60 degrees
                 fov_vertical: float = 0.785,    # 45 degrees
                 focal_length: float = 500.0,    # pixels
                 sensor_width: float = 3.2,      # mm
                 sensor_height: float = 2.4,     # mm
                 pixel_size: float = 0.005):     # mm
        self.width = width
        self.height = height
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.pixel_size = pixel_size

        # Camera intrinsic parameters
        self.cx = width / 2
        self.cy = height / 2
        self.intrinsic_matrix = np.array([
            [focal_length, 0, self.cx],
            [0, focal_length, self.cy],
            [0, 0, 1]
        ])

        # Distortion parameters (Brown-Conrady model)
        self.distortion_coeffs = np.array([0.1, -0.2, 0.001, -0.001, 0.05])

        # Camera effects parameters
        self.exposure_time = 1.0 / 30  # 30 FPS
        self.iso_sensitivity = 100
        self.shutter_speed = 1.0 / 60  # 1/60s
        self.aperture = 2.8  # f/2.8

        # Noise parameters
        self.readout_noise = 2.0      # electrons RMS
        self.dark_current = 0.1       # electrons/pixel/second
        self.photo_response_nonuniformity = 0.02  # 2% variation
        self.quantization_noise = 0.5  # AD conversion noise

        # Initialize camera state
        self.last_capture_time = 0.0
        self.dark_frame = self.generate_dark_frame()

    def generate_dark_frame(self) -> np.ndarray:
        """Generate a dark frame with thermal noise"""
        dark_frame = np.random.poisson(self.dark_current * self.exposure_time,
                                     (self.height, self.width, 3))
        return dark_frame.astype(np.float32)

    def simulate_optics(self, scene_image: np.ndarray) -> np.ndarray:
        """Simulate optical effects including aberrations and diffraction"""
        # Apply lens distortion
        distorted_image = self.apply_lens_distortion(scene_image)

        # Apply blurring due to diffraction and aberrations
        blur_kernel = self.calculate_psf()
        blurred_image = cv2.filter2D(distorted_image, -1, blur_kernel)

        return blurred_image

    def apply_lens_distortion(self, image: np.ndarray) -> np.ndarray:
        """Apply lens distortion using OpenCV"""
        # Separate distortion coefficients
        k1, k2, p1, p2, k3 = self.distortion_coeffs

        # Generate coordinate maps
        map_x = np.zeros((self.height, self.width), dtype=np.float32)
        map_y = np.zeros((self.height, self.width), dtype=np.float32)

        for y in range(self.height):
            for x in range(self.width):
                # Normalize coordinates
                x_norm = (x - self.cx) / self.focal_length
                y_norm = (y - self.cy) / self.focal_length

                # Apply radial distortion
                r_squared = x_norm**2 + y_norm**2
                radial_distortion = 1 + k1*r_squared + k2*r_squared**2 + k3*r_squared**3

                # Apply tangential distortion
                x_tangential = 2*p1*x_norm*y_norm + p2*(r_squared + 2*x_norm**2)
                y_tangential = p1*(r_squared + 2*y_norm**2) + 2*p2*x_norm*y_norm

                # Calculate distorted coordinates
                x_distorted = self.focal_length * (x_norm * radial_distortion + x_tangential) + self.cx
                y_distorted = self.focal_length * (y_norm * radial_distortion + y_tangential) + self.cy

                map_x[y, x] = x_distorted
                map_y[y, x] = y_distorted

        # Apply distortion using remapping
        distorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

        return distorted_image

    def calculate_psf(self) -> np.ndarray:
        """Calculate Point Spread Function for the lens"""
        # Simplified PSF calculation
        # In reality, this would be more complex and include diffraction effects
        kernel_size = 5
        sigma = 0.8  # Adjust based on f-number and wavelength

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

        return kernel / kernel.sum()  # Normalize

    def simulate_sensor_effects(self, image: np.ndarray, exposure_time: float) -> np.ndarray:
        """Simulate sensor-specific effects"""
        # Convert to photon counts (simplified)
        photon_image = image * (self.iso_sensitivity / 100.0) * exposure_time

        # Add photon shot noise (Poisson noise)
        photon_image = np.random.poisson(np.maximum(photon_image, 0))

        # Add readout noise (Gaussian)
        readout_noise = np.random.normal(0, self.readout_noise, photon_image.shape)
        sensor_image = photon_image + readout_noise

        # Add photo response nonuniformity
        prnu = np.random.normal(1.0, self.photo_response_nonuniformity, photon_image.shape)
        sensor_image = sensor_image * prnu

        # Apply quantization (simplified 8-bit)
        sensor_image = np.clip(sensor_image, 0, 255)
        sensor_image = np.round(sensor_image).astype(np.uint8)

        return sensor_image

    def simulate_motion_blur(self, image: np.ndarray, robot_velocity: np.ndarray,
                           robot_angular_velocity: np.ndarray) -> np.ndarray:
        """Simulate motion blur based on robot movement during exposure"""
        if np.linalg.norm(robot_velocity) < 0.01 and np.linalg.norm(robot_angular_velocity) < 0.01:
            return image  # No significant motion

        # Calculate motion blur kernel based on movement during exposure
        # This is a simplified approach - real implementation would be more complex
        motion_vector = robot_velocity * self.exposure_time
        motion_length = int(np.linalg.norm(motion_vector[:2]) * self.width / self.sensor_width)

        if motion_length > 1:
            # Create motion blur kernel
            kernel = np.zeros((motion_length, motion_length))
            kernel[motion_length//2, :] = 1.0 / motion_length
            blurred_image = cv2.filter2D(image, -1, kernel)
            return blurred_image

        return image

    def simulate_atmospheric_effects(self, image: np.ndarray, distance_map: np.ndarray,
                                   weather_conditions: Dict) -> np.ndarray:
        """Simulate atmospheric effects like fog, haze, etc."""
        # Simplified atmospheric simulation
        visibility = weather_conditions.get('visibility', 100.0)  # meters
        fog_density = weather_conditions.get('fog_density', 0.0)

        if fog_density > 0:
            # Apply fog based on distance
            attenuation = np.exp(-distance_map * fog_density)
            fog_color = np.array([150, 150, 150], dtype=np.float32)  # Light gray fog

            # Blend image with fog color
            image = image.astype(np.float32)
            image = image * attenuation[..., np.newaxis] + fog_color * (1 - attenuation[..., np.newaxis])
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def capture_image(self, scene_data: Dict, robot_state: Dict,
                     weather_conditions: Dict = None) -> np.ndarray:
        """Complete image capture pipeline"""
        if weather_conditions is None:
            weather_conditions = {'visibility': 100.0, 'fog_density': 0.0}

        # Start with the rendered scene
        scene_image = scene_data.get('rendered_image', np.zeros((self.height, self.width, 3), dtype=np.uint8))

        # Apply optical effects
        optical_image = self.simulate_optics(scene_image)

        # Apply motion blur if moving
        velocity = robot_state.get('linear_velocity', np.array([0, 0, 0]))
        angular_velocity = robot_state.get('angular_velocity', np.array([0, 0, 0]))
        motion_blurred_image = self.simulate_motion_blur(optical_image, velocity, angular_velocity)

        # Apply atmospheric effects
        distance_map = scene_data.get('distance_map', np.ones((self.height, self.width)))
        atmospheric_image = self.simulate_atmospheric_effects(
            motion_blurred_image, distance_map, weather_conditions
        )

        # Apply sensor effects
        final_image = self.simulate_sensor_effects(atmospheric_image, self.exposure_time)

        return final_image

class CameraSensorNode:
    """ROS 2 node for camera sensor simulation"""

    def __init__(self):
        self.camera_simulator = AdvancedCameraSimulator()
        self.last_capture_time = 0.0
        self.update_rate = 30.0  # Hz

    def update(self, current_time: float, world_state: Dict) -> Optional[np.ndarray]:
        """Update camera and return image if it's time"""
        if current_time - self.last_capture_time >= 1.0 / self.update_rate:
            self.last_capture_time = current_time

            # Create dummy scene data for simulation
            scene_data = {
                'rendered_image': np.random.randint(0, 255, (self.camera_simulator.height,
                                                           self.camera_simulator.width, 3), dtype=np.uint8),
                'distance_map': np.random.uniform(1.0, 10.0, (self.camera_simulator.height,
                                                            self.camera_simulator.width))
            }

            return self.camera_simulator.capture_image(scene_data, world_state)
        return None

# Example usage
camera_node = CameraSensorNode()
print(f"Advanced camera simulator initialized with resolution: "
      f"{camera_node.camera_simulator.width}x{camera_node.camera_simulator.height}")
```

## LiDAR Simulation with Realistic Physics

### Advanced LiDAR Modeling

LiDAR simulation models the complete sensing process, from laser emission to return detection:

```python
class AdvancedLidarSimulator:
    """Advanced LiDAR simulator with realistic beam physics"""

    def __init__(self, num_beams: int = 720,
                 fov_horizontal: float = 2 * math.pi,  # 360 degrees
                 fov_vertical: float = 0.174,          # 10 degrees
                 min_range: float = 0.1,
                 max_range: float = 30.0,
                 resolution: float = 0.01,             # 1cm resolution
                 divergence: float = 0.003,            # 3 mrad beam divergence
                 wavelength: float = 905e-9,           # 905 nm
                 update_rate: float = 10.0):
        self.num_beams = num_beams
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.min_range = min_range
        self.max_range = max_range
        self.resolution = resolution
        self.divergence = divergence
        self.wavelength = wavelength
        self.update_rate = update_rate

        # Generate beam angles
        self.azimuth_angles = np.linspace(0, fov_horizontal, num_beams, endpoint=False)
        self.elevation_angles = np.array([0.0])  # For simplicity, single horizontal plane
        # In real 3D LiDAR, this would have multiple elevation angles

        # LiDAR noise parameters
        self.range_noise_std = 0.02  # 2cm standard deviation
        self.intensity_noise_std = 0.1
        self.missed_detection_rate = 0.01  # 1% of returns may be missed
        self.false_alarm_rate = 0.001     # 0.1% false returns

        # Atmospheric effects
        self.atmospheric_attenuation = 0.1  # 0.1 dB/km for 905nm in clear weather

    def simulate_beam(self, beam_angle: float, world_state: Dict) -> Tuple[float, float, bool]:
        """Simulate a single LiDAR beam"""
        # Calculate beam direction
        beam_dir = np.array([math.cos(beam_angle), math.sin(beam_angle), 0])

        # Find intersection with scene objects
        intersection = self.find_beam_intersection(beam_dir, world_state)

        if intersection is None:
            # No intersection - return maximum range
            return self.max_range, 0.0, False

        distance, intensity, surface_normal = intersection

        # Apply range noise
        noisy_distance = distance + np.random.normal(0, self.range_noise_std)

        # Apply intensity noise
        noisy_intensity = intensity + np.random.normal(0, self.intensity_noise_std)

        # Apply atmospheric attenuation
        atmospheric_factor = np.exp(-self.atmospheric_attenuation * distance / 1000)
        noisy_intensity *= atmospheric_factor

        # Check for missed detection
        missed_detection = np.random.random() < self.missed_detection_rate

        return max(self.min_range, min(self.max_range, noisy_distance)), \
               max(0, noisy_intensity), \
               not missed_detection

    def find_beam_intersection(self, beam_dir: np.ndarray, world_state: Dict) -> Optional[Tuple[float, float, np.ndarray]]:
        """Find intersection of beam with scene objects"""
        # Simplified intersection with known objects
        # In real implementation, this would use ray-scene intersection algorithms

        objects = world_state.get('scene_objects', [
            {'position': np.array([2.0, 0.0, 0.0]), 'size': 0.5, 'reflectivity': 0.8},
            {'position': np.array([-1.5, 1.0, 0.0]), 'size': 0.3, 'reflectivity': 0.6},
            {'position': np.array([0.0, -3.0, 0.0]), 'size': 0.4, 'reflectivity': 0.9},
        ])

        min_distance = float('inf')
        closest_intersection = None

        for obj in objects:
            obj_pos = obj['position']
            obj_size = obj['size']
            reflectivity = obj['reflectivity']

            # Calculate distance to object sphere
            # Vector from origin to object center
            to_obj = obj_pos
            # Project onto beam direction
            proj_length = np.dot(to_obj, beam_dir)
            if proj_length < 0:  # Object behind sensor
                continue

            # Closest point on beam to object center
            closest_point = proj_length * beam_dir
            distance_to_obj = np.linalg.norm(to_obj - closest_point)

            if distance_to_obj <= obj_size:  # Beam intersects object
                intersection_distance = proj_length - math.sqrt(obj_size**2 - distance_to_obj**2)
                if intersection_distance < min_distance:
                    min_distance = intersection_distance
                    # Calculate surface normal at intersection
                    intersection_point = intersection_distance * beam_dir
                    surface_normal = (intersection_point - obj_pos) / obj_size
                    intensity = reflectivity * 100  # Simplified intensity calculation
                    closest_intersection = (intersection_distance, intensity, surface_normal)

        return closest_intersection

    def simulate_complete_scan(self, world_state: Dict) -> Dict[str, np.ndarray]:
        """Simulate a complete LiDAR scan"""
        ranges = np.full(self.num_beams, self.max_range, dtype=np.float32)
        intensities = np.zeros(self.num_beams, dtype=np.float32)
        valid_returns = np.zeros(self.num_beams, dtype=bool)

        for i, angle in enumerate(self.azimuth_angles):
            distance, intensity, valid = self.simulate_beam(angle, world_state)
            ranges[i] = distance
            intensities[i] = intensity
            valid_returns[i] = valid

        # Apply false alarms
        false_alarm_indices = np.random.choice(
            self.num_beams,
            size=int(self.false_alarm_rate * self.num_beams),
            replace=False
        )
        for idx in false_alarm_indices:
            if not valid_returns[idx]:  # Only add false alarms where no real return exists
                ranges[idx] = np.random.uniform(self.min_range, self.max_range)
                intensities[idx] = np.random.uniform(0, 10)
                valid_returns[idx] = True

        return {
            'ranges': ranges,
            'intensities': intensities,
            'valid_returns': valid_returns,
            'azimuth_angles': self.azimuth_angles,
            'timestamp': world_state.get('current_time', 0.0)
        }

    def generate_point_cloud(self, scan_data: Dict) -> np.ndarray:
        """Generate point cloud from scan data"""
        valid_indices = scan_data['valid_returns']
        valid_ranges = scan_data['ranges'][valid_indices]
        valid_angles = scan_data['azimuth_angles'][valid_indices]

        # Convert to Cartesian coordinates
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        z = np.zeros_like(x)  # For 2D LiDAR, z is 0

        return np.column_stack([x, y, z])

    def simulate_multi_beam_lidar(self, world_state: Dict) -> Dict[str, any]:
        """Simulate a multi-beam LiDAR (3D LiDAR)"""
        # For this example, we'll simulate a simple vertical multi-beam setup
        vertical_beams = 16  # Common for Velodyne-style LiDARs
        vertical_fov = 30.0 * math.pi / 180  # 30 degrees vertical FOV
        vertical_angles = np.linspace(-vertical_fov/2, vertical_fov/2, vertical_beams)

        all_points = []
        all_intensities = []

        for v_angle in vertical_angles:
            # Create rotated beam directions
            ranges = []
            intensities = []

            for h_angle in self.azimuth_angles:
                # Combine horizontal and vertical angles
                total_angle = h_angle
                beam_dir = np.array([
                    math.cos(total_angle) * math.cos(v_angle),
                    math.sin(total_angle) * math.cos(v_angle),
                    math.sin(v_angle)
                ])

                # Simplified intersection for this vertical angle
                distance, intensity, valid = self.simulate_beam(total_angle, world_state)
                ranges.append(distance if valid else self.max_range)
                intensities.append(intensity if valid else 0)

            # Convert to 3D points
            ranges = np.array(ranges)
            valid_mask = ranges < self.max_range
            valid_ranges = ranges[valid_mask]
            valid_angles = self.azimuth_angles[valid_mask]

            x = valid_ranges * np.cos(valid_angles) * math.cos(v_angle)
            y = valid_ranges * np.sin(valid_angles) * math.cos(v_angle)
            z = valid_ranges * math.sin(v_angle)

            points = np.column_stack([x, y, z])
            all_points.append(points)
            all_intensities.append(np.array(intensities)[valid_mask])

        # Combine all points
        if all_points:
            combined_points = np.vstack(all_points)
            combined_intensities = np.concatenate(all_intensities)
        else:
            combined_points = np.empty((0, 3))
            combined_intensities = np.array([])

        return {
            'point_cloud': combined_points,
            'intensities': combined_intensities,
            'num_beams': vertical_beams,
            'vertical_angles': vertical_angles
        }

class LidarSensorNode:
    """ROS 2 node for LiDAR sensor simulation"""

    def __init__(self):
        self.lidar_simulator = AdvancedLidarSimulator(
            num_beams=1080,  # Higher resolution
            fov_horizontal=2 * math.pi,  # 360 degrees
            max_range=20.0,
            update_rate=10.0
        )
        self.last_scan_time = 0.0

    def update(self, current_time: float, world_state: Dict) -> Optional[Dict]:
        """Update LiDAR and return scan if it's time"""
        if current_time - self.last_scan_time >= 1.0 / self.lidar_simulator.update_rate:
            self.last_scan_time = current_time
            return self.lidar_simulator.simulate_complete_scan(world_state)
        return None

# Example usage
lidar_node = LidarSensorNode()
print(f"Advanced LiDAR simulator initialized with {lidar_node.lidar_simulator.num_beams} beams "
      f"and {lidar_node.lidar_simulator.max_range}m max range")
```

## IMU Simulation with Advanced Physics

### Comprehensive IMU Modeling

IMU simulation models the complete behavior of accelerometers, gyroscopes, and magnetometers:

```python
class AdvancedIMUSimulator:
    """Advanced IMU simulator with realistic sensor physics"""

    def __init__(self, update_rate: float = 100.0,
                 accelerometer_params: Dict = None,
                 gyroscope_params: Dict = None,
                 magnetometer_params: Dict = None):
        self.update_rate = update_rate
        self.time_step = 1.0 / update_rate

        # Accelerometer parameters (typical for automotive-grade IMU)
        self.accel_params = accelerometer_params or {
            'range': 16.0 * 9.81,  # ±16g
            'resolution': 24,      # 24-bit ADC
            'noise_density': 100e-6,  # 100 μg/√Hz
            'bias_instability': 50e-6,  # 50 μg
            'in_run_bias_stability': 100e-6,  # 100 μg
            'non_linearity': 0.1,   # 0.1% of full scale
            'cross_axis_sensitivity': 0.02,  # 2%
            'temperature_coefficient': 2e-6,  # 2 μg/°C
            'bandwidth': 200.0     # Hz
        }

        # Gyroscope parameters (typical for tactical-grade IMU)
        self.gyro_params = gyroscope_params or {
            'range': 200.0 * math.pi / 180,  # ±200 deg/s
            'resolution': 24,      # 24-bit
            'noise_density': 0.2e-3,  # 0.2 mdeg/s/√Hz
            'bias_instability': 10e-3,  # 10 deg/hr
            'in_run_bias_stability': 20e-3,  # 20 deg/hr
            'non_linearity': 0.1,   # 0.1% of full scale
            'cross_axis_sensitivity': 0.01,  # 1%
            'temperature_coefficient': 1e-6,  # 1 deg/s/°C
            'bandwidth': 200.0     # Hz
        }

        # Magnetometer parameters
        self.mag_params = magnetometer_params or {
            'range': 1300e-6,     # ±1300 μT
            'resolution': 24,     # 24-bit
            'noise': 100e-9,      # 100 nT RMS
            'non_linearity': 0.1, # 0.1% of full scale
            'temperature_coefficient': 1e-6,  # 1 μT/°C
            'bandwidth': 100.0    # Hz
        }

        # Initialize internal states
        self.accel_bias = np.random.normal(0, self.accel_params['in_run_bias_stability'], 3)
        self.gyro_bias = np.random.normal(0, self.gyro_params['in_run_bias_stability'], 3) * math.pi/180/3600  # Convert to rad/s
        self.mag_bias = np.random.normal(0, 100e-9, 3)  # Magnetic bias in Tesla

        # Bias walk processes (random walk)
        self.accel_bias_walk = np.random.normal(0, self.accel_params['bias_instability'] * math.sqrt(self.time_step), 3)
        self.gyro_bias_walk = np.random.normal(0, self.gyro_params['bias_instability'] * math.pi/180/3600 * math.sqrt(self.time_step), 3)
        self.mag_bias_walk = np.random.normal(0, 10e-9 * math.sqrt(self.time_step), 3)

        # Temperature (simplified)
        self.temperature = 25.0  # degrees Celsius
        self.temperature_drift = 0.0  # cumulative temperature drift

    def update_temperature(self, ambient_temperature: float, dt: float):
        """Update internal temperature based on ambient conditions"""
        # Simple first-order thermal model
        thermal_time_constant = 100.0  # seconds
        self.temperature += (ambient_temperature - self.temperature) * dt / thermal_time_constant

        # Update temperature-related drifts
        self.temperature_drift = (self.temperature - 25.0)  # Reference temperature is 25°C

    def simulate_accelerometer(self, true_accel: np.ndarray, dt: float) -> np.ndarray:
        """Simulate accelerometer with full error model"""
        # Apply full scale range limits
        limited_accel = np.clip(true_accel, -self.accel_params['range'], self.accel_params['range'])

        # Update bias through random walk
        self.accel_bias += self.accel_bias_walk
        self.accel_bias_walk = np.random.normal(0, self.accel_params['bias_instability'] * math.sqrt(dt), 3)

        # Apply temperature effects
        temp_effect = self.temperature_drift * self.accel_params['temperature_coefficient']
        temp_biased_accel = limited_accel + temp_effect

        # Add bias
        biased_accel = temp_biased_accel + self.accel_bias

        # Add noise (considering noise density and bandwidth)
        noise_std = self.accel_params['noise_density'] * math.sqrt(min(self.accel_params['bandwidth'], self.update_rate) / 2)
        noise = np.random.normal(0, noise_std, 3)

        # Apply non-linearity (simplified)
        nonlinearity = self.accel_params['non_linearity'] * 0.01 * biased_accel * np.abs(biased_accel) / self.accel_params['range']

        # Apply cross-axis sensitivity (simplified)
        cross_axis_matrix = np.array([
            [1.0, self.accel_params['cross_axis_sensitivity'], self.accel_params['cross_axis_sensitivity']],
            [self.accel_params['cross_axis_sensitivity'], 1.0, self.accel_params['cross_axis_sensitivity']],
            [self.accel_params['cross_axis_sensitivity'], self.accel_params['cross_axis_sensitivity'], 1.0]
        ])
        cross_axis_accel = cross_axis_matrix @ biased_accel

        # Final measurement
        measurement = cross_axis_accel + noise + nonlinearity

        # Apply ADC quantization (simplified)
        resolution = self.accel_params['range'] / (2**(self.accel_params['resolution']-1))
        measurement = np.round(measurement / resolution) * resolution

        return measurement

    def simulate_gyroscope(self, true_gyro: np.ndarray, dt: float) -> np.ndarray:
        """Simulate gyroscope with full error model"""
        # Apply full scale range limits
        limited_gyro = np.clip(true_gyro, -self.gyro_params['range'], self.gyro_params['range'])

        # Update bias through random walk
        self.gyro_bias += self.gyro_bias_walk
        self.gyro_bias_walk = np.random.normal(0, self.gyro_params['bias_instability'] * math.pi/180/3600 * math.sqrt(dt), 3)

        # Apply temperature effects
        temp_effect = self.temperature_drift * self.gyro_params['temperature_coefficient']
        temp_biased_gyro = limited_gyro + temp_effect

        # Add bias
        biased_gyro = temp_biased_gyro + self.gyro_bias

        # Add noise
        noise_std = self.gyro_params['noise_density'] * math.sqrt(min(self.gyro_params['bandwidth'], self.update_rate) / 2)
        noise = np.random.normal(0, noise_std, 3)

        # Apply non-linearity
        nonlinearity = self.gyro_params['non_linearity'] * 0.01 * biased_gyro * np.abs(biased_gyro) / self.gyro_params['range']

        # Apply cross-axis sensitivity
        cross_axis_matrix = np.array([
            [1.0, self.gyro_params['cross_axis_sensitivity'], self.gyro_params['cross_axis_sensitivity']],
            [self.gyro_params['cross_axis_sensitivity'], 1.0, self.gyro_params['cross_axis_sensitivity']],
            [self.gyro_params['cross_axis_sensitivity'], self.gyro_params['cross_axis_sensitivity'], 1.0]
        ])
        cross_axis_gyro = cross_axis_matrix @ biased_gyro

        # Final measurement
        measurement = cross_axis_gyro + noise + nonlinearity

        # Apply ADC quantization
        resolution = self.gyro_params['range'] / (2**(self.gyro_params['resolution']-1))
        measurement = np.round(measurement / resolution) * resolution

        return measurement

    def simulate_magnetometer(self, true_mag: np.ndarray, dt: float) -> np.ndarray:
        """Simulate magnetometer with full error model"""
        # Apply full scale range limits
        limited_mag = np.clip(true_mag, -self.mag_params['range'], self.mag_params['range'])

        # Update bias
        self.mag_bias += self.mag_bias_walk
        self.mag_bias_walk = np.random.normal(0, self.mag_params['noise'] * math.sqrt(dt), 3)

        # Apply temperature effects
        temp_effect = self.temperature_drift * self.mag_params['temperature_coefficient']
        temp_biased_mag = limited_mag + temp_effect

        # Add bias
        biased_mag = temp_biased_mag + self.mag_bias

        # Add noise
        noise = np.random.normal(0, self.mag_params['noise'], 3)

        # Apply non-linearity
        nonlinearity = self.mag_params['non_linearity'] * 0.01 * biased_mag * np.abs(biased_mag) / self.mag_params['range']

        # Final measurement
        measurement = biased_mag + noise + nonlinearity

        # Apply ADC quantization
        resolution = self.mag_params['range'] / (2**(self.mag_params['resolution']-1))
        measurement = np.round(measurement / resolution) * resolution

        return measurement

    def simulate_complete_imu(self, true_state: Dict, dt: float,
                            ambient_temperature: float = 25.0) -> Dict[str, np.ndarray]:
        """Simulate complete IMU output"""
        # Update temperature
        self.update_temperature(ambient_temperature, dt)

        # Extract true values
        true_accel = np.array(true_state.get('linear_acceleration', [0, 0, 9.81]))
        true_gyro = np.array(true_state.get('angular_velocity', [0, 0, 0]))
        true_mag = np.array(true_state.get('magnetic_field', [25000e-9, 0, 45000e-9]))  # Earth's field in Tesla

        # Simulate each sensor
        accel_measurement = self.simulate_accelerometer(true_accel, dt)
        gyro_measurement = self.simulate_gyroscope(true_gyro, dt)
        mag_measurement = self.simulate_magnetometer(true_mag, dt)

        return {
            'accelerometer': accel_measurement,
            'gyroscope': gyro_measurement,
            'magnetometer': mag_measurement,
            'temperature': self.temperature,
            'timestamp': true_state.get('timestamp', 0.0)
        }

class IMUSensorNode:
    """ROS 2 node for IMU sensor simulation"""

    def __init__(self):
        self.imu_simulator = AdvancedIMUSimulator(update_rate=100.0)
        self.last_update_time = 0.0

    def update(self, current_time: float, world_state: Dict) -> Optional[Dict]:
        """Update IMU and return data if it's time"""
        dt = current_time - self.last_update_time
        if dt >= 1.0 / self.imu_simulator.update_rate:
            self.last_update_time = current_time
            ambient_temp = world_state.get('ambient_temperature', 25.0)
            return self.imu_simulator.simulate_complete_imu(world_state, dt, ambient_temp)
        return None

# Example usage
imu_node = IMUSensorNode()
print("Advanced IMU simulator initialized with realistic error models")

# Example world state for testing
test_world_state = {
    'linear_acceleration': [0.1, 0.05, 9.7],
    'angular_velocity': [0.01, -0.02, 0.005],
    'magnetic_field': [24500e-9, -1000e-9, 44000e-9],
    'timestamp': 1.0
}

imu_data = imu_node.imu_simulator.simulate_complete_imu(test_world_state, 0.01)
print(f"IMU simulation results:")
print(f"  Accelerometer: [{imu_data['accelerometer'][0]:.6f}, {imu_data['accelerometer'][1]:.6f}, {imu_data['accelerometer'][2]:.6f}] m/s²")
print(f"  Gyroscope: [{imu_data['gyroscope'][0]:.6f}, {imu_data['gyroscope'][1]:.6f}, {imu_data['gyroscope'][2]:.6f}] rad/s")
print(f"  Temperature: {imu_data['temperature']:.2f} °C")
```

## Sensor Fusion and Integration

### Combining Multiple Sensors

Realistic humanoid robots use sensor fusion to combine data from multiple sensors:

```python
class SensorFusionNode:
    """Fuses data from multiple sensors for improved perception"""

    def __init__(self):
        self.camera = CameraSensorNode()
        self.lidar = LidarSensorNode()
        self.imu = IMUSensorNode()

        # State estimation components
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.orientation = R.from_quat([0, 0, 0, 1])  # Identity rotation
        self.last_update_time = 0.0

    def update_sensors(self, current_time: float, world_state: Dict) -> Dict[str, any]:
        """Update all sensors and fuse their data"""
        fused_data = {}

        # Update individual sensors
        camera_data = self.camera.update(current_time, world_state)
        lidar_data = self.lidar.update(current_time, world_state)
        imu_data = self.imu.update(current_time, world_state)

        # Store raw sensor data
        fused_data['camera'] = camera_data
        fused_data['lidar'] = lidar_data
        fused_data['imu'] = imu_data

        # Perform sensor fusion
        if imu_data is not None:
            # Update state estimate using IMU data
            self.update_state_with_imu(imu_data, current_time - self.last_update_time)

        if lidar_data is not None:
            # Extract features from LiDAR for localization
            features = self.extract_features_from_lidar(lidar_data)
            fused_data['features'] = features

        if camera_data is not None:
            # Extract visual features
            visual_features = self.extract_visual_features(camera_data)
            fused_data['visual_features'] = visual_features

        # Combine sensor data for better state estimation
        if imu_data is not None and lidar_data is not None:
            combined_estimate = self.fuse_imu_lidar(imu_data, lidar_data)
            fused_data['combined_estimate'] = combined_estimate

        self.last_update_time = current_time
        return fused_data

    def update_state_with_imu(self, imu_data: Dict, dt: float):
        """Update position/velocity/attitude using IMU data"""
        # Get measurements
        accel = imu_data['accelerometer']
        gyro = imu_data['gyroscope']

        # Update orientation using gyroscope
        angular_velocity = gyro
        rotation_vector = angular_velocity * dt
        rotation_update = R.from_rotvec(rotation_vector)
        self.orientation = rotation_update * self.orientation

        # Transform acceleration to global frame and integrate
        global_accel = self.orientation.apply(accel)
        # Remove gravity from vertical component
        gravity = np.array([0, 0, 9.81])
        linear_accel = global_accel - gravity

        # Update velocity and position
        self.velocity += linear_accel * dt
        self.position += self.velocity * dt

    def extract_features_from_lidar(self, lidar_data: Dict) -> Dict:
        """Extract features from LiDAR data for localization"""
        ranges = lidar_data['ranges']
        valid_returns = lidar_data['valid_returns']

        # Simple feature extraction
        # Find obstacles at different distances
        obstacle_distances = ranges[valid_returns]
        obstacle_angles = lidar_data['azimuth_angles'][valid_returns]

        # Calculate statistics
        features = {
            'min_distance': np.min(obstacle_distances) if len(obstacle_distances) > 0 else float('inf'),
            'max_distance': np.max(obstacle_distances) if len(obstacle_distances) > 0 else 0,
            'avg_distance': np.mean(obstacle_distances) if len(obstacle_distances) > 0 else 0,
            'num_obstacles': np.sum(valid_returns),
            'free_space_ratio': np.sum(ranges > 5.0) / len(ranges) if len(ranges) > 0 else 0
        }

        return features

    def extract_visual_features(self, image: np.ndarray) -> Dict:
        """Extract features from camera image"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Simple feature extraction using OpenCV
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate features
        features = {
            'num_contours': len(contours),
            'total_edge_pixels': np.sum(edges > 0),
            'image_entropy': self.calculate_image_entropy(gray),
            'dominant_colors': self.get_dominant_colors(image)
        }

        return features

    def calculate_image_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy as a measure of information content"""
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize to get probabilities
        hist = hist[hist > 0]  # Remove zero probabilities
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def get_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[np.ndarray]:
        """Get dominant colors using K-means clustering"""
        # Reshape image for clustering
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert centers back to uint8
        centers = centers.astype(np.uint8)
        return [center for center in centers]

    def fuse_imu_lidar(self, imu_data: Dict, lidar_data: Dict) -> Dict:
        """Fuse IMU and LiDAR data for improved state estimation"""
        # This would implement an Extended Kalman Filter or similar
        # For this example, we'll provide a simplified fusion

        # Get IMU-based velocity estimate
        imu_velocity = self.velocity  # From our internal integration

        # Get LiDAR-based velocity estimate (simplified)
        # This would typically come from scan matching or ICP algorithms
        lidar_velocity = self.estimate_velocity_from_lidar(lidar_data)

        # Simple weighted fusion (in reality, this would use proper Kalman filtering)
        fused_velocity = 0.7 * imu_velocity + 0.3 * lidar_velocity

        return {
            'position': self.position,
            'velocity': fused_velocity,
            'orientation': self.orientation.as_quat(),
            'imu_confidence': 0.7,
            'lidar_confidence': 0.3
        }

    def estimate_velocity_from_lidar(self, lidar_data: Dict) -> np.ndarray:
        """Estimate velocity from LiDAR data (simplified)"""
        # In reality, this would use scan matching or other algorithms
        # For now, return zero velocity as a placeholder
        return np.array([0.0, 0.0, 0.0])

# Example usage of sensor fusion
fusion_node = SensorFusionNode()
world_state = {
    'linear_acceleration': [0.1, 0.05, 9.7],
    'angular_velocity': [0.01, -0.02, 0.005],
    'magnetic_field': [24500e-9, -1000e-9, 44000e-9],
    'scene_objects': [
        {'position': np.array([2.0, 0.0, 0.0]), 'size': 0.5, 'reflectivity': 0.8},
        {'position': np.array([-1.5, 1.0, 0.0]), 'size': 0.3, 'reflectivity': 0.6}
    ]
}

fused_result = fusion_node.update_sensors(1.0, world_state)
print(f"Sensor fusion completed with {len(fused_result)} data streams")
if 'combined_estimate' in fused_result:
    est = fused_result['combined_estimate']
    print(f"Fused position estimate: [{est['position'][0]:.3f}, {est['position'][1]:.3f}, {est['position'][2]:.3f}]")
    print(f"Fused velocity estimate: [{est['velocity'][0]:.3f}, {est['velocity'][1]:.3f}, {est['velocity'][2]:.3f}]")
```

Sensor simulation in Gazebo provides realistic representations of real-world sensors, enabling humanoid robots to develop perception capabilities that can transfer from simulation to reality. By accurately modeling sensor physics, noise characteristics, and environmental effects, developers can create robust perception systems that handle the challenges of real-world deployment.