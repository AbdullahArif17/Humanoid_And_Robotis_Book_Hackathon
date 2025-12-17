# Visual SLAM (VSLAM) for Humanoid Robotics

## Overview

Visual Simultaneous Localization and Mapping (VSLAM) is a critical technology for humanoid robots operating in unknown or dynamic environments. VSLAM enables robots to simultaneously build a map of their surroundings while tracking their position within that map using visual sensors such as cameras. For humanoid robots, VSLAM provides the spatial awareness necessary for navigation, interaction, and manipulation tasks.

This chapter explores the principles, algorithms, and implementation of VSLAM systems specifically tailored for humanoid robotics applications, including the challenges of operating with the unique kinematics and sensor configurations of humanoid platforms.

## VSLAM Fundamentals

VSLAM combines computer vision and robotics to solve two interconnected problems:
1. **Localization**: Determining the robot's position and orientation in the environment
2. **Mapping**: Creating a representation of the environment

The process involves:
- Feature extraction from visual input
- Feature matching across frames
- Pose estimation and tracking
- Map building and maintenance
- Loop closure detection

```python
# VSLAM Core Implementation for Humanoid Robotics
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import threading
import time
from scipy.spatial.transform import Rotation as R
import open3d as o3d

@dataclass
class VSLAMConfig:
    """Configuration for VSLAM system"""
    feature_detector: str = "ORB"  # Options: ORB, SIFT, SURF, AKAZE
    descriptor_matcher: str = "BF"  # Options: BF, FLANN
    min_matches: int = 10
    max_features: int = 1000
    tracking_threshold: float = 2.0  # Maximum reprojection error
    map_size: int = 1000  # Maximum number of map points
    keyframe_threshold: float = 10.0  # Minimum distance for keyframe
    loop_closure_threshold: float = 5.0  # Threshold for loop closure

class MapPoint:
    """Represents a 3D point in the map"""
    def __init__(self, point_id: int, position: np.ndarray, descriptor: np.ndarray):
        self.id = point_id
        self.position = position  # 3D position [x, y, z]
        self.descriptor = descriptor  # Feature descriptor
        self.observations = []  # List of (frame_id, keypoint_idx)
        self.is_bad = False

class KeyFrame:
    """Represents a keyframe in the VSLAM system"""
    def __init__(self, frame_id: int, image: np.ndarray, pose: np.ndarray, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray):
        self.id = frame_id
        self.image = image
        self.pose = pose  # 4x4 transformation matrix
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.map_points = {}  # keypoint_idx -> MapPoint

class VSLAMSystem:
    """Core VSLAM system for humanoid robotics"""

    def __init__(self, config: VSLAMConfig):
        self.config = config
        self.map_points = {}  # point_id -> MapPoint
        self.keyframes = {}   # frame_id -> KeyFrame
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.current_frame_id = 0

        # Feature detector and descriptor
        if config.feature_detector == "ORB":
            self.detector = cv2.ORB_create(nfeatures=config.max_features)
        elif config.feature_detector == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=config.max_features)
        elif config.feature_detector == "AKAZE":
            self.detector = cv2.AKAZE_create()

        # Descriptor matcher
        if config.descriptor_matcher == "BF":
            self.matcher = cv2.BFMatcher()
        elif config.descriptor_matcher == "FLANN":
            self.matcher = cv2.FlannBasedMatcher()

        # Visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="VSLAM Map", width=800, height=600)
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)

        # Threading
        self.lock = threading.Lock()
        self.running = True

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        """Process a single frame and return the current pose"""
        with self.lock:
            # Extract features
            keypoints, descriptors = self.detector.detectAndCompute(image, None)

            if descriptors is None or len(keypoints) < self.config.min_matches:
                return self.current_pose

            # If this is the first frame, initialize the map
            if self.current_frame_id == 0:
                self.initialize_map(image, keypoints, descriptors)
                self.current_frame_id += 1
                return self.current_pose

            # Track features from previous frame
            prev_keyframe = self.keyframes[self.current_frame_id - 1]
            pose = self.track_frame(image, keypoints, descriptors, prev_keyframe)

            # Update current pose
            self.current_pose = pose
            self.current_frame_id += 1

            # Add keyframe if significant motion occurred
            if self.should_add_keyframe(pose, prev_keyframe.pose):
                self.add_keyframe(image, keypoints, descriptors, pose)

            # Update visualization
            self.update_visualization()

            return pose

    def initialize_map(self, image: np.ndarray, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray):
        """Initialize the map with the first frame"""
        # Create initial keyframe at origin
        initial_pose = np.eye(4)
        keyframe = KeyFrame(0, image, initial_pose, keypoints, descriptors)
        self.keyframes[0] = keyframe

        # Initialize map points (simplified - in reality, would need stereo or motion)
        # For this example, we'll create placeholder map points
        for i, kp in enumerate(keypoints[:50]):  # Limit for performance
            # Placeholder 3D position (in reality, would be triangulated)
            pos = np.array([kp.pt[0] * 0.001, kp.pt[1] * 0.001, 1.0])
            map_point = MapPoint(i, pos, descriptors[i])
            self.map_points[i] = map_point
            keyframe.map_points[i] = map_point

    def track_frame(self, image: np.ndarray, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray, prev_keyframe: KeyFrame) -> np.ndarray:
        """Track the current frame against the previous keyframe"""
        # Match descriptors with previous keyframe
        matches = self.matcher.match(prev_keyframe.descriptors, descriptors)

        if len(matches) < self.config.min_matches:
            return self.current_pose  # Return previous pose if insufficient matches

        # Extract matched points
        prev_pts = np.float32([prev_keyframe.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate motion using essential matrix
        E, mask = cv2.findEssentialMat(curr_pts, prev_pts, method=cv2.RANSAC, threshold=1.0)

        if E is not None:
            # Extract rotation and translation
            _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts)

            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

            # Update pose (compose with previous pose)
            new_pose = T @ prev_keyframe.pose
            return new_pose

        return self.current_pose

    def should_add_keyframe(self, current_pose: np.ndarray, prev_pose: np.ndarray) -> bool:
        """Determine if a new keyframe should be added"""
        # Calculate distance between poses
        pos_diff = current_pose[:3, 3] - prev_pose[:3, 3]
        distance = np.linalg.norm(pos_diff)

        return distance > self.config.keyframe_threshold

    def add_keyframe(self, image: np.ndarray, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray, pose: np.ndarray):
        """Add a new keyframe to the map"""
        keyframe = KeyFrame(self.current_frame_id, image, pose, keypoints, descriptors)
        self.keyframes[self.current_frame_id] = keyframe

        # Update map points based on current frame
        self.update_map_points(keyframe)

    def update_map_points(self, keyframe: KeyFrame):
        """Update map points based on the current keyframe"""
        # This is a simplified version - in practice, would involve triangulation
        # and optimization techniques
        pass

    def update_visualization(self):
        """Update the 3D visualization"""
        # Extract map points for visualization
        points = []
        colors = []

        for point in self.map_points.values():
            if not point.is_bad:
                points.append(point.position)
                # Color based on number of observations
                color_intensity = min(1.0, len(point.observations) * 0.1)
                colors.append([color_intensity, 0.5, 1.0 - color_intensity])

        if points:
            self.point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
            self.point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

        self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        """Run the VSLAM system continuously"""
        while self.running:
            time.sleep(0.01)  # 100 Hz

    def stop(self):
        """Stop the VSLAM system"""
        self.running = False
        self.vis.destroy_window()
```

## Humanoid-Specific VSLAM Challenges

Humanoid robots present unique challenges for VSLAM systems:

### 1. Dynamic Body Parts

Humanoid robots have moving limbs that can interfere with feature tracking:

```python
# Humanoid-specific VSLAM with body part filtering
import numpy as np
import cv2
from typing import List, Tuple

class HumanoidVSLAM(VSLAMSystem):
    """VSLAM system adapted for humanoid robots"""

    def __init__(self, config: VSLAMConfig):
        super().__init__(config)
        # Define regions to exclude (e.g., robot body parts)
        self.exclusion_masks = []
        self.body_part_regions = [
            # Example: torso region to exclude from tracking
            np.array([[100, 100], [200, 100], [200, 300], [100, 300]], dtype=np.int32),
            # Add more regions as needed
        ]

    def filter_features(self, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Filter out features from robot body parts"""
        filtered_keypoints = []
        filtered_descriptors = []

        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])

            # Check if keypoint is in exclusion region
            is_excluded = False
            for region in self.body_part_regions:
                if cv2.pointPolygonTest(region, (x, y), False) >= 0:
                    is_excluded = True
                    break

            if not is_excluded:
                filtered_keypoints.append(kp)
                if descriptors is not None and i < len(descriptors):
                    filtered_descriptors.append(descriptors[i])

        return filtered_keypoints, np.array(filtered_descriptors) if filtered_descriptors else None

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        """Process frame with humanoid-specific filtering"""
        with self.lock:
            # Extract features
            keypoints, descriptors = self.detector.detectAndCompute(image, None)

            # Filter out robot body parts
            keypoints, descriptors = self.filter_features(keypoints, descriptors)

            if descriptors is None or len(keypoints) < self.config.min_matches:
                return self.current_pose

            # Continue with standard processing
            if self.current_frame_id == 0:
                self.initialize_map(image, keypoints, descriptors)
                self.current_frame_id += 1
                return self.current_pose

            prev_keyframe = self.keyframes[self.current_frame_id - 1]
            pose = self.track_frame(image, keypoints, descriptors, prev_keyframe)

            self.current_pose = pose
            self.current_frame_id += 1

            if self.should_add_keyframe(pose, prev_keyframe.pose):
                self.add_keyframe(image, keypoints, descriptors, pose)

            self.update_visualization()

            return pose
```

### 2. Head Movement Compensation

Humanoid robots often move their heads for visual attention, which affects VSLAM:

```python
# Head movement compensation for VSLAM
class HeadCompensatedVSLAM(HumanoidVSLAM):
    """VSLAM with head movement compensation"""

    def __init__(self, config: VSLAMConfig):
        super().__init__(config)
        self.head_pose_history = []  # Track head movements
        self.max_history = 10

    def compensate_head_movement(self, image: np.ndarray, head_pose: np.ndarray) -> np.ndarray:
        """Compensate for head movement before processing"""
        # Apply inverse head transformation to image
        # This is a simplified approach - in practice, would use more sophisticated methods
        R_inv = head_pose[:3, :3].T
        t_inv = -R_inv @ head_pose[:3, 3]

        transform = np.eye(4)
        transform[:3, :3] = R_inv
        transform[:3, 3] = t_inv

        # Apply transformation to image (simplified)
        # In practice, would use geometric transformation
        return image

    def process_frame_with_head_pose(self, image: np.ndarray, head_pose: np.ndarray) -> np.ndarray:
        """Process frame with head pose compensation"""
        # Compensate for head movement
        compensated_image = self.compensate_head_movement(image, head_pose)

        # Store head pose in history
        self.head_pose_history.append(head_pose)
        if len(self.head_pose_history) > self.max_history:
            self.head_pose_history.pop(0)

        # Process compensated image
        return self.process_frame(compensated_image)
```

## Advanced VSLAM Techniques for Humanoid Robotics

### 1. Multi-Camera VSLAM

Humanoid robots often have multiple cameras (stereo, RGB-D, etc.):

```python
# Multi-camera VSLAM for humanoid robots
class MultiCameraVSLAM:
    """VSLAM system for multi-camera humanoid robots"""

    def __init__(self, camera_configs: List[VSLAMConfig]):
        self.cameras = [VSLAMSystem(config) for config in camera_configs]
        self.camera_poses = []  # Relative poses of cameras to robot base
        self.fusion_threshold = 0.1  # Threshold for measurement fusion

    def add_camera_pose(self, camera_id: int, base_to_camera: np.ndarray):
        """Add camera pose relative to robot base"""
        if camera_id >= len(self.camera_poses):
            self.camera_poses.extend([None] * (camera_id - len(self.camera_poses) + 1))
        self.camera_poses[camera_id] = base_to_camera

    def process_multi_camera_frame(self, images: List[np.ndarray]) -> np.ndarray:
        """Process frames from multiple cameras"""
        if len(images) != len(self.cameras):
            raise ValueError("Number of images must match number of cameras")

        # Process each camera independently
        camera_poses = []
        for i, (image, camera) in enumerate(zip(images, self.cameras)):
            pose = camera.process_frame(image)
            camera_poses.append(pose)

        # Fuse camera measurements
        fused_pose = self.fuse_camera_measurements(camera_poses)

        return fused_pose

    def fuse_camera_measurements(self, camera_poses: List[np.ndarray]) -> np.ndarray:
        """Fuse poses from multiple cameras"""
        # Simple fusion - average poses weighted by confidence
        # In practice, would use more sophisticated fusion (EKF, particle filter, etc.)

        if not camera_poses:
            return np.eye(4)

        # Average rotation (using quaternion averaging)
        quats = []
        translations = []

        for pose in camera_poses:
            # Extract rotation and translation
            R = pose[:3, :3]
            t = pose[:3, 3]

            # Convert to quaternion
            r = R.from_matrix(R)
            quat = r.as_quat()

            quats.append(quat)
            translations.append(t)

        # Average quaternions (simplified)
        avg_quat = np.mean(quats, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)  # Normalize

        # Average translations
        avg_translation = np.mean(translations, axis=0)

        # Create fused pose
        fused_pose = np.eye(4)
        fused_pose[:3, :3] = R.from_quat(avg_quat).as_matrix()
        fused_pose[:3, 3] = avg_translation

        return fused_pose
```

### 2. Semantic VSLAM

Integrating semantic information for better mapping:

```python
# Semantic VSLAM for humanoid robots
import torch
import torchvision.transforms as T

class SemanticVSLAM(VSLAMSystem):
    """VSLAM with semantic information"""

    def __init__(self, config: VSLAMConfig, semantic_model_path: str = None):
        super().__init__(config)
        self.semantic_model = self.load_semantic_model(semantic_model_path)
        self.semantic_threshold = 0.7  # Confidence threshold for semantic segmentation
        self.semantic_classes = {
            0: 'background',
            1: 'human',
            2: 'robot',
            3: 'furniture',
            4: 'obstacle',
            # Add more classes as needed
        }

    def load_semantic_model(self, model_path: str):
        """Load semantic segmentation model"""
        # In practice, would load a pre-trained model
        # For this example, we'll use a placeholder
        if model_path:
            # Load actual model
            pass
        else:
            # Use placeholder model
            return None

    def get_semantic_mask(self, image: np.ndarray) -> np.ndarray:
        """Get semantic segmentation mask for the image"""
        # In practice, would run semantic segmentation model
        # For this example, return a placeholder mask
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.int32)

        # Placeholder: simple color-based segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask[gray > 128] = 1  # Background vs foreground

        return mask

    def filter_features_by_semantics(self, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray,
                                   semantic_mask: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Filter features based on semantic information"""
        filtered_keypoints = []
        filtered_descriptors = []

        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])

            # Get semantic class at keypoint location
            if 0 <= y < semantic_mask.shape[0] and 0 <= x < semantic_mask.shape[1]:
                semantic_class = semantic_mask[y, x]

                # Filter out dynamic objects (humans, robots, etc.)
                if semantic_class not in [1, 2]:  # Not human or robot
                    filtered_keypoints.append(kp)
                    if descriptors is not None and i < len(descriptors):
                        filtered_descriptors.append(descriptors[i])

        return filtered_keypoints, np.array(filtered_descriptors) if filtered_descriptors else None

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        """Process frame with semantic filtering"""
        with self.lock:
            # Get semantic segmentation
            semantic_mask = self.get_semantic_mask(image)

            # Extract features
            keypoints, descriptors = self.detector.detectAndCompute(image, None)

            # Filter features based on semantics
            keypoints, descriptors = self.filter_features_by_semantics(keypoints, descriptors, semantic_mask)

            if descriptors is None or len(keypoints) < self.config.min_matches:
                return self.current_pose

            # Continue with standard processing
            if self.current_frame_id == 0:
                self.initialize_map(image, keypoints, descriptors)
                self.current_frame_id += 1
                return self.current_pose

            prev_keyframe = self.keyframes[self.current_frame_id - 1]
            pose = self.track_frame(image, keypoints, descriptors, prev_keyframe)

            self.current_pose = pose
            self.current_frame_id += 1

            if self.should_add_keyframe(pose, prev_keyframe.pose):
                self.add_keyframe(image, keypoints, descriptors, pose)

            self.update_visualization()

            return pose
```

## Real-Time Performance Optimization

### GPU-Accelerated VSLAM

```python
# GPU-accelerated VSLAM components
import cupy as cp
import cupy.sparse as sparse

class GPUAcceleratedVSLAM(VSLAMSystem):
    """VSLAM system with GPU acceleration"""

    def __init__(self, config: VSLAMConfig):
        super().__init__(config)
        # Initialize GPU arrays for performance
        self.gpu_keypoints = None
        self.gpu_descriptors = None
        self.gpu_matches = None

    def detect_and_compute_gpu(self, image: np.ndarray):
        """GPU-accelerated feature detection and description"""
        # In practice, would use GPU-accelerated OpenCV or custom CUDA kernels
        # For this example, we'll simulate GPU processing

        # Convert image to GPU memory
        gpu_image = cp.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # Perform feature detection on GPU (simulated)
        # In real implementation, would use cv2.cuda or custom CUDA kernels
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        return keypoints, descriptors

    def match_descriptors_gpu(self, desc1: np.ndarray, desc2: np.ndarray):
        """GPU-accelerated descriptor matching"""
        # Convert to GPU arrays
        gpu_desc1 = cp.asarray(desc1) if desc1 is not None else None
        gpu_desc2 = cp.asarray(desc2) if desc2 is not None else None

        if gpu_desc1 is None or gpu_desc2 is None:
            # Fallback to CPU if descriptors are None
            return self.matcher.match(desc1, desc2)

        # Perform matching on GPU (simulated)
        # In real implementation, would use GPU-accelerated matching
        matches = self.matcher.match(desc1, desc2)

        return matches
```

## Integration with Navigation Systems

### VSLAM for Path Planning

```python
# VSLAM integration with navigation
class VSLAMNavigator:
    """Navigation system using VSLAM map"""

    def __init__(self, vslam_system: VSLAMSystem):
        self.vslam = vslam_system
        self.path = []
        self.current_goal = None

    def plan_path(self, start_pose: np.ndarray, goal_pose: np.ndarray) -> List[np.ndarray]:
        """Plan path using VSLAM map"""
        # Extract map points for path planning
        map_points = []
        for point in self.vslam.map_points.values():
            if not point.is_bad:
                map_points.append(point.position)

        # Create occupancy grid from map points
        occupancy_grid = self.create_occupancy_grid(map_points)

        # Plan path using A* or other algorithm
        path = self.a_star_pathfinding(occupancy_grid, start_pose[:3, 3], goal_pose[:3, 3])

        return path

    def create_occupancy_grid(self, map_points: List[np.ndarray]) -> np.ndarray:
        """Create occupancy grid from VSLAM map points"""
        # Define grid parameters
        grid_size = 100  # 100x100 grid
        grid_resolution = 0.1  # 10cm resolution
        grid_origin = np.array([-5.0, -5.0])  # Grid origin in world coordinates

        # Create occupancy grid
        occupancy_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

        for point in map_points:
            # Convert world coordinates to grid coordinates
            grid_x = int((point[0] - grid_origin[0]) / grid_resolution)
            grid_y = int((point[1] - grid_origin[1]) / grid_resolution)

            # Check bounds
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                occupancy_grid[grid_y, grid_x] = 1  # Mark as occupied

        return occupancy_grid

    def a_star_pathfinding(self, grid: np.ndarray, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """A* pathfinding on occupancy grid"""
        # Simplified A* implementation
        # In practice, would use a more robust implementation

        start_idx = self.world_to_grid(start, grid)
        goal_idx = self.world_to_grid(goal, grid)

        # Placeholder for A* result
        path = [start, goal]  # In reality, would compute actual path

        return path

    def world_to_grid(self, world_pos: np.ndarray, grid: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        # Simplified conversion
        grid_resolution = 0.1
        grid_origin = np.array([-5.0, -5.0])

        grid_x = int((world_pos[0] - grid_origin[0]) / grid_resolution)
        grid_y = int((world_pos[1] - grid_origin[1]) / grid_resolution)

        return (grid_x, grid_y)
```

## Evaluation and Testing

### VSLAM Quality Metrics

```python
# VSLAM evaluation metrics
class VSLAMEvaluator:
    """Evaluation metrics for VSLAM systems"""

    def __init__(self):
        self.trajectory_errors = []
        self.map_quality_scores = []
        self.tracking_accuracy = []

    def evaluate_trajectory(self, estimated_poses: List[np.ndarray], ground_truth_poses: List[np.ndarray]) -> Dict:
        """Evaluate trajectory accuracy"""
        if len(estimated_poses) != len(ground_truth_poses):
            raise ValueError("Estimated and ground truth poses must have same length")

        errors = []
        for est, gt in zip(estimated_poses, ground_truth_poses):
            # Calculate position error
            pos_error = np.linalg.norm(est[:3, 3] - gt[:3, 3])
            errors.append(pos_error)

        # Calculate metrics
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        mean_error = np.mean(errors)
        max_error = np.max(errors)

        return {
            'rmse': rmse,
            'mean_error': mean_error,
            'max_error': max_error,
            'errors': errors
        }

    def evaluate_map_quality(self, map_points: Dict, ground_truth_map: Dict) -> float:
        """Evaluate map quality"""
        # Calculate map coverage and accuracy
        # This is a simplified evaluation

        if not map_points or not ground_truth_map:
            return 0.0

        # Calculate precision and recall of map points
        estimated_points = set(point.id for point in map_points.values())
        ground_truth_points = set(point.id for point in ground_truth_map.values())

        if not ground_truth_points:
            return 0.0

        true_positives = len(estimated_points.intersection(ground_truth_points))
        precision = true_positives / len(estimated_points) if estimated_points else 0.0
        recall = true_positives / len(ground_truth_points) if ground_truth_points else 0.0

        # F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1_score

    def evaluate_tracking(self, tracking_results: List[Dict]) -> Dict:
        """Evaluate feature tracking performance"""
        if not tracking_results:
            return {'success_rate': 0.0, 'avg_features': 0.0}

        success_count = 0
        total_features = 0

        for result in tracking_results:
            if result.get('success', False):
                success_count += 1
            total_features += result.get('num_features', 0)

        success_rate = success_count / len(tracking_results) if tracking_results else 0.0
        avg_features = total_features / len(tracking_results) if tracking_results else 0.0

        return {
            'success_rate': success_rate,
            'avg_features': avg_features
        }
```

## Summary

Visual SLAM is a fundamental technology for humanoid robotics, enabling robots to navigate and interact with unknown environments. The implementation for humanoid robots requires special considerations for dynamic body parts, head movements, and multi-sensor integration.

Key takeaways:
- VSLAM combines localization and mapping using visual sensors
- Humanoid-specific challenges include body part filtering and head movement compensation
- Multi-camera and semantic VSLAM enhance robustness and accuracy
- GPU acceleration is crucial for real-time performance
- Proper evaluation metrics ensure system reliability

The next chapter will explore Navigation with Nav2, focusing on how humanoid robots can use the maps generated by VSLAM for autonomous navigation.