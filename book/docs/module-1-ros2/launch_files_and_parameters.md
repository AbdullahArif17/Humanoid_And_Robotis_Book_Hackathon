---
sidebar_position: 6
---

# Launch Files & Parameters

## Understanding ROS 2 Launch System

The ROS 2 launch system provides a declarative way to bring up complex robotic systems with multiple nodes, configurations, and dependencies. For humanoid robots, which typically involve dozens of nodes for perception, control, planning, and interaction, the launch system is essential for managing the complexity of system startup and configuration.

### Why Launch Files Matter for Humanoid Robotics

Humanoid robots require coordination of multiple subsystems:

- **Perception Nodes**: Vision, IMU, joint encoders, force sensors
- **Control Nodes**: Joint controllers, balance control, trajectory generation
- **Planning Nodes**: Path planning, motion planning, behavior planning
- **Interaction Nodes**: Speech, gesture, user interface
- **Simulation Nodes**: Robot state publisher, transforms, physics simulation

Launch files allow you to start all these components in a coordinated manner with appropriate parameters and dependencies.

## Launch File Structure and Syntax

### Basic Launch File

```python
# launch/basic_humanoid.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_name = LaunchConfiguration('robot_name', default='humanoid_robot')
    config_file = LaunchConfiguration('config_file', default='')
    debug = LaunchConfiguration('debug', default='false')

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([FindPackageShare('humanoid_robot_description'), 'config', 'robot.yaml'])
        ],
        arguments=[PathJoinSubstitution([FindPackageShare('humanoid_robot_description'), 'urdf', 'humanoid.urdf'])]
    )

    # Joint state publisher (for simulation or testing)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'rate': 50}  # 50 Hz
        ]
    )

    # Joint state publisher GUI (optional)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(debug)
    )

    # Main control node
    humanoid_controller = Node(
        package='humanoid_robot_control',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_name': robot_name},
            {'control_loop_rate': 100},  # 100 Hz control loop
            {'safety_limits.max_velocity': 1.0},
            {'safety_limits.max_torque': 50.0}
        ],
        remappings=[
            ('/joint_states', '/joint_states'),
            ('/cmd_vel', '/cmd_vel'),
            ('/balance_commands', '/balance_commands')
        ]
    )

    # Perception node
    perception_node = Node(
        package='humanoid_robot_perception',
        executable='perception_node',
        name='perception_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'camera_topic': '/camera/image_raw'},
            {'point_cloud_topic': '/laser_scan/points'},
            {'detection_threshold': 0.5}
        ]
    )

    # Behavior manager
    behavior_manager = Node(
        package='humanoid_robot_behavior',
        executable='behavior_manager',
        name='behavior_manager',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'default_behavior': 'idle'},
            {'behavior_priority': ['emergency', 'balance', 'navigation', 'interaction']}
        ]
    )

    return LaunchDescription([
        # Declare launch arguments
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
        DeclareLaunchArgument(
            'config_file',
            default_value='',
            description='Path to custom configuration file'
        ),
        DeclareLaunchArgument(
            'debug',
            default_value='false',
            description='Enable debug nodes'
        ),

        # Launch nodes
        robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui,
        humanoid_controller,
        perception_node,
        behavior_manager
    ])
```

### Advanced Launch File with Conditions and Dependencies

```python
# launch/advanced_humanoid.launch.py
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
    RegisterEventHandler, SetEnvironmentVariable
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    enable_vision = LaunchConfiguration('enable_vision')
    enable_audio = LaunchConfiguration('enable_audio')
    log_level = LaunchConfiguration('log_level')

    # Conditional launch arguments
    sim_launch_file = LaunchConfiguration('sim_launch_file', default='')
    hardware_config = LaunchConfiguration('hardware_config', default='')

    # Set environment variables if needed
    log_config_env = SetEnvironmentVariable(
        name='RCUTILS_LOGGING_SEVERITY_THRESHOLD',
        value=log_level
    )

    # Robot state publisher with conditional parameters
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'publish_frequency': 50.0}
        ],
        arguments=[PathJoinSubstitution([FindPackageShare('humanoid_robot_description'), 'urdf', 'humanoid.urdf'])],
        respawn=True,
        respawn_delay=2.0,
        on_exit=OnProcessExit(
            target_action=None,  # This would be another action if needed
        )
    )

    # Main controller with conditional remappings
    humanoid_controller = Node(
        package='humanoid_robot_control',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_name': robot_name},
            {'control_loop_rate': 1000},  # 1kHz for high-performance control
            {'balance_control.enable': True},
            {'balance_control.kp': 100.0},
            {'balance_control.kd': 10.0},
            {'walking_controller.enable': True},
            {'walking_controller.step_height': 0.05},
            {'walking_controller.step_length': 0.3},
        ],
        remappings=[
            ('/joint_states', '/joint_states'),
            ('/imu/data', '/imu/data'),
            ('/cmd_vel', '/cmd_vel'),
            ('/balance_commands', '/balance_commands'),
        ],
        respawn=True,
        respawn_delay=1.0,
        output='screen'
    )

    # Vision processing node (conditional)
    vision_node = Node(
        package='humanoid_robot_vision',
        executable='vision_node',
        name='vision_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'camera_topic': '/camera/image_raw'},
            {'detection_model': 'yolov8n.pt'},
            {'tracking_algorithm': 'csrt'},
        ],
        condition=IfCondition(enable_vision),
        respawn=True
    )

    # Audio processing node (conditional)
    audio_node = Node(
        package='humanoid_robot_audio',
        executable='audio_node',
        name='audio_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'microphone_topic': '/audio/raw'},
            {'vad_threshold': 0.3},
            {'noise_suppression': True},
        ],
        condition=IfCondition(enable_audio),
        respawn=True
    )

    # Navigation stack
    navigation_group = GroupAction(
        condition=IfCondition(
            PythonExpression(['"', robot_name, '" != "simulation"'])
        ),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('nav2_bringup'),
                        'launch',
                        'navigation_launch.py'
                    ])
                ]),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'params_file': PathJoinSubstitution([
                        FindPackageShare('humanoid_robot_nav2_config'),
                        'config',
                        'nav2_params.yaml'
                    ])
                }.items()
            )
        ]
    )

    # Simulation integration (if specified)
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_robot_gazebo'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
        condition=IfCondition(PythonExpression(['"', sim_launch_file, '" != ""']))
    )

    # Hardware interface (if specified)
    hardware_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_robot_hardware_interface'),
                'launch',
                'hardware.launch.py'
            ])
        ]),
        condition=IfCondition(PythonExpression(['"', hardware_config, '" != ""']))
    )

    # Event handlers for coordination
    def on_controller_start(event, context):
        print("Humanoid controller started, initializing balance control...")
        # Could trigger additional initialization here

    def on_vision_start(event, context):
        print("Vision system started, calibrating cameras...")
        # Could trigger camera calibration here

    # Register event handlers
    controller_event_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=humanoid_controller,
            on_start=on_controller_start
        )
    )

    vision_event_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=vision_node,
            on_start=on_vision_start,
            condition=IfCondition(enable_vision)
        )
    )

    return LaunchDescription([
        # Environment setup
        log_config_env,

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
            'enable_vision',
            default_value='true',
            description='Enable vision processing nodes'
        ),
        DeclareLaunchArgument(
            'enable_audio',
            default_value='true',
            description='Enable audio processing nodes'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level'
        ),
        DeclareLaunchArgument(
            'sim_launch_file',
            default_value='',
            description='Path to simulation launch file (optional)'
        ),
        DeclareLaunchArgument(
            'hardware_config',
            default_value='',
            description='Path to hardware configuration (optional)'
        ),

        # Main nodes
        robot_state_publisher,
        humanoid_controller,

        # Conditional nodes
        vision_node,
        audio_node,

        # Included launch files
        simulation_launch,
        hardware_launch,

        # Group actions
        navigation_group,

        # Event handlers
        controller_event_handler,
        vision_event_handler,
    ])
```

## Parameter Management in Launch Files

### Parameter Files and YAML Configuration

```yaml
# config/humanoid_params.yaml
/**:
  ros__parameters:
    use_sim_time: false
    robot_name: "humanoid_robot"
    control:
      loop_rate: 100
      safety:
        max_velocity: 1.0
        max_torque: 50.0
        position_limits:
          hip: [-1.57, 1.57]
          knee: [-1.57, 0.0]
          ankle: [-0.5, 0.5]
    balance:
      enable: true
      kp: 100.0
      ki: 0.1
      kd: 10.0
      com_threshold: 0.05
      zmp_margin: 0.02
    walking:
      enable: true
      step_height: 0.05
      step_length: 0.3
      step_duration: 1.0
      foot_lift_duration: 0.3
      dsp_ratio: 0.2
    perception:
      camera:
        resolution: [640, 480]
        frame_rate: 30
        distortion_model: "plumb_bob"
      lidar:
        range_min: 0.1
        range_max: 10.0
        angle_min: -2.36
        angle_max: 2.36
    navigation:
      planner_frequency: 5.0
      controller_frequency: 20.0
      max_vel_x: 0.5
      min_vel_x: 0.1
      max_vel_theta: 1.0
    behavior:
      default_behavior: "idle"
      behavior_priority: ["emergency", "balance", "navigation", "interaction"]
      interaction_timeout: 30.0
```

### Loading Parameters in Launch Files

```python
# launch/parameterized_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    config_path = LaunchConfiguration('config_path')

    # Load parameters from multiple sources
    def load_params_file(filename):
        return PathJoinSubstitution([
            FindPackageShare('humanoid_robot_config'),
            'config',
            filename
        ])

    # Main controller with multiple parameter sources
    humanoid_controller = Node(
        package='humanoid_robot_control',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[
            # Default parameters
            {'use_sim_time': use_sim_time},
            {'robot_name': robot_name},

            # Load from YAML file
            load_params_file('control_params.yaml'),

            # Load from launch configuration (if provided)
            config_path,

            # Override specific parameters
            {'control_loop_rate': 1000},  # Higher rate for this specific instance
        ],
        remappings=[
            ('/joint_states', '/joint_states'),
            ('/imu/data', '/imu/data'),
        ]
    )

    # Perception node with parameter validation
    perception_node = Node(
        package='humanoid_robot_perception',
        executable='perception_node',
        name='perception_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            load_params_file('perception_params.yaml'),
            # Runtime parameter that might be overridden
            {'camera.exposure_time': 0.01},  # 10ms exposure
        ]
    )

    return LaunchDescription([
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
        DeclareLaunchArgument(
            'config_path',
            default_value='',
            description='Path to additional configuration file'
        ),
        humanoid_controller,
        perception_node,
    ])
```

## Complex Launch Scenarios for Humanoid Robots

### Multi-Robot Launch Configuration

```python
# launch/multi_humanoid.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Multi-robot configuration
    num_robots = LaunchConfiguration('num_robots', default='2')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_name = LaunchConfiguration('world_name', default='multi_room')

    # Environment setup
    log_level_env = SetEnvironmentVariable(
        name='RCUTILS_LOGGING_SEVERITY_THRESHOLD',
        value='INFO'
    )

    # Define robot configurations
    robot_configs = [
        {
            'name': 'alpha',
            'initial_pose': {'x': -2.0, 'y': 1.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        },
        {
            'name': 'beta',
            'initial_pose': {'x': 2.0, 'y': -1.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 3.14159}
        }
    ]

    launch_actions = [
        log_level_env,
        DeclareLaunchArgument('num_robots', default_value='2', description='Number of robots'),
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        DeclareLaunchArgument('world_name', default_value='multi_room', description='Gazebo world name'),
    ]

    # Create launch groups for each robot
    for config in robot_configs:
        robot_group = GroupAction(
            actions=[
                PushRosNamespace(config['name']),  # Namespace each robot

                # Robot state publisher for this robot
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name='robot_state_publisher',
                    parameters=[
                        {'use_sim_time': use_sim_time},
                        PathJoinSubstitution([
                            FindPackageShare('humanoid_robot_description'),
                            'config',
                            f"{config['name']}_robot.yaml"
                        ])
                    ],
                    arguments=[PathJoinSubstitution([
                        FindPackageShare('humanoid_robot_description'),
                        'urdf',
                        f"{config['name']}_humanoid.urdf"
                    ])],
                ),

                # Controller for this robot
                Node(
                    package='humanoid_robot_control',
                    executable='humanoid_controller',
                    name='humanoid_controller',
                    parameters=[
                        {'use_sim_time': use_sim_time},
                        {'robot_name': config['name']},
                        {'control_loop_rate': 100},
                        {'initial_pose': [config['initial_pose']['x'],
                                        config['initial_pose']['y'],
                                        config['initial_pose']['yaw']]},
                    ],
                    remappings=[
                        ('/joint_states', 'joint_states'),
                        ('/imu/data', 'imu/data'),
                        ('/cmd_vel', 'cmd_vel'),
                    ]
                ),

                # Perception for this robot
                Node(
                    package='humanoid_robot_perception',
                    executable='perception_node',
                    name='perception_node',
                    parameters=[
                        {'use_sim_time': use_sim_time},
                        {'camera_topic': 'camera/image_raw'},
                        {'lidar_topic': 'lidar/scan'},
                    ]
                ),

                # Communication bridge for multi-robot coordination
                Node(
                    package='humanoid_robot_communication',
                    executable='multi_robot_bridge',
                    name='multi_robot_bridge',
                    parameters=[
                        {'use_sim_time': use_sim_time},
                        {'robot_name': config['name']},
                        {'neighbor_names': [c['name'] for c in robot_configs if c['name'] != config['name']]},
                    ]
                ),
            ]
        )
        launch_actions.append(robot_group)

    # Shared nodes that run once for all robots
    shared_nodes = [
        # Multi-robot coordination manager
        Node(
            package='humanoid_robot_coordination',
            executable='coordination_manager',
            name='coordination_manager',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'robot_names': [c['name'] for c in robot_configs]},
                {'coordination_strategy': 'distributed'},
            ]
        ),

        # Shared visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='multi_robot_rviz',
            arguments=[
                '-d',
                PathJoinSubstitution([
                    FindPackageShare('humanoid_robot_viz'),
                    'rviz',
                    'multi_humanoid.rviz'
                ])
            ]
        ),
    ]

    launch_actions.extend(shared_nodes)

    return LaunchDescription(launch_actions)
```

### Launch File with Dynamic Parameter Updates

```python
# launch/dynamic_params.launch.py
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import json

def generate_launch_description():
    # Configuration
    use_sim_time = LaunchConfiguration('use_sim_time')
    scenario = LaunchConfiguration('scenario', default='default')
    adaptive_params = LaunchConfiguration('adaptive_params', default='false')

    # Main controller with scenario-specific parameters
    humanoid_controller = Node(
        package='humanoid_robot_control',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_name': 'adaptive_humanoid'},
            {'scenario': scenario},
        ],
        arguments=['--ros-args', '--log-level', 'INFO']
    )

    # Parameter adaptation node
    param_adaptation_node = Node(
        package='humanoid_robot_adaptation',
        executable='param_adaptation_node',
        name='param_adaptation_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'target_node': 'humanoid_controller'},
            {'adaptation_method': 'reinforcement_learning'},
            {'learning_rate': 0.01},
            {'update_frequency': 1.0},  # Update parameters every second
        ],
        condition=IfCondition(adaptive_params)
    )

    # Scenario setup based on configuration
    def get_scenario_params(scenario_name):
        scenarios = {
            'walking': {
                'control_loop_rate': 500,
                'balance_control.kp': 150.0,
                'walking_controller.step_height': 0.08,
                'walking_controller.step_length': 0.4,
            },
            'dancing': {
                'control_loop_rate': 1000,
                'balance_control.kp': 200.0,
                'motion_library': 'dance_moves.yaml',
                'rhythm_detection.enable': True,
            },
            'interaction': {
                'control_loop_rate': 200,
                'gesture_control.enable': True,
                'face_tracking.enable': True,
                'safety_limits.max_velocity': 0.5,
            },
            'default': {
                'control_loop_rate': 100,
                'balance_control.kp': 100.0,
                'safety_limits.max_velocity': 1.0,
            }
        }
        return scenarios.get(scenario_name, scenarios['default'])

    # Apply scenario-specific parameters
    scenario_params = get_scenario_params('default')  # This would be dynamic in real implementation

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'scenario',
            default_value='default',
            description='Scenario type: walking, dancing, interaction'
        ),
        DeclareLaunchArgument(
            'adaptive_params',
            default_value='false',
            description='Enable adaptive parameter tuning'
        ),
        humanoid_controller,
        param_adaptation_node,
    ])
```

## Parameter Validation and Best Practices

### Parameter Validation in Launch Files

```python
# launch/validated_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import sys

def validate_and_launch(context, *args, **kwargs):
    """Validate parameters before launching nodes"""
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    control_rate = LaunchConfiguration('control_rate').perform(context)
    robot_name = LaunchConfiguration('robot_name').perform(context)

    # Validate control rate
    try:
        rate = float(control_rate)
        if rate <= 0 or rate > 10000:  # Max 10kHz
            print(f"ERROR: Control rate {rate} is invalid (must be 0 < rate <= 10000)")
            sys.exit(1)
    except ValueError:
        print(f"ERROR: Control rate '{control_rate}' is not a valid number")
        sys.exit(1)

    # Validate robot name (simple validation)
    if not robot_name or len(robot_name) < 3:
        print(f"ERROR: Robot name '{robot_name}' is too short (minimum 3 characters)")
        sys.exit(1)

    # Validate boolean
    if use_sim_time.lower() not in ['true', 'false', '1', '0']:
        print(f"ERROR: use_sim_time '{use_sim_time}' must be true/false or 1/0")
        sys.exit(1)

    print(f"Validation passed: robot={robot_name}, rate={rate}Hz, sim_time={use_sim_time}")

    # Create nodes after validation
    humanoid_controller = Node(
        package='humanoid_robot_control',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[
            {'use_sim_time': use_sim_time.lower() in ['true', '1']},
            {'robot_name': robot_name},
            {'control_loop_rate': rate},
        ]
    )

    return [humanoid_controller]

def generate_launch_description():
    return LaunchDescription([
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
        DeclareLaunchArgument(
            'control_rate',
            default_value='100',
            description='Control loop rate in Hz'
        ),
        OpaqueFunction(function=validate_and_launch)
    ])
```

### Parameter Management Best Practices

```python
# launch/best_practices_launch.py
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction, RegisterEventHandler
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import (
    LaunchConfiguration, PathJoinSubstitution, PythonExpression
)
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Use consistent naming conventions
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_namespace = LaunchConfiguration('robot_namespace', default='humanoid')
    config_package = LaunchConfiguration('config_package', default='humanoid_robot_config')
    hardware_interface = LaunchConfiguration('hardware_interface', default='ros2_control')

    # Set global parameters
    set_global_params = [
        SetParameter(name='use_sim_time', value=use_sim_time),
        SetParameter(name='robot_namespace', value=robot_namespace),
    ]

    # Organized node groups
    control_group = GroupAction(
        actions=[
            # Main controller
            Node(
                package='humanoid_robot_control',
                executable='humanoid_controller',
                name='controller',
                namespace=robot_namespace,
                parameters=[
                    PathJoinSubstitution([
                        FindPackageShare(config_package),
                        'config',
                        'control.yaml'
                    ]),
                    {'control_loop_rate': 1000},  # High rate for real-time control
                ],
                respawn=True,
                respawn_delay=2.0,
                output='screen',
                arguments=['--ros-args', '--log-level', 'INFO']
            ),

            # Trajectory generator
            Node(
                package='humanoid_robot_control',
                executable='trajectory_generator',
                name='trajectory_generator',
                namespace=robot_namespace,
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'trajectory_planning.rate': 100},  # Planning rate
                ],
                respawn=True,
            ),
        ]
    )

    perception_group = GroupAction(
        actions=[
            # Vision processing
            Node(
                package='humanoid_robot_perception',
                executable='vision_node',
                name='vision_node',
                namespace=robot_namespace,
                parameters=[
                    {'use_sim_time': use_sim_time},
                    PathJoinSubstitution([
                        FindPackageShare(config_package),
                        'config',
                        'vision.yaml'
                    ]),
                ],
                respawn=True,
            ),

            # Sensor fusion
            Node(
                package='humanoid_robot_perception',
                executable='sensor_fusion',
                name='sensor_fusion',
                namespace=robot_namespace,
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'sensor_sources': ['imu', 'joint_states', 'vision']},
                    {'fusion_rate': 100},
                ],
                respawn=True,
            ),
        ]
    )

    # Hardware interface selection
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='ros2_control_node',
        namespace=robot_namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare(config_package),
                'config',
                PythonExpression(['"', hardware_interface, '.yaml"'])
            ]),
        ],
        condition=IfCondition(
            PythonExpression(['"', hardware_interface, '" == "ros2_control"'])
        )
    )

    # Event handling for system coordination
    def on_controller_start(event, context):
        print(f"Controller started for robot: {robot_namespace.perform(context)}")

    def on_system_shutdown(event, context):
        print("Initiating safe shutdown sequence...")

    controller_start_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=None,  # Would reference the controller node
            on_start=on_controller_start
        )
    )

    return LaunchDescription([
        # Launch arguments with clear descriptions
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'robot_namespace',
            default_value='humanoid',
            description='Robot namespace for multi-robot systems'
        ),
        DeclareLaunchArgument(
            'config_package',
            default_value='humanoid_robot_config',
            description='Package containing configuration files'
        ),
        DeclareLaunchArgument(
            'hardware_interface',
            default_value='ros2_control',
            description='Hardware interface type: ros2_control, direct, etc.'
        ),

        # Global parameter settings
        *set_global_params,

        # Node groups
        control_group,
        perception_group,
        ros2_control_node,

        # Event handlers
        controller_start_handler,
    ])
```

## Common Mistakes and Troubleshooting

### Common Launch File Mistakes

#### Parameter Conflicts
- **Overlapping Parameter Names**: Multiple nodes using the same parameter names
- **Incorrect Parameter Paths**: Parameters not reaching intended nodes
- **Type Mismatches**: Providing wrong data types for parameters

#### Launch Order Issues
- **Missing Dependencies**: Nodes starting before required services are available
- **Race Conditions**: Nodes competing for the same resources
- **Timing Issues**: Real-time constraints not met during startup

#### Resource Management
- **Memory Leaks**: Nodes not properly cleaned up on shutdown
- **Port Conflicts**: Multiple nodes trying to use the same hardware interfaces
- **CPU Overload**: Too many nodes running simultaneously

### Troubleshooting Techniques

```python
# launch/debug_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Debug configuration
    debug_mode = LaunchConfiguration('debug', default='false')
    log_level = LaunchConfiguration('log_level', default='INFO')
    enable_monitoring = LaunchConfiguration('enable_monitoring', default='true')

    # Main nodes
    humanoid_controller = Node(
        package='humanoid_robot_control',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'debug_mode': debug_mode},
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'  # Always output to screen in debug mode
    )

    # Debug tools
    rqt_gui = Node(
        package='rqt_gui',
        executable='rqt_gui',
        name='rqt_debug',
        condition=IfCondition(debug_mode)
    )

    # System monitoring
    system_monitor = Node(
        package='humanoid_robot_diagnostics',
        executable='system_monitor',
        name='system_monitor',
        parameters=[
            {'monitor_frequency': 1.0},
            {'cpu_threshold': 80.0},
            {'memory_threshold': 80.0},
        ],
        condition=IfCondition(enable_monitoring)
    )

    return LaunchDescription([
        DeclareLaunchArgument('debug', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='INFO'),
        DeclareLaunchArgument('enable_monitoring', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        humanoid_controller,
        rqt_gui,
        system_monitor,
    ])
```

## Best Practices for Humanoid Robot Launch Files

### Organizational Best Practices

1. **Modular Structure**: Separate launch files for different subsystems
2. **Consistent Naming**: Use consistent parameter and node naming conventions
3. **Documentation**: Comment launch files with clear explanations
4. **Validation**: Include parameter validation where appropriate
5. **Error Handling**: Implement graceful degradation for missing components

### Performance Considerations

- **Startup Order**: Ensure critical nodes start first
- **Resource Allocation**: Consider CPU and memory requirements
- **Real-time Constraints**: Ensure timing-critical nodes start reliably
- **Monitoring**: Include system health monitoring

### Security Considerations

- **Parameter Protection**: Validate and sanitize all parameters
- **Namespace Isolation**: Use proper namespaces for multi-robot systems
- **Access Control**: Limit parameter access where appropriate

Launch files and parameter management are critical for humanoid robotics systems. They provide the foundation for reliably starting complex multi-node systems with appropriate configurations. Well-designed launch files ensure that humanoid robots can be deployed consistently across different environments and use cases.