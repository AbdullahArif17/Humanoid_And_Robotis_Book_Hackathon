---
sidebar_position: 6
---

# Unity for Visualization

## Introduction to Unity in Robotics

Unity has emerged as a powerful platform for robotics visualization, simulation, and digital twin applications. While Gazebo remains the traditional choice for physics-based simulation in the ROS ecosystem, Unity offers unique advantages for visualization, user interaction, and creating immersive experiences for humanoid robotics. Unity's real-time rendering capabilities, extensive asset ecosystem, and cross-platform deployment options make it an attractive choice for robotics visualization applications.

### Unity's Role in Robotics

Unity serves several key functions in robotics development:

- **High-quality visualization**: Photorealistic rendering for presentations and demonstrations
- **User interface development**: Interactive control panels and monitoring interfaces
- **Virtual reality integration**: Immersive teleoperation and training environments
- **Digital twin visualization**: Real-time display of robot state and sensor data
- **Simulation**: Physics simulation (though less detailed than Gazebo for robotics)

```csharp
// Example Unity C# script for robot visualization
using UnityEngine;
using System.Collections.Generic;

public class RobotVisualizer : MonoBehaviour
{
    [Header("Robot Configuration")]
    public string robotName = "HumanoidRobot";
    public List<Transform> jointTransforms = new List<Transform>();
    public List<ArticulationBody> jointBodies = new List<ArticulationBody>();

    [Header("Visualization Settings")]
    public Color activeJointColor = Color.green;
    public Color passiveJointColor = Color.gray;
    public float jointHighlightDuration = 0.1f;

    [Header("ROS Integration")]
    public bool useROSIntegration = true;
    public string robotNamespace = "/humanoid_robot";

    private Dictionary<string, int> jointNameToIndex = new Dictionary<string, int>();
    private Material[] originalMaterials;

    void Start()
    {
        InitializeRobotStructure();
        StoreOriginalMaterials();
    }

    void InitializeRobotStructure()
    {
        // Find all joint transforms in the robot hierarchy
        jointTransforms.Clear();
        jointBodies.Clear();

        ArticulationBody[] allBodies = GetComponentsInChildren<ArticulationBody>();
        foreach (ArticulationBody body in allBodies)
        {
            jointTransforms.Add(body.transform);
            jointBodies.Add(body);
            jointNameToIndex[body.name] = jointBodies.Count - 1;
        }

        Debug.Log($"Initialized {jointBodies.Count} joints for robot {robotName}");
    }

    void StoreOriginalMaterials()
    {
        // Store original materials for highlighting
        originalMaterials = new Material[jointTransforms.Count];
        for (int i = 0; i < jointTransforms.Count; i++)
        {
            Renderer renderer = jointTransforms[i].GetComponent<Renderer>();
            if (renderer != null)
            {
                originalMaterials[i] = renderer.material;
            }
        }
    }

    public void SetJointPositions(Dictionary<string, float> jointPositions)
    {
        foreach (var kvp in jointPositions)
        {
            if (jointNameToIndex.ContainsKey(kvp.Key))
            {
                int index = jointNameToIndex[kvp.Key];
                if (index < jointBodies.Count)
                {
                    ArticulationDrive drive = jointBodies[index].jointDrive;
                    drive.target = kvp.Value;
                    jointBodies[index].jointDrive = drive;
                }
            }
        }
    }

    public void HighlightJoint(string jointName, Color highlightColor)
    {
        if (jointNameToIndex.ContainsKey(jointName))
        {
            int index = jointNameToIndex[jointName];
            if (index < jointTransforms.Count)
            {
                Renderer renderer = jointTransforms[index].GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material.color = highlightColor;
                    StartCoroutine(ResetMaterialAfterDelay(renderer, index, jointHighlightDuration));
                }
            }
        }
    }

    IEnumerator<WaitForSeconds> ResetMaterialAfterDelay(Renderer renderer, int index, float delay)
    {
        yield return new WaitForSeconds(delay);
        if (originalMaterials[index] != null)
        {
            renderer.material = originalMaterials[index];
        }
    }

    public void SetRobotColor(Color color)
    {
        foreach (Transform joint in jointTransforms)
        {
            Renderer renderer = joint.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = color;
            }
        }
    }

    void Update()
    {
        // Update robot state visualization
        UpdateRobotState();
    }

    void UpdateRobotState()
    {
        // This would be updated with real robot state data
        // For example, from ROS topics or direct robot connection
    }
}
```

## Unity Robotics Simulation Framework

### Setting up Unity for Robotics

Unity provides the Unity Robotics Simulation Framework to bridge the gap between Unity's game engine capabilities and robotics requirements:

```csharp
// Example: Unity Robotics Hub Integration
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using System.Collections.Generic;

public class UnityRoboticsBridge : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Robot Topics")]
    public string jointStateTopic = "/joint_states";
    public string cmdVelTopic = "/cmd_vel";
    public string imageTopic = "/camera/image_raw";

    private ROSConnection ros;
    private RobotVisualizer robotVisualizer;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.instance;
        ros.RegisterPublisher<Unity.Robotics.ROSTCPConnector.MessageGeneration.JointStateMsg>(cmdVelTopic);

        // Subscribe to joint states
        ros.Subscribe<Unity.Robotics.ROSTCPConnector.MessageGeneration.JointStateMsg>(
            jointStateTopic, JointStateCallback);

        robotVisualizer = GetComponent<RobotVisualizer>();
    }

    void JointStateCallback(Unity.Robotics.ROSTCPConnector.MessageGeneration.JointStateMsg jointState)
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
        }
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        var cmdVel = new Unity.Robotics.ROSTCPConnector.MessageGeneration.TwistMsg();
        cmdVel.linear = new Unity.Robotics.ROSTCPConnector.MessageGeneration.Vector3Msg();
        cmdVel.angular = new Unity.Robotics.ROSTCPConnector.MessageGeneration.Vector3Msg();

        cmdVel.linear.x = linearX;
        cmdVel.angular.z = angularZ;

        ros.Publish(cmdVelTopic, cmdVel);
    }
}
```

### Advanced Visualization Techniques

Unity's rendering pipeline allows for sophisticated visualization of robot data:

```csharp
// Advanced visualization script for sensor data
using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;

public class SensorDataVisualizer : MonoBehaviour
{
    [Header("LIDAR Visualization")]
    public GameObject lidarPointPrefab;
    public Color lidarPointColor = Color.red;
    public float lidarPointSize = 0.05f;

    [Header("Camera Feed")]
    public RawImage cameraFeedDisplay;
    public RenderTexture cameraTexture;

    [Header("IMU Visualization")]
    public GameObject imuOrientationIndicator;
    public LineRenderer imuPathRenderer;

    [Header("Path Planning")]
    public LineRenderer pathRenderer;
    public Color pathColor = Color.blue;

    private List<GameObject> lidarPoints = new List<GameObject>();
    private List<Vector3> imuPathHistory = new List<Vector3>();
    private List<Vector3> plannedPath = new List<Vector3>();

    void Start()
    {
        InitializeVisualization();
    }

    void InitializeVisualization()
    {
        // Initialize LIDAR point cloud visualization
        if (lidarPointPrefab == null)
        {
            CreateDefaultLidarPoint();
        }

        // Initialize path renderer
        if (pathRenderer != null)
        {
            pathRenderer.material = new Material(Shader.Find("Sprites/Default"));
            pathRenderer.startColor = pathColor;
            pathRenderer.endColor = pathColor;
            pathRenderer.startWidth = 0.05f;
            pathRenderer.endWidth = 0.05f;
        }

        // Initialize IMU path renderer
        if (imuPathRenderer != null)
        {
            imuPathRenderer.material = new Material(Shader.Find("Sprites/Default"));
            imuPathRenderer.startColor = Color.green;
            imuPathRenderer.endColor = Color.green;
            imuPathRenderer.startWidth = 0.02f;
            imuPathRenderer.endWidth = 0.02f;
        }
    }

    void CreateDefaultLidarPoint()
    {
        GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        point.name = "LidarPoint";
        point.GetComponent<Renderer>().material = new Material(Shader.Find("Sprites/Default"));
        point.GetComponent<Renderer>().material.color = lidarPointColor;
        point.GetComponent<SphereCollider>().enabled = false; // Remove collision
        point.SetActive(false);
        lidarPointPrefab = point;
    }

    public void UpdateLidarPointCloud(List<Vector3> points)
    {
        // Remove old points
        foreach (GameObject point in lidarPoints)
        {
            if (point != null)
            {
                DestroyImmediate(point);
            }
        }
        lidarPoints.Clear();

        // Create new points
        foreach (Vector3 point in points)
        {
            GameObject lidarPoint = Instantiate(lidarPointPrefab);
            lidarPoint.transform.position = point;
            lidarPoint.transform.localScale = Vector3.one * lidarPointSize;
            lidarPoint.SetActive(true);
            lidarPoints.Add(lidarPoint);
        }
    }

    public void UpdateCameraFeed(Texture2D cameraImage)
    {
        if (cameraFeedDisplay != null && cameraImage != null)
        {
            cameraFeedDisplay.texture = cameraImage;
        }
    }

    public void UpdateImuOrientation(Quaternion orientation)
    {
        if (imuOrientationIndicator != null)
        {
            imuOrientationIndicator.transform.rotation = orientation;
        }
    }

    public void AddImuPosition(Vector3 position)
    {
        imuPathHistory.Add(position);
        UpdateImuPath();
    }

    void UpdateImuPath()
    {
        if (imuPathRenderer != null && imuPathHistory.Count > 1)
        {
            imuPathRenderer.positionCount = imuPathHistory.Count;
            imuPathRenderer.SetPositions(imuPathHistory.ToArray());
        }
    }

    public void UpdatePlannedPath(List<Vector3> path)
    {
        plannedPath = new List<Vector3>(path);
        UpdatePathVisualization();
    }

    void UpdatePathVisualization()
    {
        if (pathRenderer != null && plannedPath.Count > 1)
        {
            pathRenderer.positionCount = plannedPath.Count;
            pathRenderer.SetPositions(plannedPath.ToArray());
        }
    }

    public void ClearAllVisualizations()
    {
        // Clear LIDAR points
        foreach (GameObject point in lidarPoints)
        {
            if (point != null)
            {
                DestroyImmediate(point);
            }
        }
        lidarPoints.Clear();

        // Clear paths
        if (pathRenderer != null)
        {
            pathRenderer.positionCount = 0;
        }

        if (imuPathRenderer != null)
        {
            imuPathRenderer.positionCount = 0;
        }

        imuPathHistory.Clear();
        plannedPath.Clear();
    }
}
```

## Creating Humanoid Robot Models in Unity

### Importing and Setting Up Robot Models

Creating humanoid robots in Unity involves proper setup of the 3D model and joint configuration:

```csharp
// Script for setting up humanoid robot model in Unity
using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(Animator))]
public class HumanoidRobotSetup : MonoBehaviour
{
    [Header("Humanoid Configuration")]
    public HumanBodyBones[] bodyParts = new HumanBodyBones[0];
    public List<ArticulationBody> articulatedParts = new List<ArticulationBody>();

    [Header("Joint Limits")]
    public bool enforceJointLimits = true;
    public float hipLimit = 45f;
    public float kneeLimit = 130f;
    public float ankleLimit = 30f;
    public float shoulderLimit = 90f;
    public float elbowLimit = 130f;

    [Header("Control Parameters")]
    public float positionDamping = 0.05f;
    public float velocityDamping = 0.01f;
    public float forceLimit = 100f;

    private Animator animator;
    private Dictionary<HumanBodyBones, Transform> boneTransforms = new Dictionary<HumanBodyBones, Transform>();

    void Start()
    {
        SetupHumanoidModel();
        ConfigureArticulationBodies();
    }

    void SetupHumanoidModel()
    {
        animator = GetComponent<Animator>();

        if (animator != null && animator.avatar != null && animator.avatar.isValid)
        {
            // Map humanoid bones to transforms
            foreach (HumanBodyBones bone in System.Enum.GetValues(typeof(HumanBodyBones)))
            {
                Transform boneTransform = animator.GetBoneTransform(bone);
                if (boneTransform != null)
                {
                    boneTransforms[bone] = boneTransform;
                    bodyParts = AddToArray(bodyParts, bone);
                }
            }
        }
        else
        {
            // Fallback: find transforms manually
            FindBodyPartsManually();
        }

        Debug.Log($"Set up {boneTransforms.Count} humanoid bones");
    }

    void FindBodyPartsManually()
    {
        // Common humanoid bone names
        string[] boneNames = {
            "Hips", "Spine", "Chest", "Neck", "Head",
            "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
            "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
            "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
            "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand"
        };

        foreach (string boneName in boneNames)
        {
            Transform boneTransform = FindTransformRecursive(transform, boneName);
            if (boneTransform != null)
            {
                // Map to closest HumanBodyBones enum value
                HumanBodyBones mappedBone = MapNameToBone(boneName);
                boneTransforms[mappedBone] = boneTransform;
            }
        }
    }

    Transform FindTransformRecursive(Transform parent, string name)
    {
        if (parent.name == name)
            return parent;

        for (int i = 0; i < parent.childCount; i++)
        {
            Transform result = FindTransformRecursive(parent.GetChild(i), name);
            if (result != null)
                return result;
        }
        return null;
    }

    HumanBodyBones MapNameToBone(string boneName)
    {
        switch (boneName.ToLower())
        {
            case "hips": return HumanBodyBones.Hips;
            case "spine": return HumanBodyBones.Spine;
            case "chest": return HumanBodyBones.Chest;
            case "neck": return HumanBodyBones.Neck;
            case "head": return HumanBodyBones.Head;
            case "leftupperleg": return HumanBodyBones.LeftUpperLeg;
            case "leftlowerleg": return HumanBodyBones.LeftLowerLeg;
            case "leftfoot": return HumanBodyBones.LeftFoot;
            case "lefttoes": return HumanBodyBones.LeftToes;
            case "rightupperleg": return HumanBodyBones.RightUpperLeg;
            case "rightlowerleg": return HumanBodyBones.RightLowerLeg;
            case "rightfoot": return HumanBodyBones.RightFoot;
            case "righttoes": return HumanBodyBones.RightToes;
            case "leftshoulder": return HumanBodyBones.LeftShoulder;
            case "leftupperarm": return HumanBodyBones.LeftUpperArm;
            case "leftlowerarm": return HumanBodyBones.LeftLowerArm;
            case "lefthand": return HumanBodyBones.LeftHand;
            case "rightshoulder": return HumanBodyBones.RightShoulder;
            case "rightupperarm": return HumanBodyBones.RightUpperArm;
            case "rightlowerarm": return HumanBodyBones.RightLowerArm;
            case "righthand": return HumanBodyBones.RightHand;
            default: return HumanBodyBones.LastBone;
        }
    }

    void ConfigureArticulationBodies()
    {
        // Configure each articulated part with appropriate joint settings
        articulatedParts.Clear();

        foreach (var kvp in boneTransforms)
        {
            ArticulationBody body = kvp.Value.GetComponent<ArticulationBody>();
            if (body != null)
            {
                ConfigureArticulationBody(body, kvp.Key);
                articulatedParts.Add(body);
            }
        }
    }

    void ConfigureArticulationBody(ArticulationBody body, HumanBodyBones bone)
    {
        // Set up joint drive for smooth control
        ArticulationDrive drive = body.jointDrive;
        drive.positionSpring = 10000f; // Stiffness
        drive.positionDamper = positionDamping * 1000f; // Damping
        drive.forceLimit = forceLimit;
        body.jointDrive = drive;

        // Configure joint limits based on bone type
        ConfigureJointLimits(body, bone);

        // Set physical properties
        body.angularDamping = velocityDamping * 10f;
        body.linearDamping = velocityDamping;
    }

    void ConfigureJointLimits(ArticulationBody body, HumanBodyBones bone)
    {
        if (!enforceJointLimits) return;

        ArticulationJoint joint = body.GetComponent<ArticulationJoint>();
        if (joint == null) return;

        switch (bone)
        {
            case HumanBodyBones.LeftUpperLeg:
            case HumanBodyBones.RightUpperLeg:
                // Hip joint limits
                joint.linearLockX = ArticulationDofLock.Locked;
                joint.linearLockY = ArticulationDofLock.Locked;
                joint.linearLockZ = ArticulationDofLock.Locked;
                joint.angularLockX = ArticulationDofLock.LimitedMotion;
                joint.angularLockY = ArticulationDofLock.LimitedMotion;
                joint.angularLockZ = ArticulationDofLock.LimitedMotion;

                ArticulationLimit hipLimitX = new ArticulationLimit();
                hipLimitX.min = -hipLimit;
                hipLimitX.max = hipLimit;
                joint.xMotion = ArticulationDofLock.LimitedMotion;
                joint.lowAngularXLimit = hipLimitX;
                joint.highAngularXLimit = hipLimitX;
                break;

            case HumanBodyBones.LeftLowerLeg:
            case HumanBodyBones.RightLowerLeg:
                // Knee joint limits (only flexion)
                joint.linearLockX = ArticulationDofLock.Locked;
                joint.linearLockY = ArticulationDofLock.Locked;
                joint.linearLockZ = ArticulationDofLock.Locked;
                joint.angularLockX = ArticulationDofLock.LimitedMotion;
                joint.angularLockY = ArticulationDofLock.Locked;
                joint.angularLockZ = ArticulationDofLock.Locked;

                ArticulationLimit kneeLimit = new ArticulationLimit();
                kneeLimit.min = 0; // Only forward flexion
                kneeLimit.max = kneeLimit; // Maximum flexion
                joint.lowAngularXLimit = kneeLimit;
                joint.highAngularXLimit = kneeLimit;
                break;

            case HumanBodyBones.LeftFoot:
            case HumanBodyBones.RightFoot:
                // Ankle joint limits
                joint.linearLockX = ArticulationDofLock.Locked;
                joint.linearLockY = ArticulationDofLock.Locked;
                joint.linearLockZ = ArticulationDofLock.Locked;
                joint.angularLockX = ArticulationDofLock.LimitedMotion;
                joint.angularLockY = ArticulationDofLock.LimitedMotion;
                joint.angularLockZ = ArticulationDofLock.LimitedMotion;

                ArticulationLimit ankleLimit = new ArticulationLimit();
                ankleLimit.min = -ankleLimit;
                ankleLimit.max = ankleLimit;
                joint.lowAngularXLimit = ankleLimit;
                joint.highAngularXLimit = ankleLimit;
                break;
        }
    }

    T[] AddToArray<T>(T[] array, T element)
    {
        T[] newArray = new T[array.Length + 1];
        for (int i = 0; i < array.Length; i++)
        {
            newArray[i] = array[i];
        }
        newArray[array.Length] = element;
        return newArray;
    }

    public void SetJointPosition(HumanBodyBones bone, float position)
    {
        if (boneTransforms.ContainsKey(bone))
        {
            ArticulationBody body = boneTransforms[bone].GetComponent<ArticulationBody>();
            if (body != null)
            {
                ArticulationDrive drive = body.jointDrive;
                drive.target = position;
                body.jointDrive = drive;
            }
        }
    }

    public void SetPose(Dictionary<HumanBodyBones, float> pose)
    {
        foreach (var kvp in pose)
        {
            SetJointPosition(kvp.Key, kvp.Value);
        }
    }
}
```

## Integration with ROS and External Systems

### ROS Integration in Unity

Unity can integrate with ROS systems for real-time data exchange:

```csharp
// ROS integration manager for Unity
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using System.Collections.Generic;
using System;

public class UnityROSIntegration : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    public float updateRate = 30f; // Hz

    [Header("Robot Control")]
    public string robotNamespace = "/humanoid_robot";
    public string jointStatesTopic = "joint_states";
    public string cmdVelTopic = "cmd_vel";
    public string imageTopic = "camera/image_raw";

    [Header("Visualization")]
    public RobotVisualizer robotVisualizer;
    public SensorDataVisualizer sensorVisualizer;

    private ROSConnection ros;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        InitializeROSConnection();
        updateInterval = 1f / updateRate;
        lastUpdateTime = 0f;
    }

    void InitializeROSConnection()
    {
        ros = ROSConnection.instance;
        if (ros == null)
        {
            GameObject rosGO = new GameObject("ROSConnection");
            ros = rosGO.AddComponent<ROSConnection>();
        }

        // Set ROS connection parameters
        ros.HostAddress = rosIPAddress;
        ros.HostPort = rosPort;

        // Subscribe to topics
        string fullJointStatesTopic = robotNamespace + "/" + jointStatesTopic;
        string fullCmdVelTopic = robotNamespace + "/" + cmdVelTopic;
        string fullImageTopic = robotNamespace + "/" + imageTopic;

        ros.Subscribe<JointStateMsg>(fullJointStatesTopic, JointStateCallback);
        ros.Subscribe<ImageMsg>(fullImageTopic, ImageCallback);

        Debug.Log($"ROS integration initialized for robot: {robotNamespace}");
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
        }
    }

    void ImageCallback(ImageMsg imageMsg)
    {
        if (sensorVisualizer != null)
        {
            // Convert ROS image to Unity texture
            Texture2D texture = ConvertRosImageToTexture(imageMsg);
            sensorVisualizer.UpdateCameraFeed(texture);
        }
    }

    Texture2D ConvertRosImageToTexture(ImageMsg imageMsg)
    {
        // Convert ROS image message to Unity Texture2D
        // This is a simplified conversion - in practice, you'd handle different encodings
        int width = (int)imageMsg.width;
        int height = (int)imageMsg.height;

        Texture2D texture = new Texture2D(width, height);

        // Handle different image encodings
        if (imageMsg.encoding == "rgb8" || imageMsg.encoding == "bgr8")
        {
            // Convert byte array to color array
            Color32[] colors = new Color32[width * height];

            for (int i = 0; i < colors.Length; i++)
            {
                if (i * 3 + 2 < imageMsg.data.Count)
                {
                    byte r = imageMsg.data[i * 3];
                    byte g = imageMsg.data[i * 3 + 1];
                    byte b = imageMsg.data[i * 3 + 2];
                    byte a = 255; // Alpha

                    // For BGR format, swap R and B
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

        return texture;
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        if (ros != null)
        {
            string fullCmdVelTopic = robotNamespace + "/" + cmdVelTopic;

            var cmdVel = new TwistMsg();
            cmdVel.linear = new Vector3Msg();
            cmdVel.angular = new Vector3Msg();

            cmdVel.linear.x = linearX;
            cmdVel.angular.z = angularZ;

            ros.Publish(fullCmdVelTopic, cmdVel);
        }
    }

    public void SendJointCommands(Dictionary<string, float> jointCommands)
    {
        if (ros != null)
        {
            // This would publish to joint trajectory topics
            // Implementation depends on specific ROS control setup
        }
    }

    void Update()
    {
        // Throttle updates to specified rate
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            UpdateRobotVisualization();
            lastUpdateTime = Time.time;
        }
    }

    void UpdateRobotVisualization()
    {
        // Update visualization based on current robot state
        // This could include updating sensor visualizations, paths, etc.
    }

    void OnDestroy()
    {
        if (ros != null)
        {
            ros.OnApplicationQuit();
        }
    }
}
```

## Advanced Visualization Features

### Creating Immersive VR/AR Experiences

Unity's capabilities extend to creating immersive experiences for robot teleoperation and monitoring:

```csharp
// VR/AR integration for robot teleoperation
using UnityEngine;
using UnityEngine.XR;
using System.Collections.Generic;

public class RobotTeleoperationVR : MonoBehaviour
{
    [Header("VR Configuration")]
    public bool enableVR = true;
    public Transform vrCameraRig;
    public GameObject leftController;
    public GameObject rightController;

    [Header("Robot Control")]
    public UnityROSIntegration rosIntegration;
    public HumanoidRobotSetup robotSetup;

    [Header("UI Elements")]
    public GameObject robotStatusPanel;
    public GameObject sensorDataPanel;
    public GameObject controlPanel;

    [Header("Interaction")]
    public float movementSpeed = 1.0f;
    public float rotationSpeed = 1.0f;
    public float joystickDeadzone = 0.2f;

    private bool vrEnabled = false;
    private Vector2 leftJoystickInput;
    private Vector2 rightJoystickInput;

    void Start()
    {
        InitializeVR();
        SetupUI();
    }

    void InitializeVR()
    {
        if (enableVR && XRSettings.enabled)
        {
            vrEnabled = true;
            ConfigureXRSettings();
            Debug.Log("VR mode enabled for robot teleoperation");
        }
        else
        {
            vrEnabled = false;
            Debug.Log("VR mode disabled, using desktop interface");
        }
    }

    void ConfigureXRSettings()
    {
        // Configure XR settings for optimal robot teleoperation
        XRSettings.showDeviceView = true;

        // Set up controller mappings
        SetupControllerInput();
    }

    void SetupControllerInput()
    {
        // Map controller inputs to robot commands
        // This would typically use Unity's Input System
    }

    void SetupUI()
    {
        // Position UI elements appropriately for VR
        if (vrCameraRig != null)
        {
            PositionUIRelativeToCamera();
        }
    }

    void PositionUIRelativeToCamera()
    {
        if (robotStatusPanel != null)
        {
            robotStatusPanel.transform.position = vrCameraRig.position + vrCameraRig.forward * 2f;
            robotStatusPanel.transform.rotation = Quaternion.LookRotation(
                vrCameraRig.position - robotStatusPanel.transform.position,
                vrCameraRig.up
            );
        }

        if (sensorDataPanel != null)
        {
            sensorDataPanel.transform.position = vrCameraRig.position + vrCameraRig.right * -1.5f + vrCameraRig.up * 0.5f;
            sensorDataPanel.transform.rotation = Quaternion.LookRotation(
                vrCameraRig.position - sensorDataPanel.transform.position,
                vrCameraRig.up
            );
        }

        if (controlPanel != null)
        {
            controlPanel.transform.position = vrCameraRig.position + vrCameraRig.right * 1.5f + vrCameraRig.up * 0.5f;
            controlPanel.transform.rotation = Quaternion.LookRotation(
                vrCameraRig.position - controlPanel.transform.position,
                vrCameraRig.up
            );
        }
    }

    void Update()
    {
        if (vrEnabled)
        {
            UpdateVRInput();
            UpdateRobotControl();
        }

        UpdateUI();
    }

    void UpdateVRInput()
    {
        // Get input from VR controllers
        // This is a simplified example - in practice, use Unity's XR input system
        leftJoystickInput = GetControllerAxis("Left", "Joystick");
        rightJoystickInput = GetControllerAxis("Right", "Joystick");
    }

    Vector2 GetControllerAxis(string controller, string axis)
    {
        // Simplified input handling
        // In practice, use Unity's XR input system
        return new Vector2(0, 0);
    }

    void UpdateRobotControl()
    {
        // Convert VR input to robot commands
        float linearVel = 0f;
        float angularVel = 0f;

        // Forward/backward movement
        if (Mathf.Abs(leftJoystickInput.y) > joystickDeadzone)
        {
            linearVel = leftJoystickInput.y * movementSpeed;
        }

        // Left/right rotation
        if (Mathf.Abs(rightJoystickInput.x) > joystickDeadzone)
        {
            angularVel = rightJoystickInput.x * rotationSpeed;
        }

        // Send commands to robot
        if (rosIntegration != null)
        {
            rosIntegration.SendVelocityCommand(linearVel, angularVel);
        }

        // Handle other robot controls based on controller input
        HandleRobotArmControl();
        HandleRobotHeadControl();
    }

    void HandleRobotArmControl()
    {
        // Map controller input to robot arm movements
        if (rightJoystickInput.magnitude > joystickDeadzone && robotSetup != null)
        {
            // Example: Move robot arm based on right controller position
            Vector3 controllerPos = rightController.transform.position;
            // Convert to robot coordinate space and send commands
        }
    }

    void HandleRobotHeadControl()
    {
        // Control robot head orientation based on HMD orientation
        if (vrCameraRig != null && robotSetup != null)
        {
            // Extract head orientation and send to robot
            Quaternion headOrientation = vrCameraRig.rotation;
            // Convert to robot head control commands
        }
    }

    void UpdateUI()
    {
        // Update UI with real-time robot data
        UpdateRobotStatusUI();
        UpdateSensorDataUI();
    }

    void UpdateRobotStatusUI()
    {
        // Update robot status panel with current information
        if (robotStatusPanel != null)
        {
            // This would update with actual robot status data
        }
    }

    void UpdateSensorDataUI()
    {
        // Update sensor data panel with current sensor information
        if (sensorDataPanel != null)
        {
            // This would update with actual sensor data
        }
    }

    public void ToggleVRMode()
    {
        enableVR = !enableVR;
        InitializeVR();
    }

    public void SetRobotFocus(bool focus)
    {
        // Highlight robot when in focus
        if (robotSetup != null)
        {
            robotSetup.GetComponent<RobotVisualizer>().SetRobotColor(
                focus ? Color.yellow : Color.white
            );
        }
    }
}
```

## Performance Optimization

### Optimizing Unity Visualizations

Performance is crucial when visualizing complex humanoid robots with multiple sensors:

```csharp
// Performance optimization manager
using UnityEngine;
using System.Collections.Generic;

public class VisualizationOptimizer : MonoBehaviour
{
    [Header("LOD Configuration")]
    public int maxVisibleRobots = 10;
    public float lodDistance = 10f;
    public int maxSensorPoints = 1000;

    [Header("Quality Settings")]
    public int targetFrameRate = 60;
    public bool enableOcclusionCulling = true;
    public bool enableDynamicBatching = true;

    [Header("Visualization Settings")]
    public bool enableLidarVisualization = true;
    public bool enableCameraVisualization = true;
    public float visualizationUpdateRate = 15f; // Lower rate for better performance

    private List<Renderer> robotRenderers = new List<Renderer>();
    private List<GameObject> sensorVisualizations = new List<GameObject>();
    private float lastVisualizationUpdate = 0f;
    private float visualizationInterval;

    void Start()
    {
        ConfigureQualitySettings();
        visualizationInterval = 1f / visualizationUpdateRate;
        FindRobotRenderers();
    }

    void ConfigureQualitySettings()
    {
        Application.targetFrameRate = targetFrameRate;

        if (enableOcclusionCulling)
        {
            Camera.main.layerCullDistances = new float[32]; // Enable occlusion culling
        }

        // Enable batching
        // Dynamic batching is enabled by default if quality settings allow
        // Static batching needs to be configured in build settings
    }

    void FindRobotRenderers()
    {
        // Find all robot renderers for LOD management
        Renderer[] allRenderers = FindObjectsOfType<Renderer>();

        foreach (Renderer renderer in allRenderers)
        {
            if (renderer.CompareTag("RobotPart") ||
                renderer.name.Contains("Robot") ||
                renderer.name.Contains("Joint"))
            {
                robotRenderers.Add(renderer);
            }
        }
    }

    void Update()
    {
        ManageLOD();
        ThrottleVisualizationUpdates();
    }

    void ManageLOD()
    {
        // Implement Level of Detail for distant robots
        foreach (Renderer robotRenderer in robotRenderers)
        {
            float distance = Vector3.Distance(Camera.main.transform.position, robotRenderer.transform.position);

            if (distance > lodDistance)
            {
                // Reduce detail for distant robots
                robotRenderer.enabled = false;
            }
            else
            {
                robotRenderer.enabled = true;
            }
        }
    }

    void ThrottleVisualizationUpdates()
    {
        if (Time.time - lastVisualizationUpdate >= visualizationInterval)
        {
            UpdateSensorVisualizations();
            lastVisualizationUpdate = Time.time;
        }
    }

    void UpdateSensorVisualizations()
    {
        // Update sensor visualizations at reduced rate
        if (!enableLidarVisualization && !enableCameraVisualization)
        {
            return;
        }

        // Limit number of sensor points for performance
        LimitSensorVisualizationPoints();
    }

    void LimitSensorVisualizationPoints()
    {
        // For LIDAR visualizations, limit number of points
        if (sensorVisualizations.Count > maxSensorPoints)
        {
            // Remove oldest sensor visualization objects
            int excess = sensorVisualizations.Count - maxSensorPoints;
            for (int i = 0; i < excess; i++)
            {
                if (sensorVisualizations.Count > 0)
                {
                    GameObject oldPoint = sensorVisualizations[0];
                    sensorVisualizations.RemoveAt(0);
                    if (oldPoint != null)
                    {
                        DestroyImmediate(oldPoint);
                    }
                }
            }
        }
    }

    public void SetVisualizationQuality(int qualityLevel)
    {
        // Adjust visualization quality based on performance needs
        switch (qualityLevel)
        {
            case 0: // Low
                enableLidarVisualization = false;
                visualizationUpdateRate = 5f;
                break;
            case 1: // Medium
                enableLidarVisualization = true;
                visualizationUpdateRate = 15f;
                break;
            case 2: // High
                enableLidarVisualization = true;
                visualizationUpdateRate = 30f;
                break;
        }

        visualizationInterval = 1f / visualizationUpdateRate;
    }

    public void OptimizeForMobile()
    {
        // Apply optimizations for mobile devices
        SetVisualizationQuality(0); // Low quality
        Application.targetFrameRate = 30; // Lower target frame rate
        QualitySettings.SetQualityLevel(0); // Lowest quality preset
    }

    public void OptimizeForDesktop()
    {
        // Apply optimizations for desktop systems
        SetVisualizationQuality(2); // High quality
        Application.targetFrameRate = 60; // Higher target frame rate
        QualitySettings.SetQualityLevel(4); // Higher quality preset
    }
}
```

Unity provides powerful visualization capabilities that complement traditional robotics simulation tools like Gazebo. Its real-time rendering, VR/AR support, and user interface capabilities make it an excellent choice for creating immersive and interactive visualization experiences for humanoid robots. When combined with ROS integration, Unity becomes a comprehensive platform for robot monitoring, teleoperation, and human-robot interaction development.