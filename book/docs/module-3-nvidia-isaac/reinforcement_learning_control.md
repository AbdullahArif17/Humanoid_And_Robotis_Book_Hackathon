# Reinforcement Learning for Control in Humanoid Robotics

## Overview

Reinforcement Learning (RL) has emerged as a powerful paradigm for developing adaptive control strategies for humanoid robots. Unlike traditional control methods that rely on predefined models and controllers, RL enables robots to learn complex behaviors through interaction with their environment. For humanoid robots, which must navigate the complexities of bipedal locomotion and dynamic environments, RL offers the potential for more robust and adaptive control policies.

This chapter explores the application of reinforcement learning techniques to humanoid robot control, covering fundamental concepts, specialized algorithms, and practical implementations for various humanoid tasks.

## Fundamentals of Reinforcement Learning for Robotics

Reinforcement learning in robotics involves an agent (the robot) that learns to perform tasks by receiving rewards or penalties based on its actions. The key components are:

- **State (s)**: The current configuration of the robot and environment
- **Action (a)**: The control command executed by the robot
- **Reward (r)**: Feedback signal indicating the quality of the action
- **Policy (π)**: Strategy that maps states to actions
- **Value function (V)**: Expected cumulative reward from a given state

```python
# Reinforcement Learning Environment for Humanoid Robot
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any
import gym
from gym import spaces
import random
from collections import deque

class HumanoidRLEnvironment:
    """RL environment for humanoid robot control"""

    def __init__(self, robot_config: Dict[str, Any]):
        self.robot_config = robot_config
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(robot_config['num_joints'],), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(robot_config['observation_dim'],),
            dtype=np.float32
        )

        # Robot state
        self.current_state = None
        self.target_pose = None
        self.episode_step = 0
        self.max_episode_steps = 1000

        # Reward weights
        self.reward_weights = {
            'forward_progress': 1.0,
            'stability': 0.8,
            'energy_efficiency': 0.2,
            'balance': 1.0,
            'survival': 0.1
        }

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        # Initialize robot in standing position
        self.current_state = self.initialize_robot_state()
        self.target_pose = self.generate_random_target()
        self.episode_step = 0

        return self.get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        # Apply action to robot
        next_state = self.apply_action(action)

        # Calculate reward
        reward = self.calculate_reward(next_state, action)

        # Check if episode is done
        done = self.is_episode_done(next_state)

        # Update state
        self.current_state = next_state
        self.episode_step += 1

        # Get additional info
        info = self.get_info()

        return self.get_observation(), reward, done, info

    def initialize_robot_state(self) -> Dict:
        """Initialize robot to standing position"""
        return {
            'joint_positions': np.zeros(self.robot_config['num_joints']),
            'joint_velocities': np.zeros(self.robot_config['num_joints']),
            'base_position': np.array([0.0, 0.0, self.robot_config['base_height']]),
            'base_orientation': np.array([0.0, 0.0, 0.0, 1.0]),  # quaternion
            'linear_velocity': np.array([0.0, 0.0, 0.0]),
            'angular_velocity': np.array([0.0, 0.0, 0.0]),
            'center_of_mass': np.array([0.0, 0.0, self.robot_config['com_height']])
        }

    def apply_action(self, action: np.ndarray) -> Dict:
        """Apply action to robot and return new state"""
        # This would interface with the actual robot simulation
        # For this example, we'll simulate the effect
        new_state = self.current_state.copy()

        # Apply joint torques based on action
        torque = action * self.robot_config['max_torque']

        # Simulate physics (simplified)
        new_state['joint_positions'] += new_state['joint_velocities'] * 0.01  # dt
        new_state['joint_velocities'] += torque / self.robot_config['joint_inertia'] * 0.01

        # Update base pose based on joint movements
        self.update_base_pose(new_state)

        # Update center of mass
        self.update_center_of_mass(new_state)

        return new_state

    def update_base_pose(self, state: Dict):
        """Update base pose based on joint movements"""
        # Simplified base pose update
        # In reality, this would use forward kinematics
        pass

    def update_center_of_mass(self, state: Dict):
        """Update center of mass based on joint positions"""
        # Simplified COM calculation
        # In reality, this would use the robot's kinematic model
        pass

    def calculate_reward(self, state: Dict, action: np.ndarray) -> float:
        """Calculate reward based on current state and action"""
        reward = 0.0

        # Forward progress reward
        if self.target_pose is not None:
            forward_reward = self.calculate_forward_progress_reward(state)
            reward += self.reward_weights['forward_progress'] * forward_reward

        # Stability reward
        stability_reward = self.calculate_stability_reward(state)
        reward += self.reward_weights['stability'] * stability_reward

        # Balance reward
        balance_reward = self.calculate_balance_reward(state)
        reward += self.reward_weights['balance'] * balance_reward

        # Energy efficiency reward (penalize large actions)
        energy_reward = -np.sum(np.abs(action)) * self.reward_weights['energy_efficiency']
        reward += energy_reward

        # Survival bonus
        if not self.is_fallen(state):
            reward += self.reward_weights['survival']

        return reward

    def calculate_forward_progress_reward(self, state: Dict) -> float:
        """Calculate reward for forward progress toward target"""
        if self.target_pose is None:
            return 0.0

        current_pos = state['base_position'][:2]  # x, y
        target_pos = self.target_pose[:2]

        # Calculate distance to target
        current_dist = np.linalg.norm(target_pos - current_pos)

        # Reward based on reduction in distance
        return max(0.0, 1.0 - current_dist / self.robot_config['max_distance'])

    def calculate_stability_reward(self, state: Dict) -> float:
        """Calculate reward for stability"""
        # Check if robot is stable based on COM position
        com = state['center_of_mass']
        base_pos = state['base_position']

        # Distance from COM to base (simplified stability measure)
        stability = 1.0 - min(1.0, np.abs(com[0] - base_pos[0]) / 0.1)  # 10cm threshold
        stability *= 1.0 - min(1.0, np.abs(com[1] - base_pos[1]) / 0.1)

        return stability

    def calculate_balance_reward(self, state: Dict) -> float:
        """Calculate reward for balance"""
        # Check orientation (should be upright)
        orientation = state['base_orientation']

        # Convert quaternion to Euler angles (simplified)
        # In reality, would use proper quaternion to Euler conversion
        roll = np.arctan2(2 * (orientation[3] * orientation[0] + orientation[1] * orientation[2]),
                         1 - 2 * (orientation[0]**2 + orientation[1]**2))
        pitch = np.arcsin(2 * (orientation[3] * orientation[1] - orientation[2] * orientation[0]))

        # Penalize deviation from upright position
        balance_penalty = abs(roll) + abs(pitch)
        balance_reward = max(0.0, 1.0 - balance_penalty)

        return balance_reward

    def is_fallen(self, state: Dict) -> bool:
        """Check if robot has fallen"""
        # Check if base height is too low (robot has fallen)
        return state['base_position'][2] < self.robot_config['base_height'] * 0.5

    def is_episode_done(self, state: Dict) -> bool:
        """Check if episode is done"""
        return (self.episode_step >= self.max_episode_steps or
                self.is_fallen(state))

    def get_observation(self) -> np.ndarray:
        """Get current observation from robot state"""
        obs = []

        # Joint positions and velocities
        obs.extend(self.current_state['joint_positions'])
        obs.extend(self.current_state['joint_velocities'])

        # Base pose and velocity
        obs.extend(self.current_state['base_position'])
        obs.extend(self.current_state['base_orientation'])
        obs.extend(self.current_state['linear_velocity'])
        obs.extend(self.current_state['angular_velocity'])

        # Center of mass
        obs.extend(self.current_state['center_of_mass'])

        # Target position (relative to robot)
        if self.target_pose is not None:
            target_rel = self.target_pose - self.current_state['base_position']
            obs.extend(target_rel)

        return np.array(obs, dtype=np.float32)

    def get_info(self) -> Dict:
        """Get additional information"""
        return {
            'episode_step': self.episode_step,
            'is_fallen': self.is_fallen(self.current_state),
            'base_height': self.current_state['base_position'][2]
        }

    def generate_random_target(self) -> np.ndarray:
        """Generate random target position"""
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(1.0, 5.0)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        return np.array([x, y, 0.0])
```

## Deep Reinforcement Learning Algorithms for Humanoid Control

### Deep Deterministic Policy Gradient (DDPG)

DDPG is well-suited for continuous control tasks like humanoid robotics:

```python
# Deep Deterministic Policy Gradient for Humanoid Control
class ActorNetwork(nn.Module):
    """Actor network for DDPG"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(ActorNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class CriticNetwork(nn.Module):
    """Critic network for DDPG"""

    def __init__(self, state_dim: int, action_dim: int):
        super(CriticNetwork, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1

class DDPGAgent:
    """DDPG agent for humanoid robot control"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Hyperparameters
        self.gamma = gamma  # discount factor
        self.tau = tau      # soft update parameter
        self.max_action = max_action

        # Replay buffer
        self.replay_buffer = deque(maxlen=1000000)
        self.batch_size = 100

    def select_action(self, state: np.ndarray, add_noise: bool = True,
                     noise_scale: float = 0.1) -> np.ndarray:
        """Select action using the current policy"""
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise

        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size: int = 100) -> Dict[str, float]:
        """Train the DDPG agent"""
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))

        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        done = torch.BoolTensor(done).to(self.device).unsqueeze(1)

        # Compute target Q-value
        next_action = self.actor_target(next_state)
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (self.gamma * target_Q * (1 - done.float()))

        # Get current Q-value estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        next_state: np.ndarray, reward: float, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, next_state, reward, done))
```

### Twin Delayed DDPG (TD3)

TD3 addresses some of DDPG's limitations:

```python
# Twin Delayed DDPG for more stable training
class TD3Agent:
    """TD3 agent for humanoid robot control"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005,
                 policy_noise: float = 0.2, noise_clip: float = 0.5,
                 policy_freq: int = 2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action

        # Training variables
        self.total_it = 0

    def select_action(self, state: np.ndarray, add_noise: bool = True,
                     noise_scale: float = 0.1) -> np.ndarray:
        """Select action using the current policy"""
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise

        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size: int = 100) -> Dict[str, float]:
        """Train the TD3 agent"""
        if len(self.replay_buffer) < batch_size:
            return {}

        for _ in range(batch_size):
            # Sample replay buffer
            batch = random.sample(self.replay_buffer, batch_size)
            state, action, next_state, reward, done = map(np.stack, zip(*batch))

            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
            done = torch.BoolTensor(done).to(self.device).unsqueeze(1)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(action).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (self.gamma * target_Q * (1 - done.float()))

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            self.total_it += 1
            if self.total_it % self.policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if self.total_it % self.policy_freq == 0 else 0
        }

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        next_state: np.ndarray, reward: float, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, next_state, reward, done))
```

## Humanoid-Specific RL Challenges

### Stability and Safety

Humanoid robots require stable policies during training:

```python
# Stability-aware RL training
class StabilityConstrainedRL:
    """RL with stability constraints for humanoid robots"""

    def __init__(self, base_agent, safety_threshold: float = 0.1):
        self.base_agent = base_agent
        self.safety_threshold = safety_threshold
        self.safety_buffer = deque(maxlen=100)  # Track recent stability measures

    def safe_action_selection(self, state: np.ndarray,
                            stability_measure: float) -> np.ndarray:
        """Select action with stability constraints"""
        if stability_measure < self.safety_threshold:
            # Robot is unstable, select conservative action
            return self.get_stabilizing_action(state)

        # Robot is stable, use normal policy
        return self.base_agent.select_action(state)

    def get_stabilizing_action(self, state: np.ndarray) -> np.ndarray:
        """Get action that prioritizes stabilization"""
        # This would implement a stabilizing controller
        # For example, move joints toward neutral positions
        neutral_pos = np.zeros(len(state))  # Simplified
        current_pos = state[:len(neutral_pos)]  # Extract joint positions

        # Return action that moves toward neutral position
        return -0.1 * current_pos  # Simplified stabilizing action

    def update_safety_buffer(self, stability_measure: float):
        """Update safety buffer with current stability measure"""
        self.safety_buffer.append(stability_measure)

    def get_average_stability(self) -> float:
        """Get average stability over recent steps"""
        if not self.safety_buffer:
            return 1.0  # Assume stable if no data
        return sum(self.safety_buffer) / len(self.safety_buffer)
```

### Curriculum Learning

Training humanoid robots effectively requires gradual complexity increase:

```python
# Curriculum learning for humanoid control
class CurriculumRL:
    """Curriculum learning for humanoid robot control"""

    def __init__(self, env_config: Dict):
        self.env_config = env_config
        self.current_level = 0
        self.level_thresholds = [0.5, 0.7, 0.85, 1.0]  # Performance thresholds
        self.curriculum_levels = [
            'basic_balance',      # Level 0: Basic balance
            'simple_locomotion',  # Level 1: Simple walking
            'obstacle_avoidance', # Level 2: Obstacle avoidance
            'complex_tasks'       # Level 3: Complex tasks
        ]

    def update_curriculum(self, performance: float) -> bool:
        """Update curriculum level based on performance"""
        if (self.current_level < len(self.level_thresholds) - 1 and
            performance >= self.level_thresholds[self.current_level]):
            self.current_level += 1
            self.update_environment()
            return True
        return False

    def update_environment(self):
        """Update environment based on current curriculum level"""
        level = self.curriculum_levels[self.current_level]

        if level == 'basic_balance':
            # Simple balance task
            self.env_config['target_distance'] = 1.0
            self.env_config['disturbance_frequency'] = 0.1
        elif level == 'simple_locomotion':
            # Add forward motion requirement
            self.env_config['target_distance'] = 3.0
            self.env_config['disturbance_frequency'] = 0.2
        elif level == 'obstacle_avoidance':
            # Add obstacles
            self.env_config['target_distance'] = 5.0
            self.env_config['disturbance_frequency'] = 0.3
            self.env_config['num_obstacles'] = 3
        elif level == 'complex_tasks':
            # Complex multi-goal tasks
            self.env_config['target_distance'] = 10.0
            self.env_config['disturbance_frequency'] = 0.4
            self.env_config['num_obstacles'] = 5
            self.env_config['dynamic_obstacles'] = True

    def get_current_task(self) -> str:
        """Get current curriculum task"""
        return self.curriculum_levels[self.current_level]

    def reset_to_level(self, level: int):
        """Reset to specific curriculum level"""
        if 0 <= level < len(self.curriculum_levels):
            self.current_level = level
            self.update_environment()
```

## Advanced RL Techniques for Humanoid Robotics

### Hierarchical RL

Complex humanoid behaviors can be decomposed using hierarchical structures:

```python
# Hierarchical RL for complex humanoid behaviors
class HierarchicalRLAgent:
    """Hierarchical RL agent for humanoid robots"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        # High-level policy (option policy)
        self.option_policy = ActorNetwork(state_dim, 3, 1.0)  # 3 options: walk, turn, balance

        # Low-level policies for each option
        self.low_level_policies = {
            'walk': ActorNetwork(state_dim, action_dim, max_action),
            'turn': ActorNetwork(state_dim, action_dim, max_action),
            'balance': ActorNetwork(state_dim, action_dim, max_action)
        }

        # Option termination conditions
        self.termination_functions = {
            'walk': self.is_walk_complete,
            'turn': self.is_turn_complete,
            'balance': self.is_balance_complete
        }

        self.current_option = None
        self.option_steps = 0
        self.max_option_steps = 50  # Max steps per option

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using hierarchical policy"""
        # Check if current option should terminate
        if (self.current_option is not None and
            (self.termination_functions[self.current_option](state) or
             self.option_steps >= self.max_option_steps)):
            self.current_option = None

        # If no active option, select new one
        if self.current_option is None:
            self.current_option = self.select_option(state)
            self.option_steps = 0

        # Execute low-level policy for current option
        self.option_steps += 1
        return self.low_level_policies[self.current_option](torch.FloatTensor(state).unsqueeze(0)).detach().numpy()

    def select_option(self, state: np.ndarray) -> str:
        """Select high-level option"""
        option_probs = self.option_policy(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]
        options = ['walk', 'turn', 'balance']
        return options[np.argmax(option_probs)]

    def is_walk_complete(self, state: np.ndarray) -> bool:
        """Check if walk option is complete"""
        # Simplified: walk for a certain distance or time
        return False

    def is_turn_complete(self, state: np.ndarray) -> bool:
        """Check if turn option is complete"""
        # Simplified: turn for a certain angle
        return False

    def is_balance_complete(self, state: np.ndarray) -> bool:
        """Check if balance option is complete"""
        # Simplified: robot is stable
        return True
```

### Multi-Task Learning

Learning multiple humanoid skills simultaneously:

```python
# Multi-task RL for humanoid robots
class MultiTaskRLAgent:
    """Multi-task RL agent for humanoid robots"""

    def __init__(self, state_dim: int, action_dim: int, num_tasks: int):
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            ) for _ in range(num_tasks)
        ])

        # Task selector network
        self.task_selector = nn.Linear(256, num_tasks)

        self.num_tasks = num_tasks
        self.current_task = 0

    def select_action(self, state: np.ndarray, task_id: int = None) -> np.ndarray:
        """Select action for specific task"""
        if task_id is None:
            task_id = self.current_task

        features = self.feature_extractor(torch.FloatTensor(state))
        action = self.task_heads[task_id](features)
        return torch.tanh(action).detach().numpy()

    def select_task(self, state: np.ndarray) -> int:
        """Select which task to perform"""
        features = self.feature_extractor(torch.FloatTensor(state))
        task_logits = self.task_selector(features)
        task_probs = F.softmax(task_logits, dim=-1)
        return torch.multinomial(task_probs, 1).item()

    def forward(self, state: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass for training"""
        features = self.feature_extractor(state)
        action = torch.tanh(self.task_heads[task_id](features))
        return action
```

## Simulation and Transfer Learning

### Sim-to-Real Transfer

Training in simulation and transferring to real robots:

```python
# Sim-to-real transfer for humanoid RL
class SimToRealTransfer:
    """Sim-to-real transfer for humanoid RL"""

    def __init__(self, sim_agent, real_robot_interface):
        self.sim_agent = sim_agent
        self.real_robot = real_robot_interface
        self.domain_rand_params = {
            'mass_variance': 0.1,
            'friction_range': [0.5, 1.5],
            'actuator_noise': 0.01,
            'sensor_noise': 0.005
        }

    def apply_domain_randomization(self, sim_env):
        """Apply domain randomization to simulation"""
        # Randomize physical parameters
        mass_multiplier = 1.0 + np.random.uniform(-self.domain_rand_params['mass_variance'],
                                                 self.domain_rand_params['mass_variance'])
        friction = np.random.uniform(self.domain_rand_params['friction_range'][0],
                                    self.domain_rand_params['friction_range'][1])

        # Apply randomization to simulation
        sim_env.set_mass_multiplier(mass_multiplier)
        sim_env.set_friction(friction)

        # Add noise to actuators and sensors
        sim_env.set_actuator_noise(self.domain_rand_params['actuator_noise'])
        sim_env.set_sensor_noise(self.domain_rand_params['sensor_noise'])

    def train_with_domain_rand(self, sim_env, episodes: int = 1000):
        """Train with domain randomization"""
        for episode in range(episodes):
            self.apply_domain_randomization(sim_env)

            # Train on randomized environment
            state = sim_env.reset()
            done = False

            while not done:
                action = self.sim_agent.select_action(state)
                next_state, reward, done, info = sim_env.step(action)

                # Store transition
                self.sim_agent.store_transition(state, action, next_state, reward, done)

                # Train agent
                if episode > 100:  # Start training after initial exploration
                    self.sim_agent.train()

                state = next_state

    def adapt_to_real_robot(self, real_episodes: int = 100):
        """Adapt policy to real robot with minimal real-world training"""
        for episode in range(real_episodes):
            state = self.real_robot.reset()
            done = False

            while not done:
                # Use sim-trained policy with noise for exploration
                action = self.sim_agent.select_action(state, add_noise=True, noise_scale=0.05)

                # Execute on real robot
                next_state, reward, done, info = self.real_robot.step(action)

                # Store real-world experience
                self.sim_agent.store_transition(state, action, next_state, reward, done)

                # Fine-tune with real data
                self.sim_agent.train()

                state = next_state
```

## Practical Implementation Considerations

### Training Optimization

```python
# Training optimization for humanoid RL
class HumanoidRLTrainer:
    """Training optimization for humanoid RL"""

    def __init__(self, agent, env, log_dir: str = "./logs"):
        self.agent = agent
        self.env = env
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []

        # Training parameters
        self.max_episodes = 10000
        self.save_freq = 100
        self.eval_freq = 50
        self.target_performance = 0.8

    def train(self):
        """Main training loop"""
        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                # Store transition
                self.agent.store_transition(state, action, next_state, reward, done)

                # Train agent
                if len(self.agent.replay_buffer) > self.agent.batch_size:
                    train_info = self.agent.train()

                state = next_state
                episode_reward += reward
                episode_length += 1

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Log progress
            if episode % self.save_freq == 0:
                self.save_checkpoint(episode)

            if episode % self.eval_freq == 0:
                eval_reward = self.evaluate_policy()
                print(f"Episode {episode}, Eval Reward: {eval_reward:.2f}")

                # Check if target performance reached
                if eval_reward >= self.target_performance:
                    print(f"Target performance reached at episode {episode}")
                    break

    def evaluate_policy(self, eval_episodes: int = 10) -> float:
        """Evaluate current policy"""
        total_reward = 0

        for _ in range(eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state, add_noise=False)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward

            total_reward += episode_reward

        return total_reward / eval_episodes

    def save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint = {
            'episode': episode,
            'agent_state_dict': self.agent.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }

        torch.save(checkpoint, f"{self.log_dir}/checkpoint_{episode}.pth")
```

## Applications and Use Cases

### Walking Control

```python
# RL-based walking control for humanoid robots
class RLWalkingController:
    """Reinforcement learning walking controller"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.gait_phase = 0
        self.step_frequency = 1.0  # Hz

        # Initialize RL agent for walking
        state_dim = self.robot_config['observation_dim']
        action_dim = self.robot_config['num_joints']
        self.walking_agent = TD3Agent(state_dim, action_dim, max_action=1.0)

    def generate_walking_pattern(self, state: np.ndarray) -> np.ndarray:
        """Generate walking pattern using RL policy"""
        # Add gait phase information to state
        extended_state = np.append(state, self.gait_phase)

        # Get action from RL policy
        action = self.walking_agent.select_action(extended_state)

        # Update gait phase
        self.gait_phase = (self.gait_phase + self.step_frequency * 0.01) % (2 * np.pi)

        return action

    def train_walking_policy(self, env, episodes: int = 5000):
        """Train walking policy"""
        trainer = HumanoidRLTrainer(self.walking_agent, env)
        trainer.train()
```

### Manipulation Tasks

```python
# RL for manipulation tasks
class RLManipulationController:
    """Reinforcement learning manipulation controller"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config

        # Separate agents for different manipulation tasks
        self.grasping_agent = TD3Agent(
            state_dim=robot_config['grasping_state_dim'],
            action_dim=robot_config['gripper_action_dim'],
            max_action=1.0
        )

        self.reaching_agent = TD3Agent(
            state_dim=robot_config['reaching_state_dim'],
            action_dim=robot_config['arm_action_dim'],
            max_action=1.0
        )

    def grasp_object(self, object_state: np.ndarray) -> np.ndarray:
        """Generate grasping action"""
        return self.grasping_agent.select_action(object_state)

    def reach_target(self, target_state: np.ndarray) -> np.ndarray:
        """Generate reaching action"""
        return self.reaching_agent.select_action(target_state)
```

## Summary

Reinforcement learning offers significant potential for developing adaptive and robust control strategies for humanoid robots. The approach allows robots to learn complex behaviors through interaction with their environment, potentially achieving better performance than traditional control methods.

Key takeaways:
- RL enables adaptive control policies for complex humanoid behaviors
- DDPG and TD3 are well-suited for continuous control tasks
- Stability constraints and curriculum learning improve training safety
- Hierarchical and multi-task approaches enable complex behaviors
- Sim-to-real transfer techniques bridge simulation and reality
- Proper training optimization is crucial for practical deployment

The next chapter will explore Sim-to-Real Transfer, focusing on the methodologies and techniques for transferring skills learned in simulation to real humanoid robots.