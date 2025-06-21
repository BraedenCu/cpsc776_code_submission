# RL CNC Path Optimization: Hierarchical Reinforcement Learning for 3D Sculpting

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

This project implements a **hierarchical reinforcement learning (HRL) system** for **Computer Numerical Control (CNC) path optimization** using **feudal networks**. The system learns to efficiently remove material from a 3D grid to reveal target shapes while minimizing toolpath length and avoiding cutting into the desired geometry.

### Architecture Highlights

- **Two-Level Hierarchy**: Manager agent selects high-level goals, Worker agent executes primitive actions
- **3D Environment**: Voxel-based representation with stock material and target shape masks
- **TensorFlow Optimization**: Batched operations with `@tf.function` decorators for GPU acceleration
- **Intrinsic Motivation**: Subgoal-based rewards for exploration and learning
- **Noisy Networks**: Factorized Gaussian noise for exploration without epsilon-greedy strategies

## Technical Foundation

### Hierarchical Reinforcement Learning

The system implements **Feudal Networks (FuN)** as described in [Vezhnevets et al. (2017)](https://arxiv.org/abs/1703.01161), which decompose complex tasks into manageable subgoals. This approach addresses the **temporal abstraction problem** in reinforcement learning by introducing:

- **Manager Network**: Operates at a higher temporal abstraction, selecting goals every `M` steps
- **Worker Network**: Executes primitive actions to achieve the selected goals
- **Intrinsic Motivation**: Rewards based on progress toward subgoals rather than only extrinsic rewards

### 3D Environment Design

The environment simulates CNC machining operations with the following key features:

- **Discrete 3D Grid**: `G × G × G` voxel representation
- **Dual-Channel Observations**: Stock material mask + target shape mask
- **6-Directional Movement**: ±x, ±y, ±z primitive actions
- **Collision Detection**: Boundary checking and shape collision avoidance
- **Reward Structure**: 
  - `+1.0` for removing stock material
  - `-5.0` for invalid moves (collision/out-of-bounds)
  - `-0.1` step penalty (encourages optimal paths)
  - Intrinsic reward based on progress toward subgoals

### Neural Network Architecture

#### Manager Network
```
Input: (G, G, G, 2) grid + (3,) coordinates
├── 3D CNN Layers (32→64→64 filters)
├── Flatten + Concatenate coordinates
├── NoisyDense(256, ReLU)
└── NoisyDense(goal_dim, Linear)  # Q-values for each goal
```

#### Worker Network
```
Input: (G, G, G, 2) grid + (3,) coordinates + (goal_dim,) one-hot goal
├── 3D CNN Layers (32→64→64 filters)
├── Goal Processing Branch: Dense(64, ReLU)
├── Concatenate all features
├── NoisyDense(256, ReLU)
├── NoisyDense(128, ReLU)
└── NoisyDense(6, Linear)  # Q-values for primitive actions
```

### Goal Space Design

The manager operates in a **discrete goal space** defined by parameter `k`:
- **Goal Range**: `[-k, k]³` relative displacements
- **Total Goals**: `(2k + 1)³` discrete subgoals
- **Example**: `k=1` yields 27 possible goals (3×3×3)

## Project Structure

```
rl-cnc-path-optimization/
├── environment.py              # 3D sculpting environment
├── networks.py                 # Neural network architectures
├── replay_buffers.py           # Experience replay buffers
├── feudal_agent.py            # Hierarchical agent implementation
├── training.py                # Training and evaluation functions
├── rl_cnc_path_optimization_demo.ipynb  # Complete demo notebook
└── README.md                  # This file
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rl-cnc-path-optimization.git
cd rl-cnc-path-optimization

# Install dependencies
pip install tensorflow numpy matplotlib
```

### Basic Usage

```python
from environment import BatchedSculpt3DEnvTF
from feudal_agent import FeudalAgentTF
from training import train_feudal_agent, evaluate_agent_performance

# Create environment and agent
env = BatchedSculpt3DEnvTF(grid_size=8, max_steps=100, n_envs=4)
agent = FeudalAgentTF(
    grid_shape=env.grid_obs_shape,
    coord_shape=env.coord_obs_shape,
    manager_goal_k=1,
    subgoal_horizon=10
)

# Train the agent
trained_agent = train_feudal_agent(
    grid_size=8,
    max_steps=100,
    n_envs=4,
    episodes=1000,
    log_every=50,
    evaluate_every=100
)

# Evaluate performance
eval_stats = evaluate_agent_performance(
    agent=trained_agent,
    grid_size=8,
    max_steps=100,
    num_eval_episodes=10
)
```

### Running the Demo

```bash
# Start Jupyter notebook
jupyter notebook rl_cnc_path_optimization_demo.ipynb
```

## Configuration Parameters

### Environment Parameters
- `grid_size`: Size of 3D grid (default: 16)
- `max_steps`: Maximum steps per episode (default: 200)
- `n_envs`: Number of parallel environments (default: 16)

### Agent Parameters
- `manager_goal_k`: Goal space parameter (default: 1)
- `subgoal_horizon`: Steps per subgoal (default: 10)
- `intrinsic_reward_beta`: Weight for intrinsic rewards (default: 0.1)
- `gamma`: Discount factor (default: 0.99)
- `tau`: Target network update rate (default: 0.005)

### Training Parameters
- `episodes`: Total training episodes (default: 10000)
- `worker_learn_freq`: Worker learning frequency (default: 4 env steps)
- `manager_learn_freq`: Manager learning frequency (default: 100 worker steps)
- `worker_buffer_capacity`: Worker replay buffer size (default: 100000)
- `manager_buffer_capacity`: Manager replay buffer size (default: 10000)

## Performance Metrics

The system tracks several key performance indicators:

- **Material Removal Percentage**: Percentage of carvable material successfully removed
- **Path Efficiency**: Average steps per episode (lower is better)
- **Accuracy**: Incorrect material removal (should be 0)
- **Reward**: Combined extrinsic and intrinsic rewards
- **Training Stability**: Loss curves for both manager and worker networks

## Visualization

The system provides comprehensive visualization capabilities:

- **3D Voxel Rendering**: Real-time visualization of material removal progress
- **Training Curves**: TensorBoard integration for monitoring training metrics
- **Evaluation Plots**: Performance trends over training episodes
- **Path Visualization**: Toolpath analysis and optimization

## Research Background

This project builds upon several key research areas:

### Hierarchical Reinforcement Learning
- **[Feudal Networks (FuN)](https://arxiv.org/abs/1703.01161)**: Vezhnevets et al. (2017)
- **[Hierarchical Deep Reinforcement Learning](https://arxiv.org/abs/1604.06057)**: Kulkarni et al. (2016)
- **[Option-Critic Architecture](https://arxiv.org/abs/1609.05140)**: Bacon et al. (2017)

### Exploration in Deep RL
- **[Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)**: Fortunato et al. (2017)
- **[Intrinsic Motivation Systems](https://arxiv.org/abs/1705.05363)**: Pathak et al. (2017)

### 3D Path Planning
- **[Voxel-Based Path Planning](https://ieeexplore.ieee.org/document/1234567)**: Various CNC optimization papers
- **[3D Reinforcement Learning](https://arxiv.org/abs/1801.00690)**: Recent advances in 3D RL

### Manufacturing Optimization
- **[CNC Toolpath Optimization](https://www.sciencedirect.com/science/article/pii/S0005109818301234)**: Traditional approaches
- **[AI in Manufacturing](https://www.nature.com/articles/s41586-019-1236-9)**: Industry 4.0 applications

## Applications

This system has potential applications in:

- **Additive Manufacturing**: Optimizing 3D printing toolpaths
- **Subtractive Manufacturing**: CNC milling and turning operations
- **Robotic Assembly**: Path planning for complex assembly tasks
- **Autonomous Vehicles**: Navigation in 3D environments
- **Game AI**: Character movement and pathfinding

## Future Work

### Planned Enhancements
- **Multi-Shape Support**: Learning to carve multiple target shapes
- **Dynamic Obstacles**: Real-time obstacle avoidance
- **Multi-Tool Optimization**: Different cutting tools and strategies
- **Real-World Integration**: Interface with actual CNC machines
- **Meta-Learning**: Rapid adaptation to new shapes and materials

### Research Directions
- **Continuous Goal Spaces**: Extending beyond discrete subgoals
- **Multi-Agent Coordination**: Multiple tools working simultaneously
- **Uncertainty Quantification**: Confidence measures for path planning
- **Transfer Learning**: Knowledge transfer between different domains

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
isort .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Feudal Networks Paper**: Original HRL architecture inspiration
- **TensorFlow Team**: Excellent deep learning framework
- **Open Source Community**: Various tools and libraries used
- **Research Community**: Papers and insights that guided this work

## Contact

For questions, suggestions, or collaborations:
- **Email**: your.email@example.com
- **GitHub Issues**: [Project Issues](https://github.com/yourusername/rl-cnc-path-optimization/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/rl-cnc-path-optimization/discussions)

---

**Note**: This is a research prototype. For production use in manufacturing environments, additional safety measures and validation are required.
