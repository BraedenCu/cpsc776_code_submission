#!/usr/bin/env python3
"""
Example Usage: RL CNC Path Optimization
=======================================

This script demonstrates how to use the modular components of the RL CNC path
optimization system. It shows basic usage patterns for training and evaluation.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Import our modules
from environment import BatchedSculpt3DEnvTF, create_custom_shape_mask
from feudal_agent import FeudalAgentTF
from training import train_feudal_agent, evaluate_agent_performance


def basic_environment_demo():
    """Demonstrate basic environment functionality."""
    print("=== Environment Demo ===")
    
    # Create a small environment for demonstration
    env = BatchedSculpt3DEnvTF(grid_size=8, max_steps=50, n_envs=4)
    
    # Reset environment
    obs_tuple = env.reset()
    grid_obs, coord_obs = obs_tuple
    
    print(f"Environment created:")
    print(f"  Grid size: {env.G}")
    print(f"  Number of environments: {env.N}")
    print(f"  Grid observation shape: {grid_obs.shape}")
    print(f"  Coordinate observation shape: {coord_obs.shape}")
    
    # Take a few random steps
    for step in range(5):
        actions = tf.random.uniform([env.N], 0, 6, dtype=tf.int32)
        obs, rewards, done = env.step(actions)
        
        print(f"Step {step}: Avg reward = {tf.reduce_mean(rewards):.3f}, "
              f"Done = {tf.reduce_sum(tf.cast(done, tf.int32))}/{env.N}")
    
    # Get final statistics
    stats = env.get_stats()
    print(f"Final stats: {stats}")
    print()


def agent_demo():
    """Demonstrate agent functionality."""
    print("=== Agent Demo ===")
    
    # Create environment and agent
    env = BatchedSculpt3DEnvTF(grid_size=8, max_steps=50, n_envs=4)
    agent = FeudalAgentTF(
        grid_shape=env.grid_obs_shape,
        coord_shape=env.coord_obs_shape,
        manager_goal_k=1,
        subgoal_horizon=5,
        worker_buffer_capacity=1000,
        manager_buffer_capacity=100
    )
    
    print(f"Agent created:")
    print(f"  Manager goal dimension: {agent.manager_goal_dim}")
    print(f"  Worker action dimension: {agent.primitive_action_dim}")
    print(f"  Subgoal horizon: {agent.subgoal_horizon}")
    
    # Test action selection
    obs_tuple = env.reset()
    
    # Manager selects goals
    goals = agent.manager_act_batch(obs_tuple, deterministic=False)
    print(f"Manager selected goals: {goals.numpy()}")
    
    # Worker selects actions
    actions = agent.worker_act_batch(obs_tuple, goals, deterministic=False)
    print(f"Worker selected actions: {actions.numpy()}")
    
    # Take a step and store experience
    next_obs, rewards, done = env.step(actions)
    
    # Store in replay buffers
    agent.worker_remember_batch(obs_tuple, goals, actions, rewards, next_obs, done)
    
    print(f"Experience stored in buffers")
    print(f"  Worker buffer size: {len(agent.worker_buffer)}")
    print(f"  Manager buffer size: {len(agent.manager_buffer)}")
    print()


def quick_training_demo():
    """Demonstrate a quick training run."""
    print("=== Quick Training Demo ===")
    
    # Small training run for demonstration
    print("Starting quick training run...")
    
    trained_agent = train_feudal_agent(
        grid_size=8,
        max_steps=50,
        n_envs=4,
        episodes=20,  # Very short for demo
        worker_buffer_capacity=1000,
        manager_buffer_capacity=100,
        worker_learn_batch_size=8,
        manager_learn_batch_size=8,
        worker_learn_freq=2,
        manager_learn_freq=5,
        log_every=5,
        evaluate_every=10,
        save_every_episodes=20
    )
    
    print("Training completed!")
    print(f"Final agent stats: {trained_agent.get_stats()}")
    print()


def evaluation_demo():
    """Demonstrate evaluation functionality."""
    print("=== Evaluation Demo ===")
    
    # Create a simple agent for evaluation
    env = BatchedSculpt3DEnvTF(grid_size=8, max_steps=50, n_envs=1)
    agent = FeudalAgentTF(
        grid_shape=env.grid_obs_shape,
        coord_shape=env.coord_obs_shape,
        manager_goal_k=1,
        subgoal_horizon=5
    )
    
    # Run evaluation
    eval_stats = evaluate_agent_performance(
        agent=agent,
        grid_size=8,
        max_steps=50,
        num_eval_episodes=5,
        render=False  # Disable rendering for demo
    )
    
    print("Evaluation completed!")
    print(f"Evaluation stats: {eval_stats}")
    print()


def custom_shape_demo():
    """Demonstrate custom shape creation."""
    print("=== Custom Shape Demo ===")
    
    # Create different shape types
    grid_size = 8
    
    # Sphere shape
    sphere_mask = create_custom_shape_mask(grid_size, 'sphere', radius=3)
    print(f"Sphere shape created: {np.sum(sphere_mask)} voxels")
    
    # Cube shape
    cube_mask = create_custom_shape_mask(grid_size, 'cube', size=4)
    print(f"Cube shape created: {np.sum(cube_mask)} voxels")
    
    # Cylinder shape
    cylinder_mask = create_custom_shape_mask(grid_size, 'cylinder', radius=2, height=6)
    print(f"Cylinder shape created: {np.sum(cylinder_mask)} voxels")
    
    print("Custom shapes created successfully!")
    print()


def main():
    """Run all demonstrations."""
    print("RL CNC Path Optimization - Example Usage")
    print("=" * 50)
    print()
    
    # Check if TensorFlow is available
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print()
    
    try:
        # Run demonstrations
        basic_environment_demo()
        agent_demo()
        custom_shape_demo()
        
        # Uncomment the following lines for training and evaluation demos
        # (These take longer to run)
        # quick_training_demo()
        # evaluation_demo()
        
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 