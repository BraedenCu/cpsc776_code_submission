#!/usr/bin/env python3
"""
Feudal Network Agent for Hierarchical Reinforcement Learning
============================================================

This module implements a feudal network agent for hierarchical reinforcement learning
in CNC path optimization. The agent consists of a manager that selects high-level goals
and a worker that executes primitive actions to achieve those goals.

Key Features:
- Hierarchical architecture with manager and worker agents
- Intrinsic motivation through subgoal-based rewards
- Discrete goal space for manager actions
- Noisy networks for exploration
- Target networks for stable training
- TensorFlow-optimized operations

Author: RL CNC Path Optimization Team
License: MIT
"""

import tensorflow as tf
import numpy as np
import datetime
import os
from typing import Tuple, Optional, Dict, Any

from environment import flat_to_xyz, xyz_to_flat, clamp_pos_tf, calculate_manhattan_distance
from replay_buffers import WorkerReplayBuffer, ManagerReplayBuffer
from networks import (build_manager_network, build_worker_network, 
                     build_target_network, update_target_network)


@tf.function
def goal_index_to_delta_xyz(goal_index_batch: tf.Tensor, k: int) -> tf.Tensor:
    """
    Convert a batch of discrete goal indices to relative vectors.
    
    Args:
        goal_index_batch: Goal indices of shape [N]
        k: Goal space parameter (goals are in [-k, k]^3)
        
    Returns:
        Delta vectors of shape [N, 3] representing relative displacements
    """
    base = tf.cast(2 * k + 1, tf.int32)
    shift = tf.constant(k, dtype=tf.int32)
    
    dz_batch = goal_index_batch % base
    dy_batch = tf.cast((goal_index_batch // base) % base, tf.int32)
    dx_batch = tf.cast((goal_index_batch // (base * base)), tf.int32)
    
    # Shift from [0, 2k] range to [-k, k] range
    return tf.stack([dx_batch - shift, dy_batch - shift, dz_batch - shift], axis=-1)


@tf.function
def calculate_subgoal_target_pos_flat(start_pos_flat_batch: tf.Tensor, 
                                     goal_index_batch: tf.Tensor, 
                                     G: int, k: int) -> tf.Tensor:
    """
    Calculate the target flat position for the subgoal based on start position and goal.
    
    Args:
        start_pos_flat_batch: Starting positions [N]
        goal_index_batch: Goal indices [N]
        G: Grid size
        k: Goal space parameter
        
    Returns:
        Target positions [N]
    """
    start_pos_xyz_batch = flat_to_xyz(start_pos_flat_batch, G)
    delta_xyz_batch = goal_index_to_delta_xyz(goal_index_batch, k)
    
    target_xyz_raw_batch = start_pos_xyz_batch + delta_xyz_batch
    target_xyz_clamped_batch = clamp_pos_tf(target_xyz_raw_batch, G)
    
    target_pos_flat_batch = xyz_to_flat(
        target_xyz_clamped_batch[:, 0],
        target_xyz_clamped_batch[:, 1],
        target_xyz_clamped_batch[:, 2], G
    )
    return target_pos_flat_batch


@tf.function
def calculate_intrinsic_reward_batch(pos_t_flat_batch: tf.Tensor, 
                                   pos_tplus1_flat_batch: tf.Tensor,
                                   subgoal_start_pos_flat_batch: tf.Tensor,
                                   current_goal_index_batch: tf.Tensor,
                                   G: int, k: int) -> tf.Tensor:
    """
    Calculate intrinsic reward for a batch based on progress towards subgoal target.
    
    Args:
        pos_t_flat_batch: Positions before the step [N]
        pos_tplus1_flat_batch: Positions after the step [N]
        subgoal_start_pos_flat_batch: Positions where current goal was set [N]
        current_goal_index_batch: Current goal indices [N]
        G: Grid size
        k: Goal space parameter
        
    Returns:
        Intrinsic rewards [N]
    """
    target_pos_flat_batch = calculate_subgoal_target_pos_flat(
        subgoal_start_pos_flat_batch, current_goal_index_batch, G, k
    )
    
    # Convert target positions to xyz for distance calculation
    target_pos_xyz_batch = flat_to_xyz(target_pos_flat_batch, G)
    pos_t_xyz_batch = flat_to_xyz(pos_t_flat_batch, G)
    pos_tplus1_xyz_batch = flat_to_xyz(pos_tplus1_flat_batch, G)
    
    dist_t = calculate_manhattan_distance(pos_t_xyz_batch, target_pos_xyz_batch)
    dist_tplus1 = calculate_manhattan_distance(pos_tplus1_xyz_batch, target_pos_xyz_batch)
    
    # Intrinsic reward: decrease in distance
    intrinsic_reward_batch = tf.cast(dist_t, tf.float32) - tf.cast(dist_tplus1, tf.float32)
    
    return intrinsic_reward_batch


class FeudalAgentTF:
    """
    Feudal Network Agent for hierarchical reinforcement learning.
    
    This agent implements a two-level hierarchy:
    - Manager: Selects high-level goals (subgoals) based on current state
    - Worker: Executes primitive actions to achieve the selected goals
    
    The agent uses intrinsic motivation through subgoal-based rewards and
    maintains separate replay buffers and networks for each level.
    """
    
    def __init__(self, grid_shape: Tuple[int, ...], coord_shape: Tuple[int, ...],
                 primitive_action_dim: int = 6, manager_goal_k: int = 1,
                 subgoal_horizon: int = 10, intrinsic_reward_beta: float = 0.1,
                 manager_lr: float = 1e-4, worker_lr: float = 1e-4,
                 gamma: float = 0.99, tau: float = 0.005,
                 worker_buffer_capacity: int = 100000,
                 manager_buffer_capacity: int = 10000):
        """
        Initialize the feudal agent.
        
        Args:
            grid_shape: Shape of grid observations
            coord_shape: Shape of coordinate observations
            primitive_action_dim: Number of primitive actions
            manager_goal_k: Goal space parameter (goals in [-k, k]^3)
            subgoal_horizon: Number of steps per subgoal
            intrinsic_reward_beta: Weight for intrinsic rewards
            manager_lr: Learning rate for manager
            worker_lr: Learning rate for worker
            gamma: Discount factor
            tau: Target network update rate
            worker_buffer_capacity: Capacity of worker replay buffer
            manager_buffer_capacity: Capacity of manager replay buffer
        """
        self.grid_shape = grid_shape
        self.coord_shape = coord_shape
        self.primitive_action_dim = primitive_action_dim
        self.gamma = gamma
        self.tau = tau
        self.subgoal_horizon = subgoal_horizon
        self.intrinsic_reward_beta = intrinsic_reward_beta
        
        # Manager goal space
        self.manager_goal_k = manager_goal_k
        self.manager_goal_base = 2 * manager_goal_k + 1
        self.manager_goal_dim = self.manager_goal_base ** 3  # Number of discrete goals
        
        print(f"Feudal Agent initialized:")
        print(f"  Subgoal Horizon (M): {self.subgoal_horizon}")
        print(f"  Intrinsic Reward Beta: {self.intrinsic_reward_beta}")
        print(f"  Manager Goal K: {self.manager_goal_k} (Discrete goals: {self.manager_goal_dim})")
        
        # Build networks
        self.manager_model = build_manager_network(
            grid_shape, coord_shape, self.manager_goal_dim
        )
        self.worker_model = build_worker_network(
            grid_shape, coord_shape, self.manager_goal_dim, primitive_action_dim
        )
        
        # Build target networks
        self.manager_target = build_target_network(self.manager_model)
        self.worker_target = build_target_network(self.worker_model)
        
        # Initialize optimizers
        self.manager_optimizer = tf.keras.optimizers.Adam(learning_rate=manager_lr)
        self.worker_optimizer = tf.keras.optimizers.Adam(learning_rate=worker_lr)
        
        # Initialize replay buffers
        self.worker_buffer = WorkerReplayBuffer(capacity=worker_buffer_capacity)
        self.manager_buffer = ManagerReplayBuffer(capacity=manager_buffer_capacity)
        
        # Training step counters
        self.worker_train_step_count = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.manager_train_step_count = tf.Variable(0, dtype=tf.int32, trainable=False)
        
        # TensorBoard logging
        log_dir = os.path.join("runs", f"feudal_dqn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.writer = tf.summary.create_file_writer(log_dir)
        print(f"TensorBoard log directory: {log_dir}")
    
    def manager_act_batch(self, obs_tuple: Tuple[tf.Tensor, tf.Tensor], 
                         deterministic: bool = False) -> tf.Tensor:
        """
        Manager selects goals for a batch of environments.
        
        Args:
            obs_tuple: (grid_obs, coord_obs) observations
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Goal indices for each environment
        """
        grid_obs, coord_obs = obs_tuple
        
        # Get Q-values from manager network
        q_values = self.manager_model([grid_obs, coord_obs], training=False)
        
        if deterministic:
            # Select best action
            actions = tf.argmax(q_values, axis=-1, output_type=tf.int32)
        else:
            # Sample from softmax distribution
            logits = q_values / 0.1  # Temperature parameter
            actions = tf.random.categorical(logits, 1)
            actions = tf.squeeze(actions, axis=-1)
        
        return actions
    
    def worker_act_batch(self, obs_tuple: Tuple[tf.Tensor, tf.Tensor],
                        goal_indices: tf.Tensor, deterministic: bool = False) -> tf.Tensor:
        """
        Worker selects primitive actions for a batch of environments.
        
        Args:
            obs_tuple: (grid_obs, coord_obs) observations
            goal_indices: Goal indices from manager
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Primitive actions for each environment
        """
        grid_obs, coord_obs = obs_tuple
        
        # Convert goal indices to one-hot encoding
        goal_one_hot = tf.one_hot(goal_indices, self.manager_goal_dim, dtype=tf.float32)
        
        # Get Q-values from worker network
        q_values = self.worker_model([grid_obs, coord_obs, goal_one_hot], training=False)
        
        if deterministic:
            # Select best action
            actions = tf.argmax(q_values, axis=-1, output_type=tf.int32)
        else:
            # Sample from softmax distribution
            logits = q_values / 0.1  # Temperature parameter
            actions = tf.random.categorical(logits, 1)
            actions = tf.squeeze(actions, axis=-1)
        
        return actions
    
    @tf.function
    def worker_train_step(self, S_grid: tf.Tensor, S_coord: tf.Tensor,
                         G: tf.Tensor, A: tf.Tensor, R_total: tf.Tensor,
                         S2_grid: tf.Tensor, S2_coord: tf.Tensor, D: tf.Tensor) -> tf.Tensor:
        """
        Perform a training step for the worker network.
        
        Args:
            S_grid, S_coord: Current state observations
            G: Goal indices
            A: Actions taken
            R_total: Total rewards received
            S2_grid, S2_coord: Next state observations
            D: Done flags
            
        Returns:
            Loss value
        """
        with tf.GradientTape() as tape:
            # Convert goals to one-hot
            G_one_hot = tf.one_hot(G, self.manager_goal_dim, dtype=tf.float32)
            G2_one_hot = tf.one_hot(G, self.manager_goal_dim, dtype=tf.float32)  # Same goal for next state
            
            # Current Q-values
            current_q_values = self.worker_model([S_grid, S_coord, G_one_hot], training=True)
            current_q = tf.reduce_sum(current_q_values * tf.one_hot(A, self.primitive_action_dim), axis=-1)
            
            # Next Q-values (from target network)
            next_q_values = self.worker_target([S2_grid, S2_coord, G2_one_hot], training=False)
            next_q = tf.reduce_max(next_q_values, axis=-1)
            
            # Target Q-values
            target_q = R_total + self.gamma * next_q * (1.0 - tf.cast(D, tf.float32))
            
            # Loss
            loss = tf.reduce_mean(tf.square(target_q - current_q))
        
        # Apply gradients
        gradients = tape.gradient(loss, self.worker_model.trainable_variables)
        self.worker_optimizer.apply_gradients(zip(gradients, self.worker_model.trainable_variables))
        
        return loss
    
    @tf.function
    def manager_train_step(self, S_grid: tf.Tensor, S_coord: tf.Tensor,
                          G: tf.Tensor, R_accumulated: tf.Tensor,
                          S2_grid: tf.Tensor, S2_coord: tf.Tensor, D: tf.Tensor) -> tf.Tensor:
        """
        Perform a training step for the manager network.
        
        Args:
            S_grid, S_coord: Current state observations
            G: Goal indices selected
            R_accumulated: Accumulated extrinsic rewards
            S2_grid, S2_coord: Next state observations
            D: Done flags
            
        Returns:
            Loss value
        """
        with tf.GradientTape() as tape:
            # Current Q-values
            current_q_values = self.manager_model([S_grid, S_coord], training=True)
            current_q = tf.reduce_sum(current_q_values * tf.one_hot(G, self.manager_goal_dim), axis=-1)
            
            # Next Q-values (from target network)
            next_q_values = self.manager_target([S2_grid, S2_coord], training=False)
            next_q = tf.reduce_max(next_q_values, axis=-1)
            
            # Target Q-values
            target_q = R_accumulated + self.gamma * next_q * (1.0 - tf.cast(D, tf.float32))
            
            # Loss
            loss = tf.reduce_mean(tf.square(target_q - current_q))
        
        # Apply gradients
        gradients = tape.gradient(loss, self.manager_model.trainable_variables)
        self.manager_optimizer.apply_gradients(zip(gradients, self.manager_model.trainable_variables))
        
        return loss
    
    def worker_learn(self, batch_size: int = 32) -> Optional[float]:
        """
        Sample from worker buffer and perform training step.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Loss value or None if insufficient data
        """
        if len(self.worker_buffer) < batch_size:
            return None
        
        sampled_data = self.worker_buffer.sample(batch_size)
        if sampled_data is None:
            return None
        
        S_grid_s, S_coord_s, G_s, A_s, R_total_s, S2_grid_s, S2_coord_s, D_s = sampled_data
        
        loss = self.worker_train_step(
            S_grid_s, S_coord_s, tf.cast(G_s, tf.int32), A_s, R_total_s,
            S2_grid_s, S2_coord_s, D_s
        )
        self.worker_train_step_count.assign_add(1)
        
        # Update target network
        update_target_network(self.worker_target, self.worker_model, self.tau)
        
        return loss.numpy()
    
    def manager_learn(self, batch_size: int = 32) -> Optional[float]:
        """
        Sample from manager buffer and perform training step.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Loss value or None if insufficient data
        """
        if len(self.manager_buffer) < batch_size:
            return None
        
        sampled_data = self.manager_buffer.sample(batch_size)
        if sampled_data is None:
            return None
        
        S_grid_s, S_coord_s, G_s, R_accumulated_s, S2_grid_s, S2_coord_s, D_s = sampled_data
        
        loss = self.manager_train_step(
            S_grid_s, S_coord_s, tf.cast(G_s, tf.int32), R_accumulated_s,
            S2_grid_s, S2_coord_s, D_s
        )
        self.manager_train_step_count.assign_add(1)
        
        # Update target network
        update_target_network(self.manager_target, self.manager_model, self.tau)
        
        return loss.numpy()
    
    def worker_remember_batch(self, S_tuple: Tuple[tf.Tensor, tf.Tensor],
                            G_batch: tf.Tensor, A: tf.Tensor, R_total: tf.Tensor,
                            S2_tuple: Tuple[tf.Tensor, tf.Tensor], D: tf.Tensor):
        """Add a batch of worker transitions to the replay buffer."""
        self.worker_buffer.add_batch(S_tuple, G_batch, A, R_total, S2_tuple, D)
    
    def manager_remember_batch(self, S_tuple: Tuple[tf.Tensor, tf.Tensor],
                             G: tf.Tensor, R_accumulated: tf.Tensor,
                             S2_tuple: Tuple[tf.Tensor, tf.Tensor], D: tf.Tensor):
        """Add a batch of manager transitions to the replay buffer."""
        self.manager_buffer.add_batch(S_tuple, G, R_accumulated, S2_tuple, D)
    
    def save_weights(self, manager_path: str, worker_path: str):
        """Save model weights to files."""
        self.manager_model.save_weights(manager_path)
        self.worker_model.save_weights(worker_path)
    
    def load_weights(self, manager_path: str, worker_path: str):
        """Load model weights from files."""
        self.manager_model.load_weights(manager_path)
        self.worker_model.load_weights(worker_path)
        # Also update target networks
        self.manager_target.set_weights(self.manager_model.get_weights())
        self.worker_target.set_weights(self.worker_model.get_weights())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'worker_train_steps': self.worker_train_step_count.numpy(),
            'manager_train_steps': self.manager_train_step_count.numpy(),
            'worker_buffer_size': len(self.worker_buffer),
            'manager_buffer_size': len(self.manager_buffer),
            'worker_buffer_utilization': self.worker_buffer.get_stats()['utilization'],
            'manager_buffer_utilization': self.manager_buffer.get_stats()['utilization']
        }


if __name__ == "__main__":
    # Example usage
    print("Testing feudal agent...")
    
    # Create agent
    grid_shape = (8, 8, 8, 2)
    coord_shape = (3,)
    agent = FeudalAgentTF(grid_shape, coord_shape)
    
    print(f"Agent created successfully!")
    print(f"Manager goal dimension: {agent.manager_goal_dim}")
    print(f"Worker action dimension: {agent.primitive_action_dim}")
    
    # Test action selection
    batch_size = 4
    grid_obs = tf.random.normal([batch_size] + list(grid_shape))
    coord_obs = tf.random.normal([batch_size] + list(coord_shape))
    
    # Manager action
    goals = agent.manager_act_batch((grid_obs, coord_obs))
    print(f"Manager selected goals: {goals.numpy()}")
    
    # Worker action
    actions = agent.worker_act_batch((grid_obs, coord_obs), goals)
    print(f"Worker selected actions: {actions.numpy()}")
    
    # Test training
    print(f"\nTesting training...")
    print(f"Initial stats: {agent.get_stats()}")
    
    # Add some dummy transitions
    R_total = tf.random.normal([batch_size])
    D = tf.random.uniform([batch_size], 0, 2, dtype=tf.bool)
    
    agent.worker_remember_batch(
        (grid_obs, coord_obs), goals, actions, R_total, (grid_obs, coord_obs), D
    )
    
    # Try to learn
    worker_loss = agent.worker_learn(batch_size=2)
    if worker_loss is not None:
        print(f"Worker loss: {worker_loss:.4f}")
    
    print(f"Final stats: {agent.get_stats()}")
    print("Feudal agent tests completed!") 