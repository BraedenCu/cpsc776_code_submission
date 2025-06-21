#!/usr/bin/env python3
"""
3D Sculpting Environment for CNC Path Optimization
==================================================

This module implements a batched 3D sculpting environment optimized for TensorFlow.
The environment simulates CNC machining operations where an agent must remove material
from a 3D grid to reveal a target shape while avoiding cutting into the desired shape.

Key Features:
- Batched environment for parallel simulation
- TensorFlow-optimized operations with @tf.function decorators
- 3D grid representation with stock and shape masks
- 6-directional movement (x, y, z axes)
- Collision detection and boundary checking
- Reward system for material removal and penalty for invalid moves

Author: RL CNC Path Optimization Team
License: MIT
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional


@tf.function
def flat_to_xyz(flat_idx, G):
    """Convert flat index to (x, y, z) coordinates.
    
    Args:
        flat_idx: Flat index or batch of flat indices
        G: Grid size
        
    Returns:
        Tensor of shape [N, 3] or [3] with (x, y, z) coordinates
    """
    g_tf = tf.cast(G, tf.int32)
    z = flat_idx % g_tf
    y = (flat_idx // g_tf) % g_tf
    x = flat_idx // (g_tf * g_tf)
    
    if tf.rank(flat_idx) == 0:  # Single index
        return tf.stack([x, y, z], axis=-1)
    else:
        return tf.stack([x, y, z], axis=-1)


@tf.function
def xyz_to_flat(x, y, z, G):
    """Convert (x, y, z) coordinates to flat index.
    
    Args:
        x, y, z: Coordinate tensors
        G: Grid size
        
    Returns:
        Flat index tensor
    """
    g_tf = tf.cast(G, tf.int32)
    return x * g_tf * g_tf + y * g_tf + z


@tf.function
def clamp_pos_tf(pos_xyz, G):
    """Clamp positions to valid grid bounds.
    
    Args:
        pos_xyz: Position tensor of shape [N, 3]
        G: Grid size
        
    Returns:
        Clamped position tensor
    """
    g_minus_1 = tf.cast(G - 1, tf.int32)
    return tf.clip_by_value(pos_xyz, 0, g_minus_1)


@tf.function
def calculate_manhattan_distance(pos1_xyz, pos2_xyz):
    """Calculate Manhattan distance between two sets of coordinates.
    
    Args:
        pos1_xyz, pos2_xyz: Position tensors of shape [N, 3]
        
    Returns:
        Distance tensor of shape [N]
    """
    return tf.reduce_sum(tf.abs(pos1_xyz - pos2_xyz), axis=-1)


class BatchedSculpt3DEnvTF:
    """
    Batched 3D Sculpting Environment optimized for TensorFlow.
    
    This environment simulates CNC machining where an agent navigates a 3D grid
    and removes material to reveal a target shape. The environment supports
    batched operations for efficient parallel simulation.
    
    Attributes:
        G (int): Grid size (G x G x G)
        N (int): Number of parallel environments
        max_steps (int): Maximum steps per episode
        flat_dim (int): Total number of voxels (G^3)
        grid_obs_shape (tuple): Shape of grid observations (G, G, G, 2)
        coord_obs_shape (tuple): Shape of coordinate observations (3,)
        shape_mask (tf.Variable): Boolean mask of target shape
        stock (tf.Variable): Boolean mask of remaining material
        pos (tf.Variable): Current positions for all environments
        steps (tf.Variable): Step counts for all environments
        done (tf.Variable): Done flags for all environments
        shifts (tf.Tensor): Movement vectors for 6 directions
    """
    
    def __init__(self, grid_size=16, max_steps=200, n_envs=16):
        """
        Initialize the batched 3D sculpting environment.
        
        Args:
            grid_size (int): Size of the 3D grid (default: 16)
            max_steps (int): Maximum steps per episode (default: 200)
            n_envs (int): Number of parallel environments (default: 16)
        """
        G, N = grid_size, n_envs
        if N <= 0:
            raise ValueError("n_envs must be positive.")
            
        self.G, self.N, self.max_steps = G, N, max_steps
        self.flat_dim = G * G * G
        self.grid_obs_shape = (G, G, G, 2)  # channels: Stock, ShapeMask
        self.coord_obs_shape = (3,)          # channels: X, Y, Z (normalized)
        
        # Create spherical target shape
        coords_range = tf.range(G, dtype=tf.float32)
        coords = tf.stack(tf.meshgrid(coords_range, coords_range, coords_range, indexing='ij'), axis=-1)
        center = tf.constant([G/2 - 0.5, G/2 - 0.5, G/2 - 0.5], tf.float32)
        dist2 = tf.reduce_sum(tf.square(coords - center), axis=-1)
        radius_sq = tf.square(tf.cast(G // 2 - 1, tf.float32))
        mask3d = dist2 <= radius_sq
        mask_flat = tf.reshape(mask3d, [-1])
        
        # Initialize environment state variables
        self.shape_mask = tf.Variable(
            tf.tile(mask_flat[None, :], [N, 1]), 
            trainable=False, 
            dtype=tf.bool, 
            name="shape_mask"
        )
        self.stock = tf.Variable(
            tf.ones([N, self.flat_dim], dtype=tf.bool), 
            trainable=False, 
            name="stock"
        )
        self.pos = tf.Variable(
            tf.zeros([N], dtype=tf.int32), 
            trainable=False, 
            name="pos"
        )
        self.steps = tf.Variable(
            tf.zeros([N], dtype=tf.int32), 
            trainable=False, 
            name="steps"
        )
        self.done = tf.Variable(
            tf.zeros([N], dtype=tf.bool), 
            trainable=False, 
            name="done"
        )
        
        # Define movement vectors (6 directions: ±x, ±y, ±z)
        G_py = grid_size
        def to_flat_py(dx, dy, dz): 
            return dx * G_py * G_py + dy * G_py + dz
            
        moves = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        shifts_py = [to_flat_py(*m) for m in moves]
        self.shifts = tf.constant(shifts_py, dtype=tf.int32, name="shifts")
    
    @tf.function
    def reset(self):
        """
        Reset all environments to initial state.
        
        Returns:
            tuple: (grid_obs, coord_obs) - Initial observations for all environments
        """
        # Reset stock to full
        self.stock.assign(tf.ones_like(self.stock))
        self.steps.assign(tf.zeros_like(self.steps))
        self.done.assign(tf.zeros_like(self.done))
        
        # Find safe starting positions (not in target shape)
        safe_indices = tf.where(tf.logical_not(self.shape_mask[0]))[:, 0]
        num_safe = tf.shape(safe_indices)[0]
        tf.debugging.assert_greater_equal(
            num_safe, self.N, 
            message="Not enough safe starting positions available."
        )
        
        # Randomly select starting positions
        shuffled_safe_indices = tf.random.shuffle(safe_indices)[:self.N]
        self.pos.assign(tf.cast(shuffled_safe_indices, tf.int32))
        
        return self._get_obs()
    
    @tf.function
    def step(self, actions):
        """
        Execute actions in all environments.
        
        Args:
            actions: Action tensor of shape [N] with values in [0, 5]
            
        Returns:
            tuple: (next_obs, rewards, done) - Environment step results
        """
        # Calculate potential new positions based on actions
        action_shifts = tf.gather(self.shifts, actions)
        new_pos = self.pos + action_shifts
        
        # Check boundaries and collisions
        in_bounds = tf.logical_and(new_pos >= 0, new_pos < self.flat_dim)
        safe_new_pos = tf.clip_by_value(new_pos, 0, self.flat_dim - 1)
        
        # Get shape mask and stock at potential new positions
        shape_mask_at_new = tf.gather(self.shape_mask, safe_new_pos, axis=1, batch_dims=1)
        stock_at_new = tf.gather(self.stock, safe_new_pos, axis=1, batch_dims=1)
        
        # Determine invalid moves (hit shape or out of bounds)
        hit_shape_or_oob = tf.logical_or(tf.logical_not(in_bounds), shape_mask_at_new)
        
        # Determine if stock can be removed (valid move AND stock exists at new pos)
        can_remove = tf.logical_and(tf.logical_not(hit_shape_or_oob), stock_at_new)
        
        # Calculate rewards
        reward = tf.where(hit_shape_or_oob, -5.0, 0.0)  # Invalid move penalty
        reward = tf.where(can_remove, reward + 1.0, reward)  # Reward for removing stock
        reward = reward - 0.1  # Step penalty (encourage optimal toolpath)
        
        # Update stock (remove material where applicable)
        remove_indices = tf.where(can_remove)
        num_removals = tf.shape(remove_indices)[0]
        
        # Conditional update to avoid empty tensor issues
        def perform_update():
            env_indices_to_update = tf.squeeze(tf.cast(remove_indices, tf.int32), axis=1)
            pos_to_remove = tf.gather(new_pos, env_indices_to_update)
            scatter_indices = tf.stack([env_indices_to_update, pos_to_remove], axis=1)
            updates = tf.zeros(num_removals, dtype=tf.bool)
            return tf.tensor_scatter_nd_update(self.stock, scatter_indices, updates)
        
        maybe_updated_stock = tf.cond(
            tf.greater(num_removals, 0), 
            true_fn=perform_update, 
            false_fn=lambda: self.stock
        )
        self.stock.assign(maybe_updated_stock)
        
        # Update positions (only for valid moves)
        is_valid_move = tf.logical_not(hit_shape_or_oob)
        next_pos = tf.where(is_valid_move, new_pos, self.pos)
        self.pos.assign(next_pos)
        
        # Update step counts and done flags
        self.steps.assign_add(tf.ones_like(self.steps))
        newly_done = (self.steps >= self.max_steps)
        self.done.assign(tf.logical_or(self.done, newly_done))
        
        next_obs = self._get_obs()
        return next_obs, tf.cast(reward, tf.float32), self.done
    
    @tf.function
    def _get_obs(self):
        """
        Get current observations for all environments.
        
        Returns:
            tuple: (grid_obs, coord_obs) - Current observations
        """
        G = self.G
        N = self.N
        
        # Reshape stock and shape mask to 3D grids
        stock_grid = tf.reshape(self.stock, [N, G, G, G])
        shape_mask_grid = tf.reshape(self.shape_mask, [N, G, G, G])
        
        # Convert to float for neural network input
        stock_float = tf.cast(stock_grid, tf.float32)
        shape_mask_float = tf.cast(shape_mask_grid, tf.float32)
        grid_obs = tf.stack([stock_float, shape_mask_float], axis=-1)
        
        # Calculate normalized coordinates
        g_tf = tf.constant(G, dtype=tf.int32)
        z = self.pos % g_tf
        y = (self.pos // g_tf) % g_tf
        x = self.pos // (g_tf * g_tf)
        
        # Normalize coordinates to [0, 1] range
        g_minus_1_float = tf.cast(tf.maximum(1, G - 1), tf.float32)
        x_norm = tf.cast(x, tf.float32) / g_minus_1_float
        y_norm = tf.cast(y, tf.float32) / g_minus_1_float
        z_norm = tf.cast(z, tf.float32) / g_minus_1_float
        
        coord_obs = tf.stack([x_norm, y_norm, z_norm], axis=-1)
        
        return (grid_obs, coord_obs)
    
    def get_stats(self):
        """
        Get current environment statistics.
        
        Returns:
            dict: Statistics including material removed, steps taken, etc.
        """
        stock_np = self.stock.numpy()
        shape_mask_np = self.shape_mask.numpy()
        
        total_removed = np.sum(~stock_np & ~shape_mask_np, axis=1)  # Correctly removed
        incorrect_removed = np.sum(~stock_np & shape_mask_np, axis=1)  # Incorrectly removed
        steps_taken = self.steps.numpy()
        done_flags = self.done.numpy()
        
        return {
            'total_removed': total_removed,
            'incorrect_removed': incorrect_removed,
            'steps_taken': steps_taken,
            'done_flags': done_flags
        }


def create_custom_shape_mask(grid_size: int, shape_type: str = 'sphere', **kwargs) -> np.ndarray:
    """
    Create custom shape masks for different target geometries.
    
    Args:
        grid_size (int): Size of the 3D grid
        shape_type (str): Type of shape ('sphere', 'cube', 'cylinder', 'custom')
        **kwargs: Additional parameters for shape generation
        
    Returns:
        np.ndarray: Boolean mask of shape [grid_size, grid_size, grid_size]
    """
    if shape_type == 'sphere':
        radius = kwargs.get('radius', grid_size // 2 - 1)
        center = kwargs.get('center', [grid_size//2, grid_size//2, grid_size//2])
        
        coords = np.indices((grid_size, grid_size, grid_size))
        dist2 = np.sum((coords - np.array(center)[:, None, None, None])**2, axis=0)
        return dist2 <= radius**2
        
    elif shape_type == 'cube':
        size = kwargs.get('size', grid_size // 2)
        center = kwargs.get('center', [grid_size//2, grid_size//2, grid_size//2])
        
        mask = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
        x_start = max(0, center[0] - size // 2)
        x_end = min(grid_size, center[0] + size // 2)
        y_start = max(0, center[1] - size // 2)
        y_end = min(grid_size, center[1] + size // 2)
        z_start = max(0, center[2] - size // 2)
        z_end = min(grid_size, center[2] + size // 2)
        
        mask[x_start:x_end, y_start:y_end, z_start:z_end] = True
        return mask
        
    elif shape_type == 'cylinder':
        radius = kwargs.get('radius', grid_size // 4)
        height = kwargs.get('height', grid_size // 2)
        center = kwargs.get('center', [grid_size//2, grid_size//2, grid_size//2])
        
        coords = np.indices((grid_size, grid_size, grid_size))
        xy_dist2 = (coords[0] - center[0])**2 + (coords[1] - center[1])**2
        z_range = (coords[2] >= center[2] - height//2) & (coords[2] <= center[2] + height//2)
        
        return (xy_dist2 <= radius**2) & z_range
        
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")


if __name__ == "__main__":
    # Example usage
    env = BatchedSculpt3DEnvTF(grid_size=8, max_steps=100, n_envs=4)
    obs = env.reset()
    
    print(f"Environment initialized:")
    print(f"  Grid size: {env.G}")
    print(f"  Number of environments: {env.N}")
    print(f"  Max steps: {env.max_steps}")
    print(f"  Grid observation shape: {obs[0].shape}")
    print(f"  Coordinate observation shape: {obs[1].shape}")
    
    # Test a few steps
    for step in range(5):
        actions = tf.random.uniform([env.N], 0, 6, dtype=tf.int32)
        obs, rewards, done = env.step(actions)
        stats = env.get_stats()
        
        print(f"Step {step}: Avg reward = {tf.reduce_mean(rewards):.3f}, "
              f"Done = {tf.reduce_sum(tf.cast(done, tf.int32))}/{env.N}") 