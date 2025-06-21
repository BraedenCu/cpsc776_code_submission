#!/usr/bin/env python3
"""
Replay Buffers for Feudal Networks
==================================

This module implements specialized replay buffers for hierarchical reinforcement learning
using feudal networks. It provides separate buffers for worker and manager agents
with appropriate data structures for their respective learning objectives.

Key Features:
- WorkerReplayBuffer: Stores low-level transitions for worker policy learning
- ManagerReplayBuffer: Stores high-level transitions for manager policy learning
- Efficient batch sampling and storage
- Support for hierarchical action spaces and rewards

Author: RL CNC Path Optimization Team
License: MIT
"""

import tensorflow as tf
import numpy as np
import random
from collections import deque
from typing import Tuple, Optional, List, Any


class WorkerReplayBuffer:
    """
    Replay buffer for worker agent transitions.
    
    Stores transitions of the form: (S_w_grid, S_w_coord, G_w_index, A_w, R_w_total, S2_w_grid, S2_w_coord, D_w)
    
    Where:
    - S_w_grid, S2_w_grid: Grid observations (stock and shape masks)
    - S_w_coord, S2_w_coord: Coordinate observations (normalized x, y, z)
    - G_w_index: Goal index from manager
    - A_w: Worker action (primitive action)
    - R_w_total: Total reward (extrinsic + intrinsic)
    - D_w: Done flag
    """
    
    def __init__(self, capacity: int = 50000):
        """
        Initialize worker replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.cap = capacity
        self.buf = deque(maxlen=capacity)
    
    def add_batch(self, S_tuple: Tuple[tf.Tensor, tf.Tensor], 
                  G_batch: tf.Tensor, A: tf.Tensor, R_total: tf.Tensor,
                  S2_tuple: Tuple[tf.Tensor, tf.Tensor], D: tf.Tensor):
        """
        Add a batch of transitions to the buffer.
        
        Args:
            S_tuple: (grid_obs, coord_obs) for current state [N, ...]
            G_batch: Goal index batch [N]
            A: Action batch [N]
            R_total: Total reward batch [N]
            S2_tuple: (grid_obs, coord_obs) for next state [N, ...]
            D: Done flag batch [N]
        """
        S_grid_batch, S_coord_batch = S_tuple
        S2_grid_batch, S2_coord_batch = S2_tuple
        N = tf.shape(A)[0].numpy()  # Get batch size
        
        # Store each transition individually
        for i in range(N):
            self.buf.append((
                S_grid_batch[i], S_coord_batch[i],  # state S (single env)
                G_batch[i],                         # goal G (single env)
                A[i],                               # action A (single env)
                R_total[i],                         # reward R (single env)
                S2_grid_batch[i], S2_coord_batch[i],  # next state S2 (single env)
                D[i]                                # done D (single env)
            ))
    
    def sample(self, batch_size: int = 32) -> Optional[Tuple]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            Tuple of batched tensors or None if insufficient data
        """
        if len(self.buf) < batch_size:
            return None
        
        batch = random.sample(self.buf, batch_size)
        
        # Unzip the batch
        S_grid_list, S_coord_list, G_list, A_list, R_list, S2_grid_list, S2_coord_list, D_list = zip(*batch)
        
        return (
            tf.stack(S_grid_list, axis=0), tf.stack(S_coord_list, axis=0),  # state S [B, ...]
            tf.stack(G_list, axis=0),                                       # goal G [B]
            tf.stack(A_list, axis=0),                                       # action A [B]
            tf.stack(R_list, axis=0),                                       # reward R [B]
            tf.stack(S2_grid_list, axis=0), tf.stack(S2_coord_list, axis=0),  # next state S2 [B, ...]
            tf.stack(D_list, axis=0)                                        # done D [B]
        )
    
    def __len__(self) -> int:
        """Return the number of transitions in the buffer."""
        return len(self.buf)
    
    def clear(self):
        """Clear all transitions from the buffer."""
        self.buf.clear()
    
    def get_stats(self) -> dict:
        """
        Get statistics about the buffer contents.
        
        Returns:
            dict: Statistics including size, capacity, etc.
        """
        return {
            'size': len(self.buf),
            'capacity': self.cap,
            'utilization': len(self.buf) / self.cap if self.cap > 0 else 0.0
        }


class ManagerReplayBuffer:
    """
    Replay buffer for manager agent transitions.
    
    Stores transitions of the form: (S_m_grid, S_m_coord, G, R_m_accumulated_extrinsic, S'_m_grid, S'_m_coord, D_m)
    
    Where:
    - S_m_grid, S'_m_grid: Grid observations at start/end of horizon
    - S_m_coord, S'_m_coord: Coordinate observations at start/end of horizon
    - G: Goal index selected by manager
    - R_m_accumulated_extrinsic: Accumulated extrinsic reward over horizon
    - D_m: Done flag (episode ended)
    """
    
    def __init__(self, capacity: int = 5000):
        """
        Initialize manager replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.cap = capacity
        self.buf = deque(maxlen=capacity)
    
    def add_batch(self, S_tuple: Tuple[tf.Tensor, tf.Tensor], G: tf.Tensor,
                  R_accumulated: tf.Tensor, S2_tuple: Tuple[tf.Tensor, tf.Tensor], D: tf.Tensor):
        """
        Add a batch of transitions to the buffer.
        
        Args:
            S_tuple: (grid_obs, coord_obs) at start of horizon [N, ...]
            G: Goal index batch [N]
            R_accumulated: Accumulated extrinsic reward batch [N]
            S2_tuple: (grid_obs, coord_obs) at end of horizon [N, ...]
            D: Done flag batch [N]
        """
        S_grid_batch, S_coord_batch = S_tuple
        S2_grid_batch, S2_coord_batch = S2_tuple
        N = tf.shape(G)[0].numpy()  # Get batch size
        
        # Store each transition individually
        for i in range(N):
            self.buf.append((
                S_grid_batch[i], S_coord_batch[i],  # state S (single env, start of horizon)
                G[i],                               # goal G (single env)
                R_accumulated[i],                   # accumulated reward R (single env)
                S2_grid_batch[i], S2_coord_batch[i],  # next state S2 (single env, end of horizon)
                D[i]                                # done D (single env, episode ended)
            ))
    
    def sample(self, batch_size: int = 32) -> Optional[Tuple]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            Tuple of batched tensors or None if insufficient data
        """
        if len(self.buf) < batch_size:
            return None
        
        batch = random.sample(self.buf, batch_size)
        
        # Unzip the batch
        S_grid_list, S_coord_list, G_list, R_list, S2_grid_list, S2_coord_list, D_list = zip(*batch)
        
        return (
            tf.stack(S_grid_list, axis=0), tf.stack(S_coord_list, axis=0),  # state S [B, ...]
            tf.stack(G_list, axis=0),                                       # goal G [B]
            tf.stack(R_list, axis=0),                                       # reward R [B]
            tf.stack(S2_grid_list, axis=0), tf.stack(S2_coord_list, axis=0),  # next state S2 [B, ...]
            tf.stack(D_list, axis=0)                                        # done D [B]
        )
    
    def __len__(self) -> int:
        """Return the number of transitions in the buffer."""
        return len(self.buf)
    
    def clear(self):
        """Clear all transitions from the buffer."""
        self.buf.clear()
    
    def get_stats(self) -> dict:
        """
        Get statistics about the buffer contents.
        
        Returns:
            dict: Statistics including size, capacity, etc.
        """
        return {
            'size': len(self.buf),
            'capacity': self.cap,
            'utilization': len(self.buf) / self.cap if self.cap > 0 else 0.0
        }


class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer with importance sampling.
    
    This is an optional enhancement that can be used instead of uniform sampling
    to prioritize transitions based on their TD-error magnitude.
    """
    
    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
            alpha (float): Priority exponent (0 = uniform, 1 = pure priority)
            beta (float): Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, transition: Tuple, priority: float = None):
        """
        Add a transition to the buffer.
        
        Args:
            transition: The transition tuple
            priority: Priority value (if None, use max priority)
        """
        if priority is None:
            priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List, List[int], np.ndarray]:
        """
        Sample a batch of transitions with importance sampling weights.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            Tuple of (transitions, indices, weights)
        """
        if self.size < batch_size:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get transitions
        transitions = [self.buffer[idx] for idx in indices]
        
        return transitions, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return self.size


def create_replay_buffer(buffer_type: str = 'uniform', **kwargs) -> Any:
    """
    Factory function to create replay buffers.
    
    Args:
        buffer_type (str): Type of buffer ('worker', 'manager', 'prioritized')
        **kwargs: Additional arguments for buffer initialization
        
    Returns:
        Replay buffer instance
    """
    if buffer_type == 'worker':
        return WorkerReplayBuffer(**kwargs)
    elif buffer_type == 'manager':
        return ManagerReplayBuffer(**kwargs)
    elif buffer_type == 'prioritized':
        return PrioritizedReplayBuffer(**kwargs)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")


if __name__ == "__main__":
    # Example usage
    print("Testing replay buffers...")
    
    # Test worker buffer
    worker_buffer = WorkerReplayBuffer(capacity=1000)
    
    # Create dummy data
    batch_size = 4
    grid_shape = (8, 8, 8, 2)
    coord_shape = (3,)
    
    S_grid = tf.random.normal([batch_size] + list(grid_shape))
    S_coord = tf.random.normal([batch_size] + list(coord_shape))
    G_batch = tf.random.uniform([batch_size], 0, 27, dtype=tf.int32)
    A = tf.random.uniform([batch_size], 0, 6, dtype=tf.int32)
    R_total = tf.random.normal([batch_size])
    D = tf.random.uniform([batch_size], 0, 2, dtype=tf.bool)
    
    # Add to buffer
    worker_buffer.add_batch(
        (S_grid, S_coord), G_batch, A, R_total, (S_grid, S_coord), D
    )
    
    print(f"Worker buffer size: {len(worker_buffer)}")
    print(f"Worker buffer stats: {worker_buffer.get_stats()}")
    
    # Test sampling
    sample = worker_buffer.sample(batch_size=2)
    if sample is not None:
        print(f"Sampled batch shapes: {[t.shape for t in sample]}")
    
    # Test manager buffer
    manager_buffer = ManagerReplayBuffer(capacity=100)
    
    manager_buffer.add_batch(
        (S_grid, S_coord), G_batch, R_total, (S_grid, S_coord), D
    )
    
    print(f"Manager buffer size: {len(manager_buffer)}")
    print(f"Manager buffer stats: {manager_buffer.get_stats()}")
    
    print("Replay buffer tests completed!") 