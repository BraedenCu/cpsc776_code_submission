#!/usr/bin/env python3
"""
Neural Network Architectures for Feudal Networks
================================================

This module implements the neural network architectures for hierarchical reinforcement
learning using feudal networks. It includes specialized networks for manager and worker
agents, along with exploration mechanisms like noisy layers.

Key Features:
- Manager Network: High-level policy for goal selection
- Worker Network: Low-level policy for primitive actions
- NoisyDense Layer: Factorized Gaussian noise for exploration
- 3D CNN architectures for grid observations
- Target networks for stable training

Author: RL CNC Path Optimization Team
License: MIT
"""

import tensorflow as tf
import numpy as np
import math
from typing import Tuple, Optional, Callable


class NoisyDense(tf.keras.layers.Layer):
    """
    Dense layer with factorized Gaussian noise for exploration.
    
    This layer implements the factorized Gaussian noise technique from
    "Noisy Networks for Exploration" (Fortunato et al., 2017) to enable
    exploration without requiring epsilon-greedy strategies.
    
    The noise is generated using the factorized Gaussian formula:
    noise = sign(ε) * sqrt(|ε|) where ε ~ N(0, 1)
    """
    
    def __init__(self, units: int, activation: Optional[str] = None, 
                 sigma0: float = 0.5, **kwargs):
        """
        Initialize noisy dense layer.
        
        Args:
            units (int): Number of output units
            activation (str, optional): Activation function
            sigma0 (float): Initial noise standard deviation
            **kwargs: Additional arguments for tf.keras.layers.Layer
        """
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.sigma0 = sigma0
    
    def build(self, input_shape):
        """Build the layer by creating trainable weights."""
        in_features = input_shape[-1]
        out_features = self.units
        dtype = tf.float32
        
        # Weight parameters (mean and standard deviation)
        sigma_init_val = self.sigma0 / math.sqrt(float(in_features))
        sigma_initializer = tf.constant_initializer(sigma_init_val)
        
        self.kernel_mean = self.add_weight(
            name="kernel_mean",
            shape=(in_features, out_features),
            initializer="he_uniform",
            trainable=True,
            dtype=dtype
        )
        self.kernel_sigma = self.add_weight(
            name="kernel_sigma",
            shape=(in_features, out_features),
            initializer=sigma_initializer,
            trainable=True,
            dtype=dtype
        )
        
        # Bias parameters (mean and standard deviation)
        self.bias_mean = self.add_weight(
            name="bias_mean",
            shape=(out_features,),
            initializer="zeros",
            trainable=True,
            dtype=dtype
        )
        self.bias_sigma = self.add_weight(
            name="bias_sigma",
            shape=(out_features,),
            initializer=sigma_initializer,
            trainable=True,
            dtype=dtype
        )
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Forward pass with optional noise injection.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor with optional noise
        """
        if training:
            # Generate noise for input and output dimensions
            noise_in = self._factorized_noise(tf.shape(inputs)[-1])
            noise_out = self._factorized_noise(self.units)
            
            # Combine noise for weight matrix: outer product
            kernel_noise = tf.tensordot(
                tf.expand_dims(noise_in, -1), 
                tf.expand_dims(noise_out, 0), 
                axes=1
            )
            
            # Noise for bias is just the output noise
            bias_noise = noise_out
            
            # Apply noise: W = W_mu + W_sigma * noise_W, b = b_mu + b_sigma * noise_b
            kernel = self.kernel_mean + self.kernel_sigma * kernel_noise
            bias = self.bias_mean + self.bias_sigma * bias_noise
        else:
            # In inference mode use only the mean weights and biases
            kernel = self.kernel_mean
            bias = self.bias_mean
        
        output = tf.matmul(inputs, kernel) + bias
        
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    @tf.function
    def _factorized_noise(self, num_elements: int) -> tf.Tensor:
        """
        Generate factorized Gaussian noise.
        
        Args:
            num_elements (int): Number of elements to generate noise for
            
        Returns:
            Noise tensor of shape [num_elements]
        """
        noise = tf.random.normal(shape=[num_elements], dtype=tf.float32)
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'sigma0': self.sigma0
        })
        return config


def build_manager_network(grid_shape: Tuple[int, ...], coord_shape: Tuple[int, ...],
                         goal_dim: int, name: str = "Manager_Network") -> tf.keras.Model:
    """
    Build the manager network for high-level goal selection.
    
    The manager network takes the current state and outputs Q-values for each
    possible goal. It uses a 3D CNN to process grid observations and combines
    them with coordinate observations.
    
    Args:
        grid_shape (tuple): Shape of grid observations (G, G, G, 2)
        coord_shape (tuple): Shape of coordinate observations (3,)
        goal_dim (int): Number of possible goals
        name (str): Model name
        
    Returns:
        tf.keras.Model: Manager network
    """
    grid_input = tf.keras.layers.Input(shape=grid_shape, name="manager_grid_input")
    coord_input = tf.keras.layers.Input(shape=coord_shape, name="manager_coord_input")
    
    # CNN part for grid observations
    x_cnn = tf.keras.layers.Conv3D(
        filters=32, kernel_size=5, strides=2, activation='relu', 
        padding='same', name="m_conv1"
    )(grid_input)
    x_cnn = tf.keras.layers.Conv3D(
        filters=64, kernel_size=3, strides=2, activation='relu', 
        padding='same', name="m_conv2"
    )(x_cnn)
    x_cnn = tf.keras.layers.Conv3D(
        filters=64, kernel_size=3, strides=1, activation='relu', 
        padding='same', name="m_conv3"
    )(x_cnn)
    cnn_features = tf.keras.layers.Flatten(name="m_flatten")(x_cnn)
    
    # Concatenate CNN features with coordinate observations
    concat_features = tf.keras.layers.Concatenate(name="m_concat")(
        [cnn_features, coord_input]
    )
    
    # Dense layers with noisy exploration
    x = NoisyDense(256, activation='relu', name="manager_dense1")(concat_features)
    outputs = NoisyDense(goal_dim, activation='linear', name="manager_output")(x)
    
    return tf.keras.Model(
        inputs=[grid_input, coord_input], 
        outputs=outputs, 
        name=name
    )


def build_worker_network(grid_shape: Tuple[int, ...], coord_shape: Tuple[int, ...],
                        goal_dim: int, action_dim: int, 
                        name: str = "Worker_Network") -> tf.keras.Model:
    """
    Build the worker network for low-level primitive actions.
    
    The worker network takes the current state and goal (one-hot encoded)
    and outputs Q-values for each primitive action. It uses a 3D CNN for
    grid observations and processes goals through a separate branch.
    
    Args:
        grid_shape (tuple): Shape of grid observations (G, G, G, 2)
        coord_shape (tuple): Shape of coordinate observations (3,)
        goal_dim (int): Number of possible goals
        action_dim (int): Number of primitive actions
        name (str): Model name
        
    Returns:
        tf.keras.Model: Worker network
    """
    grid_input = tf.keras.layers.Input(shape=grid_shape, name="worker_grid_input")
    coord_input = tf.keras.layers.Input(shape=coord_shape, name="worker_coord_input")
    goal_input = tf.keras.layers.Input(shape=(goal_dim,), name="worker_goal_input")
    
    # CNN part for grid observations
    x_cnn = tf.keras.layers.Conv3D(
        filters=32, kernel_size=5, strides=2, activation='relu', 
        padding='same', name="w_conv1"
    )(grid_input)
    x_cnn = tf.keras.layers.Conv3D(
        filters=64, kernel_size=3, strides=2, activation='relu', 
        padding='same', name="w_conv2"
    )(x_cnn)
    x_cnn = tf.keras.layers.Conv3D(
        filters=64, kernel_size=3, strides=1, activation='relu', 
        padding='same', name="w_conv3"
    )(x_cnn)
    cnn_features = tf.keras.layers.Flatten(name="w_flatten")(x_cnn)
    
    # Process goal through a separate branch
    goal_features = tf.keras.layers.Dense(64, activation='relu', name="goal_dense")(goal_input)
    
    # Concatenate all features
    concat_features = tf.keras.layers.Concatenate(name="w_concat")(
        [cnn_features, coord_input, goal_features]
    )
    
    # Dense layers with noisy exploration
    x = NoisyDense(256, activation='relu', name="worker_dense1")(concat_features)
    x = NoisyDense(128, activation='relu', name="worker_dense2")(x)
    outputs = NoisyDense(action_dim, activation='linear', name="worker_output")(x)
    
    return tf.keras.Model(
        inputs=[grid_input, coord_input, goal_input], 
        outputs=outputs, 
        name=name
    )


def build_target_network(source_model: tf.keras.Model, 
                        name: str = None) -> tf.keras.Model:
    """
    Build a target network by copying the architecture and weights of a source model.
    
    Args:
        source_model (tf.keras.Model): Source model to copy
        name (str, optional): Name for the target model
        
    Returns:
        tf.keras.Model: Target network with copied weights
    """
    if name is None:
        name = f"{source_model.name}_target"
    
    # Clone the model architecture
    target_model = tf.keras.models.clone_model(source_model)
    target_model.set_weights(source_model.get_weights())
    target_model._name = name
    
    return target_model


def update_target_network(target_model: tf.keras.Model, 
                         source_model: tf.keras.Model, 
                         tau: float = 0.005):
    """
    Soft update of target network weights.
    
    Args:
        target_model (tf.keras.Model): Target network to update
        source_model (tf.keras.Model): Source network
        tau (float): Soft update coefficient (0 = no update, 1 = hard update)
    """
    target_weights = target_model.get_weights()
    source_weights = source_model.get_weights()
    
    new_weights = []
    for target_w, source_w in zip(target_weights, source_weights):
        new_w = tau * source_w + (1 - tau) * target_w
        new_weights.append(new_w)
    
    target_model.set_weights(new_weights)


def create_feudal_networks(grid_shape: Tuple[int, ...], coord_shape: Tuple[int, ...],
                          goal_dim: int, action_dim: int,
                          use_target_networks: bool = True,
                          tau: float = 0.005) -> dict:
    """
    Create all networks for the feudal agent.
    
    Args:
        grid_shape (tuple): Shape of grid observations
        coord_shape (tuple): Shape of coordinate observations
        goal_dim (int): Number of possible goals
        action_dim (int): Number of primitive actions
        use_target_networks (bool): Whether to create target networks
        tau (float): Soft update coefficient for target networks
        
    Returns:
        dict: Dictionary containing all networks
    """
    networks = {}
    
    # Create main networks
    networks['manager'] = build_manager_network(grid_shape, coord_shape, goal_dim)
    networks['worker'] = build_worker_network(grid_shape, coord_shape, goal_dim, action_dim)
    
    # Create target networks if requested
    if use_target_networks:
        networks['manager_target'] = build_target_network(networks['manager'])
        networks['worker_target'] = build_target_network(networks['worker'])
        networks['tau'] = tau
    
    return networks


def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get a formatted summary of model parameters.
    
    Args:
        model (tf.keras.Model): Model to summarize
        
    Returns:
        str: Formatted summary string
    """
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    return '\n'.join(stringlist)


if __name__ == "__main__":
    # Example usage
    print("Testing neural network architectures...")
    
    # Define shapes
    grid_shape = (8, 8, 8, 2)
    coord_shape = (3,)
    goal_dim = 27
    action_dim = 6
    
    # Create networks
    networks = create_feudal_networks(grid_shape, coord_shape, goal_dim, action_dim)
    
    print(f"Manager network:")
    print(get_model_summary(networks['manager']))
    
    print(f"\nWorker network:")
    print(get_model_summary(networks['worker']))
    
    # Test forward pass
    batch_size = 4
    grid_input = tf.random.normal([batch_size] + list(grid_shape))
    coord_input = tf.random.normal([batch_size] + list(coord_shape))
    goal_input = tf.one_hot(tf.random.uniform([batch_size], 0, goal_dim, dtype=tf.int32), goal_dim)
    
    # Manager forward pass
    manager_output = networks['manager']([grid_input, coord_input])
    print(f"\nManager output shape: {manager_output.shape}")
    
    # Worker forward pass
    worker_output = networks['worker']([grid_input, coord_input, goal_input])
    print(f"Worker output shape: {worker_output.shape}")
    
    # Test target network update
    if 'manager_target' in networks:
        print(f"\nTesting target network update...")
        old_weights = networks['manager_target'].get_weights()[0][0, 0]
        update_target_network(networks['manager_target'], networks['manager'], tau=0.1)
        new_weights = networks['manager_target'].get_weights()[0][0, 0]
        print(f"Weight change: {abs(new_weights - old_weights):.6f}")
    
    print("\nNeural network tests completed!") 