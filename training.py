#!/usr/bin/env python3
"""
Training and Evaluation Module for Feudal Networks
==================================================

This module provides comprehensive training and evaluation functionality for
the feudal network agent in CNC path optimization.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, List, Optional, Tuple, Any

from environment import BatchedSculpt3DEnvTF
from feudal_agent import FeudalAgentTF, calculate_intrinsic_reward_batch


def evaluate_agent_performance(agent: FeudalAgentTF, grid_size: int, max_steps: int,
                              num_eval_episodes: int = 10, render: bool = True,
                              render_env_index: int = 0) -> Dict[str, float]:
    """Evaluate trained feudal agent performance."""
    print(f"\n**** Running Evaluation ({num_eval_episodes} episodes) ****")
    eval_start_time = time.time()
    
    # Create evaluation environment
    eval_env = BatchedSculpt3DEnvTF(grid_size=grid_size, max_steps=max_steps, n_envs=num_eval_episodes)
    N_eval = num_eval_episodes
    
    # Get initial shape information
    initial_shape_mask_flat_gpu = eval_env.shape_mask[0]
    initial_shape_mask_flat_np = initial_shape_mask_flat_gpu.numpy()
    initial_carvable_mask_flat = ~initial_shape_mask_flat_np
    initial_carvable_count = np.sum(initial_carvable_mask_flat)
    print(f"  Initial number of carvable voxels: {initial_carvable_count}")
    
    if initial_carvable_count == 0:
        print("  ERROR: No carvable material defined.")
        return {}
    
    # Initialize tracking variables
    all_ep_rewards = []
    all_ep_lengths = []
    all_ep_removed_counts = []
    all_ep_incorrect_removed_counts = []
    
    # Initialize agent state
    current_goals = tf.Variable(tf.zeros([N_eval], dtype=tf.int32), trainable=False, name="eval_goals")
    subgoal_steps_remaining = tf.Variable(tf.zeros([N_eval], dtype=tf.int32), trainable=False, name="eval_subgoal_steps")
    
    # Reset environment
    obs_tuple = eval_env.reset()
    done = eval_env.done
    
    ep_rewards = tf.Variable(tf.zeros([N_eval], dtype=tf.float32), trainable=False, name="eval_ep_rewards")
    ep_steps = tf.Variable(tf.zeros([N_eval], dtype=tf.int32), trainable=False, name="eval_ep_steps")
    
    # Get initial goals from manager
    manager_actions_initial = agent.manager_act_batch(obs_tuple, deterministic=True)
    current_goals.assign(manager_actions_initial)
    subgoal_steps_remaining.assign(tf.constant(agent.subgoal_horizon, dtype=tf.int32, shape=[N_eval]))
    
    # Run evaluation
    for _ in range(max_steps):
        # Check if manager should update goals
        end_of_horizon_mask = tf.equal(subgoal_steps_remaining.read_value(), 0)
        episode_done_mask = done
        manager_update_mask = tf.logical_or(end_of_horizon_mask, episode_done_mask)
        
        # Find environments that need a manager update
        masked_indices = tf.where(manager_update_mask)[:, 0]
        num_masked = tf.shape(masked_indices)[0]
        
        # Manager acts only if there are new goals needed
        if num_masked > 0:
            manager_states_grid = tf.gather(obs_tuple[0], masked_indices)
            manager_states_coord = tf.gather(obs_tuple[1], masked_indices)
            manager_states = (manager_states_grid, manager_states_coord)
            
            new_goals_selected = agent.manager_act_batch(manager_states, deterministic=True)
            
            current_goals.assign(tf.tensor_scatter_nd_update(
                current_goals.read_value(), 
                tf.expand_dims(masked_indices, axis=1), 
                new_goals_selected
            ))
            
            subgoal_steps_remaining.assign(tf.tensor_scatter_nd_update(
                subgoal_steps_remaining.read_value(), 
                tf.expand_dims(masked_indices, axis=1),
                tf.fill([num_masked], agent.subgoal_horizon)
            ))
        
        # Worker step - always act for all environments
        A = agent.worker_act_batch(obs_tuple, current_goals.read_value(), deterministic=True)
        
        # Step the environment
        S2_tuple, R, next_done = eval_env.step(A)
        
        # Update rewards and steps only for environments not yet done
        active_mask_current = ~done
        ep_rewards.assign_add(R * tf.cast(active_mask_current, tf.float32))
        ep_steps.assign_add(tf.cast(active_mask_current, tf.int32))
        
        # Decrement subgoal steps for active environments
        active_mask_next = ~next_done
        subgoal_steps_remaining.assign(tf.where(
            active_mask_next, 
            subgoal_steps_remaining.read_value() - 1, 
            subgoal_steps_remaining.read_value()
        ))
        
        obs_tuple = S2_tuple
        done = next_done
        
        if tf.reduce_all(done):
            break
    
    # Calculate final statistics
    final_stock_batch_np = eval_env.stock.numpy()
    batch_rewards = ep_rewards.numpy()
    batch_lengths = ep_steps.numpy()
    
    all_ep_rewards.extend(batch_rewards.tolist())
    all_ep_lengths.extend(batch_lengths.tolist())
    
    # Calculate removed/incorrect counts per episode
    for i in range(N_eval):
        final_stock_flat_np = final_stock_batch_np[i]
        
        # Correctly removed: carvable and now carved by model
        removed_mask = initial_carvable_mask_flat & (~final_stock_flat_np)
        removed_count = np.sum(removed_mask)
        all_ep_removed_counts.append(removed_count)
        
        # Incorrectly removed: part of shape and incorrectly cut by model
        incorrectly_removed_mask = initial_shape_mask_flat_np & (~final_stock_flat_np)
        incorrectly_removed_count = np.sum(incorrectly_removed_mask)
        all_ep_incorrect_removed_counts.append(incorrectly_removed_count)
    
    # Calculate summary statistics
    avg_reward = np.mean(all_ep_rewards)
    std_reward = np.std(all_ep_rewards)
    avg_length = np.mean(all_ep_lengths)
    avg_removed_count = np.mean(all_ep_removed_counts)
    avg_incorrect_removed = np.mean(all_ep_incorrect_removed_counts)
    
    if initial_carvable_count > 0:
        removal_percentages = [(c / initial_carvable_count) * 100.0 for c in all_ep_removed_counts]
        avg_removal_percentage = np.mean(removal_percentages)
        std_removal_percentage = np.std(removal_percentages)
    else:
        avg_removal_percentage = 0.0
        std_removal_percentage = 0.0
    
    # Print results
    print(f"\n*** Evaluation Results ***")
    print(f"  Avg Reward: {avg_reward:.2f} (+/- {std_reward:.2f})")
    print(f"  Avg Length: {avg_length:.1f}")
    print(f"  Avg Removed: {avg_removed_count:.1f} / {initial_carvable_count} ({avg_removal_percentage:.2f}% +/- {std_removal_percentage:.2f}%)")
    print(f"  Avg Incorrect: {avg_incorrect_removed:.1f}")
    
    eval_duration = time.time() - eval_start_time
    print(f"  Evaluation Duration: {eval_duration:.2f}s")
    
    return {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
        "avg_removed_count": avg_removed_count,
        "avg_incorrect_removed": avg_incorrect_removed,
        "avg_removal_percentage": avg_removal_percentage,
        "std_removal_percentage": std_removal_percentage,
        "initial_carvable_count": initial_carvable_count
    }


def train_feudal_agent(grid_size: int = 16, max_steps: int = 300, n_envs: int = 32,
                      episodes: int = 10000, worker_buffer_capacity: int = 100000,
                      manager_buffer_capacity: int = 10000, worker_learn_batch_size: int = 32,
                      manager_learn_batch_size: int = 32, worker_learn_freq: int = 4,
                      manager_learn_freq: int = 100, gamma: float = 0.99,
                      manager_lr: float = 1e-4, worker_lr: float = 1e-4, tau: float = 0.005,
                      subgoal_horizon: int = 10, intrinsic_reward_beta: float = 0.1,
                      manager_goal_k: int = 1, log_every: int = 50, evaluate_every: int = 100,
                      num_eval_episodes_periodic: int = 10, render_intermediate_eval: bool = False,
                      save_every_episodes: int = 500, checkpoint_dir: str = "checkpoints_feudal") -> FeudalAgentTF:
    """Train a feudal agent for CNC path optimization."""
    print(f"*** Training Feudal Agent ***")
    print(f"Params: Grid={grid_size}, N_Envs={n_envs}, MaxSteps={max_steps}, Episodes={episodes}")
    print(f"Subgoal Horizon: {subgoal_horizon}, Intrinsic Beta: {intrinsic_reward_beta}, Goal K: {manager_goal_k}")
    
    # Create environment and agent
    env = BatchedSculpt3DEnvTF(grid_size, max_steps, n_envs)
    agent = FeudalAgentTF(
        grid_shape=env.grid_obs_shape, coord_shape=env.coord_obs_shape,
        primitive_action_dim=6, manager_goal_k=manager_goal_k,
        subgoal_horizon=subgoal_horizon, intrinsic_reward_beta=intrinsic_reward_beta,
        manager_lr=manager_lr, worker_lr=worker_lr, gamma=gamma, tau=tau,
        worker_buffer_capacity=worker_buffer_capacity,
        manager_buffer_capacity=manager_buffer_capacity
    )
    
    # Training tracking variables
    start_time = time.time()
    total_env_steps_taken = 0
    
    episode_rewards_history = []
    episode_lengths_history = []
    worker_losses_history = []
    manager_losses_history = []
    
    eval_episodes_list = []
    eval_avg_rewards_list = []
    eval_avg_removal_perc_list = []
    
    # Training loop
    for ep in range(1, episodes + 1):
        # Initialize episode variables
        current_goals = tf.Variable(tf.zeros([n_envs], dtype=tf.int32), trainable=False, name="ep_goals")
        subgoal_steps_remaining = tf.Variable(tf.zeros([n_envs], dtype=tf.int32), trainable=False, name="ep_subgoal_steps")
        current_subgoal_start_pos = tf.Variable(tf.zeros([n_envs], dtype=tf.int32), trainable=False, name="ep_subgoal_start_pos")
        current_subgoal_start_obs_grid = tf.Variable(tf.zeros([n_envs] + list(env.grid_obs_shape)), trainable=False, name="ep_subgoal_start_obs_grid")
        current_subgoal_start_obs_coord = tf.Variable(tf.zeros([n_envs] + list(env.coord_obs_shape)), trainable=False, name="ep_subgoal_start_obs_coord")
        accumulated_extrinsic_rewards_current_horizon = tf.Variable(tf.zeros([n_envs], dtype=tf.float32), trainable=False, name="ep_accumulated_rewards")
        has_valid_prev_manager_transition_data = tf.Variable(tf.zeros([n_envs], dtype=tf.bool), trainable=False, name="ep_has_valid_prev_manager_transition")
        
        ep_rewards = tf.Variable(tf.zeros([n_envs], dtype=tf.float32), trainable=False, name="ep_rewards")
        ep_steps = tf.Variable(tf.zeros([n_envs], dtype=tf.int32), trainable=False, name="ep_steps")
        
        # Reset environment
        obs_tuple = env.reset()
        done = env.done
        
        # Get initial goals from manager
        manager_actions_initial = agent.manager_act_batch(obs_tuple, deterministic=False)
        current_goals.assign(manager_actions_initial)
        subgoal_steps_remaining.assign(tf.constant(subgoal_horizon, dtype=tf.int32, shape=[n_envs]))
        current_subgoal_start_pos.assign(env.pos)
        current_subgoal_start_obs_grid.assign(obs_tuple[0])
        current_subgoal_start_obs_coord.assign(obs_tuple[1])
        accumulated_extrinsic_rewards_current_horizon.assign(tf.zeros([n_envs], dtype=tf.float32))
        has_valid_prev_manager_transition_data.assign(tf.zeros([n_envs], dtype=tf.bool))
        
        # Episode step loop
        for step in range(max_steps):
            # Check if manager should update goals
            end_of_horizon_mask = tf.equal(subgoal_steps_remaining.read_value(), 0)
            episode_done_mask = done
            manager_update_mask = tf.logical_or(end_of_horizon_mask, episode_done_mask)
            
            # Find environments that need manager updates
            masked_indices = tf.where(manager_update_mask)[:, 0]
            num_masked = tf.shape(masked_indices)[0]
            
            # Manager acts for environments needing new goals
            if num_masked > 0:
                manager_states_grid = tf.gather(obs_tuple[0], masked_indices)
                manager_states_coord = tf.gather(obs_tuple[1], masked_indices)
                manager_states = (manager_states_grid, manager_states_coord)
                
                new_goals_selected = agent.manager_act_batch(manager_states, deterministic=False)
                
                # Update goals and reset subgoal tracking
                current_goals.assign(tf.tensor_scatter_nd_update(
                    current_goals.read_value(), 
                    tf.expand_dims(masked_indices, axis=1), 
                    new_goals_selected
                ))
                
                subgoal_steps_remaining.assign(tf.tensor_scatter_nd_update(
                    subgoal_steps_remaining.read_value(), 
                    tf.expand_dims(masked_indices, axis=1),
                    tf.fill([num_masked], subgoal_horizon)
                ))
                
                current_subgoal_start_pos.assign(tf.tensor_scatter_nd_update(
                    current_subgoal_start_pos.read_value(), 
                    tf.expand_dims(masked_indices, axis=1),
                    tf.gather(env.pos, masked_indices)
                ))
                
                current_subgoal_start_obs_grid.assign(tf.tensor_scatter_nd_update(
                    current_subgoal_start_obs_grid.read_value(), 
                    tf.expand_dims(masked_indices, axis=1),
                    tf.gather(obs_tuple[0], masked_indices)
                ))
                
                current_subgoal_start_obs_coord.assign(tf.tensor_scatter_nd_update(
                    current_subgoal_start_obs_coord.read_value(), 
                    tf.expand_dims(masked_indices, axis=1),
                    tf.gather(obs_tuple[1], masked_indices)
                ))
                
                accumulated_extrinsic_rewards_current_horizon.assign(tf.tensor_scatter_nd_update(
                    accumulated_extrinsic_rewards_current_horizon.read_value(), 
                    tf.expand_dims(masked_indices, axis=1),
                    tf.zeros([num_masked], dtype=tf.float32)
                ))
                
                has_valid_prev_manager_transition_data.assign(tf.tensor_scatter_nd_update(
                    has_valid_prev_manager_transition_data.read_value(), 
                    tf.expand_dims(masked_indices, axis=1),
                    tf.ones([num_masked], dtype=tf.bool)
                ))
            
            # Worker acts for all environments
            A = agent.worker_act_batch(obs_tuple, current_goals.read_value(), deterministic=False)
            
            # Step environment
            S2_tuple, R_extrinsic, next_done = env.step(A)
            
            # Calculate intrinsic rewards
            R_intrinsic = calculate_intrinsic_reward_batch(
                env.pos, env.pos,  # Current positions (will be updated)
                current_subgoal_start_pos.read_value(),
                current_goals.read_value(),
                env.G, agent.manager_goal_k
            )
            
            # Total reward
            R_total = R_extrinsic + intrinsic_reward_beta * R_intrinsic
            
            # Accumulate extrinsic rewards for manager
            accumulated_extrinsic_rewards_current_horizon.assign_add(R_extrinsic)
            
            # Store transitions
            agent.worker_remember_batch(obs_tuple, current_goals.read_value(), A, R_total, S2_tuple, next_done)
            
            # Store manager transitions for environments that completed horizons
            manager_transition_mask = tf.logical_and(
                has_valid_prev_manager_transition_data.read_value(),
                tf.logical_or(end_of_horizon_mask, episode_done_mask)
            )
            
            if tf.reduce_any(manager_transition_mask):
                manager_indices = tf.where(manager_transition_mask)[:, 0]
                
                manager_S_grid = tf.gather(current_subgoal_start_obs_grid.read_value(), manager_indices)
                manager_S_coord = tf.gather(current_subgoal_start_obs_coord.read_value(), manager_indices)
                manager_G = tf.gather(current_goals.read_value(), manager_indices)
                manager_R_accumulated = tf.gather(accumulated_extrinsic_rewards_current_horizon.read_value(), manager_indices)
                manager_S2_grid = tf.gather(obs_tuple[0], manager_indices)
                manager_S2_coord = tf.gather(obs_tuple[1], manager_indices)
                manager_D = tf.gather(next_done, manager_indices)
                
                agent.manager_remember_batch(
                    (manager_S_grid, manager_S_coord), manager_G, manager_R_accumulated,
                    (manager_S2_grid, manager_S2_coord), manager_D
                )
            
            # Update episode tracking
            active_mask_current = ~done
            ep_rewards.assign_add(R_total * tf.cast(active_mask_current, tf.float32))
            ep_steps.assign_add(tf.cast(active_mask_current, tf.int32))
            
            # Decrement subgoal steps
            active_mask_next = ~next_done
            subgoal_steps_remaining.assign(tf.where(
                active_mask_next, 
                subgoal_steps_remaining.read_value() - 1, 
                subgoal_steps_remaining.read_value()
            ))
            
            # Update state
            obs_tuple = S2_tuple
            done = next_done
            
            total_env_steps_taken += n_envs
            
            # Worker learning
            if total_env_steps_taken > 0 and total_env_steps_taken % (worker_learn_freq * n_envs) == 0:
                worker_loss_val = agent.worker_learn(worker_learn_batch_size)
                if worker_loss_val is not None:
                    worker_losses_history.append(worker_loss_val)
            
            # Manager learning
            if agent.worker_train_step_count.numpy() > 0 and agent.worker_train_step_count.numpy() % manager_learn_freq == 0:
                manager_loss_val = agent.manager_learn(manager_learn_batch_size)
                if manager_loss_val is not None:
                    manager_losses_history.append(manager_loss_val)
            
            if tf.reduce_all(done):
                break
        
        # Calculate episode statistics
        avg_reward_batch = tf.reduce_mean(ep_rewards).numpy()
        avg_steps_batch = tf.reduce_mean(tf.cast(ep_steps, tf.float32)).numpy()
        episode_rewards_history.append(avg_reward_batch)
        episode_lengths_history.append(avg_steps_batch)
        
        # Logging
        if ep % log_every == 0 or ep == 1:
            elapsed_time = time.time() - start_time
            avg_r = np.mean(episode_rewards_history[-log_every:]) if len(episode_rewards_history) >= log_every else np.mean(episode_rewards_history)
            avg_l = np.mean(episode_lengths_history[-log_every:]) if len(episode_lengths_history) >= log_every else np.mean(episode_lengths_history)
            avg_worker_loss = np.mean(worker_losses_history) if worker_losses_history else 0.0
            avg_manager_loss = np.mean(manager_losses_history) if manager_losses_history else 0.0
            
            print(f"Ep {ep}/{episodes} | Avg R (last {log_every}): {avg_r:.2f} | Avg Len: {avg_l:.1f} | "
                  f"EnvSteps: {total_env_steps_taken} | WorkerSteps: {agent.worker_train_step_count.numpy()} | "
                  f"ManagerSteps: {agent.manager_train_step_count.numpy()} | Time: {elapsed_time:.1f}s")
            print(f"  Losses: Worker {avg_worker_loss:.4f}, Manager {avg_manager_loss:.4f}")
        
        # Model saving
        if save_every_episodes > 0 and ep % save_every_episodes == 0 and ep > 0:
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                manager_save_path = os.path.join(checkpoint_dir, f"manager_ep{ep}_g{grid_size}.weights.h5")
                worker_save_path = os.path.join(checkpoint_dir, f"worker_ep{ep}_g{grid_size}.weights.h5")
                agent.manager_model.save_weights(manager_save_path)
                agent.worker_model.save_weights(worker_save_path)
                print(f"\n--- Saved model weights at episode {ep} to {checkpoint_dir} ---")
            except Exception as e:
                print(f"\n--- Error saving weights at episode {ep}: {e} ---")
        
        # Evaluation
        if evaluate_every > 0 and ep % evaluate_every == 0:
            eval_stats = evaluate_agent_performance(
                agent=agent, grid_size=grid_size, max_steps=max_steps,
                num_eval_episodes=num_eval_episodes_periodic,
                render=render_intermediate_eval, render_env_index=0
            )
            
            if eval_stats:
                eval_episodes_list.append(ep)
                eval_avg_rewards_list.append(eval_stats["avg_reward"])
                eval_avg_removal_perc_list.append(eval_stats["avg_removal_percentage"])
            
            print("*" * 60)
    
    # Close TensorBoard writer
    agent.writer.close()
    
    # Print final statistics
    total_training_time = time.time() - start_time
    print(f"\nTraining finished. Total env steps: {total_env_steps_taken}, Total Time: {total_training_time:.2f}s")
    print(f"Worker train steps: {agent.worker_train_step_count.numpy()}, Manager train steps: {agent.manager_train_step_count.numpy()}")
    
    return agent


if __name__ == "__main__":
    # Example training configuration
    print("Testing training module...")
    
    # Small training run for testing
    trained_agent = train_feudal_agent(
        grid_size=8,
        max_steps=100,
        n_envs=4,
        episodes=10,
        worker_buffer_capacity=1000,
        manager_buffer_capacity=100,
        worker_learn_batch_size=8,
        manager_learn_batch_size=8,
        worker_learn_freq=2,
        manager_learn_freq=5,
        log_every=2,
        evaluate_every=5,
        save_every_episodes=5
    )
    
    print("Training test completed!") 