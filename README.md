# GPU Batched Router ENV Leveraging a Feudal Agent

- State Representation: Hybrid - CNN processes Stock/Mask grids, concatenates
  with normalized XYZ coordinates.
- Agent Architecture: Feudal - High-level Manager, Low-level Worker.
- Manager: Sets a goal (relative displacement). Uses DQN.
- Worker: Takes primitive actions based on state + goal. Uses DQN with intrinsic reward.
- Networks: Use 3D CNN feature extractor + Dense heads with Noisy Layers.
- Exploration: Noisy Networks instead of epsilon-greedy.
- N envs stepped in parallel during training.
- Batched replay buffer insertion (separate for Manager and Worker).
- Includes periodic evaluation during training and final evaluation/rendering.
- Plots carving performance trend at the end of training.
- Periodic saving of model weights during training
