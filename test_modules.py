#!/usr/bin/env python3
"""
Test Script: RL CNC Path Optimization Modules
=============================================

This script tests that all modules can be imported and basic functionality works.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    # Test our custom modules
    try:
        from environment import BatchedSculpt3DEnvTF, create_custom_shape_mask
        print("✓ Environment module")
    except ImportError as e:
        print(f"✗ Environment module import failed: {e}")
        return False
    
    try:
        from replay_buffers import WorkerReplayBuffer, ManagerReplayBuffer
        print("✓ Replay buffers module")
    except ImportError as e:
        print(f"✗ Replay buffers module import failed: {e}")
        return False
    
    try:
        from networks import NoisyDense, build_manager_network, build_worker_network
        print("✓ Networks module")
    except ImportError as e:
        print(f"✗ Networks module import failed: {e}")
        return False
    
    try:
        from feudal_agent import FeudalAgentTF
        print("✓ Feudal agent module")
    except ImportError as e:
        print(f"✗ Feudal agent module import failed: {e}")
        return False
    
    try:
        from training import train_feudal_agent, evaluate_agent_performance
        print("✓ Training module")
    except ImportError as e:
        print(f"✗ Training module import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of each module."""
    print("\nTesting basic functionality...")
    
    try:
        # Test environment
        from environment import BatchedSculpt3DEnvTF
        env = BatchedSculpt3DEnvTF(grid_size=4, max_steps=10, n_envs=2)
        obs = env.reset()
        print("✓ Environment creation and reset")
        
        # Test replay buffers
        from replay_buffers import WorkerReplayBuffer
        worker_buffer = WorkerReplayBuffer(capacity=100)
        print("✓ Replay buffer creation")
        
        # Test networks
        from networks import build_manager_network
        manager_net = build_manager_network((4, 4, 4, 2), (3,), 27)
        print("✓ Network creation")
        
        # Test agent
        from feudal_agent import FeudalAgentTF
        agent = FeudalAgentTF(
            grid_shape=(4, 4, 4, 2),
            coord_shape=(3,),
            manager_goal_k=1,
            subgoal_horizon=5,
            worker_buffer_capacity=100,
            manager_buffer_capacity=10
        )
        print("✓ Agent creation")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_gpu_availability():
    """Test GPU availability."""
    print("\nTesting GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("⚠ No GPU detected - will use CPU")
        return True
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("RL CNC Path Optimization - Module Tests")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed.")
        return False
    
    # Test GPU
    test_gpu_availability()
    
    print("\n✅ All tests passed! The system is ready to use.")
    print("\nNext steps:")
    print("1. Run 'python example_usage.py' for usage examples")
    print("2. Open 'rl_cnc_path_optimization_demo.ipynb' for the full demo")
    print("3. Check the README.md for detailed documentation")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 