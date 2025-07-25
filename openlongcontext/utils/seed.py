"""
Seed Utilities

Comprehensive seed management utilities for reproducible experiments.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import random
import numpy as np
import torch
import os
import logging
from typing import Optional, Dict, Any
import hashlib

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for all random number generators.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    try:
        # Python's random module
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch CPU
        torch.manual_seed(seed)
        
        # PyTorch GPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Environment variable for Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # PyTorch deterministic behavior
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Use deterministic algorithms where available
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        logger.info(f"Set random seed to {seed} (deterministic={deterministic})")
        
    except Exception as e:
        logger.error(f"Failed to set seed {seed}: {e}")
        raise


def get_seed_from_string(text: str) -> int:
    """
    Generate a seed from a string using hashing.
    
    Args:
        text: Input string
        
    Returns:
        Integer seed value
    """
    # Create hash of the string
    hash_object = hashlib.md5(text.encode())
    hex_dig = hash_object.hexdigest()
    
    # Convert to integer and limit to reasonable range
    seed = int(hex_dig, 16) % (2**31 - 1)
    return seed


def create_seed_sequence(base_seed: int, num_seeds: int) -> list[int]:
    """
    Create a sequence of seeds from a base seed.
    
    Args:
        base_seed: Base seed value
        num_seeds: Number of seeds to generate
        
    Returns:
        List of seed values
    """
    # Use NumPy's SeedSequence for proper seed generation
    ss = np.random.SeedSequence(base_seed)
    child_seeds = ss.spawn(num_seeds)
    
    # Convert to integers
    seeds = [int(seed.entropy) for seed in child_seeds]
    return seeds


def save_seed_state(filepath: str, seed: int, additional_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Save current random state to file.
    
    Args:
        filepath: Path to save the state
        seed: Seed value used
        additional_info: Additional information to save
    """
    try:
        state_info = {
            'seed': seed,
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state().tolist(),
        }
        
        # Add CUDA state if available
        if torch.cuda.is_available():
            state_info['torch_cuda_random_state'] = [
                state.tolist() for state in torch.cuda.get_rng_state_all()
            ]
        
        # Add additional info
        if additional_info:
            state_info['additional_info'] = additional_info
        
        # Save to file
        import json
        with open(filepath, 'w') as f:
            json.dump(state_info, f, indent=2)
        
        logger.info(f"Saved random state to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save random state: {e}")
        raise


def load_seed_state(filepath: str) -> Dict[str, Any]:
    """
    Load random state from file.
    
    Args:
        filepath: Path to load the state from
        
    Returns:
        Dictionary containing state information
    """
    try:
        import json
        with open(filepath, 'r') as f:
            state_info = json.load(f)
        
        # Restore Python random state
        if 'python_random_state' in state_info:
            random.setstate(tuple(state_info['python_random_state']))
        
        # Restore NumPy random state
        if 'numpy_random_state' in state_info:
            np_state = state_info['numpy_random_state']
            # Convert back to proper format
            np_state = (np_state[0], np.array(np_state[1]), np_state[2], np_state[3], np_state[4])
            np.random.set_state(np_state)
        
        # Restore PyTorch random state
        if 'torch_random_state' in state_info:
            torch_state = torch.tensor(state_info['torch_random_state'], dtype=torch.uint8)
            torch.set_rng_state(torch_state)
        
        # Restore CUDA random state
        if 'torch_cuda_random_state' in state_info and torch.cuda.is_available():
            cuda_states = [
                torch.tensor(state, dtype=torch.uint8) 
                for state in state_info['torch_cuda_random_state']
            ]
            torch.cuda.set_rng_state_all(cuda_states)
        
        logger.info(f"Loaded random state from {filepath}")
        return state_info
        
    except Exception as e:
        logger.error(f"Failed to load random state: {e}")
        raise


class SeedContext:
    """Context manager for temporary seed setting."""
    
    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.original_state = None
    
    def __enter__(self):
        # Save current state
        self.original_state = {
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'torch_deterministic': torch.backends.cudnn.deterministic,
            'torch_benchmark': torch.backends.cudnn.benchmark,
        }
        
        # Add CUDA state if available
        if torch.cuda.is_available():
            self.original_state['torch_cuda_random_state'] = torch.cuda.get_rng_state_all()
        
        # Set new seed
        set_seed(self.seed, self.deterministic)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        if self.original_state:
            random.setstate(self.original_state['python_random_state'])
            np.random.set_state(self.original_state['numpy_random_state'])
            torch.set_rng_state(self.original_state['torch_random_state'])
            
            torch.backends.cudnn.deterministic = self.original_state['torch_deterministic']
            torch.backends.cudnn.benchmark = self.original_state['torch_benchmark']
            
            if 'torch_cuda_random_state' in self.original_state:
                torch.cuda.set_rng_state_all(self.original_state['torch_cuda_random_state'])


def generate_experiment_seeds(
    experiment_name: str,
    num_runs: int = 5,
    base_seed: Optional[int] = None
) -> Dict[str, int]:
    """
    Generate seeds for multiple experimental runs.
    
    Args:
        experiment_name: Name of the experiment
        num_runs: Number of experimental runs
        base_seed: Base seed (if None, generate from experiment name)
        
    Returns:
        Dictionary mapping run names to seeds
    """
    if base_seed is None:
        base_seed = get_seed_from_string(experiment_name)
    
    seeds = create_seed_sequence(base_seed, num_runs)
    
    return {
        f"{experiment_name}_run_{i+1}": seed
        for i, seed in enumerate(seeds)
    }


def verify_reproducibility(
    func,
    seed: int,
    num_trials: int = 3,
    *args,
    **kwargs
) -> bool:
    """
    Verify that a function produces reproducible results with the same seed.
    
    Args:
        func: Function to test
        seed: Seed to use
        num_trials: Number of trials to run
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        True if results are reproducible, False otherwise
    """
    results = []
    
    for _ in range(num_trials):
        with SeedContext(seed):
            result = func(*args, **kwargs)
            results.append(result)
    
    # Check if all results are the same
    first_result = results[0]
    
    for result in results[1:]:
        if isinstance(first_result, torch.Tensor):
            if not torch.equal(first_result, result):
                return False
        elif isinstance(first_result, np.ndarray):
            if not np.array_equal(first_result, result):
                return False
        else:
            if first_result != result:
                return False
    
    return True


def get_random_state_info() -> Dict[str, Any]:
    """
    Get information about current random states.
    
    Returns:
        Dictionary containing random state information
    """
    info = {
        'python_random_state_type': type(random.getstate()).__name__,
        'numpy_random_state_type': type(np.random.get_state()).__name__,
        'torch_random_state_shape': torch.get_rng_state().shape,
        'torch_cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['torch_cuda_device_count'] = torch.cuda.device_count()
        info['torch_cuda_random_states_count'] = len(torch.cuda.get_rng_state_all())
    
    return info


# Convenience function for common use case
def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed everything for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    set_seed(seed, deterministic)
