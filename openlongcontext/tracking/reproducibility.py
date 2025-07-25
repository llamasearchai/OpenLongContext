"""
Reproducibility

Comprehensive reproducibility utilities for ensuring consistent experiment results.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import random
import numpy as np
import torch
import os
import hashlib
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import subprocess
import sys
import platform
from datetime import datetime

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for all random number generators to ensure reproducibility.
    
    Args:
        seed: Random seed value
    """
    try:
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Additional PyTorch settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        logger.info(f"Set random seeds to {seed} for reproducibility")
        
    except Exception as e:
        logger.error(f"Failed to set random seeds: {e}")
        raise


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information for reproducibility tracking.
    
    Returns:
        Dictionary containing system information
    """
    try:
        info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
            },
            "environment": {
                "python_executable": sys.executable,
                "python_path": sys.path,
                "cwd": os.getcwd(),
                "user": os.environ.get("USER", "unknown"),
                "hostname": platform.node(),
            },
            "hardware": {
                "cpu_count": os.cpu_count(),
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            info["hardware"]["cuda_available"] = True
            info["hardware"]["cuda_version"] = torch.version.cuda
            info["hardware"]["cudnn_version"] = torch.backends.cudnn.version()
            info["hardware"]["gpu_count"] = torch.cuda.device_count()
            info["hardware"]["gpu_devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory": torch.cuda.get_device_properties(i).total_memory,
                    "compute_capability": torch.cuda.get_device_properties(i).major,
                }
                for i in range(torch.cuda.device_count())
            ]
        else:
            info["hardware"]["cuda_available"] = False
        
        # Add package versions
        info["packages"] = get_package_versions()
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {"error": str(e)}


def get_package_versions() -> Dict[str, str]:
    """
    Get versions of installed packages relevant to the project.
    
    Returns:
        Dictionary mapping package names to versions
    """
    packages = {}
    
    try:
        # Core packages
        import torch
        packages["torch"] = torch.__version__
        
        import numpy
        packages["numpy"] = numpy.__version__
        
        try:
            import transformers
            packages["transformers"] = transformers.__version__
        except ImportError:
            pass
        
        try:
            import datasets
            packages["datasets"] = getattr(datasets, '__version__', 'unknown')
        except ImportError:
            pass
        
        try:
            import mlflow
            packages["mlflow"] = mlflow.__version__
        except ImportError:
            pass
        
        try:
            import wandb
            packages["wandb"] = wandb.__version__
        except ImportError:
            pass
        
        try:
            import tensorboard
            packages["tensorboard"] = tensorboard.__version__
        except ImportError:
            pass
        
        try:
            import omegaconf
            packages["omegaconf"] = omegaconf.__version__
        except ImportError:
            pass
        
        try:
            import hydra
            packages["hydra"] = hydra.__version__
        except ImportError:
            pass
        
    except Exception as e:
        logger.warning(f"Failed to get some package versions: {e}")
    
    return packages


def get_git_info() -> Dict[str, Any]:
    """
    Get Git repository information for reproducibility.
    
    Returns:
        Dictionary containing Git information
    """
    git_info = {}
    
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                git_info["commit_hash"] = result.stdout.strip()
            
            # Get branch name
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()
            
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                git_info["dirty"] = bool(result.stdout.strip())
                git_info["uncommitted_changes"] = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Get remote URL
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                git_info["remote_url"] = result.stdout.strip()
        
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.warning(f"Failed to get git info: {e}")
        git_info["error"] = str(e)
    
    return git_info


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a hash of the configuration for reproducibility tracking.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SHA256 hash of the configuration
    """
    try:
        # Convert config to a sorted JSON string for consistent hashing
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute config hash: {e}")
        return "unknown"


def save_reproducibility_info(
    output_dir: Union[str, Path],
    config: Dict[str, Any],
    seed: int,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save comprehensive reproducibility information to a file.
    
    Args:
        output_dir: Directory to save the reproducibility info
        config: Experiment configuration
        seed: Random seed used
        additional_info: Additional information to include
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        repro_info = {
            "seed": seed,
            "config": config,
            "config_hash": compute_config_hash(config),
            "system_info": get_system_info(),
            "git_info": get_git_info(),
            "timestamp": datetime.now().isoformat(),
        }
        
        if additional_info:
            repro_info["additional_info"] = additional_info
        
        # Save to JSON file
        repro_file = output_dir / "reproducibility_info.json"
        with open(repro_file, 'w') as f:
            json.dump(repro_info, f, indent=2, default=str)
        
        logger.info(f"Saved reproducibility info to {repro_file}")
        
        return repro_info
        
    except Exception as e:
        logger.error(f"Failed to save reproducibility info: {e}")
        raise


def load_reproducibility_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load reproducibility information from a file.
    
    Args:
        file_path: Path to the reproducibility info file
        
    Returns:
        Dictionary containing reproducibility information
    """
    try:
        with open(file_path, 'r') as f:
            repro_info = json.load(f)
        
        logger.info(f"Loaded reproducibility info from {file_path}")
        return repro_info
        
    except Exception as e:
        logger.error(f"Failed to load reproducibility info: {e}")
        raise


def verify_reproducibility(
    config: Dict[str, Any],
    reference_file: Union[str, Path],
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Verify that current setup matches reference reproducibility info.
    
    Args:
        config: Current configuration
        reference_file: Path to reference reproducibility info
        tolerance: Tolerance for numerical comparisons
        
    Returns:
        Dictionary containing verification results
    """
    try:
        reference_info = load_reproducibility_info(reference_file)
        current_info = {
            "config": config,
            "config_hash": compute_config_hash(config),
            "system_info": get_system_info(),
            "git_info": get_git_info(),
        }
        
        verification = {
            "config_match": reference_info["config_hash"] == current_info["config_hash"],
            "system_differences": [],
            "git_differences": [],
            "package_differences": [],
        }
        
        # Check system differences
        ref_system = reference_info.get("system_info", {})
        curr_system = current_info.get("system_info", {})
        
        for key in ["platform", "packages"]:
            if key in ref_system and key in curr_system:
                if ref_system[key] != curr_system[key]:
                    verification["system_differences"].append({
                        "key": key,
                        "reference": ref_system[key],
                        "current": curr_system[key]
                    })
        
        # Check git differences
        ref_git = reference_info.get("git_info", {})
        curr_git = current_info.get("git_info", {})
        
        for key in ["commit_hash", "branch", "dirty"]:
            if key in ref_git and key in curr_git:
                if ref_git[key] != curr_git[key]:
                    verification["git_differences"].append({
                        "key": key,
                        "reference": ref_git[key],
                        "current": curr_git[key]
                    })
        
        verification["reproducible"] = (
            verification["config_match"] and
            len(verification["system_differences"]) == 0 and
            len(verification["git_differences"]) == 0
        )
        
        return verification
        
    except Exception as e:
        logger.error(f"Failed to verify reproducibility: {e}")
        return {"error": str(e), "reproducible": False}


class ReproducibilityContext:
    """Context manager for reproducibility setup."""
    
    def __init__(
        self,
        seed: int = 42,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.seed = seed
        self.output_dir = output_dir
        self.config = config or {}
        self.repro_info = None
    
    def __enter__(self):
        # Set random seeds
        set_random_seeds(self.seed)
        
        # Save reproducibility info if output directory is provided
        if self.output_dir:
            self.repro_info = save_reproducibility_info(
                self.output_dir, self.config, self.seed
            )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Additional cleanup if needed
        pass


def ensure_reproducibility(
    seed: int = 42,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Ensure reproducibility by setting seeds and saving environment info.
    
    Args:
        seed: Random seed to use
        output_dir: Directory to save reproducibility info
        config: Configuration to include in reproducibility info
        
    Returns:
        Dictionary containing reproducibility information
    """
    set_random_seeds(seed)
    
    if output_dir:
        return save_reproducibility_info(output_dir, config or {}, seed)
    
    return {
        "seed": seed,
        "config": config or {},
        "system_info": get_system_info(),
        "git_info": get_git_info(),
    }
