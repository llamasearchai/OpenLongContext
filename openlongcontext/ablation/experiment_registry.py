"""
Experiment Registry for Ablation Studies
Author: Nik Jois <nikjois@llamasearch.ai>
"""

from typing import Dict, Callable, Any, Optional, List
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentRegistry:
    def __init__(self):
        self._experiments: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]):
        if name in self._experiments:
            raise ValueError(f"Experiment '{name}' is already registered.")
        self._experiments[name] = func

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._experiments:
            raise KeyError(f"Experiment '{name}' not found.")
        return self._experiments[name]

    def list_experiments(self):
        return list(self._experiments.keys())


# Global registry instance
experiment_registry = ExperimentRegistry()


def register_experiment(name: str):
    def decorator(func: Callable[..., Any]):
        experiment_registry.register(name, func)
        return func
    return decorator


def run_ablation(config_name: str, output_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Run ablation study based on configuration name.
    
    Args:
        config_name: Name of the ablation configuration
        output_dir: Optional output directory for results
        **kwargs: Additional parameters for the ablation
        
    Returns:
        Dictionary containing ablation results
    """
    from ..evaluation.ablation import run_ablation as run_ablation_impl
    
    # Resolve config path
    config_path = Path("configs/experiments/ablation.yaml")
    if not config_path.exists():
        # Try alternate locations
        alt_paths = [
            Path(f"configs/{config_name}.yaml"),
            Path(f"configs/ablation/{config_name}.yaml"),
            Path(config_name)  # Direct path
        ]
        
        for path in alt_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(f"Configuration '{config_name}' not found")
    
    # Set default output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ablation_{config_name}_{timestamp}"
    
    logger.info(f"Running ablation study with config: {config_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Run the ablation
    results = run_ablation_impl(
        config_path=str(config_path),
        output_dir=output_dir,
        **kwargs
    )
    
    # Print summary
    print(f"\nAblation study completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Total experiments: {len(results.get('results', []))}")
    
    if 'analysis' in results and 'best_configuration' in results['analysis']:
        best = results['analysis']['best_configuration']
        print(f"\nBest configuration found:")
        for param, value in best.get('parameters', {}).items():
            print(f"  {param}: {value}")
        print(f"  Score: {best.get('metrics', {}).get('loss', 'N/A')}")
    
    return results


def run_sweep(config_name: str, multirun: bool = False, output_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Run hyperparameter sweep based on configuration.
    
    Args:
        config_name: Name of the sweep configuration
        multirun: Whether to run multiple configurations in parallel
        output_dir: Optional output directory for results
        **kwargs: Additional parameters for the sweep
        
    Returns:
        Dictionary containing sweep results
    """
    from ..ablation.hyperparameter_sweep import HyperparameterSweep
    from omegaconf import OmegaConf
    import concurrent.futures
    
    # Resolve config path
    config_path = Path("configs/experiments/hyperparams.yaml")
    if not config_path.exists():
        # Try alternate locations
        alt_paths = [
            Path(f"configs/{config_name}.yaml"),
            Path(f"configs/sweep/{config_name}.yaml"),
            Path(config_name)  # Direct path
        ]
        
        for path in alt_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(f"Configuration '{config_name}' not found")
    
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Set default output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/sweep_{config_name}_{timestamp}"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running hyperparameter sweep with config: {config_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Multirun mode: {multirun}")
    
    # Extract sweep parameters from config
    param_space = config.get("param_space", {
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "batch_size": [16, 32, 64],
        "dropout": [0.1, 0.2, 0.3],
        "hidden_size": [256, 512, 1024]
    })
    
    # Define objective function
    def objective_fn(params: Dict[str, Any]) -> float:
        """Objective function for sweep."""
        from ..core.experiment import Experiment
        from ..core.config import Config
        
        try:
            # Merge params with base config
            exp_config = OmegaConf.to_container(config, resolve=True)
            exp_config.update(params)
            
            # Run experiment
            experiment = Experiment(Config(**exp_config))
            metrics = experiment.run()
            
            # Return loss metric
            return metrics.get("loss", float('inf'))
            
        except Exception as e:
            logger.error(f"Experiment failed with params {params}: {e}")
            return float('inf')
    
    # Create sweep instance
    sweep = HyperparameterSweep(
        objective_fn=objective_fn,
        param_space=param_space,
        n_iter=config.get("n_iter", 50)
    )
    
    # Run sweep
    if multirun:
        # Parallel execution
        logger.info("Running sweep in parallel mode")
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.get("max_workers", 4)) as executor:
            results = sweep.optimize_parallel(executor)
    else:
        # Sequential execution
        logger.info("Running sweep in sequential mode")
        results = sweep.optimize()
    
    # Save results
    results_path = Path(output_dir) / "sweep_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "config_name": config_name,
            "param_space": param_space,
            "best_params": results[0],
            "best_score": results[1],
            "history": sweep.history,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    # Print summary
    print(f"\nHyperparameter sweep completed!")
    print(f"Results saved to: {results_path}")
    print(f"Total evaluations: {len(sweep.history)}")
    print(f"\nBest parameters found:")
    for param, value in results[0].items():
        print(f"  {param}: {value}")
    print(f"  Score: {results[1]:.6f}")
    
    return {
        "best_params": results[0],
        "best_score": results[1],
        "history": sweep.history,
        "output_dir": str(output_dir)
    }