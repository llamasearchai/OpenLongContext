"""
Hyperparameter Sweep for Ablation Studies
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import itertools
import json
import logging
import random
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    """Container for sweep results."""
    best_params: Dict[str, Any]
    best_score: float
    history: List[Tuple[Dict[str, Any], float]]
    param_importance: Optional[Dict[str, float]] = None


class HyperparameterSweep:
    """
    Hyperparameter sweep with multiple search strategies.
    
    Supports random search, grid search, and quasi-random search using Sobol sequences.
    """

    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, Union[List[Any], Tuple[float, float]]],
        n_iter: int = 50,
        search_strategy: str = "random",
        random_state: Optional[int] = None
    ):
        """
        Initialize hyperparameter sweep.
        
        Args:
            objective_fn: Function to minimize
            param_space: Parameter space - either list of values or (min, max) for continuous
            n_iter: Number of iterations (ignored for grid search)
            search_strategy: One of "random", "grid", "sobol"
            random_state: Random seed for reproducibility
        """
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.n_iter = n_iter
        self.search_strategy = search_strategy
        self.history = []

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # Validate parameter space
        self._validate_param_space()

    def _validate_param_space(self):
        """Validate parameter space definition."""
        for param, values in self.param_space.items():
            if isinstance(values, tuple) and len(values) == 2:
                # Continuous parameter
                if not all(isinstance(v, (int, float)) for v in values):
                    raise ValueError(f"Continuous parameter {param} must have numeric bounds")
            elif isinstance(values, list):
                # Discrete parameter
                if len(values) == 0:
                    raise ValueError(f"Discrete parameter {param} must have at least one value")
            else:
                raise ValueError(f"Parameter {param} must be either a list or a (min, max) tuple")

    def sample(self) -> Dict[str, Any]:
        """Sample parameters based on search strategy."""
        if self.search_strategy == "random":
            return self._random_sample()
        elif self.search_strategy == "sobol":
            return self._sobol_sample()
        else:
            raise ValueError(f"Unknown search strategy: {self.search_strategy}")

    def _random_sample(self) -> Dict[str, Any]:
        """Random sampling from parameter space."""
        params = {}
        for param, values in self.param_space.items():
            if isinstance(values, tuple):
                # Continuous parameter
                params[param] = random.uniform(values[0], values[1])
            else:
                # Discrete parameter
                params[param] = random.choice(values)
        return params

    def _sobol_sample(self) -> Dict[str, Any]:
        """Quasi-random sampling using Sobol sequences."""
        # Simple Sobol-like sampling (simplified version)
        params = {}
        n = len(self.history)

        for i, (param, values) in enumerate(self.param_space.items()):
            # Use bit-reversal sequence for quasi-random sampling
            sobol_value = self._bit_reversal(n, base=2) + 0.5 * self._bit_reversal(n, base=3)
            sobol_value = sobol_value % 1.0

            if isinstance(values, tuple):
                # Continuous parameter
                params[param] = values[0] + sobol_value * (values[1] - values[0])
            else:
                # Discrete parameter
                idx = int(sobol_value * len(values))
                params[param] = values[idx]

        return params

    def _bit_reversal(self, n: int, base: int = 2) -> float:
        """Compute bit reversal of n in given base."""
        if n == 0:
            return 0

        result = 0
        denominator = 1
        while n > 0:
            denominator *= base
            result += (n % base) / denominator
            n //= base

        return result

    def _grid_search(self) -> Tuple[Dict[str, Any], float]:
        """Perform grid search over parameter space."""
        # Convert continuous parameters to discrete grid
        grid_params = {}
        for param, values in self.param_space.items():
            if isinstance(values, tuple):
                # Create grid for continuous parameter
                n_points = int(np.sqrt(self.n_iter))  # Adjust grid density based on budget
                grid_params[param] = np.linspace(values[0], values[1], n_points).tolist()
            else:
                grid_params[param] = values

        # Generate all combinations
        param_names = list(grid_params.keys())
        param_values = [grid_params[name] for name in param_names]

        best_params = None
        best_score = float('inf')

        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            score = self.objective_fn(params)
            self.history.append((params, score))

            if score < best_score:
                best_score = score
                best_params = params.copy()

            logger.info(f"Grid search: {len(self.history)} evaluations, best score: {best_score:.6f}")

        return best_params, best_score

    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """Run the optimization process."""
        if self.search_strategy == "grid":
            return self._grid_search()

        best_params = None
        best_score = float('inf')

        for i in range(self.n_iter):
            params = self.sample()

            try:
                score = float(self.objective_fn(params))
            except Exception as e:
                logger.error(f"Evaluation failed for params {params}: {e}")
                score = float('inf')

            self.history.append((params, score))

            if score < best_score:
                best_score = score
                best_params = params.copy()
                logger.info(f"Iteration {i+1}/{self.n_iter}: New best score = {best_score:.6f}")

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{self.n_iter} evaluations completed")

        return best_params, best_score

    def optimize_parallel(self, executor: Executor, batch_size: int = 10) -> Tuple[Dict[str, Any], float]:
        """Run optimization in parallel using provided executor."""
        best_params = None
        best_score = float('inf')

        for batch_start in range(0, self.n_iter, batch_size):
            batch_end = min(batch_start + batch_size, self.n_iter)
            batch_size_actual = batch_end - batch_start

            # Generate batch of parameters
            param_batch = [self.sample() for _ in range(batch_size_actual)]

            # Evaluate in parallel
            futures = [executor.submit(self.objective_fn, params) for params in param_batch]

            # Collect results
            for params, future in zip(param_batch, futures):
                try:
                    score = float(future.result())
                except Exception as e:
                    logger.error(f"Evaluation failed for params {params}: {e}")
                    score = float('inf')

                self.history.append((params, score))

                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    logger.info(f"New best score = {best_score:.6f}")

            logger.info(f"Progress: {len(self.history)}/{self.n_iter} evaluations completed")

        return best_params, best_score

    def analyze_results(self) -> SweepResult:
        """Analyze sweep results and compute parameter importance."""
        if not self.history:
            return SweepResult(best_params={}, best_score=float('inf'), history=[])

        # Find best result
        best_idx = np.argmin([score for _, score in self.history])
        best_params, best_score = self.history[best_idx]

        # Compute parameter importance
        param_importance = self._compute_param_importance()

        return SweepResult(
            best_params=best_params,
            best_score=best_score,
            history=self.history,
            param_importance=param_importance
        )

    def _compute_param_importance(self) -> Dict[str, float]:
        """Compute parameter importance based on score variance."""
        if len(self.history) < 10:
            return {}

        importance = {}
        all_scores = [score for _, score in self.history]

        for param in self.param_space:
            # Group scores by parameter value
            param_values = {}
            for params, score in self.history:
                value = params.get(param)
                if value is not None:
                    if value not in param_values:
                        param_values[value] = []
                    param_values[value].append(score)

            # Compute variance explained by this parameter
            if len(param_values) > 1:
                group_means = [np.mean(scores) for scores in param_values.values()]
                overall_mean = np.mean(all_scores)

                # Between-group variance
                between_var = sum(
                    len(scores) * (mean - overall_mean) ** 2
                    for scores, mean in zip(param_values.values(), group_means)
                )

                # Total variance
                total_var = np.var(all_scores) * len(all_scores)

                # Importance as variance ratio
                importance[param] = between_var / total_var if total_var > 0 else 0
            else:
                importance[param] = 0

        # Normalize importances
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}

        return importance

    def save_results(self, path: str):
        """Save sweep results to file."""
        results = self.analyze_results()

        data = {
            "best_params": results.best_params,
            "best_score": results.best_score,
            "param_importance": results.param_importance,
            "history": [
                {"params": params, "score": score}
                for params, score in results.history
            ],
            "search_strategy": self.search_strategy,
            "n_evaluations": len(self.history)
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {path}")

    # Maintain backward compatibility
    def run(self):
        """Backward compatible method name."""
        return self.optimize()


# Example usage with enhanced features:
if __name__ == "__main__":
    # Define objective function
    def objective(params):
        # Simple quadratic function for testing
        x = params.get("x", 0)
        y = params.get("y", 0)
        return (x - 1) ** 2 + (y + 2) ** 2

    # Define parameter space
    param_space = {
        "x": (-5.0, 5.0),  # Continuous
        "y": [-5, -2, 0, 2, 5],  # Discrete
    }

    # Run sweep
    sweep = HyperparameterSweep(
        objective_fn=objective,
        param_space=param_space,
        n_iter=100,
        search_strategy="random",
        random_state=42
    )

    best_params, best_score = sweep.optimize()
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.6f}")

    # Analyze results
    results = sweep.analyze_results()
    print(f"Parameter importance: {results.param_importance}")
