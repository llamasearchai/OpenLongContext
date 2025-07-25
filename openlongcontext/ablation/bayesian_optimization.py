"""
Bayesian Optimization for Ablation Studies
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, float]
    best_score: float
    history: List[Tuple[Dict[str, float], float]]
    convergence_info: Dict[str, Any]


class BayesianOptimizer:
    """
    Bayesian Optimization using Gaussian Processes.
    
    This implementation uses Expected Improvement (EI) as the acquisition function
    and supports multiple acquisition strategies for robust optimization.
    """

    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        bounds: Dict[str, Tuple[float, float]],
        n_iter: int = 25,
        n_initial: int = 5,
        acquisition: str = "ei",
        xi: float = 0.01,
        kappa: float = 2.576,
        random_state: Optional[int] = None,
        normalize_y: bool = True,
        alpha: float = 1e-6,
        n_restarts_optimizer: int = 10
    ):
        """
        Initialize Bayesian Optimizer.
        
        Args:
            objective_fn: Function to minimize
            bounds: Parameter bounds as {param_name: (min, max)}
            n_iter: Number of optimization iterations
            n_initial: Number of initial random samples
            acquisition: Acquisition function ('ei', 'ucb', 'pi')
            xi: Exploration parameter for EI/PI
            kappa: Exploration parameter for UCB
            random_state: Random seed for reproducibility
            normalize_y: Whether to normalize objective values
            alpha: Noise level for GP
            n_restarts_optimizer: Number of restarts for GP hyperparameter optimization
        """
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.n_iter = n_iter
        self.n_initial = max(n_initial, len(bounds) + 1)
        self.acquisition = acquisition.lower()
        self.xi = xi
        self.kappa = kappa
        self.random_state = random_state
        self.normalize_y = normalize_y
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer

        self.param_names = list(bounds.keys())
        self.param_bounds = np.array([bounds[k] for k in self.param_names])

        self.history = []
        self.X_observed = []
        self.y_observed = []

        np.random.seed(random_state)

        # Initialize Gaussian Process
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5
        ) + WhiteKernel(noise_level=alpha)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state
        )

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to numpy array."""
        return np.array([params[k] for k in self.param_names])

    def _array_to_params(self, x: np.ndarray) -> Dict[str, float]:
        """Convert numpy array to parameter dictionary."""
        return {k: float(v) for k, v in zip(self.param_names, x)}

    def _normalize_params(self, x: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1] range."""
        return (x - self.param_bounds[:, 0]) / (self.param_bounds[:, 1] - self.param_bounds[:, 0])

    def _denormalize_params(self, x: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [0, 1] range."""
        return x * (self.param_bounds[:, 1] - self.param_bounds[:, 0]) + self.param_bounds[:, 0]

    def _acquisition_ei(self, x: np.ndarray, gp: GaussianProcessRegressor, y_min: float) -> float:
        """Expected Improvement acquisition function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)

        if sigma < 1e-10:
            return 0.0

        z = (y_min - mu - self.xi) / sigma
        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        return float(ei)

    def _acquisition_ucb(self, x: np.ndarray, gp: GaussianProcessRegressor, y_min: float) -> float:  # noqa: ARG002
        """Upper Confidence Bound acquisition function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)

        return float(-(mu - self.kappa * sigma))

    def _acquisition_pi(self, x: np.ndarray, gp: GaussianProcessRegressor, y_min: float) -> float:
        """Probability of Improvement acquisition function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)

        if sigma < 1e-10:
            return 0.0

        z = (y_min - mu - self.xi) / sigma
        return float(norm.cdf(z))

    def _get_acquisition_function(self) -> Callable:
        """Get the appropriate acquisition function."""
        acquisition_functions = {
            "ei": self._acquisition_ei,
            "ucb": self._acquisition_ucb,
            "pi": self._acquisition_pi
        }

        if self.acquisition not in acquisition_functions:
            logger.warning(f"Unknown acquisition function '{self.acquisition}', using EI")
            return self._acquisition_ei

        return acquisition_functions[self.acquisition]

    def suggest(self) -> Dict[str, Any]:
        """Suggest next parameters to evaluate."""
        # Initial random sampling
        if len(self.X_observed) < self.n_initial:
            x = np.array([
                np.random.uniform(low, high)
                for low, high in self.param_bounds
            ])
            return self._array_to_params(x)

        # Fit Gaussian Process
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)

        # Normalize features for better GP performance
        X_norm = self._normalize_params(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(X_norm, y)

        # Find best observation so far
        y_min = np.min(y)

        # Get acquisition function
        acq_func = self._get_acquisition_function()

        # Optimize acquisition function
        best_x = None
        best_acq = -np.inf

        # Multiple random restarts for acquisition optimization
        n_restarts = max(20, 5 * len(self.param_names))

        for _ in range(n_restarts):
            # Random starting point
            x0 = np.random.uniform(0, 1, size=len(self.param_names))

            # Minimize negative acquisition (maximize acquisition)
            res = minimize(
                lambda x: -acq_func(x, self.gp, y_min),
                x0,
                bounds=[(0, 1)] * len(self.param_names),
                method='L-BFGS-B'
            )

            if res.fun < -best_acq:
                best_acq = -res.fun
                best_x = res.x

        # Denormalize and return
        x_denorm = self._denormalize_params(best_x)
        return self._array_to_params(x_denorm)

    def optimize(self) -> OptimizationResult:
        best_params: Dict[str, float] = {k: float(v[0]) for k, v in self.bounds.items()}  # Default to lower bounds
        best_score = float('inf')
        convergence_scores = []

        logger.info(f"Starting Bayesian optimization with {self.n_iter} iterations")

        for i in range(self.n_iter):
            # Get next parameters to evaluate
            params = self.suggest()

            # Evaluate objective function
            try:
                score = float(self.objective_fn(params))
            except Exception as e:
                logger.error(f"Error evaluating objective function: {e}")
                score = float('inf')

            # Update observations
            x_array = self._params_to_array(params)
            self.X_observed.append(x_array)
            self.y_observed.append(score)
            self.history.append((params, score))

            # Track best parameters
            if score < best_score:
                best_score = score
                best_params = params.copy()
                logger.info(f"Iteration {i+1}/{self.n_iter}: New best score = {best_score:.6f}")
            else:
                logger.info(f"Iteration {i+1}/{self.n_iter}: Score = {score:.6f} (best = {best_score:.6f})")

            convergence_scores.append(best_score)

        # Compute convergence metrics
        convergence_info = {
            "final_score": best_score,
            "n_evaluations": len(self.history),
            "convergence_history": convergence_scores,
            "improvement_rate": (convergence_scores[0] - best_score) / convergence_scores[0] if convergence_scores[0] != 0 else 0
        }

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=self.history,
            convergence_info=convergence_info
        )

    def get_incumbent(self) -> Tuple[Dict[str, float], float]:
        """Get the best parameters found so far."""
        if not self.history:
            # Return default lower bounds if no history
            default_params = {k: float(v[0]) for k, v in self.bounds.items()}
            return default_params, float('inf')

        best_idx = np.argmin([score for _, score in self.history])
        return self.history[best_idx]

    def plot_convergence(self) -> None:
        """Plot optimization convergence (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            scores = [score for _, score in self.history]
            best_scores = np.minimum.accumulate(scores)

            plt.figure(figsize=(10, 6))
            plt.plot(scores, 'o-', alpha=0.5, label='Observed')
            plt.plot(best_scores, 'r-', linewidth=2, label='Best')
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title('Bayesian Optimization Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


# Example usage with complete implementation:
if __name__ == "__main__":
    # Define a simple test function (Rosenbrock)
    def rosenbrock(params):
        x, y = params["x"], params["y"]
        return (1 - x)**2 + 100 * (y - x**2)**2

    # Define bounds
    bounds = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}

    # Run optimization
    optimizer = BayesianOptimizer(
        objective_fn=rosenbrock,
        bounds=bounds,
        n_iter=50,
        acquisition="ei",
        random_state=42
    )

    result = optimizer.optimize()
    print(f"Best parameters: {result.best_params}")
    print(f"Best score: {result.best_score:.6f}")
    print(f"Improvement rate: {result.convergence_info['improvement_rate']:.2%}")
