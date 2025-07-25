"""
Unit tests for Bayesian optimization module.
"""

from typing import Any, Dict

import numpy as np
import pytest

from openlongcontext.ablation.bayesian_optimization import BayesianOptimizer, OptimizationResult


class TestBayesianOptimizer:
    """Test suite for BayesianOptimizer class."""

    @pytest.fixture
    def simple_objective(self):
        """Simple quadratic objective function for testing."""
        def objective(params: Dict[str, Any]) -> float:
            x = params.get("x", 0)
            y = params.get("y", 0)
            # Minimum at (1, -2)
            return (x - 1) ** 2 + (y + 2) ** 2
        return objective

    @pytest.fixture
    def bounds(self):
        """Standard bounds for testing."""
        return {
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0)
        }

    def test_initialization(self, simple_objective, bounds):
        """Test optimizer initialization."""
        optimizer = BayesianOptimizer(
            objective_fn=simple_objective,
            bounds=bounds,
            n_iter=10
        )

        assert optimizer.objective_fn == simple_objective
        assert optimizer.bounds == bounds
        assert optimizer.n_iter == 10
        assert optimizer.acquisition == "ei"
        assert len(optimizer.history) == 0
        assert len(optimizer.X_observed) == 0
        assert len(optimizer.y_observed) == 0

    def test_params_conversion(self, simple_objective, bounds):
        """Test parameter conversion between dict and array."""
        optimizer = BayesianOptimizer(simple_objective, bounds)

        # Test dict to array
        params_dict = {"x": 1.5, "y": -2.5}
        params_array = optimizer._params_to_array(params_dict)
        assert np.allclose(params_array, [1.5, -2.5])

        # Test array to dict
        params_dict_back = optimizer._array_to_params(params_array)
        assert params_dict_back["x"] == 1.5
        assert params_dict_back["y"] == -2.5

    def test_normalization(self, simple_objective, bounds):
        """Test parameter normalization and denormalization."""
        optimizer = BayesianOptimizer(simple_objective, bounds)

        # Test normalization
        x = np.array([0.0, 0.0])  # Center of bounds
        x_norm = optimizer._normalize_params(x)
        assert np.allclose(x_norm, [0.5, 0.5])

        # Test edge cases
        x_min = np.array([-5.0, -5.0])
        x_norm_min = optimizer._normalize_params(x_min)
        assert np.allclose(x_norm_min, [0.0, 0.0])

        x_max = np.array([5.0, 5.0])
        x_norm_max = optimizer._normalize_params(x_max)
        assert np.allclose(x_norm_max, [1.0, 1.0])

        # Test denormalization
        x_denorm = optimizer._denormalize_params(x_norm)
        assert np.allclose(x_denorm, x)

    def test_initial_sampling(self, simple_objective, bounds):
        """Test initial random sampling phase."""
        optimizer = BayesianOptimizer(
            simple_objective,
            bounds,
            n_iter=10,
            n_initial=5,
            random_state=42
        )

        # First few suggestions should be random
        for i in range(5):
            params = optimizer.suggest()
            assert isinstance(params, dict)
            assert "x" in params and "y" in params
            assert bounds["x"][0] <= params["x"] <= bounds["x"][1]
            assert bounds["y"][0] <= params["y"] <= bounds["y"][1]

            # Add observation
            score = simple_objective(params)
            optimizer.X_observed.append(optimizer._params_to_array(params))
            optimizer.y_observed.append(score)

    def test_acquisition_functions(self, simple_objective, bounds):
        """Test different acquisition functions."""
        # Test EI
        optimizer_ei = BayesianOptimizer(
            simple_objective, bounds, n_iter=20, acquisition="ei"
        )
        result_ei = optimizer_ei.optimize()
        assert isinstance(result_ei, OptimizationResult)
        assert result_ei.best_score < 1.0  # Should find near optimum

        # Test UCB
        optimizer_ucb = BayesianOptimizer(
            simple_objective, bounds, n_iter=20, acquisition="ucb"
        )
        result_ucb = optimizer_ucb.optimize()
        assert isinstance(result_ucb, OptimizationResult)
        assert result_ucb.best_score < 1.0

        # Test PI
        optimizer_pi = BayesianOptimizer(
            simple_objective, bounds, n_iter=20, acquisition="pi"
        )
        result_pi = optimizer_pi.optimize()
        assert isinstance(result_pi, OptimizationResult)
        assert result_pi.best_score < 1.0

    def test_optimization_result(self, simple_objective, bounds):
        """Test optimization result structure."""
        optimizer = BayesianOptimizer(
            simple_objective, bounds, n_iter=30, random_state=42
        )
        result = optimizer.optimize()

        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_params, dict)
        assert isinstance(result.best_score, float)
        assert isinstance(result.history, list)
        assert len(result.history) == 30
        assert isinstance(result.convergence_info, dict)

        # Check convergence info
        assert "final_score" in result.convergence_info
        assert "n_evaluations" in result.convergence_info
        assert "convergence_history" in result.convergence_info
        assert "improvement_rate" in result.convergence_info

        # Verify optimization worked
        assert result.best_score < 1.0  # Should find near optimum
        assert abs(result.best_params["x"] - 1.0) < 0.5
        assert abs(result.best_params["y"] - (-2.0)) < 0.5

    def test_convergence_tracking(self, simple_objective, bounds):
        """Test that convergence is properly tracked."""
        optimizer = BayesianOptimizer(
            simple_objective, bounds, n_iter=50, random_state=42
        )
        result = optimizer.optimize()

        convergence_history = result.convergence_info["convergence_history"]

        # Check monotonic improvement
        for i in range(1, len(convergence_history)):
            assert convergence_history[i] <= convergence_history[i-1]

        # Check improvement over time
        assert convergence_history[-1] < convergence_history[0]
        assert result.convergence_info["improvement_rate"] > 0

    def test_get_incumbent(self, simple_objective, bounds):
        """Test getting incumbent (best) solution."""
        optimizer = BayesianOptimizer(simple_objective, bounds, n_iter=10)

        # Before optimization
        params, score = optimizer.get_incumbent()
        assert params is None
        assert score == float('inf')

        # After optimization
        optimizer.optimize()
        params, score = optimizer.get_incumbent()
        assert isinstance(params, dict)
        assert isinstance(score, float)
        assert score < float('inf')

    def test_noisy_objective(self):
        """Test optimization with noisy objective function."""
        def noisy_objective(params: Dict[str, Any]) -> float:
            x = params.get("x", 0)
            noise = np.random.normal(0, 0.1)
            return x ** 2 + noise

        bounds = {"x": (-2.0, 2.0)}
        optimizer = BayesianOptimizer(
            noisy_objective,
            bounds,
            n_iter=50,
            random_state=42,
            alpha=0.1  # Account for noise
        )

        result = optimizer.optimize()
        # Should still find near optimum despite noise
        assert abs(result.best_params["x"]) < 0.5

    def test_high_dimensional(self):
        """Test optimization in higher dimensions."""
        def high_dim_objective(params: Dict[str, Any]) -> float:
            # Sum of squares with minimum at origin
            return sum(params[f"x{i}"] ** 2 for i in range(5))

        bounds = {f"x{i}": (-1.0, 1.0) for i in range(5)}
        optimizer = BayesianOptimizer(
            high_dim_objective,
            bounds,
            n_iter=100,
            random_state=42
        )

        result = optimizer.optimize()
        # Should find near origin
        assert result.best_score < 0.5
        for i in range(5):
            assert abs(result.best_params[f"x{i}"]) < 0.5

    def test_discrete_parameters(self):
        """Test with mixed continuous and discrete parameters."""
        def mixed_objective(params: Dict[str, Any]) -> float:
            x = params.get("x", 0)
            category = params.get("category", "A")

            # Different optima for different categories
            if category == "A":
                return (x - 1) ** 2
            elif category == "B":
                return (x + 1) ** 2
            else:
                return x ** 2

        # Note: Current implementation treats all as continuous
        # This test documents expected behavior
        bounds = {
            "x": (-3.0, 3.0),
            "category": (0.0, 2.0)  # Will be treated as continuous
        }

        optimizer = BayesianOptimizer(
            mixed_objective,
            bounds,
            n_iter=50,
            random_state=42
        )

        result = optimizer.optimize()
        assert result.best_score < 1.0

    def test_invalid_bounds(self):
        """Test error handling for invalid bounds."""
        with pytest.raises(ValueError):
            BayesianOptimizer(
                lambda x: 0,
                {"x": (1.0, 1.0)},  # Invalid: min == max
                n_iter=10
            )

    def test_objective_failure(self, bounds):
        """Test handling of objective function failures."""
        def failing_objective(params: Dict[str, Any]) -> float:
            if params["x"] > 0:
                raise ValueError("Simulated failure")
            return params["x"] ** 2

        optimizer = BayesianOptimizer(
            failing_objective,
            bounds,
            n_iter=20,
            random_state=42
        )

        # Should handle failures gracefully
        result = optimizer.optimize()
        assert len(result.history) == 20
        # Should find solutions in valid region (x <= 0)
        assert result.best_params["x"] <= 0


class TestAcquisitionFunctions:
    """Test acquisition function implementations."""

    @pytest.fixture
    def mock_gp(self):
        """Mock Gaussian Process for testing acquisition functions."""
        from unittest.mock import Mock
        gp = Mock()
        return gp

    def test_expected_improvement(self, mock_gp):
        """Test Expected Improvement acquisition function."""
        optimizer = BayesianOptimizer(
            lambda x: 0,
            {"x": (0, 1)},
            acquisition="ei"
        )

        # Test with high uncertainty
        mock_gp.predict.return_value = (np.array([[0.5]]), np.array([[1.0]]))
        x = np.array([0.5])
        ei = optimizer._acquisition_ei(x, mock_gp, y_min=0.0)
        assert ei > 0

        # Test with zero uncertainty
        mock_gp.predict.return_value = (np.array([[0.5]]), np.array([[1e-12]]))
        ei = optimizer._acquisition_ei(x, mock_gp, y_min=0.0)
        assert ei == 0.0

    def test_upper_confidence_bound(self, mock_gp):
        """Test Upper Confidence Bound acquisition function."""
        optimizer = BayesianOptimizer(
            lambda x: 0,
            {"x": (0, 1)},
            acquisition="ucb",
            kappa=2.0
        )

        mock_gp.predict.return_value = (np.array([[0.5]]), np.array([[1.0]]))
        x = np.array([0.5])
        ucb = optimizer._acquisition_ucb(x, mock_gp, y_min=0.0)

        # UCB should be negative (since we minimize negative UCB)
        assert ucb < 0

    def test_probability_improvement(self, mock_gp):
        """Test Probability of Improvement acquisition function."""
        optimizer = BayesianOptimizer(
            lambda x: 0,
            {"x": (0, 1)},
            acquisition="pi"
        )

        # Test with improvement likely
        mock_gp.predict.return_value = (np.array([[0.5]]), np.array([[1.0]]))
        x = np.array([0.5])
        pi = optimizer._acquisition_pi(x, mock_gp, y_min=1.0)
        assert 0 <= pi <= 1

        # Test with zero uncertainty
        mock_gp.predict.return_value = (np.array([[0.5]]), np.array([[1e-12]]))
        pi = optimizer._acquisition_pi(x, mock_gp, y_min=0.0)
        assert pi == 0.0


def test_rosenbrock_optimization():
    """Test optimization on the Rosenbrock function."""
    def rosenbrock(params):
        x, y = params["x"], params["y"]
        return (1 - x)**2 + 100 * (y - x**2)**2

    bounds = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}

    optimizer = BayesianOptimizer(
        objective_fn=rosenbrock,
        bounds=bounds,
        n_iter=100,
        acquisition="ei",
        random_state=42
    )

    result = optimizer.optimize()

    # Rosenbrock minimum is at (1, 1)
    assert abs(result.best_params["x"] - 1.0) < 0.2
    assert abs(result.best_params["y"] - 1.0) < 0.2
    assert result.best_score < 0.1
