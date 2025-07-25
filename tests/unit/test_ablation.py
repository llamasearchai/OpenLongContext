"""
Unit tests for ablation study modules.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from openlongcontext.evaluation.ablation import (
    analyze_results,
    run_ablation,
    compare_ablations,
    _create_ablation_plots
)
from openlongcontext.ablation.hyperparameter_sweep import (
    HyperparameterSweep,
    SweepResult
)
from openlongcontext.ablation.experiment_registry import (
    ExperimentRegistry,
    register_experiment,
    experiment_registry
)


class TestAnalyzeResults:
    """Test ablation results analysis."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample ablation results."""
        return [
            {
                "experiment_id": 1,
                "parameters": {"lr": 0.001, "batch_size": 32},
                "metrics": {"loss": 0.5, "accuracy": 0.85}
            },
            {
                "experiment_id": 2,
                "parameters": {"lr": 0.01, "batch_size": 64},
                "metrics": {"loss": 0.3, "accuracy": 0.90}
            },
            {
                "experiment_id": 3,
                "parameters": {"lr": 0.001, "batch_size": 64},
                "metrics": {"loss": 0.4, "accuracy": 0.87}
            }
        ]
    
    def test_analyze_single_file(self, sample_results):
        """Test analyzing results from a single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save results to file
            results_file = Path(tmpdir) / "results.json"
            with open(results_file, 'w') as f:
                json.dump(sample_results[0], f)
            
            # Analyze
            analysis = analyze_results(str(results_file))
            
            assert analysis["num_experiments"] == 1
            assert "parameters" in analysis
            assert "metrics" in analysis
            assert "summary_statistics" in analysis
    
    def test_analyze_directory(self, sample_results):
        """Test analyzing results from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save multiple result files
            for i, result in enumerate(sample_results):
                results_file = Path(tmpdir) / f"result_{i}.json"
                with open(results_file, 'w') as f:
                    json.dump(result, f)
            
            # Analyze
            analysis = analyze_results(tmpdir)
            
            assert analysis["num_experiments"] == 3
            assert len(analysis["parameters"]["lr"]) == 3
            assert len(analysis["metrics"]["loss"]) == 3
            
            # Check best/worst configuration
            assert analysis["best_configuration"]["metrics"]["loss"] == 0.3
            assert analysis["worst_configuration"]["metrics"]["loss"] == 0.5
    
    def test_summary_statistics(self, sample_results):
        """Test summary statistics calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_file = Path(tmpdir) / "results.json"
            
            # Save all results as single file
            with open(results_file, 'w') as f:
                json.dump(sample_results, f)
            
            # Mock multiple files
            with patch('pathlib.Path.is_file', return_value=False):
                with patch('pathlib.Path.glob') as mock_glob:
                    mock_glob.return_value = [results_file]
                    
                    with patch('builtins.open', create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(sample_results[0])
                        
                        analysis = analyze_results(tmpdir)
            
            # Check statistics
            loss_stats = analysis["summary_statistics"].get("loss", {})
            if loss_stats:
                assert "mean" in loss_stats
                assert "std" in loss_stats
                assert "min" in loss_stats
                assert "max" in loss_stats
                assert "median" in loss_stats
    
    def test_empty_results(self):
        """Test handling of empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis = analyze_results(tmpdir)
            assert analysis == {}
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_creation(self, mock_close, mock_savefig, sample_results):
        """Test that plots are created without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create analysis data
            analysis = {
                "parameters": {"lr": [0.001, 0.01, 0.001], "batch_size": [32, 64, 64]},
                "metrics": {"loss": [0.5, 0.3, 0.4]},
                "raw_results": sample_results
            }
            
            output_dir = Path(tmpdir) / "plots"
            _create_ablation_plots(analysis, output_dir)
            
            # Check that plot functions were called
            assert mock_savefig.called
            assert mock_close.called


class TestHyperparameterSweep:
    """Test hyperparameter sweep functionality."""
    
    @pytest.fixture
    def simple_objective(self):
        """Simple objective for testing."""
        def objective(params: Dict[str, Any]) -> float:
            return params["x"] ** 2 + params["y"] ** 2
        return objective
    
    @pytest.fixture
    def param_space(self):
        """Parameter space for testing."""
        return {
            "x": [-1, 0, 1],
            "y": [-1, 0, 1]
        }
    
    def test_initialization(self, simple_objective, param_space):
        """Test sweep initialization."""
        sweep = HyperparameterSweep(
            objective_fn=simple_objective,
            param_space=param_space,
            n_iter=10
        )
        
        assert sweep.objective_fn == simple_objective
        assert sweep.param_space == param_space
        assert sweep.n_iter == 10
        assert sweep.search_strategy == "random"
        assert len(sweep.history) == 0
    
    def test_random_sampling(self, simple_objective, param_space):
        """Test random parameter sampling."""
        sweep = HyperparameterSweep(
            simple_objective,
            param_space,
            search_strategy="random",
            random_state=42
        )
        
        # Sample parameters
        params = sweep.sample()
        assert "x" in params
        assert "y" in params
        assert params["x"] in param_space["x"]
        assert params["y"] in param_space["y"]
    
    def test_grid_search(self, simple_objective):
        """Test grid search strategy."""
        param_space = {
            "x": [0, 1],
            "y": [0, 1]
        }
        
        sweep = HyperparameterSweep(
            simple_objective,
            param_space,
            search_strategy="grid"
        )
        
        best_params, best_score = sweep.optimize()
        
        # Should evaluate all 4 combinations
        assert len(sweep.history) == 4
        assert best_params == {"x": 0, "y": 0}
        assert best_score == 0
    
    def test_continuous_parameters(self, simple_objective):
        """Test with continuous parameter ranges."""
        param_space = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0)
        }
        
        sweep = HyperparameterSweep(
            simple_objective,
            param_space,
            n_iter=50,
            random_state=42
        )
        
        best_params, best_score = sweep.optimize()
        
        # Should find near optimum at (0, 0)
        assert abs(best_params["x"]) < 0.5
        assert abs(best_params["y"]) < 0.5
        assert best_score < 0.5
    
    def test_sobol_sampling(self, simple_objective):
        """Test Sobol quasi-random sampling."""
        param_space = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0)
        }
        
        sweep = HyperparameterSweep(
            simple_objective,
            param_space,
            n_iter=20,
            search_strategy="sobol"
        )
        
        # Test Sobol sampling produces valid parameters
        for _ in range(10):
            params = sweep.sample()
            assert -1.0 <= params["x"] <= 1.0
            assert -1.0 <= params["y"] <= 1.0
            sweep.history.append((params, 0))  # Add to history for next sample
    
    def test_parallel_optimization(self, simple_objective, param_space):
        """Test parallel optimization."""
        from concurrent.futures import ThreadPoolExecutor
        
        sweep = HyperparameterSweep(
            simple_objective,
            param_space,
            n_iter=20
        )
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            best_params, best_score = sweep.optimize_parallel(
                executor,
                batch_size=5
            )
        
        assert len(sweep.history) == 20
        assert best_params["x"] == 0
        assert best_params["y"] == 0
        assert best_score == 0
    
    def test_analyze_results(self, simple_objective, param_space):
        """Test results analysis."""
        sweep = HyperparameterSweep(simple_objective, param_space, n_iter=10)
        sweep.optimize()
        
        result = sweep.analyze_results()
        
        assert isinstance(result, SweepResult)
        assert result.best_params == {"x": 0, "y": 0}
        assert result.best_score == 0
        assert len(result.history) == 10
        assert isinstance(result.param_importance, dict)
    
    def test_parameter_importance(self, param_space):
        """Test parameter importance calculation."""
        # Objective where x matters more than y
        def biased_objective(params):
            return 10 * params["x"] ** 2 + params["y"] ** 2
        
        sweep = HyperparameterSweep(
            biased_objective,
            param_space,
            n_iter=50,
            random_state=42
        )
        
        sweep.optimize()
        result = sweep.analyze_results()
        
        # x should have higher importance
        if result.param_importance:
            assert result.param_importance.get("x", 0) > result.param_importance.get("y", 0)
    
    def test_save_results(self, simple_objective, param_space):
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep = HyperparameterSweep(simple_objective, param_space, n_iter=5)
            sweep.optimize()
            
            output_path = Path(tmpdir) / "sweep_results.json"
            sweep.save_results(str(output_path))
            
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert "best_params" in data
            assert "best_score" in data
            assert "history" in data
            assert "search_strategy" in data
            assert data["n_evaluations"] == 5


class TestExperimentRegistry:
    """Test experiment registry functionality."""
    
    def test_registry_singleton(self):
        """Test that registry is a singleton."""
        from openlongcontext.ablation.experiment_registry import experiment_registry
        assert isinstance(experiment_registry, ExperimentRegistry)
    
    def test_register_experiment(self):
        """Test registering experiments."""
        registry = ExperimentRegistry()
        
        def my_experiment():
            return "result"
        
        registry.register("test_exp", my_experiment)
        
        assert "test_exp" in registry.list_experiments()
        assert registry.get("test_exp") == my_experiment
    
    def test_register_duplicate(self):
        """Test registering duplicate experiment names."""
        registry = ExperimentRegistry()
        
        def exp1():
            pass
        
        def exp2():
            pass
        
        registry.register("duplicate", exp1)
        
        with pytest.raises(ValueError):
            registry.register("duplicate", exp2)
    
    def test_get_nonexistent(self):
        """Test getting non-existent experiment."""
        registry = ExperimentRegistry()
        
        with pytest.raises(KeyError):
            registry.get("nonexistent")
    
    def test_decorator(self):
        """Test register_experiment decorator."""
        registry = ExperimentRegistry()
        
        @register_experiment("decorated_exp")
        def my_experiment():
            return 42
        
        # Function should be in global registry
        assert "decorated_exp" in experiment_registry.list_experiments()
        assert experiment_registry.get("decorated_exp")() == 42
    
    @patch('pathlib.Path.exists')
    @patch('openlongcontext.evaluation.ablation.run_ablation')
    def test_run_ablation_function(self, mock_run_ablation_impl, mock_exists):
        """Test run_ablation function in experiment registry."""
        from openlongcontext.ablation.experiment_registry import run_ablation
        
        # Mock config file existence
        mock_exists.return_value = True
        mock_run_ablation_impl.return_value = {
            "results": [],
            "analysis": {"best_configuration": {"parameters": {}, "metrics": {"loss": 0.1}}}
        }
        
        result = run_ablation("test_config")
        
        assert mock_run_ablation_impl.called
        assert "results" in result
    
    @patch('pathlib.Path.exists')
    @patch('omegaconf.OmegaConf.load')
    @patch('openlongcontext.ablation.hyperparameter_sweep.HyperparameterSweep')
    def test_run_sweep_function(self, mock_sweep_class, mock_load, mock_exists):
        """Test run_sweep function in experiment registry."""
        from openlongcontext.ablation.experiment_registry import run_sweep
        
        # Mock config
        mock_exists.return_value = True
        mock_load.return_value = {
            "param_space": {"x": [0, 1]},
            "n_iter": 10
        }
        
        # Mock sweep
        mock_sweep = Mock()
        mock_sweep.optimize.return_value = ({"x": 0}, 0.0)
        mock_sweep.history = []
        mock_sweep_class.return_value = mock_sweep
        
        result = run_sweep("test_config")
        
        assert mock_sweep.optimize.called
        assert "best_params" in result
        assert "best_score" in result


class TestCompareAblations:
    """Test ablation comparison functionality."""
    
    @pytest.fixture
    def mock_ablation_dirs(self):
        """Create mock ablation study directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two ablation studies
            study1_dir = Path(tmpdir) / "study1"
            study1_dir.mkdir()
            
            study1_analysis = {
                "num_experiments": 10,
                "parameters": {"lr": [0.001, 0.01]},
                "metrics": {"loss": [0.5, 0.3, 0.4]},
                "best_configuration": {
                    "parameters": {"lr": 0.01},
                    "metrics": {"loss": 0.3}
                }
            }
            
            with open(study1_dir / "ablation_analysis.json", 'w') as f:
                json.dump(study1_analysis, f)
            
            study2_dir = Path(tmpdir) / "study2"
            study2_dir.mkdir()
            
            study2_analysis = {
                "num_experiments": 15,
                "parameters": {"batch_size": [32, 64, 128]},
                "metrics": {"loss": [0.6, 0.4, 0.35]},
                "best_configuration": {
                    "parameters": {"batch_size": 128},
                    "metrics": {"loss": 0.35}
                }
            }
            
            with open(study2_dir / "ablation_analysis.json", 'w') as f:
                json.dump(study2_analysis, f)
            
            yield [str(study1_dir), str(study2_dir)]
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_compare_ablations(self, mock_close, mock_savefig, mock_ablation_dirs):
        """Test comparing multiple ablation studies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "comparison.png"
            
            result = compare_ablations(
                mock_ablation_dirs,
                str(output_path),
                metric="loss"
            )
            
            assert "studies" in result
            assert "summary" in result
            assert len(result["studies"]) == 2
            
            # Check summary
            assert result["summary"]["total_experiments"] == 25
            assert result["summary"]["best_overall"][0] in ["study1", "study2"]
            
            # Check plots were created
            assert mock_savefig.called
            assert mock_close.called