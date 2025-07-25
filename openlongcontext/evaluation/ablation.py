"""
Ablation study analysis and evaluation utilities.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_results(results_path: str) -> Dict[str, Any]:
    """
    Analyze ablation study results from experiments.
    
    Args:
        results_path: Path to results directory or file
        
    Returns:
        Dictionary containing analysis results
    """
    results_path = Path(results_path)

    if results_path.is_file():
        # Single results file
        with open(results_path) as f:
            data = json.load(f)
        results = [data]
    else:
        # Directory with multiple result files
        results = []
        for file_path in results_path.glob("*.json"):
            with open(file_path) as f:
                results.append(json.load(f))

    if not results:
        logger.warning(f"No results found at {results_path}")
        return {}

    # Aggregate results
    analysis = {
        "num_experiments": len(results),
        "parameters": defaultdict(list),
        "metrics": defaultdict(list),
        "best_configuration": None,
        "worst_configuration": None,
        "summary_statistics": {}
    }

    # Extract parameters and metrics
    for result in results:
        if "parameters" in result:
            for param, value in result["parameters"].items():
                analysis["parameters"][param].append(value)

        if "metrics" in result:
            for metric, value in result["metrics"].items():
                analysis["metrics"][metric].append(value)

    # Calculate summary statistics
    for metric, values in analysis["metrics"].items():
        analysis["summary_statistics"][metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values)
        }

    # Find best and worst configurations
    if "loss" in analysis["metrics"]:
        best_idx = np.argmin(analysis["metrics"]["loss"])
        worst_idx = np.argmax(analysis["metrics"]["loss"])

        analysis["best_configuration"] = {
            "index": int(best_idx),
            "parameters": {k: v[best_idx] for k, v in analysis["parameters"].items()},
            "metrics": {k: v[best_idx] for k, v in analysis["metrics"].items()}
        }

        analysis["worst_configuration"] = {
            "index": int(worst_idx),
            "parameters": {k: v[worst_idx] for k, v in analysis["parameters"].items()},
            "metrics": {k: v[worst_idx] for k, v in analysis["metrics"].items()}
        }

    # Create visualizations if possible
    try:
        _create_ablation_plots(analysis, results_path.parent / "ablation_plots")
    except Exception as e:
        logger.warning(f"Could not create plots: {e}")

    # Save analysis results
    output_path = results_path.parent / "ablation_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    logger.info(f"Analysis complete. Results saved to {output_path}")

    # Print summary
    print("\n=== Ablation Study Analysis ===")
    print(f"Total experiments: {analysis['num_experiments']}")
    print("\nParameter ranges:")
    for param, values in analysis["parameters"].items():
        unique_values = set(values)
        print(f"  {param}: {len(unique_values)} unique values")

    print("\nMetric summary:")
    for metric, stats in analysis["summary_statistics"].items():
        print(f"  {metric}:")
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    if analysis["best_configuration"]:
        print("\nBest configuration:")
        for param, value in analysis["best_configuration"]["parameters"].items():
            print(f"  {param}: {value}")
        print(f"  Loss: {analysis['best_configuration']['metrics'].get('loss', 'N/A')}")

    return analysis


def _create_ablation_plots(analysis: Dict[str, Any], output_dir: Path) -> None:
    """Create visualization plots for ablation analysis."""
    output_dir.mkdir(exist_ok=True)

    # Parameter importance plot
    if len(analysis["parameters"]) > 1 and "loss" in analysis["metrics"]:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate parameter importance using correlation
        param_importance = {}
        loss_values = analysis["metrics"]["loss"]

        for param, values in analysis["parameters"].items():
            if len(set(values)) > 1:  # Only for varying parameters
                # Handle both numeric and categorical parameters
                try:
                    numeric_values = [float(v) for v in values]
                    correlation = abs(np.corrcoef(numeric_values, loss_values)[0, 1])
                    param_importance[param] = correlation
                except (ValueError, TypeError):
                    # Categorical parameter - use variance ratio
                    unique_vals = list(set(values))
                    group_losses = [[] for _ in unique_vals]
                    for v, l in zip(values, loss_values):
                        idx = unique_vals.index(v)
                        group_losses[idx].append(l)

                    # Calculate between-group variance
                    group_means = [np.mean(g) if g else 0 for g in group_losses]
                    overall_mean = np.mean(loss_values)
                    between_var = sum(len(g) * (m - overall_mean)**2 for g, m in zip(group_losses, group_means))
                    total_var = np.var(loss_values) * len(loss_values)
                    param_importance[param] = between_var / total_var if total_var > 0 else 0

        if param_importance:
            params = list(param_importance.keys())
            importances = list(param_importance.values())

            ax.bar(params, importances)
            ax.set_xlabel("Parameter")
            ax.set_ylabel("Importance Score")
            ax.set_title("Parameter Importance in Ablation Study")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / "parameter_importance.png")
            plt.close()

    # Loss distribution plot
    if "loss" in analysis["metrics"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        losses = analysis["metrics"]["loss"]

        ax.hist(losses, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(losses), color='red', linestyle='--', label=f'Mean: {np.mean(losses):.4f}')
        ax.axvline(np.median(losses), color='green', linestyle='--', label=f'Median: {np.median(losses):.4f}')
        ax.set_xlabel("Loss")
        ax.set_ylabel("Frequency")
        ax.set_title("Loss Distribution Across Ablation Experiments")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_distribution.png")
        plt.close()

    # Convergence plot if history is available
    if results := analysis.get("raw_results"):
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, result in enumerate(results[:10]):  # Plot first 10 for clarity
            if "history" in result:
                history = result["history"]
                if isinstance(history, list) and all(isinstance(h, (int, float)) for h in history):
                    ax.plot(history, alpha=0.5, label=f"Exp {i}")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Convergence Across Experiments")
        if len(ax.lines) <= 10:
            ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "convergence_curves.png")
        plt.close()


def run_ablation(
    config_path: str,
    output_dir: str,
    parameters_to_ablate: Optional[List[str]] = None,
    n_trials: int = 10
) -> Dict[str, Any]:
    """
    Run ablation study on specified parameters.
    
    Args:
        config_path: Path to base configuration
        output_dir: Directory to save results
        parameters_to_ablate: List of parameter names to ablate
        n_trials: Number of trials per parameter configuration
        
    Returns:
        Dictionary containing ablation results
    """

    from omegaconf import OmegaConf

    from ..core.config import Config
    from ..core.experiment import Experiment

    # Load base configuration
    base_config = OmegaConf.load(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define parameter variations
    ablation_params = {
        "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2],
        "batch_size": [16, 32, 64, 128],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "hidden_size": [128, 256, 512, 1024],
        "num_layers": [2, 4, 6, 8],
        "optimizer": ["adam", "sgd", "adamw"],
        "scheduler": ["constant", "linear", "cosine"],
    }

    # Filter to requested parameters
    if parameters_to_ablate:
        ablation_params = {k: v for k, v in ablation_params.items() if k in parameters_to_ablate}

    results = []
    experiment_id = 0

    # Run ablation for each parameter
    for param_name, param_values in ablation_params.items():
        logger.info(f"Ablating parameter: {param_name}")

        for value in param_values:
            # Create modified config
            config = OmegaConf.to_container(base_config, resolve=True)

            # Update parameter value (handle nested parameters)
            if "." in param_name:
                parts = param_name.split(".")
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config[param_name] = value

            # Run trials
            for trial in range(n_trials):
                experiment_id += 1
                logger.info(f"Running experiment {experiment_id}: {param_name}={value}, trial {trial+1}/{n_trials}")

                try:
                    # Create experiment
                    exp_config = Config(**config)
                    experiment = Experiment(exp_config)

                    # Run experiment
                    metrics = experiment.run()

                    # Save results
                    result = {
                        "experiment_id": experiment_id,
                        "ablation_parameter": param_name,
                        "ablation_value": value,
                        "trial": trial,
                        "parameters": config,
                        "metrics": metrics,
                        "timestamp": pd.Timestamp.now().isoformat()
                    }

                    results.append(result)

                    # Save individual result
                    result_path = output_dir / f"experiment_{experiment_id:04d}.json"
                    with open(result_path, 'w') as f:
                        json.dump(result, f, indent=2)

                except Exception as e:
                    logger.error(f"Experiment {experiment_id} failed: {e}")
                    results.append({
                        "experiment_id": experiment_id,
                        "ablation_parameter": param_name,
                        "ablation_value": value,
                        "trial": trial,
                        "error": str(e)
                    })

    # Save all results
    all_results_path = output_dir / "all_results.json"
    with open(all_results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Run analysis
    analysis = analyze_results(all_results_path)

    return {
        "results": results,
        "analysis": analysis,
        "output_dir": str(output_dir)
    }


def compare_ablations(
    ablation_dirs: List[str],
    output_path: str,
    metric: str = "loss"
) -> Dict[str, Any]:
    """
    Compare results from multiple ablation studies.
    
    Args:
        ablation_dirs: List of directories containing ablation results
        output_path: Path to save comparison results
        metric: Metric to use for comparison
        
    Returns:
        Dictionary containing comparison results
    """
    comparisons = {}

    for ablation_dir in ablation_dirs:
        name = Path(ablation_dir).name
        analysis_path = Path(ablation_dir) / "ablation_analysis.json"

        if analysis_path.exists():
            with open(analysis_path) as f:
                comparisons[name] = json.load(f)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Plot 1: Best scores comparison
    ax = axes[0]
    names = list(comparisons.keys())
    best_scores = [comp["best_configuration"]["metrics"].get(metric, np.nan)
                   for comp in comparisons.values()]

    ax.bar(names, best_scores)
    ax.set_ylabel(f"Best {metric}")
    ax.set_title(f"Best {metric} Across Ablation Studies")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Score distributions
    ax = axes[1]
    for name, comp in comparisons.items():
        if metric in comp["metrics"]:
            ax.hist(comp["metrics"][metric], alpha=0.5, label=name, bins=20)
    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{metric} Distributions")
    ax.legend()

    # Plot 3: Parameter counts
    ax = axes[2]
    param_counts = {name: len(comp["parameters"]) for name, comp in comparisons.items()}
    ax.bar(param_counts.keys(), param_counts.values())
    ax.set_ylabel("Number of Parameters")
    ax.set_title("Parameters Ablated per Study")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 4: Experiment counts
    ax = axes[3]
    exp_counts = {name: comp["num_experiments"] for name, comp in comparisons.items()}
    ax.bar(exp_counts.keys(), exp_counts.values())
    ax.set_ylabel("Number of Experiments")
    ax.set_title("Experiments per Study")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Save comparison data
    comparison_data = {
        "studies": comparisons,
        "summary": {
            "best_overall": min((name, comp["best_configuration"]["metrics"].get(metric, np.inf))
                               for name, comp in comparisons.items()
                               if comp.get("best_configuration")),
            "total_experiments": sum(comp["num_experiments"] for comp in comparisons.values())
        }
    }

    comparison_json_path = Path(output_path).with_suffix('.json')
    with open(comparison_json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    return comparison_data
