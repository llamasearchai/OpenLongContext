"""
MLflow Integration

Comprehensive MLflow integration for experiment tracking and model management.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import mlflow.pytorch
import numpy as np
import torch

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking integration."""

    def __init__(
        self,
        experiment_name: str = "openlongcontext",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: Location to store artifacts
        """
        self.experiment_name = experiment_name

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
                logger.info(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")

            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id

        except Exception as e:
            logger.error(f"Failed to initialize MLflow experiment: {e}")
            raise

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {mlflow.active_run().info.run_id}")

    def end_run(self):
        """End the current MLflow run."""
        if mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {run_id}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        try:
            # Convert complex types to strings
            processed_params = {}
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    processed_params[key] = str(value)
                else:
                    processed_params[key] = value

            mlflow.log_params(processed_params)
            logger.debug(f"Logged {len(processed_params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to MLflow."""
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    mlflow.log_metric(key, float(value), step=step)
                else:
                    logger.warning(f"Skipping non-numeric metric: {key} = {value}")

            logger.debug(f"Logged {len(metrics)} metrics to MLflow")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to MLflow."""
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log a directory of artifacts to MLflow."""
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.debug(f"Logged artifacts from: {local_dir}")
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        **kwargs
    ):
        """Log a PyTorch model to MLflow."""
        try:
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
            logger.info(f"Logged model to MLflow: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_text(self, text: str, artifact_file: str):
        """Log text content as an artifact."""
        try:
            mlflow.log_text(text, artifact_file)
            logger.debug(f"Logged text artifact: {artifact_file}")
        except Exception as e:
            logger.error(f"Failed to log text: {e}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """Log dictionary as JSON artifact."""
        try:
            mlflow.log_dict(dictionary, artifact_file)
            logger.debug(f"Logged dict artifact: {artifact_file}")
        except Exception as e:
            logger.error(f"Failed to log dict: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the current run."""
        try:
            mlflow.set_tags(tags)
            logger.debug(f"Set {len(tags)} tags")
        except Exception as e:
            logger.error(f"Failed to set tags: {e}")

    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current run."""
        if mlflow.active_run():
            run = mlflow.active_run()
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "artifact_uri": run.info.artifact_uri
            }
        return None


class MLflowContextManager:
    """Context manager for MLflow runs."""

    def __init__(
        self,
        tracker: MLflowTracker,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags

    def __enter__(self):
        self.tracker.start_run(self.run_name, self.tags)
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.end_run()


def create_mlflow_tracker(
    experiment_name: str = "openlongcontext",
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None
) -> MLflowTracker:
    """
    Create and configure an MLflow tracker.
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI
        artifact_location: Location to store artifacts
        
    Returns:
        Configured MLflowTracker instance
    """
    return MLflowTracker(experiment_name, tracking_uri, artifact_location)


def log_experiment_results(
    tracker: MLflowTracker,
    config: Dict[str, Any],
    metrics: Dict[str, Union[float, int]],
    model: Optional[torch.nn.Module] = None,
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None
):
    """
    Log complete experiment results to MLflow.
    
    Args:
        tracker: MLflowTracker instance
        config: Experiment configuration
        metrics: Performance metrics
        model: Optional model to log
        artifacts: Optional artifacts to log (path -> artifact_path mapping)
        tags: Optional tags to set
    """
    try:
        # Log parameters
        tracker.log_params(config)

        # Log metrics
        tracker.log_metrics(metrics)

        # Set tags
        if tags:
            tracker.set_tags(tags)

        # Log model
        if model:
            tracker.log_model(model)

        # Log artifacts
        if artifacts:
            for local_path, artifact_path in artifacts.items():
                if Path(local_path).is_dir():
                    tracker.log_artifacts(local_path, artifact_path)
                else:
                    tracker.log_artifact(local_path, artifact_path)

        logger.info("Successfully logged experiment results to MLflow")

    except Exception as e:
        logger.error(f"Failed to log experiment results: {e}")
        raise
