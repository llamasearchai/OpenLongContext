"""
Weights & Biases Integration

Comprehensive W&B integration for experiment tracking and visualization.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

logger = logging.getLogger(__name__)


class WandBTracker:
    """Weights & Biases experiment tracking integration."""

    def __init__(
        self,
        project: str = "openlongcontext",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online"
    ):
        """
        Initialize W&B tracker.
        
        Args:
            project: W&B project name
            entity: W&B entity (username or team)
            name: Run name
            config: Configuration dictionary
            tags: List of tags for the run
            notes: Optional notes for the run
            mode: W&B mode ('online', 'offline', 'disabled')
        """
        self.project = project
        self.entity = entity
        self.run = None

        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
                mode=mode,
                reinit=True
            )

            logger.info(f"Initialized W&B run: {self.run.name} (ID: {self.run.id})")

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            raise

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, np.number]],
        step: Optional[int] = None,
        commit: bool = True
    ):
        """Log metrics to W&B."""
        try:
            # Filter numeric values
            numeric_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    numeric_metrics[key] = float(value)
                else:
                    logger.warning(f"Skipping non-numeric metric: {key} = {value}")

            wandb.log(numeric_metrics, step=step, commit=commit)
            logger.debug(f"Logged {len(numeric_metrics)} metrics to W&B")

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to W&B config."""
        try:
            wandb.config.update(params)
            logger.debug(f"Updated W&B config with {len(params)} parameters")
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")

    def log_artifact(
        self,
        artifact_path: str,
        name: Optional[str] = None,
        type: str = "dataset",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an artifact to W&B."""
        try:
            artifact = wandb.Artifact(
                name=name or Path(artifact_path).name,
                type=type,
                description=description,
                metadata=metadata
            )

            if Path(artifact_path).is_dir():
                artifact.add_dir(artifact_path)
            else:
                artifact.add_file(artifact_path)

            wandb.log_artifact(artifact)
            logger.debug(f"Logged artifact: {artifact_path}")

        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def log_model(
        self,
        model: torch.nn.Module,
        name: str = "model",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a PyTorch model to W&B."""
        try:
            # Save model temporarily
            model_path = f"/tmp/{name}.pth"
            torch.save(model.state_dict(), model_path)

            # Create artifact
            artifact = wandb.Artifact(
                name=name,
                type="model",
                metadata=metadata or {}
            )
            artifact.add_file(model_path)

            wandb.log_artifact(artifact)
            logger.info(f"Logged model to W&B: {name}")

            # Clean up
            Path(model_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_image(
        self,
        image: Union[str, np.ndarray, plt.Figure],
        key: str = "image",
        caption: Optional[str] = None,
        step: Optional[int] = None
    ):
        """Log an image to W&B."""
        try:
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)
            logger.debug(f"Logged image: {key}")
        except Exception as e:
            logger.error(f"Failed to log image: {e}")

    def log_table(
        self,
        data: List[List[Any]],
        columns: List[str],
        key: str = "table",
        step: Optional[int] = None
    ):
        """Log a table to W&B."""
        try:
            table = wandb.Table(data=data, columns=columns)
            wandb.log({key: table}, step=step)
            logger.debug(f"Logged table: {key}")
        except Exception as e:
            logger.error(f"Failed to log table: {e}")

    def log_histogram(
        self,
        values: Union[np.ndarray, List[float]],
        key: str = "histogram",
        step: Optional[int] = None,
        num_bins: int = 64
    ):
        """Log a histogram to W&B."""
        try:
            wandb.log({key: wandb.Histogram(values, num_bins=num_bins)}, step=step)
            logger.debug(f"Logged histogram: {key}")
        except Exception as e:
            logger.error(f"Failed to log histogram: {e}")

    def log_text(
        self,
        text: str,
        key: str = "text",
        step: Optional[int] = None
    ):
        """Log text to W&B."""
        try:
            wandb.log({key: wandb.Html(text)}, step=step)
            logger.debug(f"Logged text: {key}")
        except Exception as e:
            logger.error(f"Failed to log text: {e}")

    def log_audio(
        self,
        audio_path: str,
        key: str = "audio",
        sample_rate: int = 22050,
        caption: Optional[str] = None,
        step: Optional[int] = None
    ):
        """Log audio to W&B."""
        try:
            wandb.log({
                key: wandb.Audio(audio_path, sample_rate=sample_rate, caption=caption)
            }, step=step)
            logger.debug(f"Logged audio: {key}")
        except Exception as e:
            logger.error(f"Failed to log audio: {e}")

    def log_code(self, code_dir: str = "."):
        """Log code files to W&B."""
        try:
            wandb.run.log_code(code_dir)
            logger.debug(f"Logged code from: {code_dir}")
        except Exception as e:
            logger.error(f"Failed to log code: {e}")

    def watch_model(
        self,
        model: torch.nn.Module,
        criterion: Optional[torch.nn.Module] = None,
        log: str = "gradients",
        log_freq: int = 100,
        idx: Optional[int] = None
    ):
        """Watch a model for gradient and parameter tracking."""
        try:
            wandb.watch(model, criterion, log=log, log_freq=log_freq, idx=idx)
            logger.info("Started watching model for gradient tracking")
        except Exception as e:
            logger.error(f"Failed to watch model: {e}")

    def unwatch_model(self, model: torch.nn.Module):
        """Stop watching a model."""
        try:
            wandb.unwatch(model)
            logger.info("Stopped watching model")
        except Exception as e:
            logger.error(f"Failed to unwatch model: {e}")

    def add_tags(self, tags: List[str]):
        """Add tags to the current run."""
        try:
            wandb.run.tags = wandb.run.tags + tags
            logger.debug(f"Added tags: {tags}")
        except Exception as e:
            logger.error(f"Failed to add tags: {e}")

    def set_summary(self, summary: Dict[str, Any]):
        """Set summary metrics for the run."""
        try:
            for key, value in summary.items():
                wandb.run.summary[key] = value
            logger.debug(f"Set summary with {len(summary)} metrics")
        except Exception as e:
            logger.error(f"Failed to set summary: {e}")

    def finish(self):
        """Finish the W&B run."""
        try:
            if self.run:
                wandb.finish()
                logger.info(f"Finished W&B run: {self.run.name}")
        except Exception as e:
            logger.error(f"Failed to finish W&B run: {e}")

    def get_run_url(self) -> Optional[str]:
        """Get the URL of the current run."""
        try:
            if self.run:
                return self.run.get_url()
        except Exception as e:
            logger.error(f"Failed to get run URL: {e}")
        return None


class WandBContextManager:
    """Context manager for W&B runs."""

    def __init__(
        self,
        project: str = "openlongcontext",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online"
    ):
        self.tracker = WandBTracker(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            mode=mode
        )

    def __enter__(self):
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.finish()


def create_wandb_tracker(
    project: str = "openlongcontext",
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    mode: str = "online"
) -> WandBTracker:
    """
    Create and configure a W&B tracker.
    
    Args:
        project: W&B project name
        entity: W&B entity (username or team)
        name: Run name
        config: Configuration dictionary
        tags: List of tags for the run
        notes: Optional notes for the run
        mode: W&B mode ('online', 'offline', 'disabled')
        
    Returns:
        Configured WandBTracker instance
    """
    return WandBTracker(
        project=project,
        entity=entity,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        mode=mode
    )


def log_experiment_results(
    tracker: WandBTracker,
    config: Dict[str, Any],
    metrics: Dict[str, Union[float, int]],
    model: Optional[torch.nn.Module] = None,
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[List[str]] = None,
    summary: Optional[Dict[str, Any]] = None
):
    """
    Log complete experiment results to W&B.
    
    Args:
        tracker: WandBTracker instance
        config: Experiment configuration
        metrics: Performance metrics
        model: Optional model to log
        artifacts: Optional artifacts to log
        tags: Optional tags to add
        summary: Optional summary metrics
    """
    try:
        # Log hyperparameters
        tracker.log_hyperparameters(config)

        # Log metrics
        tracker.log_metrics(metrics)

        # Add tags
        if tags:
            tracker.add_tags(tags)

        # Log model
        if model:
            tracker.log_model(model)

        # Log artifacts
        if artifacts:
            for artifact_path, artifact_name in artifacts.items():
                tracker.log_artifact(artifact_path, name=artifact_name)

        # Set summary
        if summary:
            tracker.set_summary(summary)

        logger.info("Successfully logged experiment results to W&B")

    except Exception as e:
        logger.error(f"Failed to log experiment results: {e}")
        raise
