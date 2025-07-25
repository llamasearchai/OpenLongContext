"""
Experiment tracking module.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class TrackerBackend(ABC):
    """Abstract base class for tracking backends."""

    @abstractmethod
    def init_experiment(self, name: str, config: Dict[str, Any]) -> None:
        """Initialize a new experiment."""
        pass

    @abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        pass

    @abstractmethod
    def log_text(self, key: str, text: str) -> None:
        """Log text data."""
        pass

    @abstractmethod
    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        """Log an artifact file."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize the experiment."""
        pass


class LocalTracker(TrackerBackend):
    """Local file-based experiment tracker."""

    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = Path(base_dir)
        self.experiment_dir = None
        self.metrics_file = None
        self.step = 0

    def init_experiment(self, name: str, config: Dict[str, Any]) -> None:
        """Initialize a new experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_dir / f"{name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Initialize metrics file
        self.metrics_file = self.experiment_dir / "metrics.jsonl"

        logger.info(f"Initialized local tracking in {self.experiment_dir}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if self.metrics_file is None:
            logger.warning("Tracker not initialized")
            return

        if step is None:
            step = self.step
            self.step += 1

        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "key": key,
            "value": value
        }

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric_entry) + '\n')

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_text(self, key: str, text: str) -> None:
        """Log text data."""
        if self.experiment_dir is None:
            logger.warning("Tracker not initialized")
            return

        text_dir = self.experiment_dir / "texts"
        text_dir.mkdir(exist_ok=True)

        text_path = text_dir / f"{key}.txt"
        with open(text_path, 'w') as f:
            f.write(text)

    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        """Log an artifact file."""
        if self.experiment_dir is None:
            logger.warning("Tracker not initialized")
            return

        artifact_dir = self.experiment_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)

        src_path = Path(path)
        if name is None:
            name = src_path.name

        dst_path = artifact_dir / name

        # Copy file
        import shutil
        shutil.copy2(src_path, dst_path)

    def finalize(self) -> None:
        """Finalize the experiment."""
        if self.experiment_dir is None:
            return

        # Create summary
        summary = {
            "finalized_at": datetime.now().isoformat(),
            "total_steps": self.step,
            "experiment_dir": str(self.experiment_dir)
        }

        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Finalized experiment in {self.experiment_dir}")


class ExperimentTracker:
    """Main experiment tracker that can use different backends."""

    def __init__(self, config: Any):
        """Initialize tracker based on configuration."""
        self.config = config
        self.backends: List[TrackerBackend] = []

        # Always use local tracker
        self.backends.append(LocalTracker(
            base_dir=getattr(config.experiment, "output_dir", "./experiments")
        ))

        # Add other backends based on config
        if hasattr(config.experiment, "wandb_project") and config.experiment.wandb_project:
            try:
                from ..tracking.wandb_integration import WandbTracker
                self.backends.append(WandbTracker(config))
            except ImportError:
                logger.warning("W&B not available")

        if hasattr(config.experiment, "mlflow_tracking_uri") and config.experiment.mlflow_tracking_uri:
            try:
                from ..tracking.mlflow_integration import MLFlowTracker
                self.backends.append(MLFlowTracker(config))
            except ImportError:
                logger.warning("MLflow not available")

        if hasattr(config.experiment, "tensorboard_dir") and config.experiment.tensorboard_dir:
            try:
                from ..tracking.tensorboard_integration import TensorBoardTracker
                self.backends.append(TensorBoardTracker(config))
            except ImportError:
                logger.warning("TensorBoard not available")

    def init_experiment(self, name: str, config: Dict[str, Any]) -> None:
        """Initialize experiment in all backends."""
        for backend in self.backends:
            try:
                backend.init_experiment(name, config)
            except Exception as e:
                logger.error(f"Failed to init experiment in {backend.__class__.__name__}: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log metric to all backends."""
        for backend in self.backends:
            try:
                backend.log_metric(key, value, step)
            except Exception as e:
                logger.error(f"Failed to log metric in {backend.__class__.__name__}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics to all backends."""
        for backend in self.backends:
            try:
                backend.log_metrics(metrics, step)
            except Exception as e:
                logger.error(f"Failed to log metrics in {backend.__class__.__name__}: {e}")

    def log_text(self, key: str, text: str) -> None:
        """Log text to all backends."""
        for backend in self.backends:
            try:
                backend.log_text(key, text)
            except Exception as e:
                logger.error(f"Failed to log text in {backend.__class__.__name__}: {e}")

    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        """Log artifact to all backends."""
        for backend in self.backends:
            try:
                backend.log_artifact(path, name)
            except Exception as e:
                logger.error(f"Failed to log artifact in {backend.__class__.__name__}: {e}")

    def finalize(self) -> None:
        """Finalize all backends."""
        for backend in self.backends:
            try:
                backend.finalize()
            except Exception as e:
                logger.error(f"Failed to finalize {backend.__class__.__name__}: {e}")
