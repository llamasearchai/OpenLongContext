"""
Experiment management module.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config
from .engine import Engine
from .tracking import ExperimentTracker

logger = logging.getLogger(__name__)


class Experiment:
    """Experiment runner and manager."""

    def __init__(self, config: Config, name: Optional[str] = None):
        """Initialize experiment."""
        self.config = config
        self.name = name or config.experiment.name
        self.engine = Engine(config)
        self.tracker = ExperimentTracker(config)
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.status = "initialized"

    def setup(self) -> None:
        """Setup experiment environment."""
        logger.info(f"Setting up experiment: {self.name}")

        # Create output directories
        output_dir = Path(self.config.experiment.output_dir) / self.name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        # Save configuration
        config_path = output_dir / "config.json"
        self.config.save_json(config_path)

        # Initialize tracking
        self.tracker.init_experiment(self.name, self.config.to_dict())

        # Setup engine
        self.engine.setup()

        self.status = "ready"

    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        logger.info(f"Running experiment: {self.name}")
        self.start_time = datetime.now()
        self.status = "running"

        try:
            # Setup if not already done
            if self.status != "ready":
                self.setup()

            # Log experiment start
            self.tracker.log_metric("experiment_started", 1)

            # Training phase
            if self.config.training.num_epochs > 0:
                logger.info("Starting training phase...")
                train_results = self.engine.train()
                self.results["training"] = train_results

                # Log training metrics
                for metric_name, value in train_results.get("metrics", {}).items():
                    self.tracker.log_metric(f"train_{metric_name}", value)

            # Evaluation phase
            logger.info("Starting evaluation phase...")
            eval_results = self.engine.evaluate()
            self.results["evaluation"] = eval_results

            # Log evaluation metrics
            for metric_name, value in eval_results.get("metrics", {}).items():
                self.tracker.log_metric(f"eval_{metric_name}", value)

            # Compute final metrics
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            self.results["duration_seconds"] = duration.total_seconds()
            self.results["status"] = "completed"
            self.status = "completed"

            # Save results
            self.save_results()

            # Finalize tracking
            self.tracker.log_metric("experiment_completed", 1)
            self.tracker.finalize()

            logger.info(f"Experiment completed successfully in {duration}")
            return self.results

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            self.results["error"] = str(e)
            self.results["status"] = "failed"
            self.status = "failed"

            # Log failure
            self.tracker.log_metric("experiment_failed", 1)
            self.tracker.log_text("error", str(e))

            # Save partial results
            self.save_results()

            raise

    def save_results(self) -> None:
        """Save experiment results."""
        results_path = self.output_dir / "results.json"

        # Add metadata
        results_with_metadata = {
            "name": self.name,
            "config": self.config.to_dict(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            **self.results
        }

        with open(results_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)

        logger.info(f"Results saved to {results_path}")

        # Also save to tracker
        self.tracker.log_artifact(results_path, "results")

    def load_results(self, path: str) -> Dict[str, Any]:
        """Load experiment results from file."""
        with open(path) as f:
            return json.load(f)

    def resume(self, checkpoint_path: str) -> None:
        """Resume experiment from checkpoint."""
        logger.info(f"Resuming experiment from {checkpoint_path}")

        # Load checkpoint
        self.engine.load_checkpoint(checkpoint_path)

        # Update status
        self.status = "resumed"

    def get_status(self) -> Dict[str, Any]:
        """Get current experiment status."""
        return {
            "name": self.name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "current_metrics": self.results.get("metrics", {}),
            "output_dir": str(self.output_dir) if hasattr(self, "output_dir") else None
        }

    @classmethod
    def from_config_file(cls, config_path: str, name: Optional[str] = None) -> "Experiment":
        """Create experiment from configuration file."""
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            config = Config.from_yaml(config_path)
        elif config_path.endswith(".json"):
            config = Config.from_json(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")

        return cls(config, name)
