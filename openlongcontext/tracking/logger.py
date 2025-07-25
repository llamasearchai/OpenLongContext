"""
Logger

Comprehensive logging setup for OpenLongContext.
Provides structured logging with multiple handlers and formatters.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_rich: bool = True,
    format_string: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging for OpenLongContext.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        use_rich: Whether to use rich formatting for console output
        format_string: Custom format string for log messages
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger level
    root_logger.setLevel(level)

    # Default format string
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler
    if use_rich:
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False
        )
        console_handler.setLevel(level)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        formatter = logging.Formatter(format_string)
        console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels for common libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    # Return the root logger
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name for the logger (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLogger:
    """
    Structured logger that provides consistent formatting for experiments.
    """

    def __init__(self, name: str, extra_fields: Optional[Dict[str, Any]] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            extra_fields: Additional fields to include in all log messages
        """
        self.logger = logging.getLogger(name)
        self.extra_fields = extra_fields or {}

    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Format message with structured fields."""
        fields = {**self.extra_fields}
        if extra:
            fields.update(extra)

        if fields:
            field_str = " | ".join(f"{k}={v}" for k, v in fields.items())
            return f"{message} | {field_str}"
        return message

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with structured fields."""
        self.logger.debug(self._format_message(message, extra))

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with structured fields."""
        self.logger.info(self._format_message(message, extra))

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with structured fields."""
        self.logger.warning(self._format_message(message, extra))

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message with structured fields."""
        self.logger.error(self._format_message(message, extra))

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message with structured fields."""
        self.logger.critical(self._format_message(message, extra))


class ExperimentLogger(StructuredLogger):
    """
    Specialized logger for experiment tracking.
    """

    def __init__(self, experiment_name: str, run_id: Optional[str] = None):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            run_id: Unique identifier for this run
        """
        extra_fields = {
            "experiment": experiment_name,
            "run_id": run_id or "unknown"
        }
        super().__init__(f"experiment.{experiment_name}", extra_fields)

    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters for the experiment."""
        self.info("Hyperparameters", extra={"hyperparams": hyperparams})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for the experiment."""
        extra: Dict[str, Any] = {"metrics": metrics}
        if step is not None:
            extra["step"] = step
        self.info("Metrics", extra=extra)

    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information."""
        self.info("Model Info", extra={"model": model_info})

    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information."""
        self.info("Dataset Info", extra={"dataset": dataset_info})


def configure_experiment_logging(
    experiment_name: str,
    log_dir: str = "./logs",
    level: int = logging.INFO
) -> ExperimentLogger:
    """
    Configure logging for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        level: Logging level
        
    Returns:
        Configured experiment logger
    """
    log_file = Path(log_dir) / f"{experiment_name}.log"
    setup_logging(level=level, log_file=str(log_file))

    return ExperimentLogger(experiment_name)
