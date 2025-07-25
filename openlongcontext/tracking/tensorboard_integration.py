"""
TensorBoard Integration

Comprehensive TensorBoard integration for experiment tracking and visualization.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Union, List
import logging
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor

logger = logging.getLogger(__name__)


class TensorBoardTracker:
    """TensorBoard experiment tracking integration."""
    
    def __init__(
        self,
        log_dir: str = "runs",
        comment: str = "",
        purge_step: Optional[int] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = ""
    ):
        """
        Initialize TensorBoard tracker.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            comment: Comment to append to default log_dir name
            purge_step: Step to start purging old data
            max_queue: Maximum number of items to queue before writing
            flush_secs: How often to flush data to disk
            filename_suffix: Suffix for log filename
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir),
                comment=comment,
                purge_step=purge_step,
                max_queue=max_queue,
                flush_secs=flush_secs,
                filename_suffix=filename_suffix
            )
            
            logger.info(f"Initialized TensorBoard writer: {self.writer.log_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TensorBoard writer: {e}")
            raise
    
    def log_scalar(
        self,
        tag: str,
        scalar_value: Union[float, int, np.number],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None
    ):
        """Log a scalar value to TensorBoard."""
        try:
            self.writer.add_scalar(tag, scalar_value, global_step, walltime)
            logger.debug(f"Logged scalar: {tag} = {scalar_value}")
        except Exception as e:
            logger.error(f"Failed to log scalar {tag}: {e}")
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, int, np.number]],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None
    ):
        """Log multiple scalars to TensorBoard."""
        try:
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)
            logger.debug(f"Logged scalars: {main_tag} with {len(tag_scalar_dict)} values")
        except Exception as e:
            logger.error(f"Failed to log scalars {main_tag}: {e}")
    
    def log_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.ndarray],
        global_step: Optional[int] = None,
        bins: str = "tensorflow",
        walltime: Optional[float] = None,
        max_bins: Optional[int] = None
    ):
        """Log a histogram to TensorBoard."""
        try:
            self.writer.add_histogram(tag, values, global_step, bins, walltime, max_bins)
            logger.debug(f"Logged histogram: {tag}")
        except Exception as e:
            logger.error(f"Failed to log histogram {tag}: {e}")
    
    def log_image(
        self,
        tag: str,
        img_tensor: Union[torch.Tensor, np.ndarray, PIL.Image.Image],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        dataformats: str = "CHW"
    ):
        """Log an image to TensorBoard."""
        try:
            self.writer.add_image(tag, img_tensor, global_step, walltime, dataformats)
            logger.debug(f"Logged image: {tag}")
        except Exception as e:
            logger.error(f"Failed to log image {tag}: {e}")
    
    def log_images(
        self,
        tag: str,
        img_tensor: Union[torch.Tensor, np.ndarray],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        dataformats: str = "NCHW"
    ):
        """Log multiple images to TensorBoard."""
        try:
            self.writer.add_images(tag, img_tensor, global_step, walltime, dataformats)
            logger.debug(f"Logged images: {tag}")
        except Exception as e:
            logger.error(f"Failed to log images {tag}: {e}")
    
    def log_figure(
        self,
        tag: str,
        figure: plt.Figure,
        global_step: Optional[int] = None,
        close: bool = True,
        walltime: Optional[float] = None
    ):
        """Log a matplotlib figure to TensorBoard."""
        try:
            self.writer.add_figure(tag, figure, global_step, close, walltime)
            logger.debug(f"Logged figure: {tag}")
        except Exception as e:
            logger.error(f"Failed to log figure {tag}: {e}")
    
    def log_text(
        self,
        tag: str,
        text_string: str,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None
    ):
        """Log text to TensorBoard."""
        try:
            self.writer.add_text(tag, text_string, global_step, walltime)
            logger.debug(f"Logged text: {tag}")
        except Exception as e:
            logger.error(f"Failed to log text {tag}: {e}")
    
    def log_graph(
        self,
        model: torch.nn.Module,
        input_to_model: Union[torch.Tensor, List[torch.Tensor]],
        verbose: bool = False,
        use_strict_trace: bool = True
    ):
        """Log model graph to TensorBoard."""
        try:
            self.writer.add_graph(model, input_to_model, verbose, use_strict_trace)
            logger.info("Logged model graph to TensorBoard")
        except Exception as e:
            logger.error(f"Failed to log model graph: {e}")
    
    def log_embedding(
        self,
        mat: torch.Tensor,
        metadata: Optional[List[str]] = None,
        label_img: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
        tag: str = "default",
        metadata_header: Optional[List[str]] = None
    ):
        """Log embeddings to TensorBoard."""
        try:
            self.writer.add_embedding(
                mat, metadata, label_img, global_step, tag, metadata_header
            )
            logger.debug(f"Logged embedding: {tag}")
        except Exception as e:
            logger.error(f"Failed to log embedding {tag}: {e}")
    
    def log_pr_curve(
        self,
        tag: str,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        global_step: Optional[int] = None,
        num_thresholds: int = 127,
        weights: Optional[torch.Tensor] = None,
        walltime: Optional[float] = None
    ):
        """Log precision-recall curve to TensorBoard."""
        try:
            self.writer.add_pr_curve(
                tag, labels, predictions, global_step, num_thresholds, weights, walltime
            )
            logger.debug(f"Logged PR curve: {tag}")
        except Exception as e:
            logger.error(f"Failed to log PR curve {tag}: {e}")
    
    def log_custom_scalars(
        self,
        layout: Dict[str, Any]
    ):
        """Log custom scalar layout to TensorBoard."""
        try:
            self.writer.add_custom_scalars(layout)
            logger.debug("Logged custom scalars layout")
        except Exception as e:
            logger.error(f"Failed to log custom scalars: {e}")
    
    def log_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, float],
        hparam_domain_discrete: Optional[Dict[str, List[Any]]] = None,
        run_name: Optional[str] = None
    ):
        """Log hyperparameters to TensorBoard."""
        try:
            self.writer.add_hparams(
                hparam_dict, metric_dict, hparam_domain_discrete, run_name
            )
            logger.debug("Logged hyperparameters")
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")
    
    def log_mesh(
        self,
        tag: str,
        vertices: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        faces: Optional[torch.Tensor] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None
    ):
        """Log 3D mesh to TensorBoard."""
        try:
            self.writer.add_mesh(
                tag, vertices, colors, faces, config_dict, global_step, walltime
            )
            logger.debug(f"Logged mesh: {tag}")
        except Exception as e:
            logger.error(f"Failed to log mesh {tag}: {e}")
    
    def flush(self):
        """Flush pending data to disk."""
        try:
            self.writer.flush()
            logger.debug("Flushed TensorBoard data")
        except Exception as e:
            logger.error(f"Failed to flush TensorBoard data: {e}")
    
    def close(self):
        """Close the TensorBoard writer."""
        try:
            if self.writer:
                self.writer.close()
                logger.info("Closed TensorBoard writer")
        except Exception as e:
            logger.error(f"Failed to close TensorBoard writer: {e}")
    
    def get_log_dir(self) -> str:
        """Get the log directory path."""
        return str(self.log_dir)


class TensorBoardContextManager:
    """Context manager for TensorBoard writer."""
    
    def __init__(
        self,
        log_dir: str = "runs",
        comment: str = "",
        **kwargs
    ):
        self.tracker = TensorBoardTracker(log_dir=log_dir, comment=comment, **kwargs)
    
    def __enter__(self):
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.close()


def create_tensorboard_tracker(
    log_dir: str = "runs",
    comment: str = "",
    **kwargs
) -> TensorBoardTracker:
    """
    Create and configure a TensorBoard tracker.
    
    Args:
        log_dir: Directory to save TensorBoard logs
        comment: Comment to append to default log_dir name
        **kwargs: Additional arguments for SummaryWriter
        
    Returns:
        Configured TensorBoardTracker instance
    """
    return TensorBoardTracker(log_dir=log_dir, comment=comment, **kwargs)


def log_training_metrics(
    tracker: TensorBoardTracker,
    metrics: Dict[str, Union[float, int]],
    epoch: int,
    step: Optional[int] = None
):
    """
    Log training metrics to TensorBoard.
    
    Args:
        tracker: TensorBoardTracker instance
        metrics: Dictionary of metrics to log
        epoch: Current epoch
        step: Optional global step
    """
    try:
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                tracker.log_scalar(f"training/{metric_name}", value, step or epoch)
        
        logger.debug(f"Logged training metrics for epoch {epoch}")
        
    except Exception as e:
        logger.error(f"Failed to log training metrics: {e}")


def log_model_weights(
    tracker: TensorBoardTracker,
    model: torch.nn.Module,
    step: int
):
    """
    Log model weights as histograms to TensorBoard.
    
    Args:
        tracker: TensorBoardTracker instance
        model: PyTorch model
        step: Global step
    """
    try:
        for name, param in model.named_parameters():
            if param.grad is not None:
                tracker.log_histogram(f"weights/{name}", param.data, step)
                tracker.log_histogram(f"gradients/{name}", param.grad.data, step)
        
        logger.debug(f"Logged model weights for step {step}")
        
    except Exception as e:
        logger.error(f"Failed to log model weights: {e}")


def plot_to_tensorboard_image(
    figure: plt.Figure
) -> torch.Tensor:
    """
    Convert matplotlib figure to tensor for TensorBoard.
    
    Args:
        figure: Matplotlib figure
        
    Returns:
        Tensor representation of the figure
    """
    try:
        # Save figure to buffer
        buf = io.BytesIO()
        figure.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        
        # Convert to PIL Image and then to tensor
        img = PIL.Image.open(buf)
        img_tensor = ToTensor()(img)
        
        buf.close()
        plt.close(figure)
        
        return img_tensor
        
    except Exception as e:
        logger.error(f"Failed to convert plot to tensor: {e}")
        raise
