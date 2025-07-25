"""
OpenLongContext

A comprehensive framework for long-context language models with efficient attention mechanisms.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import warnings

__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

# Core imports
from .core.config import Config
from .core.backend import get_backend_manager, set_backend, BackendType
from .core.engine import Engine
from .core.experiment import Experiment

# Model imports
from .models.flashattention import FlashAttention, FlashAttentionForQuestionAnswering
from .models.longformer import LongformerForQuestionAnswering
from .models.bigbird import BigBirdForQuestionAnswering
from .models.hyena import HyenaForQuestionAnswering
from .models.transformer_xl import TransformerXLForQuestionAnswering

# Algorithm imports
from .algorithms.efficient_attention import (
    SparseAttention,
    LinearAttention,
    SlidingWindowAttention,
    MultiScaleAttention,
    create_efficient_attention
)

# Evaluation imports
from .evaluation.perplexity import evaluate_model_perplexity, compute_perplexity
from .evaluation.copy_metrics import CopyMetrics

# Tracking imports
from .tracking.mlflow_integration import MLflowTracker, create_mlflow_tracker
from .tracking.wandb_integration import WandBTracker, create_wandb_tracker
from .tracking.tensorboard_integration import TensorBoardTracker, create_tensorboard_tracker
from .tracking.reproducibility import set_random_seeds, ensure_reproducibility

# Utility imports
from .utils.seed import set_seed, seed_everything
from .utils.math_utils import (
    safe_log, safe_exp, log_sum_exp, 
    positional_encoding, rotary_positional_encoding,
    scaled_dot_product_attention
)

# Agent imports
from .agents.agent_manager import AgentManager

# API imports (disabled temporarily due to FastAPI dependency conflicts)
api_available = False
try:
    from .api.routes import router as api_router
    from .api.auth import auth_router
    api_available = True
except Exception as e:
    warnings.warn(f"API modules not available: {e}", UserWarning)

__all__ = [
    # Meta
    "__version__", "__author__", "__email__",
    
    # Core
    "Config", "Engine", "Experiment", "get_backend_manager", "set_backend", "BackendType",
    
    # Models
    "FlashAttention", "FlashAttentionForQuestionAnswering",
    "LongformerForQuestionAnswering", "BigBirdForQuestionAnswering",
    "HyenaForQuestionAnswering", "TransformerXLForQuestionAnswering",
    
    # Algorithms
    "SparseAttention", "LinearAttention", "SlidingWindowAttention",
    "MultiScaleAttention", "create_efficient_attention",
    
    # Evaluation
    "evaluate_model_perplexity", "compute_perplexity", "CopyMetrics",
    
    # Tracking
    "MLflowTracker", "create_mlflow_tracker",
    "WandBTracker", "create_wandb_tracker", "TensorBoardTracker",
    "create_tensorboard_tracker", "set_random_seeds", "ensure_reproducibility",
    
    # Utils
    "set_seed", "seed_everything", "safe_log", "safe_exp", "log_sum_exp",
    "positional_encoding", "rotary_positional_encoding", "scaled_dot_product_attention",
    
    # Agents
    "AgentManager",
    
    # API (if available)
    "api_available"
]

# Add API exports if available
if api_available:
    __all__.extend(["api_router", "auth_router"])
