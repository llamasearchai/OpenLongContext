"""
OpenLongContext

A comprehensive framework for long-context language models with efficient attention mechanisms.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import warnings
from typing import Optional

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
from .models.memorizing_transformer import MemorizingTransformerForQuestionAnswering
from .models.rwkv import RWKVForQuestionAnswering
from .models.linear_attention import LinearAttentionForQuestionAnswering
from .models.rotary import RotaryPositionEmbedding

# Algorithm imports
from .algorithms.efficient_attention import (
    SparseAttention,
    LinearAttention,
    SlidingWindowAttention,
    MultiScaleAttention,
    create_efficient_attention
)
from .algorithms.chunking import ChunkingStrategy
from .algorithms.memory import MemoryModule
from .algorithms.recurrence import RecurrentModule
from .algorithms.sliding_window import SlidingWindowModule
from .algorithms.scaling_laws import ScalingLawAnalyzer

# Dataset imports
from .datasets.dataloader import DataLoader
from .datasets.real.pg19 import PG19Dataset
from .datasets.real.books3 import Books3Dataset
from .datasets.real.arxiv_math import ArxivMathDataset
from .datasets.real.code_continuation import CodeContinuationDataset
from .datasets.real.github_issues import GitHubIssuesDataset
from .datasets.synthetic.copy_task import CopyTask
from .datasets.synthetic.recall_task import RecallTask
from .datasets.synthetic.retrieval_task import RetrievalTask
from .datasets.synthetic.reasoning_task import ReasoningTask

# Evaluation imports
from .evaluation.perplexity import evaluate_model_perplexity, compute_perplexity
from .evaluation.copy_metrics import CopyMetrics
from .evaluation.retrieval_metrics import RetrievalMetrics
from .evaluation.reasoning_metrics import ReasoningMetrics
from .evaluation.error_analysis import ErrorAnalyzer
from .evaluation.ablation import AblationStudy

# Tracking imports
from .tracking.logger import Logger, setup_logging
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
from .utils.config_utils import load_config, save_config
from .utils.profiler import Profiler
from .utils.testing import TestSuite

# Agent imports
from .agents.agent_base import BaseAgent
from .agents.long_context_agent import LongContextAgent
from .agents.openai_agent import OpenAIAgent
from .agents.agent_manager import AgentManager

# CLI imports
from .cli.main import main as cli_main

# API imports (disabled temporarily due to FastAPI dependency conflicts)
try:
    from .api.routes import router as api_router
    from .api.auth import auth_router
    api_available = True
except Exception as e:
    api_available = False
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
    "MemorizingTransformerForQuestionAnswering", "RWKVForQuestionAnswering",
    "LinearAttentionForQuestionAnswering", "RotaryPositionEmbedding",
    
    # Algorithms
    "SparseAttention", "LinearAttention", "SlidingWindowAttention",
    "MultiScaleAttention", "create_efficient_attention",
    "ChunkingStrategy", "MemoryModule", "RecurrentModule",
    "SlidingWindowModule", "ScalingLawAnalyzer",
    
    # Datasets
    "DataLoader", "PG19Dataset", "Books3Dataset", "ArxivMathDataset",
    "CodeContinuationDataset", "GitHubIssuesDataset",
    "CopyTask", "RecallTask", "RetrievalTask", "ReasoningTask",
    
    # Evaluation
    "evaluate_model_perplexity", "compute_perplexity",
    "CopyMetrics", "RetrievalMetrics", "ReasoningMetrics",
    "ErrorAnalyzer", "AblationStudy",
    
    # Tracking
    "Logger", "setup_logging", "MLflowTracker", "create_mlflow_tracker",
    "WandBTracker", "create_wandb_tracker", "TensorBoardTracker",
    "create_tensorboard_tracker", "set_random_seeds", "ensure_reproducibility",
    
    # Utils
    "set_seed", "seed_everything", "safe_log", "safe_exp", "log_sum_exp",
    "positional_encoding", "rotary_positional_encoding", "scaled_dot_product_attention",
    "load_config", "save_config", "Profiler", "TestSuite",
    
    # Agents
    "BaseAgent", "LongContextAgent", "OpenAIAgent", "AgentManager",
    
    # CLI
    "cli_main",
    
    # API (if available)
    "api_available"
]

# Add API exports if available
if api_available:
    __all__.extend(["api_router", "auth_router"])
