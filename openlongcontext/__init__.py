"""
OpenLongContext - A comprehensive research platform for long-context scaling in transformers.

This package provides:
- Production-ready FastAPI service for document QA and retrieval
- State-of-the-art long-context models (Longformer, BigBird, Hyena, etc.)
- Comprehensive ablation study tools with Bayesian optimization
- Agent-based architecture for OpenAI integration
- Extensive evaluation metrics and benchmarking tools
"""

__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
__license__ = "Apache-2.0"

# Core imports for convenience
try:
    from .core.config import Config
    from .core.experiment import Experiment
    from .core.engine import Engine
    from .core.tracking import ExperimentTracker
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import core modules: {e}")
    Config = None
    Experiment = None
    Engine = None
    ExperimentTracker = None

# Model imports
try:
    from .models import (
        longformer,
        bigbird,
        hyena,
        transformer_xl,
        memorizing_transformer
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import model modules: {e}")
    longformer = None
    bigbird = None
    hyena = None
    transformer_xl = None
    memorizing_transformer = None

# Ablation tools
try:
    from .ablation.bayesian_optimization import BayesianOptimizer
    from .ablation.hyperparameter_sweep import HyperparameterSweep
    from .ablation.experiment_registry import experiment_registry, register_experiment
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import ablation modules: {e}")
    BayesianOptimizer = None
    HyperparameterSweep = None
    experiment_registry = None
    register_experiment = None

# Evaluation tools
try:
    from .evaluation import (
        perplexity,
        retrieval_metrics,
        reasoning_metrics,
        copy_metrics
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import evaluation modules: {e}")
    perplexity = None
    retrieval_metrics = None
    reasoning_metrics = None
    copy_metrics = None

# API components
try:
    from .api.routes import router as api_router
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import API modules: {e}")
    api_router = None

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Core components
    "Config",
    "Experiment", 
    "Engine",
    "ExperimentTracker",
    
    # Models
    "longformer",
    "bigbird",
    "hyena",
    "transformer_xl",
    "memorizing_transformer",
    
    # Ablation tools
    "BayesianOptimizer",
    "HyperparameterSweep",
    "experiment_registry",
    "register_experiment",
    
    # Evaluation
    "perplexity",
    "retrieval_metrics",
    "reasoning_metrics",
    "copy_metrics",
    
    # API
    "api_router"
]