"""
Type definitions for OpenLongContext.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypedDict

from torch import Tensor


class ModelType(str, Enum):
    """Supported model types."""
    TRANSFORMER = "transformer"
    LONGFORMER = "longformer"
    BIGBIRD = "bigbird"
    TRANSFORMER_XL = "transformer_xl"
    MEMORIZING_TRANSFORMER = "memorizing_transformer"
    HYENA = "hyena"
    RWKV = "rwkv"
    LINEAR_ATTENTION = "linear_attention"


class TaskType(str, Enum):
    """Supported task types."""
    LANGUAGE_MODELING = "language_modeling"
    QUESTION_ANSWERING = "question_answering"
    TEXT_CLASSIFICATION = "text_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    TEXT_GENERATION = "text_generation"
    DOCUMENT_RETRIEVAL = "document_retrieval"


class MetricType(str, Enum):
    """Supported metric types."""
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    BLEU = "bleu"
    ROUGE = "rouge"
    EXACT_MATCH = "exact_match"
    RETRIEVAL_ACCURACY = "retrieval_accuracy"


@dataclass
class BatchData:
    """Container for batch data."""
    input_ids: Tensor
    attention_mask: Optional[Tensor] = None
    token_type_ids: Optional[Tensor] = None
    position_ids: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelOutput:
    """Container for model outputs."""
    logits: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None
    predictions: Optional[Tensor] = None
    metrics: Optional[Dict[str, float]] = None


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer interface."""

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        ...

    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text into token IDs."""
        ...

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs into text."""
        ...

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...


class ModelProtocol(Protocol):
    """Protocol for model interface."""

    def forward(self, batch: BatchData) -> ModelOutput:
        """Forward pass through model."""
        ...

    def generate(self, input_ids: Tensor, **kwargs) -> Tensor:
        """Generate text from input."""
        ...

    def save_pretrained(self, path: str) -> None:
        """Save model to disk."""
        ...

    @classmethod
    def from_pretrained(cls, path: str) -> "ModelProtocol":
        """Load model from disk."""
        ...


class DatasetInfo(TypedDict):
    """Information about a dataset."""
    name: str
    task_type: TaskType
    num_samples: int
    max_length: int
    features: List[str]
    metadata: Dict[str, Any]


class ExperimentMetrics(TypedDict):
    """Metrics from an experiment."""
    train_loss: float
    eval_loss: float
    train_metrics: Dict[str, float]
    eval_metrics: Dict[str, float]
    speed_metrics: Dict[str, float]
    memory_metrics: Dict[str, float]


# Re-export dataclasses from config for convenience
from .config import DataConfig, EvaluationConfig, ExperimentConfig, ModelConfig, TrainingConfig

__all__ = [
    "ModelType",
    "TaskType",
    "MetricType",
    "BatchData",
    "ModelOutput",
    "TokenizerProtocol",
    "ModelProtocol",
    "DatasetInfo",
    "ExperimentMetrics",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ExperimentConfig"
]
