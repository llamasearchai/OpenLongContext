"""
Evaluation modules for OpenLongContext.
"""

# Import main evaluation functions
try:
    from .perplexity import PerplexityEvaluator, calculate_perplexity
except ImportError:
    calculate_perplexity = None
    PerplexityEvaluator = None

try:
    from .retrieval_metrics import (
        RetrievalEvaluator,
        calculate_retrieval_accuracy,
        calculate_retrieval_precision_recall,
    )
except ImportError:
    calculate_retrieval_accuracy = None
    calculate_retrieval_precision_recall = None
    RetrievalEvaluator = None

try:
    from .reasoning_metrics import (
        ReasoningEvaluator,
        calculate_reasoning_accuracy,
        evaluate_chain_of_thought,
    )
except ImportError:
    calculate_reasoning_accuracy = None
    evaluate_chain_of_thought = None
    ReasoningEvaluator = None

try:
    from .copy_metrics import CopyTaskEvaluator, calculate_copy_accuracy, calculate_exact_match
except ImportError:
    calculate_copy_accuracy = None
    calculate_exact_match = None
    CopyTaskEvaluator = None

# Convenience names
perplexity = calculate_perplexity
retrieval_metrics = {
    "accuracy": calculate_retrieval_accuracy,
    "precision_recall": calculate_retrieval_precision_recall
}
reasoning_metrics = {
    "accuracy": calculate_reasoning_accuracy,
    "chain_of_thought": evaluate_chain_of_thought
}
copy_metrics = {
    "accuracy": calculate_copy_accuracy,
    "exact_match": calculate_exact_match
}

__all__ = [
    "perplexity",
    "retrieval_metrics",
    "reasoning_metrics",
    "copy_metrics",
    "PerplexityEvaluator",
    "RetrievalEvaluator",
    "ReasoningEvaluator",
    "CopyTaskEvaluator"
]
