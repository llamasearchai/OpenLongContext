"""
Copy Task Metrics

Comprehensive evaluation metrics for copy task performance.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
from typing import Dict, List, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _convert_to_list(data: Union[List[int], torch.Tensor, np.ndarray]) -> List[int]:
    """Convert input data to a Python list."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return data
    else:
        return list(data)


class CopyMetrics:
    """Metrics for evaluating copy task performance."""
    
    def __init__(self):
        self.predictions = []
        self.targets = []
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.total_samples = 0
        self.correct_samples = 0
        self.total_tokens = 0
        self.correct_tokens = 0
    
    def update(
        self,
        predictions: Union[List[int], torch.Tensor, np.ndarray],
        targets: Union[List[int], torch.Tensor, np.ndarray]
    ):
        """
        Update metrics with new predictions and targets.
        
        Args:
            predictions: Predicted token sequences
            targets: Target token sequences
        """
        # Convert to lists for consistent processing
        pred_list = _convert_to_list(predictions)
        target_list = _convert_to_list(targets)
        
        self.predictions.append(pred_list)
        self.targets.append(target_list)
        
        # Update token-level accuracy
        self._update_token_accuracy(pred_list, target_list)
        
        # Update sequence-level accuracy
        self._update_sequence_accuracy(pred_list, target_list)
    
    def _update_token_accuracy(self, predictions: List[int], targets: List[int]):
        """Update token-level accuracy metrics."""
        min_len = min(len(predictions), len(targets))
        
        for i in range(min_len):
            self.total_tokens += 1
            if predictions[i] == targets[i]:
                self.correct_tokens += 1
    
    def _update_sequence_accuracy(self, predictions: List[int], targets: List[int]):
        """Update sequence-level accuracy metrics."""
        self.total_samples += 1
        
        # Check if sequences match exactly
        if len(predictions) == len(targets) and predictions == targets:
            self.correct_samples += 1
    
    def compute_exact_match(
        self,
        predictions: Union[List[int], torch.Tensor, np.ndarray],
        targets: Union[List[int], torch.Tensor, np.ndarray]
    ) -> float:
        """
        Compute exact match accuracy between predictions and targets.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            
        Returns:
            Exact match accuracy (0.0 to 1.0)
        """
        # Convert to lists for consistent processing
        pred_list = _convert_to_list(predictions)
        target_list = _convert_to_list(targets)
        
        if len(pred_list) != len(target_list):
            return 0.0
        
        return float(pred_list == target_list)
    
    def compute_token_accuracy(
        self,
        predictions: Union[List[int], torch.Tensor, np.ndarray],
        targets: Union[List[int], torch.Tensor, np.ndarray]
    ) -> float:
        """
        Compute token-level accuracy.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            
        Returns:
            Token-level accuracy (0.0 to 1.0)
        """
        # Convert to lists for consistent processing
        pred_list = _convert_to_list(predictions)
        target_list = _convert_to_list(targets)
        
        if not pred_list or not target_list:
            return 0.0
        
        min_len = min(len(pred_list), len(target_list))
        correct = sum(1 for i in range(min_len) if pred_list[i] == target_list[i])
        
        return correct / min_len if min_len > 0 else 0.0
    
    def compute_edit_distance(
        self,
        predictions: Union[List[int], torch.Tensor, np.ndarray],
        targets: Union[List[int], torch.Tensor, np.ndarray]
    ) -> int:
        """
        Compute Levenshtein edit distance between sequences.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            
        Returns:
            Edit distance (number of operations needed)
        """
        # Convert to lists for consistent processing
        pred_list = _convert_to_list(predictions)
        target_list = _convert_to_list(targets)
        
        return self._levenshtein_distance(pred_list, target_list)
    
    def _levenshtein_distance(self, seq1: List[int], seq2: List[int]) -> int:
        """Compute Levenshtein distance between two sequences."""
        len1, len2 = len(seq1), len(seq2)
        
        # Create distance matrix
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize base cases
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        return dp[len1][len2]
    
    def compute_prefix_accuracy(
        self,
        predictions: Union[List[int], torch.Tensor, np.ndarray],
        targets: Union[List[int], torch.Tensor, np.ndarray],
        max_prefix_len: int = 10
    ) -> Dict[str, float]:
        """
        Compute prefix accuracy for different prefix lengths.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            max_prefix_len: Maximum prefix length to evaluate
            
        Returns:
            Dictionary mapping prefix lengths to accuracies
        """
        # Convert to lists for consistent processing
        pred_list = _convert_to_list(predictions)
        target_list = _convert_to_list(targets)
        
        prefix_accuracies = {}
        max_len = min(len(pred_list), len(target_list), max_prefix_len)
        
        for prefix_len in range(1, max_len + 1):
            pred_prefix = pred_list[:prefix_len]
            target_prefix = target_list[:prefix_len]
            
            accuracy = float(pred_prefix == target_prefix)
            prefix_accuracies[f"prefix_{prefix_len}"] = accuracy
        
        return prefix_accuracies
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all accumulated metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        if self.total_samples == 0:
            return {
                "exact_match_accuracy": 0.0,
                "token_accuracy": 0.0,
                "sequence_accuracy": 0.0,
                "total_samples": 0,
                "total_tokens": 0
            }
        
        metrics = {
            "exact_match_accuracy": self.correct_samples / self.total_samples,
            "sequence_accuracy": self.correct_samples / self.total_samples,
            "token_accuracy": self.correct_tokens / self.total_tokens if self.total_tokens > 0 else 0.0,
            "total_samples": self.total_samples,
            "total_tokens": self.total_tokens,
            "correct_samples": self.correct_samples,
            "correct_tokens": self.correct_tokens
        }
        
        return metrics
    
    def get_detailed_analysis(self) -> Dict[str, any]:
        """
        Get detailed analysis of copy task performance.
        
        Returns:
            Detailed analysis including per-sample metrics
        """
        if not self.predictions or not self.targets:
            return {"error": "No data available for analysis"}
        
        sample_metrics = []
        
        for i, (pred, target) in enumerate(zip(self.predictions, self.targets)):
            sample_metric = {
                "sample_id": i,
                "exact_match": self.compute_exact_match(pred, target),
                "token_accuracy": self.compute_token_accuracy(pred, target),
                "edit_distance": self.compute_edit_distance(pred, target),
                "sequence_length": len(target),
                "prediction_length": len(pred)
            }
            
            # Add prefix accuracies
            prefix_acc = self.compute_prefix_accuracy(pred, target)
            sample_metric.update(prefix_acc)
            
            sample_metrics.append(sample_metric)
        
        # Aggregate statistics
        overall_metrics = self.compute()
        
        return {
            "overall_metrics": overall_metrics,
            "sample_metrics": sample_metrics,
            "num_samples": len(sample_metrics)
        }


def evaluate_copy_task(
    model,
    dataset,
    device: str = "cpu",
    max_samples: int = None
) -> Dict[str, float]:
    """
    Evaluate a model on copy task dataset.
    
    Args:
        model: Model to evaluate
        dataset: Copy task dataset
        device: Device to run evaluation on
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    metrics = CopyMetrics()
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch in dataset:
            if max_samples and samples_processed >= max_samples:
                break
            
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            # Generate predictions
            outputs = model.generate(
                input_ids,
                max_length=input_ids.size(1) + target_ids.size(1),
                num_return_sequences=1,
                do_sample=False
            )
            
            # Extract generated portion
            generated = outputs[:, input_ids.size(1):]
            
            # Update metrics
            for i in range(generated.size(0)):
                pred = generated[i].cpu().numpy()
                target = target_ids[i].cpu().numpy()
                
                metrics.update(pred, target)
                samples_processed += 1
                
                if max_samples and samples_processed >= max_samples:
                    break
    
    return metrics.compute()


def analyze_copy_errors(
    predictions: List[List[int]],
    targets: List[List[int]]
) -> Dict[str, any]:
    """
    Analyze common error patterns in copy task predictions.
    
    Args:
        predictions: List of predicted sequences
        targets: List of target sequences
        
    Returns:
        Analysis of error patterns
    """
    error_analysis = {
        "total_samples": len(predictions),
        "error_samples": 0,
        "error_types": {
            "length_mismatch": 0,
            "prefix_errors": 0,
            "suffix_errors": 0,
            "middle_errors": 0,
            "complete_mismatch": 0
        },
        "length_statistics": {
            "avg_pred_length": 0.0,
            "avg_target_length": 0.0,
            "length_diff_avg": 0.0
        }
    }
    
    total_pred_len = 0
    total_target_len = 0
    total_length_diff = 0
    
    for pred, target in zip(predictions, targets):
        total_pred_len += len(pred)
        total_target_len += len(target)
        total_length_diff += abs(len(pred) - len(target))
        
        if pred != target:
            error_analysis["error_samples"] += 1
            
            # Analyze error type
            if len(pred) != len(target):
                error_analysis["error_types"]["length_mismatch"] += 1
            else:
                # Same length, analyze position of errors
                errors_at_start = 0
                errors_at_end = 0
                
                # Count errors from start
                for i, (p, t) in enumerate(zip(pred, target)):
                    if p != t:
                        break
                    errors_at_start = i + 1
                
                # Count errors from end
                for i, (p, t) in enumerate(zip(reversed(pred), reversed(target))):
                    if p != t:
                        break
                    errors_at_end = i + 1
                
                if errors_at_start == 0:
                    error_analysis["error_types"]["prefix_errors"] += 1
                elif errors_at_end == 0:
                    error_analysis["error_types"]["suffix_errors"] += 1
                elif errors_at_start + errors_at_end < len(pred):
                    error_analysis["error_types"]["middle_errors"] += 1
                else:
                    error_analysis["error_types"]["complete_mismatch"] += 1
    
    # Calculate statistics
    if len(predictions) > 0:
        error_analysis["length_statistics"]["avg_pred_length"] = total_pred_len / len(predictions)
        error_analysis["length_statistics"]["avg_target_length"] = total_target_len / len(predictions)
        error_analysis["length_statistics"]["length_diff_avg"] = total_length_diff / len(predictions)
    
    return error_analysis
