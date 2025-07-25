"""
Copy Metrics

Evaluation metrics for copy task performance.
Provides comprehensive metrics for assessing model copying ability.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class CopyMetrics:
    """
    Metrics for evaluating copy task performance.
    
    Provides various metrics to assess how well models can copy
    sequences in long-context scenarios.
    """
    
    def __init__(self):
        """Initialize copy metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_samples = 0
        self.total_exact_matches = 0
        self.total_token_correct = 0
        self.total_tokens = 0
        self.total_copy_correct = 0
        self.total_copy_tokens = 0
        self.sequence_accuracies = []
        self.copy_accuracies = []
        self.edit_distances = []
    
    def compute_exact_match(
        self, 
        predictions: Union[List[int], torch.Tensor], 
        targets: Union[List[int], torch.Tensor]
    ) -> float:
        """
        Compute exact match accuracy.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            
        Returns:
            Exact match accuracy (1.0 if perfect match, 0.0 otherwise)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        
        return 1.0 if predictions == targets else 0.0
    
    def compute_token_accuracy(
        self, 
        predictions: Union[List[int], torch.Tensor], 
        targets: Union[List[int], torch.Tensor]
    ) -> float:
        """
        Compute token-level accuracy.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            
        Returns:
            Token-level accuracy
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        
        if len(targets) == 0:
            return 0.0
        
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        return correct / len(targets)
    
    def compute_copy_accuracy(
        self, 
        predictions: Union[List[int], torch.Tensor], 
        targets: Union[List[int], torch.Tensor],
        copy_start: int,
        copy_length: int
    ) -> float:
        """
        Compute accuracy for the copy region specifically.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            copy_start: Start position of copy region
            copy_length: Length of copy region
            
        Returns:
            Copy region accuracy
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        
        # Extract copy regions
        copy_end = copy_start + copy_length
        pred_copy = predictions[copy_start:copy_end] if copy_start < len(predictions) else []
        target_copy = targets[copy_start:copy_end] if copy_start < len(targets) else []
        
        if len(target_copy) == 0:
            return 0.0
        
        correct = sum(1 for p, t in zip(pred_copy, target_copy) if p == t)
        return correct / len(target_copy)
    
    def compute_edit_distance(
        self, 
        predictions: Union[List[int], torch.Tensor], 
        targets: Union[List[int], torch.Tensor]
    ) -> int:
        """
        Compute edit distance (Levenshtein distance) between sequences.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            
        Returns:
            Edit distance
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        
        m, n = len(predictions), len(targets)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if predictions[i-1] == targets[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        return dp[m][n]
    
    def update(
        self, 
        predictions: Union[List[int], torch.Tensor], 
        targets: Union[List[int], torch.Tensor],
        copy_start: Optional[int] = None,
        copy_length: Optional[int] = None
    ):
        """
        Update metrics with a new prediction.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            copy_start: Start position of copy region
            copy_length: Length of copy region
        """
        # Convert to lists if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        
        # Update counters
        self.total_samples += 1
        
        # Exact match
        exact_match = self.compute_exact_match(predictions, targets)
        self.total_exact_matches += exact_match
        
        # Token accuracy
        token_acc = self.compute_token_accuracy(predictions, targets)
        self.sequence_accuracies.append(token_acc)
        
        correct_tokens = sum(1 for p, t in zip(predictions, targets) if p == t)
        self.total_token_correct += correct_tokens
        self.total_tokens += len(targets)
        
        # Copy accuracy (if copy region specified)
        if copy_start is not None and copy_length is not None:
            copy_acc = self.compute_copy_accuracy(predictions, targets, copy_start, copy_length)
            self.copy_accuracies.append(copy_acc)
            
            # Extract copy regions for counting
            copy_end = copy_start + copy_length
            pred_copy = predictions[copy_start:copy_end] if copy_start < len(predictions) else []
            target_copy = targets[copy_start:copy_end] if copy_start < len(targets) else []
            
            copy_correct = sum(1 for p, t in zip(pred_copy, target_copy) if p == t)
            self.total_copy_correct += copy_correct
            self.total_copy_tokens += len(target_copy)
        
        # Edit distance
        edit_dist = self.compute_edit_distance(predictions, targets)
        self.edit_distances.append(edit_dist)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        if self.total_samples == 0:
            return {
                'exact_match_accuracy': 0.0,
                'token_accuracy': 0.0,
                'copy_accuracy': 0.0,
                'average_edit_distance': 0.0,
                'sequence_accuracy_std': 0.0,
                'copy_accuracy_std': 0.0,
                'total_samples': 0
            }
        
        metrics = {
            'exact_match_accuracy': self.total_exact_matches / self.total_samples,
            'token_accuracy': self.total_token_correct / self.total_tokens if self.total_tokens > 0 else 0.0,
            'average_edit_distance': np.mean(self.edit_distances) if self.edit_distances else 0.0,
            'sequence_accuracy_mean': np.mean(self.sequence_accuracies) if self.sequence_accuracies else 0.0,
            'sequence_accuracy_std': np.std(self.sequence_accuracies) if self.sequence_accuracies else 0.0,
            'total_samples': self.total_samples,
            'total_tokens': self.total_tokens
        }
        
        # Copy-specific metrics
        if self.copy_accuracies:
            metrics.update({
                'copy_accuracy': self.total_copy_correct / self.total_copy_tokens if self.total_copy_tokens > 0 else 0.0,
                'copy_accuracy_mean': np.mean(self.copy_accuracies),
                'copy_accuracy_std': np.std(self.copy_accuracies),
                'total_copy_tokens': self.total_copy_tokens
            })
        else:
            metrics.update({
                'copy_accuracy': 0.0,
                'copy_accuracy_mean': 0.0,
                'copy_accuracy_std': 0.0,
                'total_copy_tokens': 0
            })
        
        return metrics
    
    def batch_update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        copy_starts: Optional[torch.Tensor] = None,
        copy_lengths: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            predictions: Batch of predicted sequences [batch_size, seq_len]
            targets: Batch of target sequences [batch_size, seq_len]
            copy_starts: Start positions of copy regions [batch_size]
            copy_lengths: Lengths of copy regions [batch_size]
        """
        batch_size = predictions.shape[0]
        
        for i in range(batch_size):
            pred = predictions[i].tolist()
            target = targets[i].tolist()
            
            copy_start = copy_starts[i].item() if copy_starts is not None else None
            copy_length = copy_lengths[i].item() if copy_lengths is not None else None
            
            self.update(pred, target, copy_start, copy_length)
    
    def get_summary(self) -> str:
        """Get a human-readable summary of metrics."""
        metrics = self.compute()
        
        summary = f"""
Copy Task Metrics Summary:
========================
Total Samples: {metrics['total_samples']}
Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}
Token Accuracy: {metrics['token_accuracy']:.3f}
Copy Region Accuracy: {metrics['copy_accuracy']:.3f}
Average Edit Distance: {metrics['average_edit_distance']:.2f}
Sequence Accuracy (mean ± std): {metrics['sequence_accuracy_mean']:.3f} ± {metrics['sequence_accuracy_std']:.3f}
Copy Accuracy (mean ± std): {metrics['copy_accuracy_mean']:.3f} ± {metrics['copy_accuracy_std']:.3f}
"""
        return summary
