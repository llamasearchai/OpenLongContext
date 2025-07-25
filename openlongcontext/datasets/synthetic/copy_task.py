"""
Copy Task Dataset

Synthetic dataset for testing long-context copy tasks.
Generates sequences that models need to copy exactly.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class CopyTask:
    """
    Copy task dataset for testing long-context memory capabilities.
    
    Generates sequences of tokens that models need to copy exactly,
    with various patterns and lengths to test memory and attention.
    """

    def __init__(
        self,
        seq_length: int = 1024,
        vocab_size: int = 1000,
        num_samples: int = 1000,
        copy_length: Optional[int] = None,
        separator_token: int = 0,
        seed: Optional[int] = None
    ):
        """
        Initialize copy task dataset.
        
        Args:
            seq_length: Total sequence length
            vocab_size: Size of vocabulary (excluding separator)
            num_samples: Number of samples to generate
            copy_length: Length of sequence to copy (if None, uses seq_length // 4)
            separator_token: Token used to separate input and target
            seed: Random seed for reproducibility
        """
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.copy_length = copy_length or seq_length // 4
        self.separator_token = separator_token

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Generate all samples upfront
        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate all samples for the dataset."""
        samples = []

        for i in range(self.num_samples):
            # Generate random sequence to copy
            copy_sequence = [
                random.randint(1, self.vocab_size)
                for _ in range(self.copy_length)
            ]

            # Create input sequence: [copy_seq] + [separator] + [padding] + [copy_seq]
            input_sequence = copy_sequence.copy()
            input_sequence.append(self.separator_token)

            # Add padding to reach desired length
            padding_length = self.seq_length - len(input_sequence) - self.copy_length
            if padding_length > 0:
                padding = [self.separator_token] * padding_length
                input_sequence.extend(padding)

            # Target is the copy sequence at the end
            target_sequence = input_sequence.copy()
            target_start = len(input_sequence) - self.copy_length
            for j, token in enumerate(copy_sequence):
                if target_start + j < len(target_sequence):
                    target_sequence[target_start + j] = token

            # Ensure sequences are the right length
            input_sequence = input_sequence[:self.seq_length]
            target_sequence = target_sequence[:self.seq_length]

            # Pad if necessary
            while len(input_sequence) < self.seq_length:
                input_sequence.append(self.separator_token)
            while len(target_sequence) < self.seq_length:
                target_sequence.append(self.separator_token)

            sample = {
                'id': f'copy_{i}',
                'input_sequence': input_sequence,
                'target_sequence': target_sequence,
                'copy_sequence': copy_sequence,
                'copy_start': target_start,
                'copy_length': self.copy_length
            }
            samples.append(sample)

        return samples

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index."""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        return self.samples[idx]

    def get_batch(self, batch_size: int, indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Get a batch of samples as tensors.
        
        Args:
            batch_size: Size of batch to return
            indices: Specific indices to include (if None, random sample)
            
        Returns:
            Dictionary with batched tensors
        """
        if indices is None:
            indices = random.sample(range(len(self.samples)), min(batch_size, len(self.samples)))

        batch_inputs = []
        batch_targets = []
        batch_copy_starts = []

        for idx in indices:
            sample = self.samples[idx]
            batch_inputs.append(sample['input_sequence'])
            batch_targets.append(sample['target_sequence'])
            batch_copy_starts.append(sample['copy_start'])

        return {
            'input_ids': torch.tensor(batch_inputs, dtype=torch.long),
            'target_ids': torch.tensor(batch_targets, dtype=torch.long),
            'copy_starts': torch.tensor(batch_copy_starts, dtype=torch.long)
        }

    def evaluate_prediction(self, prediction: List[int], target: List[int]) -> Dict[str, float]:
        """
        Evaluate a prediction against the target.
        
        Args:
            prediction: Predicted sequence
            target: Target sequence
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Exact match accuracy
        exact_match = 1.0 if prediction == target else 0.0

        # Token-level accuracy
        correct_tokens = sum(1 for p, t in zip(prediction, target) if p == t)
        token_accuracy = correct_tokens / len(target) if target else 0.0

        # Copy region accuracy (focus on the part that needs to be copied)
        copy_start = len(prediction) - self.copy_length
        copy_prediction = prediction[copy_start:] if copy_start >= 0 else prediction
        copy_target = target[copy_start:] if copy_start >= 0 else target

        copy_correct = sum(1 for p, t in zip(copy_prediction, copy_target) if p == t)
        copy_accuracy = copy_correct / len(copy_target) if copy_target else 0.0

        return {
            'exact_match': exact_match,
            'token_accuracy': token_accuracy,
            'copy_accuracy': copy_accuracy,
            'copy_correct_tokens': copy_correct,
            'total_tokens': len(target)
        }

    def get_sample_info(self, idx: int) -> str:
        """Get human-readable information about a sample."""
        sample = self.samples[idx]
        return f"""
Sample {idx}:
- Sequence length: {len(sample['input_sequence'])}
- Copy length: {sample['copy_length']}
- Copy start position: {sample['copy_start']}
- Copy sequence: {sample['copy_sequence'][:10]}{'...' if len(sample['copy_sequence']) > 10 else ''}
- Input preview: {sample['input_sequence'][:20]}{'...' if len(sample['input_sequence']) > 20 else ''}
"""


class VariableLengthCopyTask(CopyTask):
    """Copy task with variable sequence and copy lengths."""

    def __init__(
        self,
        min_seq_length: int = 512,
        max_seq_length: int = 2048,
        min_copy_length: int = 10,
        max_copy_length: int = 100,
        **kwargs
    ):
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.min_copy_length = min_copy_length
        self.max_copy_length = max_copy_length

        # Use max length for initialization
        super().__init__(seq_length=max_seq_length, **kwargs)

    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate samples with variable lengths."""
        samples = []

        for i in range(self.num_samples):
            # Random lengths for this sample
            seq_len = random.randint(self.min_seq_length, self.max_seq_length)
            copy_len = random.randint(self.min_copy_length, min(self.max_copy_length, seq_len // 4))

            # Generate copy sequence
            copy_sequence = [
                random.randint(1, self.vocab_size)
                for _ in range(copy_len)
            ]

            # Create input sequence
            input_sequence = copy_sequence.copy()
            input_sequence.append(self.separator_token)

            # Add padding
            padding_length = seq_len - len(input_sequence) - copy_len
            if padding_length > 0:
                padding = [self.separator_token] * padding_length
                input_sequence.extend(padding)

            # Target sequence
            target_sequence = input_sequence.copy()
            target_start = len(input_sequence) - copy_len
            for j, token in enumerate(copy_sequence):
                if target_start + j < len(target_sequence):
                    target_sequence[target_start + j] = token

            sample = {
                'id': f'var_copy_{i}',
                'input_sequence': input_sequence[:seq_len],
                'target_sequence': target_sequence[:seq_len],
                'copy_sequence': copy_sequence,
                'copy_start': target_start,
                'copy_length': copy_len,
                'seq_length': seq_len
            }
            samples.append(sample)

        return samples
