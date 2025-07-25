"""
Efficient Attention

Implementation of various efficient attention mechanisms for long sequences.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SparseAttention(nn.Module):
    """Sparse attention mechanism with configurable sparsity patterns."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        block_size: int = 32,
        num_random_blocks: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.dropout = dropout

        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout_layer = nn.Dropout(dropout)

    def create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sparse attention mask with local + random blocks."""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        # Local attention (diagonal blocks)
        for i in range(0, seq_len, self.block_size):
            end_i = min(i + self.block_size, seq_len)
            for j in range(max(0, i - self.block_size), min(seq_len, i + 2 * self.block_size), self.block_size):
                end_j = min(j + self.block_size, seq_len)
                mask[i:end_i, j:end_j] = True

        # Random blocks
        num_blocks = seq_len // self.block_size
        for i in range(num_blocks):
            for _ in range(self.num_random_blocks):
                j = torch.randint(0, num_blocks, (1,)).item()
                start_i, end_i = i * self.block_size, min((i + 1) * self.block_size, seq_len)
                start_j, end_j = j * self.block_size, min((j + 1) * self.block_size, seq_len)
                mask[start_i:end_i, start_j:end_j] = True

        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to q, k, v
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply sparse mask
        sparse_mask = self.create_sparse_mask(seq_len, hidden_states.device)
        scores = scores.masked_fill(~sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(out)

        return {
            "hidden_states": output,
            "attention_weights": attn_weights,
            "sparse_mask": sparse_mask
        }


class LinearAttention(nn.Module):
    """Linear attention mechanism with O(n) complexity."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        feature_dim: int = 256,
        dropout: float = 0.1,
        kernel_type: str = "elu"
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.kernel_type = kernel_type

        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Feature mapping for linear attention
        self.feature_map = nn.Linear(self.head_dim, feature_dim, bias=False)

        self.dropout_layer = nn.Dropout(dropout)

    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply kernel feature mapping."""
        if self.kernel_type == "elu":
            return F.elu(self.feature_map(x)) + 1
        elif self.kernel_type == "relu":
            return F.relu(self.feature_map(x))
        else:
            return torch.exp(self.feature_map(x))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to q, k, v
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply kernel feature mapping
        q_features = self.kernel_feature_map(q)  # [batch, heads, seq, feature_dim]
        k_features = self.kernel_feature_map(k)  # [batch, heads, seq, feature_dim]

        # Linear attention computation: O(n) complexity
        # Compute K^T V first (feature_dim x head_dim)
        kv = torch.einsum('bhsf,bhsd->bhfd', k_features, v)

        # Then compute Q (K^T V) (seq x head_dim)
        out = torch.einsum('bhsf,bhfd->bhsd', q_features, kv)

        # Normalization
        k_sum = k_features.sum(dim=2, keepdim=True)  # [batch, heads, 1, feature_dim]
        normalizer = torch.einsum('bhsf,bhf->bhs', q_features, k_sum.squeeze(2))
        normalizer = normalizer.unsqueeze(-1) + 1e-6
        out = out / normalizer

        # Apply dropout
        out = self.dropout_layer(out)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(out)

        return {
            "hidden_states": output,
            "attention_weights": None,  # Linear attention doesn't compute explicit weights
            "q_features": q_features,
            "k_features": k_features
        }


class SlidingWindowAttention(nn.Module):
    """Sliding window attention with configurable window size."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        window_size: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.dropout = dropout

        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout_layer = nn.Dropout(dropout)

    def create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sliding window attention mask."""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = True

        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to q, k, v
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply sliding window mask
        window_mask = self.create_sliding_window_mask(seq_len, hidden_states.device)
        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(out)

        return {
            "hidden_states": output,
            "attention_weights": attn_weights,
            "window_mask": window_mask
        }


class MultiScaleAttention(nn.Module):
    """Multi-scale attention combining different attention mechanisms."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        window_sizes: Tuple[int, ...] = (128, 512, 2048),
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_sizes = window_sizes
        self.num_scales = len(window_sizes)

        # Create attention layers for each scale
        self.attention_layers = nn.ModuleList([
            SlidingWindowAttention(
                d_model=d_model // self.num_scales,
                n_heads=n_heads // self.num_scales,
                window_size=window_size,
                dropout=dropout
            )
            for window_size in window_sizes
        ])

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Split input across scales
        scale_dim = self.d_model // self.num_scales
        scale_inputs = torch.split(hidden_states, scale_dim, dim=-1)

        # Apply attention at each scale
        scale_outputs = []
        attention_weights = []

        for scale_input, attention_layer in zip(scale_inputs, self.attention_layers):
            output = attention_layer(scale_input, attention_mask)
            scale_outputs.append(output["hidden_states"])
            attention_weights.append(output.get("attention_weights"))

        # Concatenate scale outputs
        combined_output = torch.cat(scale_outputs, dim=-1)

        # Final projection
        output = self.out_proj(combined_output)

        return {
            "hidden_states": output,
            "attention_weights": attention_weights,
            "scale_outputs": scale_outputs
        }


def create_efficient_attention(
    attention_type: str = "sparse",
    d_model: int = 768,
    n_heads: int = 12,
    **kwargs
) -> nn.Module:
    """
    Factory function to create efficient attention mechanisms.
    
    Args:
        attention_type: Type of attention ("sparse", "linear", "sliding_window", "multi_scale")
        d_model: Model dimension
        n_heads: Number of attention heads
        **kwargs: Additional arguments for specific attention types
        
    Returns:
        Configured attention module
    """
    if attention_type == "sparse":
        return SparseAttention(d_model=d_model, n_heads=n_heads, **kwargs)
    elif attention_type == "linear":
        return LinearAttention(d_model=d_model, n_heads=n_heads, **kwargs)
    elif attention_type == "sliding_window":
        return SlidingWindowAttention(d_model=d_model, n_heads=n_heads, **kwargs)
    elif attention_type == "multi_scale":
        return MultiScaleAttention(d_model=d_model, n_heads=n_heads, **kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


def compute_attention_complexity(seq_len: int, attention_type: str, **kwargs) -> Dict[str, Any]:
    """
    Compute theoretical complexity of different attention mechanisms.
    
    Args:
        seq_len: Sequence length
        attention_type: Type of attention mechanism
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing complexity information
    """
    if attention_type == "standard":
        return {
            "time_complexity": f"O({seq_len}²)",
            "space_complexity": f"O({seq_len}²)",
            "operations": seq_len ** 2
        }
    elif attention_type == "sparse":
        block_size = kwargs.get("block_size", 32)
        num_random_blocks = kwargs.get("num_random_blocks", 3)
        sparsity = (3 * block_size + num_random_blocks * block_size) / seq_len
        operations = int(seq_len ** 2 * sparsity)
        return {
            "time_complexity": f"O({seq_len}² × {sparsity:.3f})",
            "space_complexity": f"O({seq_len}² × {sparsity:.3f})",
            "operations": operations,
            "sparsity": sparsity
        }
    elif attention_type == "linear":
        feature_dim = kwargs.get("feature_dim", 256)
        return {
            "time_complexity": f"O({seq_len} × {feature_dim})",
            "space_complexity": f"O({feature_dim}²)",
            "operations": seq_len * feature_dim
        }
    elif attention_type == "sliding_window":
        window_size = kwargs.get("window_size", 512)
        operations = seq_len * min(window_size, seq_len)
        return {
            "time_complexity": f"O({seq_len} × {window_size})",
            "space_complexity": f"O({seq_len} × {window_size})",
            "operations": operations
        }
    else:
        return {"error": f"Unknown attention type: {attention_type}"}
