"""
Flash Attention Implementation

Provides efficient attention computation with support for CUDA, CPU, and MLX backends.
Automatically falls back to standard attention when flash-attn is not available.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.backend import BackendType, get_backend_manager

logger = logging.getLogger(__name__)


class FlashAttention(nn.Module):
    """
    Flash Attention implementation with multi-backend support.
    
    Supports CUDA (with flash-attn), CPU (standard attention), and MLX backends
    with automatic fallback to standard attention when flash-attn is unavailable.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        dropout: float = 0.1,
        causal: bool = True,
        window_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.causal = causal
        self.window_size = window_size

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout_layer = nn.Dropout(dropout)

        # Backend manager
        self.backend_manager = get_backend_manager()

        # Check for flash attention availability
        self.flash_attn_available = self._check_flash_attention()

        if self.flash_attn_available:
            logger.info("Using Flash Attention (CUDA)")
        else:
            logger.info("Using standard attention (CPU/MLX fallback)")

    def _check_flash_attention(self) -> bool:
        """Check if flash attention is available and can be used."""
        if self.backend_manager.current_backend != BackendType.CUDA:
            return False

        try:
            import flash_attn
            del flash_attn  # Mark as accessed
            return True
        except ImportError:
            return False

    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass using flash attention (CUDA only)."""
        try:
            from flash_attn import flash_attn_func

            # Flash attention expects (batch, seq_len, n_heads, head_dim)
            batch_size, seq_len = q.shape[:2]

            q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

            # Apply flash attention
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
                window_size=(-1, -1) if self.window_size is None else (self.window_size, self.window_size)
            )

            # Reshape back to (batch, seq_len, d_model)
            out = out.view(batch_size, seq_len, self.d_model)
            return out

        except Exception as e:
            logger.warning(f"Flash attention failed: {e}, falling back to standard attention")
            return self._standard_attention_forward(q, k, v, attention_mask)

    def _standard_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard attention implementation (CPU/MLX compatible)."""
        batch_size, seq_len = q.shape[:2]

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        # Apply window attention if specified
        if self.window_size is not None:
            window_mask = torch.zeros(seq_len, seq_len, device=q.device, dtype=torch.bool)
            for i in range(seq_len):
                start = max(0, i - self.window_size)
                end = min(seq_len, i + self.window_size + 1)
                window_mask[i, start:end] = True
            scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape back to (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return out

    def _mlx_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """MLX-optimized attention implementation."""
        # For now, use standard attention for MLX
        # In future, this could be optimized for MLX operations
        return self._standard_attention_forward(q, k, v, attention_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through flash attention.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing attention output and metadata
        """
        _ = kwargs  # Mark as accessed
        _, _, _ = hidden_states.shape

        # Project to q, k, v
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Choose attention implementation based on backend
        if self.flash_attn_available and self.backend_manager.current_backend == BackendType.CUDA:
            attn_out = self._flash_attention_forward(q, k, v, attention_mask)
        elif self.backend_manager.current_backend == BackendType.MLX:
            attn_out = self._mlx_attention_forward(q, k, v, attention_mask)
        else:
            attn_out = self._standard_attention_forward(q, k, v, attention_mask)

        # Final output projection
        output = self.out_proj(attn_out)

        return {
            "hidden_states": output,
            "attention_weights": None,  # Flash attention doesn't return weights
            "backend_used": self.backend_manager.current_backend.value
        }


class FlashAttentionForQuestionAnswering(nn.Module):
    """Flash Attention model for question answering tasks."""

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            FlashAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                **kwargs
            ) for _ in range(n_layers)
        ])

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # QA head
        self.qa_outputs = nn.Linear(d_model, 2)  # start and end logits

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for question answering."""
        _ = kwargs  # Mark as accessed
        _, seq_len = input_ids.shape

        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + pos_embeds
        hidden_states = self.dropout(hidden_states)

        # Apply transformer layers
        for layer in self.layers:
            layer_out = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_out["hidden_states"]

        # Layer norm
        hidden_states = self.layer_norm(hidden_states)

        # QA outputs
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden_states": hidden_states,
            "backend_used": get_backend_manager().current_backend.value
        }
