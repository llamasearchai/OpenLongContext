"""
Hyena model implementation for efficient long-context processing.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class HyenaOperator(nn.Module):
    """Hyena operator for subquadratic attention."""

    def __init__(self, d_model: int, l_max: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max

        # Hyena filter
        self.filter_fn = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=l_max,
            padding=l_max // 2,
            groups=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hyena operator."""
        B, L, D = x.shape

        # Apply filter
        x_reshaped = x.transpose(1, 2)  # (B, D, L)
        filtered = self.filter_fn(x_reshaped.reshape(-1, 1, L))
        filtered = filtered.reshape(B, D, L).transpose(1, 2)

        return filtered


class HyenaForQuestionAnswering(nn.Module):
    """Hyena model for question answering."""

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_layers: int = 12,
        max_length: int = 4096,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)

        self.layers = nn.ModuleList([
            HyenaOperator(d_model, max_length)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.qa_outputs = nn.Linear(d_model, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B, L = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.position_embedding(pos_ids)

        # Apply Hyena layers
        for layer in self.layers:
            x = x + layer(x)

        x = self.norm(x)
        logits = self.qa_outputs(x)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
