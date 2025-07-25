"""
BigBird model implementation for long-context question answering.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn as nn
from transformers import BigBirdModel, BigBirdConfig
from typing import Dict, Any, Tuple, Optional


class BigBirdForQuestionAnswering(nn.Module):
    """BigBird model for question answering with sparse attention."""
    
    def __init__(
        self,
        max_position_embeddings: int = 4096,
        attention_type: str = "block_sparse",
        block_size: int = 64,
        num_random_blocks: int = 3,
        **kwargs
    ):
        super().__init__()
        
        config = BigBirdConfig(
            max_position_embeddings=max_position_embeddings,
            attention_type=attention_type,
            block_size=block_size,
            num_random_blocks=num_random_blocks,
            **kwargs
        )
        
        self.bigbird = BigBirdModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = self.bigbird(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
