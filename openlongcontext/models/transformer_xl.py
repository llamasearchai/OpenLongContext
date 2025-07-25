"""
Transformer-XL model implementation with recurrent memory.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn as nn
from transformers import TransfoXLModel, TransfoXLConfig
from typing import Dict, Any, Optional, List


class TransformerXLForQuestionAnswering(nn.Module):
    """Transformer-XL model for question answering with recurrent memory."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_head: int = 12,
        n_layer: int = 12,
        mem_len: int = 512,
        **kwargs
    ):
        super().__init__()
        
        config = TransfoXLConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            mem_len=mem_len,
            **kwargs
        )
        
        self.transformer = TransfoXLModel(config)
        self.qa_outputs = nn.Linear(config.d_model, 2)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        mems: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with memory."""
        outputs = self.transformer(
            input_ids=input_ids,
            mems=mems,
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
            "mems": outputs.mems,
        }
