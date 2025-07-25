"""
Longformer Model Implementation for Question Answering
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn as nn
from transformers import (
    LongformerTokenizer, 
    LongformerForQuestionAnswering as HFLongformerForQuestionAnswering,
    LongformerConfig
)
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class LongformerForQuestionAnswering:
    """
    Longformer model wrapper for question answering tasks.
    Provides a unified interface for long-context question answering.
    """
    
    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        max_length: int = 4096,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = HFLongformerForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized Longformer model: {model_name} on {self.device}")
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> 'LongformerForQuestionAnswering':
        """Create a Longformer model from a pretrained checkpoint."""
        return cls(model_name=model_name, **kwargs)
    
    def answer_question(self, context: str, question: str) -> str:
        """
        Answer a question based on the given context.
        
        Args:
            context: The document or context text
            question: The question to answer
            
        Returns:
            The answer string
        """
        try:
            # Tokenize inputs
            inputs = self.tokenizer.encode_plus(
                question,
                context,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
            
            # Find the best answer span
            start_idx = torch.argmax(start_logits, dim=1).item()
            end_idx = torch.argmax(end_logits, dim=1).item()
            
            # Ensure valid span
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Extract answer tokens
            input_ids = inputs["input_ids"][0]
            answer_tokens = input_ids[start_idx:end_idx + 1]
            
            # Decode answer
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            return answer.strip() if answer.strip() else "No answer found"
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            return f"Error processing question: {str(e)}"
    
    def get_answer_with_confidence(
        self, 
        context: str, 
        question: str
    ) -> Tuple[str, float]:
        """
        Answer a question and return confidence score.
        
        Args:
            context: The document or context text
            question: The question to answer
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        try:
            # Tokenize inputs
            inputs = self.tokenizer.encode_plus(
                question,
                context,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
            
            # Apply softmax to get probabilities
            start_probs = torch.softmax(start_logits, dim=1)
            end_probs = torch.softmax(end_logits, dim=1)
            
            # Find the best answer span
            start_idx = torch.argmax(start_logits, dim=1).item()
            end_idx = torch.argmax(end_logits, dim=1).item()
            
            # Calculate confidence as geometric mean of start and end probabilities
            start_conf = start_probs[0, start_idx].item()
            end_conf = end_probs[0, end_idx].item()
            confidence = (start_conf * end_conf) ** 0.5
            
            # Ensure valid span
            if end_idx < start_idx:
                end_idx = start_idx
                confidence = min(start_conf, end_conf)
            
            # Extract answer tokens
            input_ids = inputs["input_ids"][0]
            answer_tokens = input_ids[start_idx:end_idx + 1]
            
            # Decode answer
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            return (answer.strip() if answer.strip() else "No answer found", confidence)
            
        except Exception as e:
            logger.error(f"Error in get_answer_with_confidence: {str(e)}")
            return (f"Error processing question: {str(e)}", 0.0)
    
    def batch_answer_questions(
        self, 
        contexts: list, 
        questions: list
    ) -> list:
        """
        Answer multiple questions in batch for efficiency.
        
        Args:
            contexts: List of context texts
            questions: List of questions
            
        Returns:
            List of answers
        """
        if len(contexts) != len(questions):
            raise ValueError("Number of contexts must match number of questions")
        
        answers = []
        for context, question in zip(contexts, questions):
            answer = self.answer_question(context, question)
            answers.append(answer)
        
        return answers
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "device": self.device,
            "vocab_size": self.tokenizer.vocab_size,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def save_model(self, save_path: str):
        """Save the model and tokenizer to disk."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def to(self, device: str):
        """Move model to specified device."""
        self.device = device
        self.model.to(device)
        return self

class LongformerQA:
    """Alias for backward compatibility."""
    
    @staticmethod
    def from_pretrained(model_name: str, **kwargs) -> LongformerForQuestionAnswering:
        """Create a Longformer QA model from pretrained checkpoint."""
        return LongformerForQuestionAnswering.from_pretrained(model_name, **kwargs)