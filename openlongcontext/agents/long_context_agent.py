"""
Long Context Agent with OpenAI Integration
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
from typing import Dict, Any, Optional, List
from .agent_base import AgentBase
from .openai_agent import OpenAIAgent
from ..models.longformer import LongformerForQuestionAnswering

class LongContextAgent(AgentBase):
    """Agent that combines long-context models with OpenAI for processing extremely long documents."""
    
    def __init__(
        self,
        model,
        openai_client: OpenAIAgent,
        max_context_length: int = 16384,
        chunk_size: int = 4096,
        overlap_size: int = 256,
        name: str = "Long Context Agent",
        description: str = "Agent specialized in processing extremely long documents",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, description, config)
        self.model = model
        self.openai_client = openai_client
        self.max_context_length = max_context_length
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.tokenizer = getattr(model, 'tokenizer', None)
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a task using the long-context agent."""
        try:
            self.status = "executing"
            
            # If context contains long document, process with long-context model
            if context and "document_content" in context:
                result = await self._process_long_document(
                    context["document_content"], 
                    task
                )
            else:
                # For regular tasks, delegate to OpenAI
                result = await self.openai_client.execute(task, context)
            
            self.status = "completed"
            self.add_to_history({
                "task": task,
                "result": result,
                "context": context
            })
            
            return result
            
        except Exception as e:
            self.status = "error"
            error_result = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            self.add_to_history({
                "task": task,
                "result": error_result,
                "context": context
            })
            
            return error_result
    
    async def process_document(self, document_path: str, task: str) -> Dict[str, Any]:
        """Process a long document using the specialized long-context model."""
        try:
            # Read document
            with open(document_path, 'r', encoding='utf-8') as f:
                document_content = f.read()
            
            return await self._process_long_document(document_content, task)
            
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"Document not found: {document_path}",
                "error_type": "FileNotFoundError"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def _process_long_document(self, document_content: str, task: str) -> Dict[str, Any]:
        """Process a long document using chunking and long-context models."""
        try:
            # Check if document fits in context window
            if self.tokenizer:
                tokens = self.tokenizer.encode(document_content)
                if len(tokens) <= self.max_context_length:
                    # Document fits, process directly
                    return await self._process_single_chunk(document_content, task)
            
            # Document is too long, use chunking strategy
            chunks = self._chunk_document(document_content)
            chunk_results = []
            
            for i, chunk in enumerate(chunks):
                chunk_result = await self._process_single_chunk(
                    chunk, 
                    f"For chunk {i+1}/{len(chunks)}: {task}"
                )
                chunk_results.append(chunk_result)
            
            # Combine results using OpenAI
            combined_result = await self._combine_chunk_results(chunk_results, task)
            
            return {
                "success": True,
                "response": combined_result["response"],
                "chunks_processed": len(chunks),
                "processing_method": "chunked",
                "chunk_results": chunk_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def _process_single_chunk(self, chunk: str, task: str) -> Dict[str, Any]:
        """Process a single chunk using the long-context model."""
        try:
            # Use the model directly for question answering or analysis
            if hasattr(self.model, 'answer_question'):
                # If it's a QA model
                answer = self.model.answer_question(chunk, task)
                return {
                    "success": True,
                    "response": answer,
                    "processing_method": "direct_model"
                }
            else:
                # Fall back to OpenAI for general tasks
                context = {"document_content": chunk}
                return await self.openai_client.execute(task, context)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def _chunk_document(self, document: str) -> List[str]:
        """Split document into overlapping chunks."""
        if self.tokenizer:
            tokens = self.tokenizer.encode(document)
            chunks = []
            
            for i in range(0, len(tokens), self.chunk_size - self.overlap_size):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
            
            return chunks
        else:
            # Fallback to character-based chunking
            chunks = []
            chunk_char_size = self.chunk_size * 4  # Rough estimate
            overlap_char_size = self.overlap_size * 4
            
            for i in range(0, len(document), chunk_char_size - overlap_char_size):
                chunk = document[i:i + chunk_char_size]
                chunks.append(chunk)
            
            return chunks
    
    async def _combine_chunk_results(self, chunk_results: List[Dict], original_task: str) -> Dict[str, Any]:
        """Combine results from multiple chunks using OpenAI."""
        try:
            # Extract responses from successful chunk results
            responses = []
            for result in chunk_results:
                if result.get("success") and "response" in result:
                    responses.append(result["response"])
            
            if not responses:
                return {
                    "success": False,
                    "error": "No successful chunk results to combine"
                }
            
            # Create combination prompt
            combination_prompt = f"""
            I processed a long document in {len(responses)} chunks for the task: "{original_task}"
            
            Here are the results from each chunk:
            
            {chr(10).join(f"Chunk {i+1}: {response}" for i, response in enumerate(responses))}
            
            Please provide a comprehensive final answer that synthesizes these chunk results for the original task.
            """
            
            return await self.openai_client.execute(combination_prompt)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            } 