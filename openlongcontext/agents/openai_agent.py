"""
OpenAI Agent with SDK Integration
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .agent_base import AgentBase


class OpenAIAgent(AgentBase):
    """OpenAI agent with full SDK integration for advanced AI workflows."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        name: str = "OpenAI Agent",
        description: str = "Advanced AI agent powered by OpenAI's API",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, description, config)
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = self.config.get(
            "system_prompt",
            "You are an advanced AI assistant specialized in processing and analyzing long documents."
        )
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a task using OpenAI's API."""
        try:
            self.status = "executing"

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task}
            ]

            # Add context if provided
            if context:
                context_str = f"Context: {context}"
                messages.insert(1, {"role": "system", "content": context_str})

            # Make API call
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            result = {
                "success": True,
                "response": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": self.model
            }

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
        """Process a document using OpenAI's API."""
        try:
            # Read document
            with open(document_path, encoding='utf-8') as f:
                document_content = f.read()

            # Truncate if too long (basic handling - could be improved with chunking)
            max_content_length = 8000  # Leave room for task and system prompt
            if len(document_content) > max_content_length:
                document_content = document_content[:max_content_length] + "...[truncated]"

            # Create context with document
            context = {
                "document_path": document_path,
                "document_content": document_content
            }

            # Execute task with document context
            enhanced_task = f"Based on the following document, {task}\n\nDocument content:\n{document_content}"

            return await self.execute(enhanced_task, context)

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

    async def create_assistant(self, instructions: str, tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Create an OpenAI assistant."""
        try:
            assistant = await asyncio.to_thread(
                self.client.beta.assistants.create,
                name=self.name,
                instructions=instructions,
                tools=tools or [],
                model=self.model
            )

            return {
                "success": True,
                "assistant_id": assistant.id,
                "assistant": assistant
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    async def run_thread(self, thread_id: str, assistant_id: str) -> Dict[str, Any]:
        """Run a thread with an assistant."""
        try:
            run = await asyncio.to_thread(
                self.client.beta.threads.runs.create,
                thread_id=thread_id,
                assistant_id=assistant_id
            )

            # Wait for completion
            while run.status in ['queued', 'in_progress']:
                await asyncio.sleep(1)
                run = await asyncio.to_thread(
                    self.client.beta.threads.runs.retrieve,
                    thread_id=thread_id,
                    run_id=run.id
                )

            return {
                "success": True,
                "run_id": run.id,
                "status": run.status,
                "run": run
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
