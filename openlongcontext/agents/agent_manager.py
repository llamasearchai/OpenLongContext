"""
Agent Manager for OpenLongContext
Author: Nik Jois <nikjois@llamasearch.ai>
"""

from typing import Dict, Any, Optional, List
import asyncio
from .agent_base import AgentBase
from .openai_agent import OpenAIAgent
from .long_context_agent import LongContextAgent

class AgentManager:
    """Manager for creating, monitoring, and coordinating multiple agents."""
    
    def __init__(self):
        self.agents: Dict[str, AgentBase] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
    
    def create_openai_agent(
        self, 
        api_key: str, 
        agent_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create a new OpenAI agent."""
        agent = OpenAIAgent(api_key=api_key, **kwargs)
        agent_id = agent_id or agent.agent_id
        self.agents[agent_id] = agent
        return agent_id
    
    def create_long_context_agent(
        self,
        model,
        openai_client: OpenAIAgent,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create a new long context agent."""
        agent = LongContextAgent(
            model=model,
            openai_client=openai_client,
            **kwargs
        )
        agent_id = agent_id or agent.agent_id
        self.agents[agent_id] = agent
        return agent_id
    
    def get_agent(self, agent_id: str) -> Optional[AgentBase]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents and their status."""
        return [agent.get_status() for agent in self.agents.values()]
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent."""
        if agent_id in self.agents:
            # Cancel any active tasks for this agent
            if agent_id in self.active_tasks:
                self.active_tasks[agent_id].cancel()
                del self.active_tasks[agent_id]
            
            del self.agents[agent_id]
            return True
        return False
    
    async def execute_task(
        self, 
        agent_id: str, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a task with a specific agent."""
        agent = self.get_agent(agent_id)
        if not agent:
            return {
                "success": False,
                "error": f"Agent {agent_id} not found",
                "error_type": "AgentNotFound"
            }
        
        try:
            # Create and store the task
            task_coroutine = agent.execute(task, context)
            self.active_tasks[agent_id] = asyncio.create_task(task_coroutine)
            
            # Execute and return result
            result = await self.active_tasks[agent_id]
            
            # Clean up completed task
            if agent_id in self.active_tasks:
                del self.active_tasks[agent_id]
            
            return result
            
        except asyncio.CancelledError:
            return {
                "success": False,
                "error": "Task was cancelled",
                "error_type": "TaskCancelled"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def process_document(
        self,
        agent_id: str,
        document_path: str,
        task: str
    ) -> Dict[str, Any]:
        """Process a document with a specific agent."""
        agent = self.get_agent(agent_id)
        if not agent:
            return {
                "success": False,
                "error": f"Agent {agent_id} not found",
                "error_type": "AgentNotFound"
            }
        
        try:
            return await agent.process_document(document_path, task)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def cancel_task(self, agent_id: str) -> bool:
        """Cancel an active task for an agent."""
        if agent_id in self.active_tasks:
            self.active_tasks[agent_id].cancel()
            del self.active_tasks[agent_id]
            return True
        return False
    
    def get_agent_history(self, agent_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get execution history for an agent."""
        agent = self.get_agent(agent_id)
        if agent:
            return agent.get_history(limit)
        return []
    
    async def shutdown(self):
        """Shutdown the agent manager and cancel all active tasks."""
        # Cancel all active tasks
        for task in self.active_tasks.values():
            task.cancel()
        
        # Wait for all tasks to complete cancellation
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        # Clear all data
        self.active_tasks.clear()
        self.agents.clear() 