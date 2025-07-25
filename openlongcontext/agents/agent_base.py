"""
Base Agent Class
Author: Nik Jois <nikjois@llamasearch.ai>
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime

class AgentBase(ABC):
    """Base class for all agents in the OpenLongContext platform."""
    
    def __init__(self, name: str, description: str = "", config: Optional[Dict[str, Any]] = None):
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.config = config or {}
        self.created_at = datetime.utcnow()
        self.status = "initialized"
        self.history: List[Dict[str, Any]] = []
    
    @abstractmethod
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a task with the agent."""
        pass
    
    @abstractmethod
    async def process_document(self, document_path: str, task: str) -> Dict[str, Any]:
        """Process a document with the agent."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "history_length": len(self.history)
        }
    
    def add_to_history(self, entry: Dict[str, Any]) -> None:
        """Add an entry to the agent's history."""
        entry["timestamp"] = datetime.utcnow().isoformat()
        self.history.append(entry)
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the agent's execution history."""
        if limit:
            return self.history[-limit:]
        return self.history 