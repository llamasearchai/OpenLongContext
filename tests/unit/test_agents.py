"""
Unit tests for OpenLongContext Agents
Author: Nik Jois <nikjois@llamasearch.ai>
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from openlongcontext.agents import AgentBase, AgentManager, LongContextAgent, OpenAIAgent


class MockAgent(AgentBase):
    """Mock agent for testing AgentBase functionality."""

    async def execute(self, task: str, context=None):
        return {"success": True, "result": f"Mock result for: {task}"}

    async def process_document(self, document_path: str, task: str):
        return {"success": True, "result": f"Mock document processing: {task}"}

class TestAgentBase:
    """Test the base agent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = MockAgent("Test Agent", "Test description")

        assert agent.name == "Test Agent"
        assert agent.description == "Test description"
        assert agent.status == "initialized"
        assert len(agent.history) == 0
        assert agent.agent_id is not None
        assert agent.created_at is not None

    def test_agent_initialization_with_config(self):
        """Test agent initialization with config."""
        config = {"key": "value", "setting": 123}
        agent = MockAgent("Test Agent", config=config)

        assert agent.config == config

    def test_get_status(self):
        """Test getting agent status."""
        agent = MockAgent("Test Agent")
        status = agent.get_status()

        assert status["agent_id"] == agent.agent_id
        assert status["name"] == "Test Agent"
        assert status["status"] == "initialized"
        assert "created_at" in status
        assert status["history_length"] == 0

    def test_add_to_history(self):
        """Test adding entries to agent history."""
        agent = MockAgent("Test Agent")

        entry = {"task": "test", "result": "success"}
        agent.add_to_history(entry)

        assert len(agent.history) == 1
        assert agent.history[0]["task"] == "test"
        assert agent.history[0]["result"] == "success"
        assert "timestamp" in agent.history[0]

    def test_get_history(self):
        """Test getting agent history."""
        agent = MockAgent("Test Agent")

        # Add multiple entries
        for i in range(5):
            agent.add_to_history({"task": f"task_{i}", "result": f"result_{i}"})

        # Test getting all history
        history = agent.get_history()
        assert len(history) == 5

        # Test getting limited history
        limited_history = agent.get_history(limit=3)
        assert len(limited_history) == 3
        assert limited_history[0]["task"] == "task_2"  # Last 3 entries

    @pytest.mark.asyncio
    async def test_execute_abstract_method(self):
        """Test that execute method works in mock implementation."""
        agent = MockAgent("Test Agent")
        result = await agent.execute("test task")

        assert result["success"] is True
        assert "test task" in result["result"]

    @pytest.mark.asyncio
    async def test_process_document_abstract_method(self):
        """Test that process_document method works in mock implementation."""
        agent = MockAgent("Test Agent")
        result = await agent.process_document("test.txt", "analyze")

        assert result["success"] is True
        assert "analyze" in result["result"]

class TestOpenAIAgent:
    """Test the OpenAI agent class."""

    @patch('openlongcontext.agents.openai_agent.OpenAI')
    def test_openai_agent_initialization(self, mock_openai):
        """Test OpenAI agent initialization."""
        agent = OpenAIAgent(api_key="test-key")

        assert agent.name == "OpenAI Agent"
        assert agent.model == "gpt-4-turbo-preview"
        assert agent.max_tokens == 4096
        assert agent.temperature == 0.7
        mock_openai.assert_called_once_with(api_key="test-key")

    @patch('openlongcontext.agents.openai_agent.OpenAI')
    def test_openai_agent_custom_config(self, mock_openai):
        """Test OpenAI agent with custom configuration."""
        config = {
            "system_prompt": "Custom prompt",
            "max_tokens": 2048,
            "temperature": 0.5
        }

        agent = OpenAIAgent(
            api_key="test-key",
            model="gpt-3.5-turbo",
            name="Custom Agent",
            config=config
        )

        assert agent.name == "Custom Agent"
        assert agent.model == "gpt-3.5-turbo"
        assert agent.system_prompt == "Custom prompt"
        assert agent.max_tokens == 2048
        assert agent.temperature == 0.5

    @patch('openlongcontext.agents.openai_agent.OpenAI')
    @patch('openlongcontext.agents.openai_agent.asyncio.to_thread')
    @pytest.mark.asyncio
    async def test_execute_success(self, mock_to_thread, mock_openai):
        """Test successful task execution."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_to_thread.return_value = mock_response

        agent = OpenAIAgent(api_key="test-key")
        result = await agent.execute("Test task")

        assert result["success"] is True
        assert result["response"] == "Test response"
        assert result["usage"]["total_tokens"] == 30
        assert agent.status == "completed"
        assert len(agent.history) == 1

    @patch('openlongcontext.agents.openai_agent.OpenAI')
    @patch('openlongcontext.agents.openai_agent.asyncio.to_thread')
    @pytest.mark.asyncio
    async def test_execute_with_context(self, mock_to_thread, mock_openai):
        """Test task execution with context."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response with context"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 40

        mock_to_thread.return_value = mock_response

        agent = OpenAIAgent(api_key="test-key")
        context = {"document": "test content"}
        result = await agent.execute("Test task", context)

        assert result["success"] is True
        assert result["response"] == "Response with context"
        assert agent.history[0]["context"] == context

    @patch('openlongcontext.agents.openai_agent.OpenAI')
    @patch('openlongcontext.agents.openai_agent.asyncio.to_thread')
    @pytest.mark.asyncio
    async def test_execute_error(self, mock_to_thread, mock_openai):
        """Test task execution with error."""
        mock_to_thread.side_effect = Exception("API Error")

        agent = OpenAIAgent(api_key="test-key")
        result = await agent.execute("Test task")

        assert result["success"] is False
        assert result["error"] == "API Error"
        assert result["error_type"] == "Exception"
        assert agent.status == "error"

    @patch('openlongcontext.agents.openai_agent.OpenAI')
    @patch('builtins.open', create=True)
    @pytest.mark.asyncio
    async def test_process_document_success(self, mock_open, mock_openai):
        """Test successful document processing."""
        # Mock file reading
        mock_file = Mock()
        mock_file.read.return_value = "Document content"
        mock_open.return_value.__enter__.return_value = mock_file

        agent = OpenAIAgent(api_key="test-key")

        # Mock the execute method
        agent.execute = AsyncMock(return_value={
            "success": True,
            "response": "Document analysis"
        })

        result = await agent.process_document("test.txt", "analyze")

        assert result["success"] is True
        assert result["response"] == "Document analysis"
        mock_open.assert_called_once_with("test.txt", 'r', encoding='utf-8')

    @patch('openlongcontext.agents.openai_agent.OpenAI')
    @pytest.mark.asyncio
    async def test_process_document_file_not_found(self, mock_openai):
        """Test document processing with file not found."""
        agent = OpenAIAgent(api_key="test-key")

        result = await agent.process_document("nonexistent.txt", "analyze")

        assert result["success"] is False
        assert result["error_type"] == "FileNotFoundError"

    @patch('openlongcontext.agents.openai_agent.OpenAI')
    @patch('openlongcontext.agents.openai_agent.asyncio.to_thread')
    @pytest.mark.asyncio
    async def test_create_assistant(self, mock_to_thread, mock_openai):
        """Test creating an OpenAI assistant."""
        mock_assistant = Mock()
        mock_assistant.id = "asst_123"
        mock_to_thread.return_value = mock_assistant

        agent = OpenAIAgent(api_key="test-key")
        result = await agent.create_assistant("Test instructions")

        assert result["success"] is True
        assert result["assistant_id"] == "asst_123"
        assert result["assistant"] == mock_assistant

class TestLongContextAgent:
    """Test the long context agent class."""

    def test_long_context_agent_initialization(self):
        """Test long context agent initialization."""
        mock_model = Mock()
        mock_openai_client = Mock()

        agent = LongContextAgent(
            model=mock_model,
            openai_client=mock_openai_client
        )

        assert agent.name == "Long Context Agent"
        assert agent.model == mock_model
        assert agent.openai_client == mock_openai_client
        assert agent.max_context_length == 16384
        assert agent.chunk_size == 4096
        assert agent.overlap_size == 256

    def test_long_context_agent_custom_config(self):
        """Test long context agent with custom configuration."""
        mock_model = Mock()
        mock_openai_client = Mock()

        agent = LongContextAgent(
            model=mock_model,
            openai_client=mock_openai_client,
            max_context_length=32768,
            chunk_size=8192,
            overlap_size=512,
            name="Custom Long Context Agent"
        )

        assert agent.name == "Custom Long Context Agent"
        assert agent.max_context_length == 32768
        assert agent.chunk_size == 8192
        assert agent.overlap_size == 512

    @pytest.mark.asyncio
    async def test_execute_with_document_content(self):
        """Test execution with document content in context."""
        mock_model = Mock()
        mock_openai_client = Mock()

        agent = LongContextAgent(
            model=mock_model,
            openai_client=mock_openai_client
        )

        # Mock the _process_long_document method
        agent._process_long_document = AsyncMock(return_value={
            "success": True,
            "response": "Long document analysis"
        })

        context = {"document_content": "Very long document content"}
        result = await agent.execute("analyze", context)

        assert result["success"] is True
        assert result["response"] == "Long document analysis"
        assert agent.status == "completed"

    @pytest.mark.asyncio
    async def test_execute_without_document_content(self):
        """Test execution without document content (delegates to OpenAI)."""
        mock_model = Mock()
        mock_openai_client = AsyncMock()
        mock_openai_client.execute.return_value = {
            "success": True,
            "response": "Regular task response"
        }

        agent = LongContextAgent(
            model=mock_model,
            openai_client=mock_openai_client
        )

        result = await agent.execute("regular task")

        assert result["success"] is True
        assert result["response"] == "Regular task response"
        mock_openai_client.execute.assert_called_once()

    def test_chunk_document_with_tokenizer(self):
        """Test document chunking with tokenizer."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(10000))  # 10k tokens
        mock_tokenizer.decode.side_effect = lambda tokens: f"chunk_{len(tokens)}"

        mock_model.tokenizer = mock_tokenizer
        mock_openai_client = Mock()

        agent = LongContextAgent(
            model=mock_model,
            openai_client=mock_openai_client,
            chunk_size=4096,
            overlap_size=256
        )

        chunks = agent._chunk_document("long document")

        # Should create multiple chunks
        assert len(chunks) > 1
        mock_tokenizer.encode.assert_called_once_with("long document")

    def test_chunk_document_without_tokenizer(self):
        """Test document chunking without tokenizer (character-based)."""
        mock_model = Mock()
        mock_model.tokenizer = None
        mock_openai_client = Mock()

        agent = LongContextAgent(
            model=mock_model,
            openai_client=mock_openai_client,
            chunk_size=1000,
            overlap_size=100
        )

        long_document = "A" * 5000  # 5000 character document
        chunks = agent._chunk_document(long_document)

        # Should create multiple chunks
        assert len(chunks) > 1
        # Check overlap
        assert chunks[1].startswith(chunks[0][-400:])  # Some overlap

class TestAgentManager:
    """Test the agent manager class."""

    def test_agent_manager_initialization(self):
        """Test agent manager initialization."""
        manager = AgentManager()

        assert len(manager.agents) == 0
        assert len(manager.active_tasks) == 0

    @patch('openlongcontext.agents.agent_manager.OpenAIAgent')
    def test_create_openai_agent(self, mock_openai_agent):
        """Test creating an OpenAI agent."""
        mock_agent = Mock()
        mock_agent.agent_id = "agent_123"
        mock_openai_agent.return_value = mock_agent

        manager = AgentManager()
        agent_id = manager.create_openai_agent(api_key="test-key")

        assert agent_id == "agent_123"
        assert manager.agents[agent_id] == mock_agent
        mock_openai_agent.assert_called_once_with(api_key="test-key")

    @patch('openlongcontext.agents.agent_manager.LongContextAgent')
    def test_create_long_context_agent(self, mock_long_context_agent):
        """Test creating a long context agent."""
        mock_agent = Mock()
        mock_agent.agent_id = "agent_456"
        mock_long_context_agent.return_value = mock_agent

        mock_model = Mock()
        mock_openai_client = Mock()

        manager = AgentManager()
        agent_id = manager.create_long_context_agent(
            model=mock_model,
            openai_client=mock_openai_client
        )

        assert agent_id == "agent_456"
        assert manager.agents[agent_id] == mock_agent

    def test_get_agent(self):
        """Test getting an agent by ID."""
        manager = AgentManager()
        mock_agent = Mock()
        manager.agents["test_id"] = mock_agent

        retrieved_agent = manager.get_agent("test_id")
        assert retrieved_agent == mock_agent

        # Test non-existent agent
        assert manager.get_agent("nonexistent") is None

    def test_list_agents(self):
        """Test listing all agents."""
        manager = AgentManager()

        mock_agent1 = Mock()
        mock_agent1.get_status.return_value = {"id": "1", "name": "Agent 1"}
        mock_agent2 = Mock()
        mock_agent2.get_status.return_value = {"id": "2", "name": "Agent 2"}

        manager.agents["1"] = mock_agent1
        manager.agents["2"] = mock_agent2

        agents_list = manager.list_agents()

        assert len(agents_list) == 2
        assert {"id": "1", "name": "Agent 1"} in agents_list
        assert {"id": "2", "name": "Agent 2"} in agents_list

    def test_remove_agent(self):
        """Test removing an agent."""
        manager = AgentManager()
        mock_agent = Mock()
        manager.agents["test_id"] = mock_agent

        # Test successful removal
        result = manager.remove_agent("test_id")
        assert result is True
        assert "test_id" not in manager.agents

        # Test removing non-existent agent
        result = manager.remove_agent("nonexistent")
        assert result is False

    def test_remove_agent_with_active_task(self):
        """Test removing an agent with active task."""
        manager = AgentManager()
        mock_agent = Mock()
        mock_task = Mock()

        manager.agents["test_id"] = mock_agent
        manager.active_tasks["test_id"] = mock_task

        result = manager.remove_agent("test_id")

        assert result is True
        assert "test_id" not in manager.agents
        assert "test_id" not in manager.active_tasks
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_task_success(self):
        """Test successful task execution."""
        manager = AgentManager()
        mock_agent = AsyncMock()
        mock_agent.execute.return_value = {"success": True, "result": "test"}

        manager.agents["test_id"] = mock_agent

        result = await manager.execute_task("test_id", "test task")

        assert result["success"] is True
        assert result["result"] == "test"
        mock_agent.execute.assert_called_once_with("test task", None)

    @pytest.mark.asyncio
    async def test_execute_task_agent_not_found(self):
        """Test task execution with non-existent agent."""
        manager = AgentManager()

        result = await manager.execute_task("nonexistent", "test task")

        assert result["success"] is False
        assert result["error_type"] == "AgentNotFound"

    @pytest.mark.asyncio
    async def test_process_document_success(self):
        """Test successful document processing."""
        manager = AgentManager()
        mock_agent = AsyncMock()
        mock_agent.process_document.return_value = {
            "success": True,
            "result": "document processed"
        }

        manager.agents["test_id"] = mock_agent

        result = await manager.process_document("test_id", "doc.txt", "analyze")

        assert result["success"] is True
        assert result["result"] == "document processed"
        mock_agent.process_document.assert_called_once_with("doc.txt", "analyze")

    def test_cancel_task(self):
        """Test cancelling an active task."""
        manager = AgentManager()
        mock_task = Mock()
        manager.active_tasks["test_id"] = mock_task

        result = manager.cancel_task("test_id")

        assert result is True
        assert "test_id" not in manager.active_tasks
        mock_task.cancel.assert_called_once()

        # Test cancelling non-existent task
        result = manager.cancel_task("nonexistent")
        assert result is False

    def test_get_agent_history(self):
        """Test getting agent history."""
        manager = AgentManager()
        mock_agent = Mock()
        mock_agent.get_history.return_value = [{"task": "test", "result": "success"}]

        manager.agents["test_id"] = mock_agent

        history = manager.get_agent_history("test_id")

        assert len(history) == 1
        assert history[0]["task"] == "test"
        mock_agent.get_history.assert_called_once_with(None)

        # Test with limit
        manager.get_agent_history("test_id", limit=5)
        mock_agent.get_history.assert_called_with(5)

        # Test non-existent agent
        history = manager.get_agent_history("nonexistent")
        assert history == []

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test shutting down the agent manager."""
        manager = AgentManager()

        # Add mock tasks
        mock_task1 = AsyncMock()
        mock_task2 = AsyncMock()
        manager.active_tasks["task1"] = mock_task1
        manager.active_tasks["task2"] = mock_task2

        # Add mock agents
        manager.agents["agent1"] = Mock()
        manager.agents["agent2"] = Mock()

        await manager.shutdown()

        # Check that tasks were cancelled
        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_called_once()

        # Check that data was cleared
        assert len(manager.active_tasks) == 0
        assert len(manager.agents) == 0
