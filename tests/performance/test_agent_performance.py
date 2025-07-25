"""
Performance tests for OpenLongContext Agents
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from openlongcontext.agents import AgentManager, LongContextAgent, OpenAIAgent


class TestAgentPerformance:
    """Performance tests for agent operations."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_openai_agent_response_time(self, benchmark):
        """Test OpenAI agent response time."""
        with patch('openlongcontext.agents.openai_agent.OpenAI') as mock_openai:
            with patch('openlongcontext.agents.openai_agent.asyncio.to_thread') as mock_to_thread:
                # Mock response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "Test response"
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 20
                mock_response.usage.total_tokens = 30
                mock_to_thread.return_value = mock_response

                agent = OpenAIAgent(api_key="test-key")

                # Benchmark the execution
                result = await benchmark.pedantic(
                    agent.execute,
                    args=("Test task",),
                    iterations=10,
                    rounds=3
                )

                assert result["success"] is True
                assert "response" in result

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_agent_manager_concurrent_tasks(self, benchmark):
        """Test agent manager handling concurrent tasks."""
        manager = AgentManager()

        # Create mock agents
        mock_agents = []
        for i in range(10):
            mock_agent = AsyncMock()
            mock_agent.execute.return_value = {"success": True, "result": f"result_{i}"}
            agent_id = f"agent_{i}"
            manager.agents[agent_id] = mock_agent
            mock_agents.append((agent_id, mock_agent))

        async def run_concurrent_tasks():
            tasks = []
            for agent_id, _ in mock_agents:
                task = manager.execute_task(agent_id, f"task_for_{agent_id}")
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            return results

        # Benchmark concurrent execution
        results = await benchmark.pedantic(
            run_concurrent_tasks,
            iterations=5,
            rounds=2
        )

        assert len(results) == 10
        assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_long_context_agent_chunking_performance(self):
        """Test performance of document chunking in long context agent."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(50000))  # 50k tokens
        mock_tokenizer.decode.side_effect = lambda tokens: f"chunk_{len(tokens)}"
        mock_model.tokenizer = mock_tokenizer

        mock_openai_client = Mock()

        agent = LongContextAgent(
            model=mock_model,
            openai_client=mock_openai_client,
            chunk_size=4096,
            overlap_size=256
        )

        # Test chunking performance
        start_time = time.time()
        long_document = "A" * 200000  # 200k character document
        chunks = agent._chunk_document(long_document)
        end_time = time.time()

        # Should complete chunking quickly
        assert (end_time - start_time) < 1.0  # Less than 1 second
        assert len(chunks) > 10  # Should create multiple chunks

    @pytest.mark.asyncio
    async def test_agent_memory_usage(self):
        """Test that agents don't leak memory with repeated operations."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        with patch('openlongcontext.agents.openai_agent.OpenAI') as mock_openai:
            with patch('openlongcontext.agents.openai_agent.asyncio.to_thread') as mock_to_thread:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "Test response"
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 20
                mock_response.usage.total_tokens = 30
                mock_to_thread.return_value = mock_response

                agent = OpenAIAgent(api_key="test-key")

                # Perform many operations
                for i in range(100):
                    await agent.execute(f"Test task {i}")

                final_memory = process.memory_info().rss
                memory_increase = final_memory - initial_memory

                # Memory increase should be reasonable (less than 100MB)
                assert memory_increase < 100 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_agent_manager_scalability(self):
        """Test agent manager scalability with many agents."""
        manager = AgentManager()

        # Create many agents
        num_agents = 100
        for i in range(num_agents):
            mock_agent = Mock()
            mock_agent.get_status.return_value = {"id": f"agent_{i}", "status": "active"}
            manager.agents[f"agent_{i}"] = mock_agent

        # Test listing agents performance
        start_time = time.time()
        agents_list = manager.list_agents()
        end_time = time.time()

        assert len(agents_list) == num_agents
        assert (end_time - start_time) < 0.1  # Should be very fast

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self):
        """Test performance of batch processing operations."""
        mock_model = Mock()
        mock_model.tokenizer = None  # Use character-based chunking
        mock_openai_client = AsyncMock()

        agent = LongContextAgent(
            model=mock_model,
            openai_client=mock_openai_client
        )

        # Mock the processing methods
        agent._process_single_chunk = AsyncMock(return_value={
            "success": True,
            "response": "chunk processed"
        })
        agent._combine_chunk_results = AsyncMock(return_value={
            "success": True,
            "response": "combined result"
        })

        # Test processing multiple documents
        documents = [f"Document {i} content" * 1000 for i in range(10)]

        start_time = time.time()
        tasks = []
        for i, doc in enumerate(documents):
            task = agent._process_long_document(doc, f"task_{i}")
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        assert len(results) == 10
        assert all(r["success"] for r in results)
        # Should process all documents in reasonable time
        assert (end_time - start_time) < 5.0

    @pytest.mark.asyncio
    async def test_error_handling_performance(self):
        """Test that error handling doesn't significantly impact performance."""
        with patch('openlongcontext.agents.openai_agent.OpenAI') as mock_openai:
            with patch('openlongcontext.agents.openai_agent.asyncio.to_thread') as mock_to_thread:
                # Simulate intermittent errors
                call_count = 0
                def side_effect(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count % 3 == 0:  # Every 3rd call fails
                        raise Exception("API Error")

                    mock_response = Mock()
                    mock_response.choices = [Mock()]
                    mock_response.choices[0].message.content = "Test response"
                    mock_response.usage.prompt_tokens = 10
                    mock_response.usage.completion_tokens = 20
                    mock_response.usage.total_tokens = 30
                    return mock_response

                mock_to_thread.side_effect = side_effect

                agent = OpenAIAgent(api_key="test-key")

                # Test performance with errors
                start_time = time.time()
                results = []
                for i in range(30):
                    result = await agent.execute(f"Test task {i}")
                    results.append(result)

                end_time = time.time()

                # Should handle errors gracefully without major performance impact
                successful_results = [r for r in results if r["success"]]
                failed_results = [r for r in results if not r["success"]]

                assert len(successful_results) == 20  # 2/3 should succeed
                assert len(failed_results) == 10     # 1/3 should fail
                assert (end_time - start_time) < 3.0  # Should complete quickly

    @pytest.mark.asyncio
    async def test_agent_cleanup_performance(self):
        """Test performance of agent cleanup operations."""
        manager = AgentManager()

        # Create many agents with active tasks
        num_agents = 50
        for i in range(num_agents):
            mock_agent = Mock()
            mock_task = AsyncMock()

            manager.agents[f"agent_{i}"] = mock_agent
            manager.active_tasks[f"agent_{i}"] = mock_task

        # Test cleanup performance
        start_time = time.time()
        await manager.shutdown()
        end_time = time.time()

        assert len(manager.agents) == 0
        assert len(manager.active_tasks) == 0
        assert (end_time - start_time) < 1.0  # Should cleanup quickly
