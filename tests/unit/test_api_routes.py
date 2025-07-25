"""
Unit tests for API routes.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import uuid
import asyncio

from openlongcontext.api.routes import router


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestDocumentEndpoints:
    """Test document-related endpoints."""
    
    def test_upload_document(self, client):
        """Test document upload."""
        # Create a test file
        content = b"Test document content"
        files = {"file": ("test.txt", content, "text/plain")}
        
        response = client.post("/docs/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "doc_id" in data
        assert data["message"] == "Document uploaded and indexed."
    
    def test_query_document(self, client):
        """Test document querying."""
        # First upload a document
        content = b"The capital of France is Paris. Paris is a beautiful city."
        files = {"file": ("test.txt", content, "text/plain")}
        upload_response = client.post("/docs/upload", files=files)
        doc_id = upload_response.json()["doc_id"]
        
        # Query the document
        with patch('openlongcontext.api.routes.answer_question') as mock_answer:
            mock_answer.return_value = ("Paris", "The capital of France is Paris.")
            
            query_data = {
                "doc_id": doc_id,
                "question": "What is the capital of France?"
            }
            response = client.post("/docs/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Paris"
        assert data["doc_id"] == doc_id
    
    def test_query_nonexistent_document(self, client):
        """Test querying non-existent document."""
        query_data = {
            "doc_id": str(uuid.uuid4()),
            "question": "Test question"
        }
        response = client.post("/docs/query", json=query_data)
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]
    
    def test_get_document_metadata(self, client):
        """Test retrieving document metadata."""
        # Upload a document
        content = b"Test content"
        files = {"file": ("test.txt", content, "text/plain")}
        upload_response = client.post("/docs/upload", files=files)
        doc_id = upload_response.json()["doc_id"]
        
        # Get metadata
        response = client.get(f"/docs/{doc_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == doc_id
        assert data["filename"] == "test.txt"
        assert data["size"] == len(content)
    
    def test_list_documents(self, client):
        """Test listing all documents."""
        # Upload multiple documents
        for i in range(3):
            files = {"file": (f"test{i}.txt", f"Content {i}".encode(), "text/plain")}
            client.post("/docs/upload", files=files)
        
        # List documents
        response = client.get("/docs/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 3


class TestAgentEndpoints:
    """Test agent-related endpoints."""
    
    @patch('openlongcontext.api.routes.agent_manager')
    def test_create_openai_agent(self, mock_manager, client):
        """Test creating OpenAI agent."""
        mock_manager.create_openai_agent.return_value = "agent_123"
        
        request_data = {
            "agent_type": "openai",
            "name": "Test Agent",
            "openai_api_key": "test_key",
            "config": {"temperature": 0.7}
        }
        
        response = client.post("/agents/create", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["agent_id"] == "agent_123"
        
        mock_manager.create_openai_agent.assert_called_once()
    
    @patch('openlongcontext.api.routes.agent_manager')
    @patch('openlongcontext.api.routes.OpenAIAgent')
    @patch('openlongcontext.models.longformer.LongformerForQuestionAnswering')
    def test_create_long_context_agent(self, mock_model, mock_openai, mock_manager, client):
        """Test creating long context agent."""
        mock_manager.create_long_context_agent.return_value = "agent_456"
        
        request_data = {
            "agent_type": "long_context",
            "name": "Long Context Agent",
            "openai_api_key": "test_key"
        }
        
        response = client.post("/agents/create", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["agent_id"] == "agent_456"
    
    def test_create_invalid_agent_type(self, client):
        """Test creating agent with invalid type."""
        request_data = {
            "agent_type": "invalid_type",
            "name": "Test Agent"
        }
        
        response = client.post("/agents/create", json=request_data)
        
        assert response.status_code == 400
        assert "Unknown agent type" in response.json()["detail"]
    
    @patch('openlongcontext.api.routes.agent_manager')
    def test_execute_agent_task(self, mock_manager, client):
        """Test executing task with agent."""
        mock_manager.execute_task.return_value = asyncio.coroutine(
            lambda: {"result": "Task completed"}
        )()
        
        request_data = {
            "task": "Summarize this document",
            "context": {"doc_id": "123"}
        }
        
        response = client.post("/agents/agent_123/execute", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "Task completed"
    
    @patch('openlongcontext.api.routes.agent_manager')
    def test_list_agents(self, mock_manager, client):
        """Test listing agents."""
        mock_manager.list_agents.return_value = [
            {"agent_id": "123", "name": "Agent 1", "type": "openai"},
            {"agent_id": "456", "name": "Agent 2", "type": "long_context"}
        ]
        
        response = client.get("/agents/")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["agent_id"] == "123"
    
    @patch('openlongcontext.api.routes.agent_manager')
    def test_delete_agent(self, mock_manager, client):
        """Test deleting agent."""
        mock_manager.remove_agent.return_value = True
        
        response = client.delete("/agents/agent_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted" in data["message"]


class TestModelEndpoints:
    """Test model-related endpoints."""
    
    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/models/list")
        
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "loaded_models" in data
        assert len(data["available_models"]) > 0
        
        # Check model structure
        model = data["available_models"][0]
        assert "name" in model
        assert "type" in model
        assert "max_context_length" in model
        assert "description" in model
    
    @patch('importlib.import_module')
    def test_load_model(self, mock_import, client):
        """Test loading a model."""
        # Mock model class
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_instance.max_context_length = 4096
        mock_model_class.return_value = mock_model_instance
        
        # Mock module
        mock_module = Mock()
        setattr(mock_module, "LongformerForQuestionAnswering", mock_model_class)
        mock_import.return_value = mock_module
        
        request_data = {
            "model_name": "longformer-base-4096",
            "model_type": "question_answering",
            "config": {"num_labels": 2}
        }
        
        response = client.post("/models/load", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "model_info" in data
        assert data["model_info"]["status"] == "loaded"
    
    def test_load_invalid_model(self, client):
        """Test loading non-existent model."""
        request_data = {
            "model_name": "invalid-model",
            "model_type": "unknown"
        }
        
        response = client.post("/models/load", json=request_data)
        
        assert response.status_code == 404
        assert "not found in registry" in response.json()["detail"]
    
    def test_model_inference_no_models(self, client):
        """Test inference when no models are loaded."""
        request_data = {
            "input_text": "Test input",
            "model_name": "test-model"
        }
        
        response = client.post("/models/inference", json=request_data)
        
        assert response.status_code == 400
        assert "No models loaded" in response.json()["detail"]
    
    @patch('openlongcontext.api.routes.router._loaded_models', {})
    def test_model_inference_model_not_found(self, client):
        """Test inference with non-existent model."""
        # Patch to simulate loaded models
        with patch.object(router, '_loaded_models', {}):
            request_data = {
                "input_text": "Test input",
                "model_id": "nonexistent"
            }
            
            response = client.post("/models/inference", json=request_data)
        
        assert response.status_code == 404
        assert "Model not found" in response.json()["detail"]
    
    def test_get_model_info(self, client):
        """Test getting model information."""
        # First load a model
        with patch('importlib.import_module') as mock_import:
            mock_model_class = Mock()
            mock_model_instance = Mock()
            mock_model_instance.max_context_length = 4096
            mock_model_instance.num_parameters.return_value = 1000000
            mock_model_class.return_value = mock_model_instance
            
            mock_module = Mock()
            setattr(mock_module, "LongformerForQuestionAnswering", mock_model_class)
            mock_import.return_value = mock_module
            
            load_response = client.post("/models/load", json={
                "model_name": "longformer-base-4096",
                "model_type": "question_answering"
            })
            
            model_id = load_response.json()["model_info"]["model_id"]
            
            # Get model info
            response = client.get(f"/models/{model_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["model_id"] == model_id
            assert "capabilities" in data
            assert "stats" in data


class TestExperimentEndpoints:
    """Test experiment-related endpoints."""
    
    def test_run_experiment(self, client):
        """Test running an experiment."""
        request_data = {
            "experiment_type": "ablation",
            "config": {"param1": "value1"},
            "name": "Test Experiment"
        }
        
        response = client.post("/experiments/run", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "experiment_id" in data
        assert data["status"] == "started"
    
    def test_get_experiment_status(self, client):
        """Test getting experiment status."""
        # Run an experiment first
        run_response = client.post("/experiments/run", json={
            "experiment_type": "test",
            "config": {}
        })
        exp_id = run_response.json()["experiment_id"]
        
        # Get status
        response = client.get(f"/experiments/{exp_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == exp_id
        assert "status" in data
        assert "start_time" in data
    
    def test_get_nonexistent_experiment(self, client):
        """Test getting non-existent experiment."""
        response = client.get(f"/experiments/{uuid.uuid4()}/status")
        
        assert response.status_code == 404
        assert "Experiment not found" in response.json()["detail"]
    
    def test_list_experiments(self, client):
        """Test listing experiments."""
        # Run a few experiments
        for i in range(3):
            client.post("/experiments/run", json={
                "experiment_type": f"type_{i}",
                "config": {},
                "name": f"Experiment {i}"
            })
        
        # List experiments
        response = client.get("/experiments/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 3
        
        # Check experiment structure
        exp = data[0]
        assert "experiment_id" in exp
        assert "name" in exp
        assert "experiment_type" in exp
        assert "status" in exp


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns correct status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "services" in data
        assert data["services"]["document_store"] == "active"
        assert data["services"]["agent_manager"] == "active"
        assert data["services"]["model_inference"] == "active"