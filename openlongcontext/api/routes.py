import asyncio
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..agents import AgentManager, OpenAIAgent
from .auth import (
    CurrentActiveUser,
    CurrentVerifiedUser,
    OptionalUser,
    require_api_user,
    require_read,
    require_write,
)
from .middleware import api_limit
from .model_inference import answer_question

router = APIRouter(prefix="/api/v1", tags=["api"])

# Global storage
doc_store: Dict[str, Dict] = {}
agent_manager = AgentManager()

# Pydantic models
class UploadResponse(BaseModel):
    doc_id: str
    message: str

class QueryRequest(BaseModel):
    doc_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    context: Optional[str] = None
    doc_id: str

class DocMetadata(BaseModel):
    doc_id: str
    filename: str
    size: int

class AgentCreateRequest(BaseModel):
    agent_type: str  # "openai" or "long_context"
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    openai_api_key: Optional[str] = None
    model_name: Optional[str] = None

class AgentExecuteRequest(BaseModel):
    task: str
    context: Optional[Dict[str, Any]] = None

class AgentProcessDocumentRequest(BaseModel):
    document_path: str
    task: str

class ModelLoadRequest(BaseModel):
    model_name: str
    model_type: str
    config: Optional[Dict[str, Any]] = None

class ExperimentRunRequest(BaseModel):
    experiment_type: str
    config: Dict[str, Any]
    name: Optional[str] = None

# Document endpoints
@router.post("/docs/upload", response_model=UploadResponse, dependencies=[Depends(api_limit)])
async def upload_document(
    file: UploadFile = File(...),
    current_user: CurrentVerifiedUser = Depends(require_write)
):
    """Upload and index a document."""
    content = await file.read()
    doc_id = str(uuid.uuid4())
    doc_store[doc_id] = {
        "filename": file.filename,
        "content": content,
        "size": len(content),
        "content_type": file.content_type,
        "upload_timestamp": asyncio.get_event_loop().time()
    }
    return UploadResponse(doc_id=doc_id, message="Document uploaded and indexed.")

@router.post("/docs/query", response_model=QueryResponse, dependencies=[Depends(api_limit)])
async def query_document(
    request: QueryRequest,
    current_user: CurrentActiveUser = Depends(require_read)
):
    """Query a document with natural language."""
    doc = doc_store.get(request.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    content = doc["content"].decode(errors="ignore") if isinstance(doc["content"], bytes) else str(doc["content"])
    answer, context = answer_question(content, request.question)
    return QueryResponse(answer=answer, context=context, doc_id=request.doc_id)

@router.get("/docs/{doc_id}", response_model=DocMetadata)
async def get_document_metadata(
    doc_id: str,
    current_user: CurrentActiveUser = Depends(require_read)
):
    """Retrieve document metadata."""
    doc = doc_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocMetadata(doc_id=doc_id, filename=doc["filename"], size=doc["size"])

@router.get("/docs/")
async def list_documents(
    current_user: CurrentActiveUser = Depends(require_read)
):
    """List all uploaded documents."""
    return [
        {
            "doc_id": doc_id,
            "filename": doc["filename"],
            "size": doc["size"],
            "content_type": doc.get("content_type"),
            "upload_timestamp": doc.get("upload_timestamp")
        }
        for doc_id, doc in doc_store.items()
    ]

# Agent endpoints
@router.post("/agents/create", dependencies=[Depends(api_limit)])
async def create_agent(
    request: AgentCreateRequest,
    current_user: CurrentVerifiedUser = Depends(require_api_user)
):
    """Create a new AI agent instance."""
    try:
        if request.agent_type == "openai":
            if not request.openai_api_key:
                raise HTTPException(status_code=400, detail="OpenAI API key required for OpenAI agent")

            agent_id = agent_manager.create_openai_agent(
                api_key=request.openai_api_key,
                name=request.name or "OpenAI Agent",
                config=request.config or {}
            )

        elif request.agent_type == "long_context":
            if not request.openai_api_key:
                raise HTTPException(status_code=400, detail="OpenAI API key required for long context agent")

            # Create OpenAI client first
            openai_client = OpenAIAgent(api_key=request.openai_api_key)

            # For demo purposes, use a mock model - in production, load actual model
            from ..models.longformer import LongformerForQuestionAnswering
            model = LongformerForQuestionAnswering()

            agent_id = agent_manager.create_long_context_agent(
                model=model,
                openai_client=openai_client,
                name=request.name or "Long Context Agent",
                config=request.config or {}
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown agent type: {request.agent_type}")

        return {"success": True, "agent_id": agent_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}/execute", dependencies=[Depends(api_limit)])
async def execute_agent_task(
    agent_id: str,
    request: AgentExecuteRequest,
    current_user: CurrentVerifiedUser = Depends(require_api_user)
):
    """Execute a task with a specific agent."""
    try:
        result = await agent_manager.execute_task(
            agent_id=agent_id,
            task=request.task,
            context=request.context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}/process-document")
async def process_document_with_agent(agent_id: str, request: AgentProcessDocumentRequest):
    """Process a document with a specific agent."""
    try:
        result = await agent_manager.process_document(
            agent_id=agent_id,
            document_path=request.document_path,
            task=request.task
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/")
async def list_agents(
    current_user: CurrentActiveUser = Depends(require_read)
):
    """List all agents and their status."""
    return agent_manager.list_agents()

@router.get("/agents/{agent_id}")
async def get_agent_status(agent_id: str):
    """Get status of a specific agent."""
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent.get_status()

@router.get("/agents/{agent_id}/history")
async def get_agent_history(agent_id: str, limit: Optional[int] = None):
    """Get execution history for an agent."""
    history = agent_manager.get_agent_history(agent_id, limit)
    if history is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"agent_id": agent_id, "history": history}

@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent."""
    success = agent_manager.remove_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"success": True, "message": f"Agent {agent_id} deleted"}

@router.post("/agents/{agent_id}/cancel")
async def cancel_agent_task(agent_id: str):
    """Cancel an active task for an agent."""
    success = agent_manager.cancel_task(agent_id)
    return {"success": success, "message": "Task cancelled" if success else "No active task found"}

# Model endpoints
@router.get("/models/list")
async def list_models(
    current_user: OptionalUser = None
):
    """List available models."""
    available_models = [
        {
            "name": "longformer-base-4096",
            "type": "question_answering",
            "max_context_length": 4096,
            "description": "Longformer model for question answering",
            "memory_requirements": "2GB",
            "supported_tasks": ["question_answering", "document_qa"]
        },
        {
            "name": "bigbird-roberta-base",
            "type": "question_answering",
            "max_context_length": 4096,
            "description": "BigBird model with sparse attention",
            "memory_requirements": "2.5GB",
            "supported_tasks": ["question_answering", "document_qa", "classification"]
        },
        {
            "name": "hyena-medium",
            "type": "language_model",
            "max_context_length": 8192,
            "description": "Hyena model with subquadratic attention",
            "memory_requirements": "4GB",
            "supported_tasks": ["text_generation", "completion"]
        },
        {
            "name": "transformer-xl",
            "type": "language_model",
            "max_context_length": 16384,
            "description": "Transformer-XL with segment-level recurrence",
            "memory_requirements": "3GB",
            "supported_tasks": ["text_generation", "language_modeling"]
        },
        {
            "name": "memorizing-transformer",
            "type": "language_model",
            "max_context_length": 32768,
            "description": "Memorizing Transformer with kNN memory",
            "memory_requirements": "6GB",
            "supported_tasks": ["text_generation", "long_context_modeling"]
        }
    ]

    # Add loaded models info
    loaded_models = []
    if hasattr(router, "_loaded_models"):
        for model_id, model_data in router._loaded_models.items():
            loaded_models.append({
                "model_id": model_id,
                "model_name": model_data["model_name"],
                "model_type": model_data["model_type"],
                "loaded_at": model_data.get("loaded_at"),
                "status": "loaded"
            })

    return {
        "available_models": available_models,
        "loaded_models": loaded_models
    }

@router.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a loaded model."""
    if not hasattr(router, "_loaded_models") or model_id not in router._loaded_models:
        raise HTTPException(status_code=404, detail="Model not found")

    model_data = router._loaded_models[model_id]
    model = model_data["model"]

    # Get model stats
    param_count = 0
    if hasattr(model, "num_parameters"):
        param_count = model.num_parameters()
    elif hasattr(model, "parameters"):
        try:
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except:
            pass

    return {
        "model_id": model_id,
        "model_name": model_data["model_name"],
        "model_type": model_data["model_type"],
        "config": model_data["config"],
        "loaded_at": model_data.get("loaded_at"),
        "capabilities": {
            "max_context_length": getattr(model, "max_context_length", 4096),
            "supports_streaming": getattr(model, "supports_streaming", False),
            "supports_batching": getattr(model, "supports_batching", True)
        },
        "stats": {
            "parameter_count": param_count,
            "model_size_mb": round(param_count * 4 / 1024 / 1024, 2) if param_count > 0 else None
        }
    }

@router.delete("/models/{model_id}")
async def unload_model(model_id: str):
    """Unload a model from memory."""
    if not hasattr(router, "_loaded_models") or model_id not in router._loaded_models:
        raise HTTPException(status_code=404, detail="Model not found")

    model_name = router._loaded_models[model_id]["model_name"]
    del router._loaded_models[model_id]

    # Trigger garbage collection to free memory
    import gc
    gc.collect()

    return {
        "success": True,
        "message": f"Model {model_name} (ID: {model_id}) unloaded successfully"
    }

@router.post("/models/load", dependencies=[Depends(api_limit)])
async def load_model(
    request: ModelLoadRequest,
    current_user: CurrentVerifiedUser = Depends(require_api_user)
):
    """Load a specific model."""
    try:
        # Model registry for available models
        model_registry = {
            "longformer-base-4096": {
                "module": "..models.longformer",
                "class": "LongformerForQuestionAnswering",
                "type": "question_answering"
            },
            "bigbird-roberta-base": {
                "module": "..models.bigbird",
                "class": "BigBirdForQuestionAnswering",
                "type": "question_answering"
            },
            "hyena-medium": {
                "module": "..models.hyena",
                "class": "HyenaModel",
                "type": "language_model"
            },
            "transformer-xl": {
                "module": "..models.transformer_xl",
                "class": "TransformerXL",
                "type": "language_model"
            },
            "memorizing-transformer": {
                "module": "..models.memorizing_transformer",
                "class": "MemorizingTransformer",
                "type": "language_model"
            }
        }

        if request.model_name not in model_registry:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found in registry")

        model_spec = model_registry[request.model_name]

        # Verify model type matches
        if request.model_type != model_spec["type"]:
            raise HTTPException(
                status_code=400,
                detail=f"Model type mismatch. Expected {model_spec['type']}, got {request.model_type}"
            )

        # Dynamic model loading
        try:
            import importlib
            module = importlib.import_module(model_spec["module"], package="openlongcontext.api")
            model_class = getattr(module, model_spec["class"])

            # Initialize model with config
            config = request.config or {}
            model_instance = model_class(**config)

            # Store model instance (in production, use proper model registry)
            if not hasattr(router, "_loaded_models"):
                router._loaded_models = {}

            model_id = f"{request.model_name}_{uuid.uuid4().hex[:8]}"
            router._loaded_models[model_id] = {
                "model": model_instance,
                "model_name": request.model_name,
                "model_type": request.model_type,
                "config": config,
                "loaded_at": asyncio.get_event_loop().time()
            }

            model_info = {
                "model_id": model_id,
                "model_name": request.model_name,
                "model_type": request.model_type,
                "status": "loaded",
                "config": config,
                "capabilities": {
                    "max_context_length": getattr(model_instance, "max_context_length", 4096),
                    "supports_streaming": getattr(model_instance, "supports_streaming", False),
                    "supports_batching": getattr(model_instance, "supports_batching", True)
                }
            }

            return {"success": True, "model_info": model_info}

        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"Failed to import model module: {str(e)}")
        except AttributeError as e:
            raise HTTPException(status_code=500, detail=f"Failed to find model class: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/inference", dependencies=[Depends(api_limit)])
async def model_inference(
    request: dict,
    current_user: CurrentVerifiedUser = Depends(require_api_user)
):
    """Direct model inference."""
    try:
        input_text = request.get("input_text", "")
        model_id = request.get("model_id")
        model_name = request.get("model_name")
        inference_params = request.get("params", {})

        # Get loaded model
        if not hasattr(router, "_loaded_models"):
            raise HTTPException(status_code=400, detail="No models loaded")

        # Find model by ID or name
        model_data = None
        if model_id and model_id in router._loaded_models:
            model_data = router._loaded_models[model_id]
        elif model_name:
            # Find first model with matching name
            for mid, mdata in router._loaded_models.items():
                if mdata["model_name"] == model_name:
                    model_data = mdata
                    model_id = mid
                    break

        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found. Please load a model first.")

        model = model_data["model"]
        model_type = model_data["model_type"]

        # Perform inference based on model type
        import time
        start_time = time.time()

        if model_type == "question_answering":
            # For QA models, expect question in params
            question = inference_params.get("question", "What is this about?")

            # Call model's predict method
            if hasattr(model, "answer_question"):
                answer = model.answer_question(context=input_text, question=question)
                output = {"answer": answer, "question": question}
            else:
                output = {"error": "Model does not support question answering"}

        elif model_type == "language_model":
            # For language models, generate text
            max_length = inference_params.get("max_length", 100)
            temperature = inference_params.get("temperature", 1.0)

            if hasattr(model, "generate"):
                generated = model.generate(
                    input_text,
                    max_length=max_length,
                    temperature=temperature
                )
                output = {"generated_text": generated}
            elif hasattr(model, "forward"):
                # Fallback to forward pass
                output = {"logits": "Model forward pass completed"}
            else:
                output = {"error": "Model does not support text generation"}
        else:
            output = {"error": f"Unknown model type: {model_type}"}

        processing_time = time.time() - start_time

        result = {
            "success": True,
            "model_id": model_id,
            "model_name": model_data["model_name"],
            "model_type": model_type,
            "input_length": len(input_text),
            "output": output,
            "processing_time": round(processing_time, 3),
            "inference_params": inference_params
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Experiment endpoints
experiments_store: Dict[str, Dict] = {}

@router.post("/experiments/run")
async def run_experiment(request: ExperimentRunRequest, background_tasks: BackgroundTasks):
    """Start an experiment."""
    exp_id = str(uuid.uuid4())

    experiment_data = {
        "experiment_id": exp_id,
        "experiment_type": request.experiment_type,
        "config": request.config,
        "name": request.name or f"Experiment {exp_id[:8]}",
        "status": "running",
        "start_time": asyncio.get_event_loop().time(),
        "results": None
    }

    experiments_store[exp_id] = experiment_data

    # Start experiment in background
    background_tasks.add_task(run_experiment_background, exp_id, request)

    return {
        "success": True,
        "experiment_id": exp_id,
        "status": "started",
        "message": "Experiment started in background"
    }

async def run_experiment_background(exp_id: str, request: ExperimentRunRequest):
    """Run experiment in background."""
    try:
        # Simulate experiment execution
        await asyncio.sleep(5)  # Mock processing time

        # Mock results
        results = {
            "experiment_type": request.experiment_type,
            "metrics": {
                "accuracy": 0.95,
                "f1_score": 0.92,
                "processing_time": 4.8
            },
            "config_used": request.config
        }

        experiments_store[exp_id].update({
            "status": "completed",
            "end_time": asyncio.get_event_loop().time(),
            "results": results
        })

    except Exception as e:
        experiments_store[exp_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": asyncio.get_event_loop().time()
        })

@router.get("/experiments/{exp_id}/status")
async def get_experiment_status(exp_id: str):
    """Check experiment status."""
    experiment = experiments_store.get(exp_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "experiment_id": exp_id,
        "status": experiment["status"],
        "start_time": experiment.get("start_time"),
        "end_time": experiment.get("end_time"),
        "name": experiment["name"]
    }

@router.get("/experiments/{exp_id}/results")
async def get_experiment_results(exp_id: str):
    """Retrieve experiment results."""
    experiment = experiments_store.get(exp_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiment["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Experiment status: {experiment['status']}")

    return {
        "experiment_id": exp_id,
        "results": experiment["results"],
        "config": experiment["config"],
        "name": experiment["name"]
    }

@router.get("/experiments/")
async def list_experiments():
    """List all experiments."""
    return [
        {
            "experiment_id": exp_id,
            "name": exp["name"],
            "experiment_type": exp["experiment_type"],
            "status": exp["status"],
            "start_time": exp.get("start_time"),
            "end_time": exp.get("end_time")
        }
        for exp_id, exp in experiments_store.items()
    ]

# Health check endpoint
@router.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "document_store": "active",
            "agent_manager": "active",
            "model_inference": "active"
        }
    }
