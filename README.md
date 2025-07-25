# OpenLongContext

**Ultimate Long-Context Scaling Research Platform for Principled Algorithmic and Empirical Study**

OpenLongContext is a production-ready FastAPI service and comprehensive research platform for long-context document question-answering, retrieval, and scaling law experiments with OpenAI agent integration.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Research Capabilities](#research-capabilities)
- [Configuration](#configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Capabilities
- **Long-Context Processing**: Advanced algorithms for handling documents with extended context lengths
- **Scaling Law Research**: Comprehensive tools for studying transformer scaling behaviors
- **Bayesian Optimization**: Advanced hyperparameter tuning and ablation studies
- **Multi-Model Support**: Integration with Longformer, BigBird, Hyena, and other efficient attention mechanisms
- **Production API**: FastAPI-based service with authentication, rate limiting, and monitoring

### Research Tools
- **Ablation Studies**: Systematic component analysis and evaluation
- **Scaling Experiments**: Automated scaling law discovery and validation
- **Performance Benchmarking**: Comprehensive evaluation across multiple metrics
- **Visualization**: Advanced plotting and analysis tools

### Agent Integration
- **OpenAI Integration**: Seamless integration with OpenAI models and APIs
- **Multi-Agent Support**: Coordinated agent workflows for complex tasks
- **Authentication**: JWT and API key-based authentication systems

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM (32GB+ recommended for large models)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/llamasearchai/OpenLongContext.git
   cd OpenLongContext
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Install with all features
   pip install -e .[dev,research,agents]
   
   # Or install from requirements.txt
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize the database** (if using database features)
   ```bash
   openlongcontext-setup --init-db
   ```

## Quick Start

### 1. Start the API Server

```bash
# Development server
openlongcontext serve --dev

# Production server
openlongcontext serve --host 0.0.0.0 --port 8000
```

### 2. Run a Simple Experiment

```python
from openlongcontext import BayesianOptimizer, LongformerForQuestionAnswering

# Define optimization objective
def objective(params):
    model = LongformerForQuestionAnswering(**params)
    return model.evaluate_on_dataset("squad")

# Set up optimization
optimizer = BayesianOptimizer(
    objective_fn=objective,
    bounds={
        "learning_rate": (1e-5, 1e-3),
        "attention_window": (128, 4096),
        "hidden_size": (512, 1024)
    },
    n_iter=50
)

# Run optimization
result = optimizer.optimize()
print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score}")
```

### 3. Use the API

```python
import httpx

async with httpx.AsyncClient() as client:
    # Upload document
    response = await client.post(
        "http://localhost:8000/api/v1/documents/upload",
        files={"file": open("document.pdf", "rb")}
    )
    doc_id = response.json()["doc_id"]
    
    # Ask question
    response = await client.post(
        "http://localhost:8000/api/v1/qa/ask",
        json={
            "question": "What is the main topic of this document?",
            "doc_id": doc_id
        }
    )
    answer = response.json()["answer"]
    print(f"Answer: {answer}")
```

## API Documentation

### Authentication

The API supports two authentication methods:

1. **JWT Tokens**: For user-based authentication
2. **API Keys**: For service-to-service communication

```bash
# Get JWT token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use API key
curl -X GET "http://localhost:8000/api/v1/documents" \
  -H "X-API-Key: your-api-key"
```

### Core Endpoints

#### Document Management
- `POST /api/v1/documents/upload` - Upload document
- `GET /api/v1/documents` - List documents
- `GET /api/v1/documents/{doc_id}` - Get document details
- `DELETE /api/v1/documents/{doc_id}` - Delete document

#### Question Answering
- `POST /api/v1/qa/ask` - Ask question about document
- `POST /api/v1/qa/batch` - Batch question processing
- `GET /api/v1/qa/history` - Get QA history

#### Research
- `POST /api/v1/research/experiment` - Run experiment
- `GET /api/v1/research/experiments` - List experiments
- `POST /api/v1/research/ablation` - Run ablation study

## Research Capabilities

### Scaling Laws

Study how model performance scales with:
- Model size (parameters)
- Dataset size (tokens)
- Compute budget (FLOPs)
- Context length

```python
from openlongcontext.theory import ScalingLawAnalyzer

analyzer = ScalingLawAnalyzer()
results = analyzer.fit_scaling_law(
    model_sizes=[1e6, 1e7, 1e8, 1e9],
    dataset_sizes=[1e9, 1e10, 1e11, 1e12],
    performance_metrics=[0.8, 0.85, 0.9, 0.95]
)
analyzer.plot_scaling_curves()
```

### Ablation Studies

Systematic component analysis:

```python
from openlongcontext.ablation import run_ablation

results = run_ablation(
    base_config="configs/models/longformer.yaml",
    ablation_config="configs/ablations/attention_mechanisms.yaml",
    metrics=["perplexity", "accuracy", "inference_time"]
)
```

### Hyperparameter Optimization

Advanced optimization strategies:

```python
from openlongcontext.ablation import BayesianOptimizer, GridSearch, RandomSearch

# Bayesian optimization
optimizer = BayesianOptimizer(objective_fn, bounds, n_iter=100)
result = optimizer.optimize()

# Grid search
grid = GridSearch(objective_fn, param_grid)
result = grid.search()

# Random search
random = RandomSearch(objective_fn, bounds, n_iter=50)
result = random.search()
```

## Configuration

### Environment Variables

```bash
# API Configuration
OPENLONGCONTEXT_HOST=0.0.0.0
OPENLONGCONTEXT_PORT=8000
OPENLONGCONTEXT_DEBUG=false

# Database
DATABASE_URL=postgresql://user:pass@localhost/openlongcontext

# OpenAI
OPENAI_API_KEY=your-openai-api-key
OPENAI_ORG_ID=your-org-id

# Authentication
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Model Configuration

Models are configured via YAML files in the `configs/models/` directory:

```yaml
# configs/models/longformer.yaml
model:
  name: "longformer-base-4096"
  attention_window: 512
  max_position_embeddings: 4098
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12

training:
  learning_rate: 3e-5
  batch_size: 8
  gradient_accumulation_steps: 4
  max_epochs: 10
  warmup_steps: 1000
```

## Examples

### Complete Examples

- **Authentication Demo**: `examples/authentication_demo.py`
- **Scaling Law Analysis**: `examples/scaling_laws/analyze_capacity.py`
- **Document QA**: `examples/document_qa_demo.py`
- **Ablation Study**: `examples/ablation_study_demo.py`

### Jupyter Notebooks

- **Getting Started**: `notebooks/01_getting_started.ipynb`
- **Advanced Usage**: `notebooks/02_advanced_usage.ipynb`
- **Research Workflows**: `notebooks/03_research_workflows.ipynb`

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=openlongcontext --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

### Code Quality

```bash
# Format code
black openlongcontext/
isort openlongcontext/

# Lint code
flake8 openlongcontext/
mypy openlongcontext/

# Run pre-commit hooks
pre-commit run --all-files
```

### Building Documentation

```bash
# Build docs
cd docs/
make html

# Serve docs locally
python -m http.server 8080 -d _build/html/
```

## Performance

### Benchmarks

| Model | Context Length | Throughput (tokens/s) | Memory (GB) |
|-------|----------------|----------------------|-------------|
| Longformer-base | 4,096 | 1,200 | 8.5 |
| Longformer-large | 4,096 | 800 | 16.2 |
| BigBird-base | 4,096 | 1,100 | 9.1 |
| Hyena-small | 8,192 | 2,400 | 6.8 |

### Optimization Tips

1. **Use gradient checkpointing** for memory efficiency
2. **Enable mixed precision** training (FP16/BF16)
3. **Optimize attention window size** based on your use case
4. **Use efficient attention patterns** (sliding window, global + local)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Add unit tests for new features

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use OpenLongContext in your research, please cite:

```bibtex
@software{openlongcontext2024,
  title={OpenLongContext: Ultimate Long-Context Scaling Research Platform},
  author={Jois, Nik},
  year={2024},
  url={https://github.com/llamasearchai/OpenLongContext}
}
```

## Support

- **Documentation**: [https://openlongcontext.github.io](https://openlongcontext.github.io)
- **Issues**: [GitHub Issues](https://github.com/llamasearchai/OpenLongContext/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llamasearchai/OpenLongContext/discussions)
- **Email**: nikjois@llamasearch.ai

---

**Built with passion for advancing long-context AI research.**