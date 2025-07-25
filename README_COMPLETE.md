# OpenLongContext

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/openlongcontext.svg)](https://pypi.org/project/openlongcontext/)

A comprehensive research platform and production-ready service for long-context scaling in transformers, featuring state-of-the-art models, advanced ablation tools, and seamless OpenAI integration.

## 🚀 Features

### Production-Ready API Service
- **FastAPI-based REST API** for document processing and question answering
- **Multi-model support**: Longformer, BigBird, Hyena, Transformer-XL, and more
- **Agent-based architecture** with OpenAI GPT integration
- **Asynchronous processing** with background task support
- **Comprehensive model management** and dynamic loading

### Research Platform
- **Bayesian Optimization** for hyperparameter tuning with multiple acquisition functions
- **Advanced ablation study tools** with automated experiment tracking
- **Comprehensive evaluation metrics** for long-context understanding
- **Scaling law analysis** and theoretical bounds computation
- **Multi-strategy hyperparameter sweep** (random, grid, Sobol)

### Models & Algorithms
- **Longformer**: Efficient attention patterns for long sequences
- **BigBird**: Sparse attention with global tokens
- **Hyena**: Subquadratic attention alternative
- **Transformer-XL**: Segment-level recurrence
- **Memorizing Transformer**: kNN-augmented attention
- **Flash Attention**: Hardware-efficient implementation

## 📦 Installation

### Basic Installation
```bash
pip install openlongcontext
```

### Development Installation
```bash
git clone https://github.com/openlongcontext/openlongcontext.git
cd openlongcontext
pip install -e ".[dev,research]"
```

### Full Installation (All Features)
```bash
pip install "openlongcontext[all]"
```

## 🎯 Quick Start

### 1. API Service

Start the API server:
```bash
uvicorn openlongcontext.api.routes:router --reload
```

Upload and query documents:
```python
import requests

# Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/docs/upload",
        files={"file": f}
    )
doc_id = response.json()["doc_id"]

# Query document
response = requests.post(
    "http://localhost:8000/docs/query",
    json={
        "doc_id": doc_id,
        "question": "What is the main conclusion?"
    }
)
print(response.json()["answer"])
```

### 2. Research Tools

Run Bayesian optimization:
```python
from openlongcontext import BayesianOptimizer

def objective(params):
    # Your model training code here
    return validation_loss

optimizer = BayesianOptimizer(
    objective_fn=objective,
    bounds={
        "learning_rate": (1e-5, 1e-2),
        "dropout": (0.0, 0.5),
        "hidden_size": (128, 1024)
    },
    n_iter=50,
    acquisition="ei"  # Expected Improvement
)

result = optimizer.optimize()
print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score}")
```

### 3. Long-Context Models

Load and use models:
```python
from openlongcontext.models import longformer

# Initialize model
model = longformer.LongformerForQuestionAnswering()

# Process long document
answer = model.answer_question(
    context="Your very long document text here...",
    question="What is the main topic?"
)
```

## 🔬 Advanced Usage

### Ablation Studies

```python
from openlongcontext.evaluation.ablation import run_ablation

results = run_ablation(
    config_path="configs/base_model.yaml",
    output_dir="ablation_results/",
    parameters_to_ablate=["attention_type", "positional_encoding"],
    n_trials=10
)
```

### Hyperparameter Sweeps

```python
from openlongcontext import HyperparameterSweep

sweep = HyperparameterSweep(
    objective_fn=train_and_evaluate,
    param_space={
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "batch_size": [16, 32, 64],
        "model_size": ["small", "medium", "large"]
    },
    search_strategy="grid"
)

best_params, best_score = sweep.optimize()
```

### Agent Integration

```python
# Create an agent with long-context capabilities
response = requests.post(
    "http://localhost:8000/agents/create",
    json={
        "agent_type": "long_context",
        "name": "Research Assistant",
        "openai_api_key": "your-api-key",
        "config": {
            "max_context_length": 32768,
            "chunk_size": 4096
        }
    }
)

agent_id = response.json()["agent_id"]

# Execute complex task
response = requests.post(
    f"http://localhost:8000/agents/{agent_id}/execute",
    json={
        "task": "Summarize all research papers and identify common themes",
        "context": {"document_ids": ["doc1", "doc2", "doc3"]}
    }
)
```

## 🛠️ CLI Tools

The package includes several command-line tools:

```bash
# Run experiments
openlongcontext-experiment --config experiments/scaling_law.yaml

# Hyperparameter sweep
openlongcontext-sweep --config sweeps/transformer_xl.yaml --multirun

# Analyze results
openlongcontext-analyze results/experiment_001/

# Run ablation study
openlongcontext-ablate --config ablations/attention_patterns.yaml
```

## 📊 Evaluation Metrics

Built-in metrics for long-context evaluation:

- **Perplexity**: Standard and sliding-window variants
- **Retrieval Metrics**: Precision, recall, F1 for passage retrieval
- **Copy Task Metrics**: Accuracy for synthetic memory tasks
- **Reasoning Metrics**: Chain-of-thought evaluation

## 🔧 Configuration

Example configuration for experiments:

```yaml
# configs/experiment.yaml
model:
  name: longformer
  max_length: 16384
  attention_window: 512

training:
  batch_size: 8
  learning_rate: 5e-5
  num_epochs: 10
  gradient_checkpointing: true

data:
  dataset: "scientific_papers"
  max_samples: 10000
  preprocessing: "chunk_and_stride"
```

## 📈 Performance

Benchmark results on long-context tasks:

| Model | Max Context | PG-19 PPL | arXiv QA | Throughput |
|-------|-------------|-----------|----------|------------|
| Longformer | 4,096 | 18.3 | 0.73 | 245 tok/s |
| BigBird | 4,096 | 17.9 | 0.75 | 198 tok/s |
| Hyena | 8,192 | 19.1 | 0.71 | 412 tok/s |
| Transformer-XL | 16,384 | 16.8 | 0.77 | 156 tok/s |

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance benchmarks
pytest tests/performance/

# Full test suite with coverage
pytest --cov=openlongcontext --cov-report=html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

Full documentation is available at [https://openlongcontext.readthedocs.io](https://openlongcontext.readthedocs.io)

Key documentation sections:
- [API Reference](https://openlongcontext.readthedocs.io/api/)
- [Model Architecture Guide](https://openlongcontext.readthedocs.io/models/)
- [Experiment Design](https://openlongcontext.readthedocs.io/experiments/)
- [Scaling Laws](https://openlongcontext.readthedocs.io/scaling/)

## 📝 Citation

If you use OpenLongContext in your research, please cite:

```bibtex
@software{openlongcontext2024,
  title = {OpenLongContext: A Comprehensive Platform for Long-Context Scaling Research},
  author = {Jois, Nik},
  year = {2024},
  url = {https://github.com/openlongcontext/openlongcontext},
  version = {1.0.0}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the groundbreaking work on efficient transformers
- Built on top of Hugging Face Transformers
- Powered by PyTorch and FastAPI

## 📞 Contact

- **Author**: Nik Jois
- **Email**: nikjois@llamasearch.ai
- **Issues**: [GitHub Issues](https://github.com/openlongcontext/openlongcontext/issues)
- **Discussions**: [GitHub Discussions](https://github.com/openlongcontext/openlongcontext/discussions)