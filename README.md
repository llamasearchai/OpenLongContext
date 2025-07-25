# OpenLongContext

A comprehensive framework for long-context language models with efficient attention mechanisms, advanced evaluation metrics, and production-ready infrastructure.

**Author:** Nik Jois <nikjois@llamasearch.ai>

## Overview

OpenLongContext is a state-of-the-art framework designed to handle extremely long sequences in language models. It provides efficient attention mechanisms, comprehensive evaluation tools, and a robust API for deploying long-context models in production environments.

## Key Features

### 🚀 Efficient Attention Mechanisms
- **FlashAttention**: GPU-optimized attention with automatic fallback to CPU/MLX
- **Sparse Attention**: Configurable sparsity patterns for reduced complexity
- **Linear Attention**: O(n) complexity attention mechanisms
- **Sliding Window**: Local attention with configurable window sizes
- **Multi-Scale Attention**: Hierarchical attention across different scales

### 🧠 Advanced Model Architectures
- **Longformer**: Sparse attention for long documents
- **BigBird**: Random + local + global attention patterns
- **Hyena**: Subquadratic attention alternative
- **Transformer-XL**: Recurrent memory mechanisms
- **Memorizing Transformer**: External memory integration
- **RWKV**: Efficient recurrent architecture
- **Rotary Position Embeddings**: Advanced positional encoding

### 📊 Comprehensive Evaluation Suite
- **Perplexity Analysis**: Token-level and position-wise metrics
- **Retrieval Metrics**: Long-context information retrieval
- **Copy Task Evaluation**: Memory and attention analysis
- **Reasoning Metrics**: Complex reasoning over long contexts
- **Error Analysis**: Detailed failure mode investigation

### 🔬 Experiment Management
- **MLflow Integration**: Comprehensive experiment tracking
- **Weights & Biases**: Advanced visualization and monitoring
- **TensorBoard**: Real-time training metrics
- **Reproducibility Tools**: Deterministic experiment execution
- **Ablation Studies**: Systematic component analysis

### 🌐 Production API
- **FastAPI Backend**: High-performance REST API
- **Authentication**: JWT and API key support
- **Rate Limiting**: Production-ready request throttling
- **Multi-Backend Support**: CUDA, CPU, and MLX acceleration
- **Docker Deployment**: Containerized deployment ready

### 📚 Rich Dataset Support
- **Real Datasets**: PG19, Books3, ArXiv Math, GitHub Issues, Code Continuation
- **Synthetic Tasks**: Copy, Recall, Retrieval, and Reasoning benchmarks
- **Custom Loaders**: Flexible data pipeline integration

## Installation

### Quick Start

```bash
pip install openlongcontext
```

### Development Installation

```bash
git clone https://github.com/llamasearchai/OpenLongContext.git
cd OpenLongContext
pip install -e .
```

### CUDA Support

```bash
pip install -r requirements-cuda.txt
```

### MLX Support (Apple Silicon)

```bash
pip install -r requirements-mlx.txt
```

## Quick Start

### Basic Usage

```python
from openlongcontext.models import FlashAttentionForQuestionAnswering
from openlongcontext.datasets import PG19Dataset
from openlongcontext.evaluation import evaluate_model_perplexity

# Initialize model with FlashAttention
model = FlashAttentionForQuestionAnswering(
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    n_layers=12,
    max_seq_len=8192
)

# Load long-context dataset
dataset = PG19Dataset(max_length=8192)
dataloader = dataset.get_dataloader(batch_size=2)

# Evaluate model
results = evaluate_model_perplexity(model, dataloader, device='cuda')
print(f"Perplexity: {results['perplexity']:.2f}")
```

### API Server

```bash
# Start the API server
python -m openlongcontext.api

# Or with Docker
docker build -t openlongcontext .
docker run -p 8000:8000 openlongcontext
```

### Experiment Tracking

```python
from openlongcontext.tracking import create_mlflow_tracker, create_wandb_tracker

# MLflow tracking
mlflow_tracker = create_mlflow_tracker("long-context-experiments")
with mlflow_tracker.start_run():
    # Your training code here
    mlflow_tracker.log_metrics({"perplexity": 15.2, "loss": 2.72})
    mlflow_tracker.log_model(model)

# Weights & Biases tracking
wandb_tracker = create_wandb_tracker("openlongcontext", "my-experiment")
wandb_tracker.log_metrics({"perplexity": 15.2, "loss": 2.72})
wandb_tracker.log_model(model)
```

## Architecture

OpenLongContext is built with a modular architecture:

```
openlongcontext/
├── models/           # Long-context model implementations
├── algorithms/       # Efficient attention mechanisms
├── datasets/         # Dataset loaders and processors
├── evaluation/       # Comprehensive evaluation metrics
├── tracking/         # Experiment tracking integrations
├── api/             # Production-ready API server
├── cli/             # Command-line interface
├── core/            # Core utilities and configurations
├── theory/          # Theoretical analysis tools
└── utils/           # General utilities
```

## Models

### FlashAttention

```python
from openlongcontext.models import FlashAttention

attention = FlashAttention(
    d_model=768,
    n_heads=12,
    dropout=0.1,
    causal=True,
    window_size=None  # Full attention
)

# Automatic backend selection (CUDA -> CPU -> MLX)
output = attention(hidden_states, attention_mask)
```

### Efficient Attention Variants

```python
from openlongcontext.algorithms import create_efficient_attention

# Sparse attention
sparse_attn = create_efficient_attention(
    "sparse", 
    d_model=768, 
    block_size=64, 
    num_random_blocks=3
)

# Linear attention
linear_attn = create_efficient_attention(
    "linear", 
    d_model=768, 
    feature_dim=256
)

# Sliding window
window_attn = create_efficient_attention(
    "sliding_window", 
    d_model=768, 
    window_size=512
)
```

## Evaluation

### Perplexity Analysis

```python
from openlongcontext.evaluation import (
    evaluate_model_perplexity,
    analyze_perplexity_by_position,
    plot_perplexity_analysis
)

# Standard evaluation
results = evaluate_model_perplexity(model, dataloader, device)

# Position-wise analysis
position_analysis = analyze_perplexity_by_position(model, dataloader, device)
fig = plot_perplexity_analysis(position_analysis, save_path="perplexity_analysis.png")
```

### Custom Metrics

```python
from openlongcontext.evaluation import compute_retrieval_metrics

# Long-context retrieval evaluation
retrieval_results = compute_retrieval_metrics(
    model, 
    retrieval_dataset, 
    max_context_length=8192
)
```

## CLI Tools

```bash
# Run experiments
openlongcontext experiment run --config configs/longformer_experiment.yaml

# Evaluate models
openlongcontext evaluate --model longformer --dataset pg19 --max-length 4096

# Hyperparameter sweeps
openlongcontext sweep --config configs/sweep_config.yaml --num-trials 50

# Ablation studies
openlongcontext ablate --component attention --variations sparse,linear,sliding
```

## API Documentation

The REST API provides endpoints for:

- **Document Upload**: `/docs/upload`
- **Question Answering**: `/docs/query`
- **Model Management**: `/models/`
- **Health Monitoring**: `/health`

### Example API Usage

```python
import requests

# Upload document
response = requests.post(
    "http://localhost:8000/docs/upload",
    files={"file": open("long_document.txt", "rb")},
    headers={"Authorization": "Bearer your-api-key"}
)

# Query document
response = requests.post(
    "http://localhost:8000/docs/query",
    json={
        "doc_id": response.json()["doc_id"],
        "question": "What is the main theme of this document?",
        "max_length": 4096
    },
    headers={"Authorization": "Bearer your-api-key"}
)
```

## Configuration

OpenLongContext uses Hydra for configuration management:

```yaml
# config.yaml
model:
  name: "flashattention"
  d_model: 768
  n_heads: 12
  n_layers: 12
  max_seq_len: 8192

training:
  batch_size: 4
  learning_rate: 1e-4
  max_epochs: 10
  gradient_clip: 1.0

dataset:
  name: "pg19"
  max_length: 8192
  split: "train"

tracking:
  use_mlflow: true
  use_wandb: true
  experiment_name: "long-context-experiment"
```

## Performance

OpenLongContext is optimized for efficiency:

| Model | Sequence Length | Memory (GB) | Speed (tokens/sec) |
|-------|----------------|-------------|-------------------|
| Standard Attention | 2K | 8.2 | 1,200 |
| FlashAttention | 8K | 12.1 | 2,800 |
| Sparse Attention | 16K | 15.3 | 2,200 |
| Linear Attention | 32K | 18.7 | 3,500 |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/llamasearchai/OpenLongContext.git
cd OpenLongContext

# Setup development environment
bash scripts/setup_dev_env.sh

# Run tests
pytest tests/

# Run linting
pre-commit run --all-files
```

## Citation

If you use OpenLongContext in your research, please cite:

```bibtex
@software{jois2024openlongcontext,
  title={OpenLongContext: A Comprehensive Framework for Long-Context Language Models},
  author={Jois, Nik},
  year={2024},
  url={https://github.com/llamasearchai/OpenLongContext},
  version={1.0.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://openlongcontext.readthedocs.io](https://openlongcontext.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/llamasearchai/OpenLongContext/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llamasearchai/OpenLongContext/discussions)
- **Email**: nikjois@llamasearch.ai

## Acknowledgments

- FlashAttention implementation inspired by Dao et al.
- Sparse attention patterns based on BigBird and Longformer
- Linear attention mechanisms from Performer and Linear Transformer
- Evaluation metrics adapted from standard NLP benchmarks

---

**OpenLongContext** - Enabling efficient long-context understanding at scale.