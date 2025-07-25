# OpenLongContext Documentation

Welcome to the comprehensive documentation for OpenLongContext - the ultimate long-context scaling research platform for principled algorithmic and empirical study.

## Table of Contents

- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Research Guides](#research-guides)
- [Tutorials](#tutorials)
- [Advanced Topics](#advanced-topics)
- [FAQ](#faq)

## Getting Started

### Quick Installation

```bash
git clone https://github.com/llamasearchai/OpenLongContext.git
cd OpenLongContext
pip install -e .[dev,research,agents]
```

### Your First Experiment

```python
from openlongcontext import BayesianOptimizer

def objective(params):
    # Your model evaluation logic here
    return loss_value

optimizer = BayesianOptimizer(objective, bounds, n_iter=50)
result = optimizer.optimize()
```

## API Reference

### Core Classes

#### BayesianOptimizer
Advanced Bayesian optimization for hyperparameter tuning.

```python
class BayesianOptimizer:
    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        bounds: Dict[str, Tuple[float, float]],
        n_iter: int = 25,
        acquisition: str = "ei"
    ):
        """
        Initialize Bayesian Optimizer.
        
        Args:
            objective_fn: Function to minimize
            bounds: Parameter bounds as {param_name: (min, max)}
            n_iter: Number of optimization iterations
            acquisition: Acquisition function ('ei', 'ucb', 'pi')
        """
```

#### LongformerForQuestionAnswering
Long-context transformer model for question answering.

```python
class LongformerForQuestionAnswering:
    def __init__(
        self,
        attention_window: int = 512,
        max_position_embeddings: int = 4098,
        hidden_size: int = 768
    ):
        """
        Initialize Longformer model.
        
        Args:
            attention_window: Size of sliding window attention
            max_position_embeddings: Maximum sequence length
            hidden_size: Hidden dimension size
        """
```

### REST API Endpoints

#### Authentication

**POST /api/v1/auth/login**
```json
{
  "username": "string",
  "password": "string"
}
```

Response:
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**POST /api/v1/auth/register**
```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "full_name": "string"
}
```

#### Document Management

**POST /api/v1/documents/upload**
- Content-Type: multipart/form-data
- File: document file

Response:
```json
{
  "doc_id": "string",
  "filename": "string",
  "size": 12345,
  "status": "processed"
}
```

**GET /api/v1/documents**
```json
{
  "documents": [
    {
      "doc_id": "string",
      "filename": "string",
      "uploaded_at": "2024-01-01T00:00:00Z",
      "status": "processed"
    }
  ]
}
```

#### Question Answering

**POST /api/v1/qa/ask**
```json
{
  "question": "string",
  "doc_id": "string",
  "context_length": 4096
}
```

Response:
```json
{
  "answer": "string",
  "confidence": 0.95,
  "context": "string",
  "processing_time": 1.23
}
```

## Research Guides

### Scaling Law Analysis

Learn how to conduct comprehensive scaling law studies:

1. **Data Preparation**: Prepare datasets of varying sizes
2. **Model Training**: Train models with different parameter counts
3. **Analysis**: Fit scaling curves and extract insights

```python
from openlongcontext.theory import ScalingLawAnalyzer

analyzer = ScalingLawAnalyzer()
results = analyzer.fit_scaling_law(
    model_sizes=[1e6, 1e7, 1e8],
    dataset_sizes=[1e9, 1e10, 1e11],
    performance_metrics=[0.8, 0.85, 0.9]
)
```

### Ablation Studies

Systematic component analysis methodology:

1. **Component Identification**: Define components to ablate
2. **Experimental Design**: Create ablation matrix
3. **Execution**: Run experiments systematically
4. **Analysis**: Measure component contributions

```python
from openlongcontext.ablation import run_ablation

results = run_ablation(
    base_config="configs/models/longformer.yaml",
    components=["attention", "feedforward", "layer_norm"],
    metrics=["perplexity", "accuracy"]
)
```

### Hyperparameter Optimization

Advanced optimization strategies:

#### Bayesian Optimization
- Uses Gaussian Processes for efficient search
- Balances exploration vs exploitation
- Handles noisy evaluations

#### Grid Search
- Exhaustive search over parameter grid
- Guaranteed to find global optimum in discrete space
- Computationally expensive but reliable

#### Random Search
- Random sampling from parameter distributions
- Often outperforms grid search
- Good baseline for comparison

## Tutorials

### Tutorial 1: Basic Document QA

Step-by-step guide to implement document question answering:

```python
# 1. Initialize the API client
import httpx

client = httpx.AsyncClient(base_url="http://localhost:8000")

# 2. Upload a document
with open("document.pdf", "rb") as f:
    response = await client.post("/api/v1/documents/upload", files={"file": f})
    doc_id = response.json()["doc_id"]

# 3. Ask questions
response = await client.post("/api/v1/qa/ask", json={
    "question": "What is the main topic?",
    "doc_id": doc_id
})
answer = response.json()["answer"]
```

### Tutorial 2: Advanced Model Training

Learn to train custom long-context models:

```python
from openlongcontext.models import LongformerForQuestionAnswering
from openlongcontext.training import Trainer

# 1. Initialize model
model = LongformerForQuestionAnswering(
    attention_window=512,
    max_position_embeddings=4096
)

# 2. Prepare data
train_dataset = load_dataset("squad", split="train")
val_dataset = load_dataset("squad", split="validation")

# 3. Configure training
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    learning_rate=3e-5,
    batch_size=8
)

# 4. Train
trainer.train()
```

### Tutorial 3: Research Workflow

Complete research pipeline from hypothesis to publication:

1. **Hypothesis Formation**: Define research questions
2. **Experimental Design**: Plan experiments and metrics
3. **Implementation**: Code experiments using OpenLongContext
4. **Execution**: Run experiments with proper tracking
5. **Analysis**: Analyze results and draw conclusions
6. **Reporting**: Generate figures and write up results

## Advanced Topics

### Memory Optimization

Techniques for handling extremely long contexts:

- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision**: Use FP16/BF16 for efficiency
- **Attention Patterns**: Optimize attention mechanisms
- **Sequence Parallelism**: Distribute long sequences

### Custom Model Integration

How to integrate your own models:

```python
from openlongcontext.models import BaseModel

class CustomLongContextModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        # Your model implementation
        
    def forward(self, input_ids, attention_mask=None):
        # Forward pass implementation
        return outputs
```

### Distributed Training

Scale training across multiple GPUs and nodes:

```python
from openlongcontext.training import DistributedTrainer

trainer = DistributedTrainer(
    model=model,
    world_size=8,  # Number of GPUs
    strategy="deepspeed"  # or "fsdp"
)
```

## FAQ

### General Questions

**Q: What context lengths does OpenLongContext support?**
A: OpenLongContext supports context lengths from 512 tokens up to 1 million tokens, depending on the model and available memory.

**Q: How do I contribute to the project?**
A: See our [Contributing Guide](https://github.com/llamasearchai/OpenLongContext/blob/main/CONTRIBUTING.md) for detailed instructions.

### Technical Questions

**Q: How do I optimize memory usage for long sequences?**
A: Use gradient checkpointing, mixed precision training, and consider using efficient attention patterns like sliding window or block-sparse attention.

**Q: Can I use custom datasets?**
A: Yes, OpenLongContext supports custom datasets. Implement the `Dataset` interface or use our data loading utilities.

### Troubleshooting

**Q: I'm getting CUDA out of memory errors. What should I do?**
A: Try reducing batch size, enabling gradient checkpointing, using mixed precision, or distributing across multiple GPUs.

**Q: My experiments are running slowly. How can I speed them up?**
A: Consider using efficient attention mechanisms, optimizing your data loading pipeline, and ensuring you're using GPU acceleration.

## Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/llamasearchai/OpenLongContext/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/llamasearchai/OpenLongContext/discussions)
- **Email**: nikjois@llamasearch.ai

---

**Last updated**: January 2024  
**Version**: 1.0.0