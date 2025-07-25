# Contributing to OpenLongContext

Thank you for your interest in contributing to OpenLongContext! This guide will help you get started with contributing to our long-context AI research platform.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- CUDA-capable GPU (recommended for development)
- 16GB+ RAM (32GB+ recommended for large model development)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/OpenLongContext.git
   cd OpenLongContext
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/llamasearchai/OpenLongContext.git
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Development Dependencies

```bash
# Install package in editable mode with all dependencies
pip install -e .[dev,research,agents]

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Installation

```bash
# Run basic tests
pytest tests/unit/test_core.py

# Check code formatting
black --check openlongcontext/
isort --check-only openlongcontext/

# Type checking
mypy openlongcontext/
```

### 4. Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
OPENLONGCONTEXT_DEBUG=true
OPENLONGCONTEXT_LOG_LEVEL=DEBUG

# OpenAI (for agent features)
OPENAI_API_KEY=your-api-key-here

# Database (optional for development)
DATABASE_URL=sqlite:///./test.db
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Fixes**: Fix existing bugs or issues
2. **Feature Development**: Add new features or capabilities
3. **Documentation**: Improve or add documentation
4. **Research**: Add new models, algorithms, or experiments
5. **Performance**: Optimize existing code
6. **Testing**: Add or improve test coverage

### Branch Naming Convention

Use descriptive branch names with prefixes:

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test improvements
- `perf/` - Performance improvements

Examples:
- `feature/bayesian-optimization`
- `fix/memory-leak-longformer`
- `docs/api-reference-update`

### Commit Message Guidelines

Follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Examples:
```
feat(models): add Hyena attention mechanism

Implement efficient Hyena attention for long-context processing.
Includes memory optimization and CUDA kernel integration.

Closes #123
```

```
fix(ablation): resolve memory leak in bayesian optimizer

Fixed issue where GP models were not properly deallocated
after optimization runs, causing memory accumulation.
```

## Testing

### Test Structure

```
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # Integration tests
└── performance/    # Performance benchmarks
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/performance/             # Performance tests only

# Run with coverage
pytest --cov=openlongcontext --cov-report=html

# Run specific test file
pytest tests/unit/test_bayesian_optimization.py

# Run tests matching pattern
pytest -k "test_optimization"
```

### Writing Tests

#### Unit Tests

```python
import pytest
from openlongcontext.ablation import BayesianOptimizer


def test_bayesian_optimizer_initialization():
    """Test BayesianOptimizer initializes correctly."""
    def dummy_objective(params):
        return params["x"] ** 2
    
    bounds = {"x": (-1.0, 1.0)}
    optimizer = BayesianOptimizer(dummy_objective, bounds)
    
    assert optimizer.bounds == bounds
    assert optimizer.n_iter == 25  # default value
    assert len(optimizer.history) == 0


def test_bayesian_optimizer_optimization():
    """Test BayesianOptimizer finds minimum."""
    def quadratic(params):
        return (params["x"] - 0.5) ** 2
    
    bounds = {"x": (-2.0, 2.0)}
    optimizer = BayesianOptimizer(quadratic, bounds, n_iter=10)
    result = optimizer.optimize()
    
    assert result.best_score < 0.1  # Should find minimum near x=0.5
    assert abs(result.best_params["x"] - 0.5) < 0.3
```

#### Integration Tests

```python
import pytest
from fastapi.testclient import TestClient
from openlongcontext.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_upload_and_query_document(client):
    """Test complete document upload and query workflow."""
    # Upload document
    with open("test_document.txt", "w") as f:
        f.write("This is a test document for OpenLongContext.")
    
    with open("test_document.txt", "rb") as f:
        response = client.post("/api/v1/documents/upload", files={"file": f})
    
    assert response.status_code == 200
    doc_id = response.json()["doc_id"]
    
    # Query document
    response = client.post("/api/v1/qa/ask", json={
        "question": "What is this document about?",
        "doc_id": doc_id
    })
    
    assert response.status_code == 200
    assert "test" in response.json()["answer"].lower()
```

### Test Requirements

- All new features must include tests
- Tests should cover both happy path and edge cases
- Integration tests for API endpoints
- Performance tests for optimization algorithms
- Mock external dependencies (OpenAI API, etc.)

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use type hints for all functions
- Use descriptive variable names
- Add docstrings for all public functions and classes

### Formatting Tools

```bash
# Format code
black openlongcontext/ tests/
isort openlongcontext/ tests/

# Check formatting
black --check openlongcontext/ tests/
isort --check-only openlongcontext/ tests/

# Lint code
flake8 openlongcontext/ tests/
mypy openlongcontext/
```

### Docstring Style

Use Google-style docstrings:

```python
def train_model(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    learning_rate: float = 1e-3,
    epochs: int = 10
) -> Dict[str, float]:
    """Train a PyTorch model on the given dataset.
    
    Args:
        model: The PyTorch model to train
        dataset: Training dataset
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics:
        - 'final_loss': Final training loss
        - 'accuracy': Final training accuracy
        
    Raises:
        ValueError: If learning_rate is not positive
        RuntimeError: If CUDA is required but not available
        
    Example:
        >>> model = LongformerForQuestionAnswering()
        >>> dataset = load_dataset("squad")
        >>> metrics = train_model(model, dataset, learning_rate=3e-5)
        >>> print(f"Final loss: {metrics['final_loss']}")
    """
```

## Documentation

### Documentation Types

1. **API Documentation**: Automatically generated from docstrings
2. **User Guides**: Step-by-step tutorials and examples
3. **Research Documentation**: Experimental methodology and results
4. **Developer Documentation**: Architecture and implementation details

### Building Documentation

```bash
# Install documentation dependencies
pip install -e .[dev]

# Build documentation
cd docs/
make html

# Serve locally
python -m http.server 8080 -d _build/html/
```

### Documentation Guidelines

- Use clear, concise language
- Include code examples for all features
- Add diagrams for complex architectures
- Keep examples up-to-date with code changes
- Use consistent terminology throughout

## Pull Request Process

### Before Submitting

1. **Update from upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run full test suite**:
   ```bash
   pytest
   black --check openlongcontext/
   mypy openlongcontext/
   ```

3. **Update documentation** if needed

4. **Add entry to CHANGELOG.md** if applicable

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages follow convention

### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes

## Related Issues
Closes #[issue_number]
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and style checks
2. **Code Review**: Maintainers review code and provide feedback
3. **Discussion**: Address comments and make requested changes
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to main branch

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug.

**Steps to Reproduce**
1. Step one
2. Step two
3. ...

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- OpenLongContext version: [e.g., 1.0.0]
- GPU: [e.g., NVIDIA RTX 4090]

**Additional Context**
Any other relevant information.
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How do you envision this feature working?

**Alternatives**
Any alternative solutions you've considered.

**Additional Context**
Any other relevant information.
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions, ideas, and general discussion
- **Email**: nikjois@llamasearch.ai for private matters

### Getting Help

1. **Check Documentation**: Review docs and examples first
2. **Search Issues**: Look for existing solutions
3. **Ask in Discussions**: Post questions in GitHub Discussions
4. **Contact Maintainers**: Email for urgent or private matters

### Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes for significant contributions
- Special mentions in documentation

## Development Tips

### Performance Considerations

- Profile code before optimizing
- Use appropriate data structures
- Consider memory usage for large models
- Test with realistic data sizes

### Debugging

- Use descriptive error messages
- Add logging for complex operations
- Include context in exceptions
- Test edge cases thoroughly

### Research Contributions

- Follow reproducible research practices
- Document experimental methodology
- Include statistical significance tests
- Share datasets when possible

---

Thank you for contributing to OpenLongContext! Your contributions help advance the field of long-context AI research.

For questions about this guide, please open an issue or contact the maintainers.