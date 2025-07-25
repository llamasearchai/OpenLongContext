# CLAUDE.md - AI Assistant Context for OpenLongContext

This file provides context and guidance for AI assistants (like Claude) working with the OpenLongContext codebase.

## Project Overview

OpenLongContext is a production-ready FastAPI service and research platform for long-context document QA and retrieval with OpenAI agents integration. It's designed to handle documents with millions of tokens and provides state-of-the-art long-context processing capabilities.

## Key Features

1. **Long-Context Processing**: Handles documents up to millions of tokens
2. **Multiple Model Architectures**: Longformer, BigBird, Hyena, Transformer-XL, and more
3. **OpenAI Agent Integration**: Seamless integration with OpenAI's API for agent-based processing
4. **Production-Ready API**: FastAPI-based REST API with authentication and rate limiting
5. **Experiment Tracking**: Integration with MLflow, Weights & Biases, and TensorBoard
6. **Comprehensive Testing**: Unit, integration, and performance tests

## Codebase Structure

```
openlongcontext/
├── ablation/         # Ablation study tools (Bayesian optimization, grid search)
├── agents/           # AI agent implementations with OpenAI integration
├── algorithms/       # Core algorithms (attention, memory, chunking)
├── api/              # FastAPI service with authentication
├── cli/              # Command-line interface tools
├── core/             # Core framework (config, engine, experiment)
├── datasets/         # Dataset loaders for real and synthetic data
├── evaluation/       # Evaluation metrics and analysis
├── models/           # Model implementations
├── theory/           # Theoretical analysis modules
├── tracking/         # Experiment tracking integrations
└── utils/            # Utility functions
```

## Development Guidelines

### Code Style
- Follow PEP 8 with 100-character line length
- Use type hints for all functions
- Write Google-style docstrings
- Run `black`, `isort`, and `ruff` before committing

### Testing
- Write tests for all new features
- Aim for >90% test coverage
- Use pytest for testing
- Run: `pytest tests/`

### Commands to Run
- **Linting**: `ruff check openlongcontext tests`
- **Formatting**: `black openlongcontext tests`
- **Type checking**: `mypy openlongcontext`
- **Tests**: `pytest tests/ -v`
- **API Server**: `python -m openlongcontext.api`

## Common Tasks

### Adding a New Model
1. Create model file in `openlongcontext/models/`
2. Inherit from `BaseModel` class
3. Implement required methods: `forward`, `generate`, `encode`
4. Add tests in `tests/unit/test_models.py`
5. Update model registry in `models/__init__.py`

### Adding API Endpoints
1. Create route in `openlongcontext/api/routes.py`
2. Add authentication with `@require_auth` decorator
3. Use Pydantic models for request/response
4. Add tests in `tests/integration/test_api.py`
5. Update API documentation

### Running Experiments
```bash
openlongcontext-experiment --config configs/experiments/baseline.yaml
openlongcontext-sweep --config configs/sweep/hyperparameter.yaml
openlongcontext-analyze --experiment-id exp_123
```

## Architecture Decisions

### Authentication System
- JWT-based authentication with refresh tokens
- API key support for programmatic access
- Role-based access control (RBAC)
- Rate limiting per user/API key

### Model Design
- Modular architecture for easy model swapping
- Efficient attention mechanisms for long sequences
- Memory-optimized implementations
- Support for model quantization

### API Design
- RESTful API following OpenAPI 3.0
- Async request handling
- WebSocket support for streaming
- Comprehensive error handling

## Performance Considerations

### Memory Optimization
- Use gradient checkpointing for large models
- Implement sliding window attention
- Enable mixed precision training
- Use efficient data loaders

### Scaling
- Horizontal scaling with multiple workers
- Redis for caching and queuing
- PostgreSQL for persistent storage
- Docker/Kubernetes deployment

## Debugging Tips

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Slow Inference**: Check if GPU is being used, enable Flash Attention
3. **API Timeouts**: Increase timeout in nginx/gunicorn config
4. **Import Errors**: Ensure package is installed with `pip install -e .`

## Important Files

- `pyproject.toml`: Project configuration and dependencies
- `setup.py`: Package setup (being phased out for pyproject.toml)
- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `Dockerfile`: Container configuration
- `.github/workflows/`: CI/CD pipelines

## Environment Variables

Key environment variables to set:
- `OPENAI_API_KEY`: OpenAI API key for agent functionality
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: JWT secret key
- `API_RATE_LIMIT`: Rate limit configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following guidelines
4. Ensure all tests pass
5. Submit a pull request

## Resources

- [Documentation](https://openlongcontext.github.io/openlongcontext/)
- [API Reference](https://openlongcontext.github.io/openlongcontext/api/)
- [Contributing Guide](CONTRIBUTING.md)
- [Deployment Guide](docs/deployment.md)

## Contact

For questions or issues:
- GitHub Issues: https://github.com/openlongcontext/openlongcontext/issues
- Email: nikjois@llamasearch.ai