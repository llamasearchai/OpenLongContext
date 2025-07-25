# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive authentication system with JWT and API key support
- Role-based access control (RBAC) for fine-grained permissions
- Rate limiting and security middleware
- GitHub Actions workflows for CI/CD
- Professional documentation with MkDocs
- Pre-commit hooks configuration
- CLAUDE.md for AI assistant context
- Extensive test coverage framework
- Docker multi-platform support
- Dependabot configuration
- CodeQL security scanning

### Changed
- Standardized project metadata across pyproject.toml and setup.py
- Enhanced requirements.txt with categorized dependencies
- Improved API security with authentication requirements
- Updated pyproject.toml with comprehensive tool configurations

### Fixed
- Malformed requirements.txt file
- Missing deployment documentation
- Missing contributing guidelines

## [1.0.0] - 2024-01-XX

### Added
- Initial release of OpenLongContext
- FastAPI-based REST API service
- Support for multiple long-context models:
  - Longformer for question answering
  - BigBird sparse attention
  - Hyena model
  - Transformer-XL with recurrence
  - Memorizing Transformer
  - FlashAttention implementation
  - Linear attention mechanisms
  - RWKV model
- OpenAI agent integration
- Comprehensive dataset loaders:
  - Real datasets: PG19, Books3, ArXiv Math, Code Continuation, GitHub Issues
  - Synthetic datasets: Copy Task, Recall Task, Retrieval Task, Reasoning Task
- Experiment tracking with MLflow, Weights & Biases, and TensorBoard
- Ablation study tools with Bayesian optimization
- CLI tools for experiments and analysis
- Docker support with GPU acceleration
- Comprehensive configuration system using Hydra/OmegaConf

### Security
- API authentication system
- Rate limiting
- CORS configuration
- Security headers middleware

## [0.9.0] - 2023-12-XX (Pre-release)

### Added
- Beta version with core functionality
- Basic API endpoints
- Initial model implementations
- Testing framework setup

## Links
- [Compare v1.0.0...HEAD](https://github.com/openlongcontext/openlongcontext/compare/v1.0.0...HEAD)
- [Compare v0.9.0...v1.0.0](https://github.com/openlongcontext/openlongcontext/compare/v0.9.0...v1.0.0)