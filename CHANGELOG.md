# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-01-25

### Fixed
- Fixed import errors and module compatibility issues
- Fixed Pydantic v2 compatibility by updating to use pydantic-settings
- Fixed torch tensor comparison errors in unit tests
- Fixed unused variable warnings across the codebase
- Fixed List[int].tolist() AttributeError in copy_metrics.py

### Changed
- Removed all emoji icons from documentation for professional appearance
- Improved error handling in import statements with proper stacklevel
- Updated type annotations to be more accurate
- Enhanced code quality with proper linting fixes

### Dependencies
- Added pydantic-settings for Pydantic v2 compatibility
- Added email-validator for authentication module

### Development
- Applied ruff auto-fixes for consistent code formatting
- Fixed all linting errors and warnings
- Ensured zero stub implementations remain

## [1.0.0] - 2025-01-24

### Added
- Initial release of OpenLongContext framework
- Production-ready FastAPI service for document QA and retrieval
- State-of-the-art long-context models (Longformer, BigBird, Hyena, etc.)
- Comprehensive ablation study tools with Bayesian optimization
- Agent-based architecture for OpenAI integration
- Extensive evaluation metrics and benchmarking tools
- Multi-backend support (CUDA, CPU, MLX)
- Authentication system with JWT and API key support
- Experiment tracking with MLflow, Weights & Biases, and TensorBoard
- Docker deployment support