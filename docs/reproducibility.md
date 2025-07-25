# Reproducibility Guide

OpenLongContext is designed for rigorous, reproducible research and production deployment. This guide explains how to ensure your experiments and API results are fully reproducible.

## Configuration Management

- All experiments and API runs are controlled by YAML configs in `configs/`.
- Use Hydra for hierarchical, composable config management.
- Always specify config versions in experiment logs and results.

## Versioning

- The codebase is versioned via Git and PyPI releases.
- All API and CLI runs log the current code version (commit hash or release tag).
- Use `git tag` and `git describe` to track experiment provenance.

## Experiment Tracking

- Integrate with MLflow or Weights & Biases (W&B) for full experiment tracking.
- Log all hyperparameters, configs, and results.
- Store random seeds and environment info for each run.

## API Reproducibility

- All API requests are deterministic given the same model, config, and document.
- Log all API requests and responses for auditability.
- Use versioned models and configs in production.

## Best Practices

- Use `pytest --cov` to ensure all code paths are tested.
- Use CI/CD to enforce reproducibility and prevent regressions.
- Document all experiment and deployment steps in Markdown (see `docs/`).
