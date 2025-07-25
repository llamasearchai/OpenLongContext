# File path: /Users/nemesis/OpenPre-Training/pyproject.toml

[build-system]
requires = ["setuptools>=61.0", "wheel", "torch", "numpy"]
build-backend = "setuptools.build_meta"

[project]
name = "openlongcontext"
version = "1.0.0"
description = "Ultimate Long-Context Scaling Research Platform for Principled Algorithmic and Empirical Study"
authors = [
    {name = "OpenLongContext Research Team", email = "research@openlongcontext.ai"}
]
readme = "README.md"
license = {text = "Apache-2.0"}
keywords = [
    "long-context", "scaling-laws", "transformer-xl", "efficient-attention", "memory", "deep-learning", "openai"
]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
requires-python = ">=3.9"
dependencies = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.35.0",
    "datasets>=2.14.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "pandas>=2.1.0",
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.0",
    "wandb>=0.16.0",
    "mlflow>=2.8.0",
    "tensorboard>=2.15.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "scikit-learn>=1.3.0",
    "pyyaml>=6.0.0",
    "tqdm>=4.65.0",
    "rich>=13.6.0",
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "pytest-xdist>=3.4.0",
]

[project.optional-dependencies]
dev = [
    "jupyter>=1.0.0",
    "jupyterlab>=4.0.0",
    "notebook>=7.0.0",
    "ipywidgets>=8.1.0",
    "nbconvert>=7.0.0",
    "black>=23.10.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "mypy>=1.6.0",
]
research = [
    "optuna>=3.4.0",
    "ax-platform>=0.3.4",
    "hyperopt>=0.2.7",
    "pymc>=5.9.0",
]

[project.urls]
Homepage = "https://github.com/openlongcontext/openlongcontext"
Documentation = "https://openlongcontext.readthedocs.io"
Repository = "https://github.com/openlongcontext/openlongcontext"
"Bug Tracker" = "https://github.com/openlongcontext/openlongcontext/issues"

[project.scripts]
openlongcontext = "openlongcontext.cli:main"
openlongcontext-experiment = "openlongcontext.cli.run_experiment:main"
openlongcontext-sweep = "openlongcontext.cli.sweep:main"
openlongcontext-analyze = "openlongcontext.cli.analyze_results:main"
openlongcontext-ablate = "openlongcontext.cli.ablate:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["openlongcontext*"]

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true