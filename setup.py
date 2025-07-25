from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Read the contents of README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="openlongcontext",
    version="1.0.0",
    description="Advanced Long-Context Language Model Framework with CUDA, CPU, and MLX support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    url="https://github.com/nikjois/OpenContext",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
        "rich>=13.6.0",
        
        # API dependencies
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "httpx>=0.25.0",
        "python-multipart>=0.0.6",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "slowapi>=0.1.9",
        
        # Model dependencies
        "einops>=0.7.0",
        
        # Agent dependencies
        "openai>=1.3.0",
        "langchain>=0.0.340",
        "tiktoken>=0.5.0",
        
        # Evaluation dependencies
        "rouge-score>=0.1.2",
        "bert-score>=0.3.13",
        "sacrebleu>=2.3.0",
        "nltk>=3.8.0",
        
        # Utilities
        "click>=8.1.0",
        "python-dotenv>=1.0.0",
        "tenacity>=8.2.0",
        "joblib>=1.3.0",
        "filelock>=3.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-xdist>=3.4.0",
            "pytest-benchmark>=4.0.0",
            "hypothesis>=6.92.0",
            "mypy>=1.6.0",
            "ruff>=0.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "pre-commit>=3.5.0",
        ],
        "cuda": [
            "flash-attn>=2.3.0",
            "triton>=2.1.0",
            "apex>=0.1",
            "bitsandbytes>=0.41.0",
            "xformers>=0.0.22",
        ],
        "mlx": [
            "mlx>=0.0.1",
            "mlx-lm>=0.0.1",
        ],
        "research": [
            "wandb>=0.16.0",
            "mlflow>=2.8.0",
            "tensorboard>=2.15.0",
            "prometheus-client>=0.19.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
            "optuna>=3.4.0",
            "scikit-optimize>=0.9.0",
            "hyperopt>=0.2.7",
            "ray[tune]>=2.8.0",
        ],
        "data": [
            "h5py>=3.10.0",
            "zarr>=2.16.0",
            "pyarrow>=14.0.0",
            "faiss-cpu>=1.7.4",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
            "myst-parser>=2.0.0",
            "autodoc-pydantic>=2.0.0",
        ],
        "all": [
            # Include all optional dependencies
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-xdist>=3.4.0",
            "pytest-benchmark>=4.0.0",
            "hypothesis>=6.92.0",
            "mypy>=1.6.0",
            "ruff>=0.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "pre-commit>=3.5.0",
            "wandb>=0.16.0",
            "mlflow>=2.8.0",
            "tensorboard>=2.15.0",
            "prometheus-client>=0.19.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
            "optuna>=3.4.0",
            "scikit-optimize>=0.9.0",
            "hyperopt>=0.2.7",
            "ray[tune]>=2.8.0",
            "h5py>=3.10.0",
            "zarr>=2.16.0",
            "pyarrow>=14.0.0",
            "faiss-cpu>=1.7.4",
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
            "myst-parser>=2.0.0",
            "autodoc-pydantic>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "openlongcontext=openlongcontext.cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Framework :: FastAPI",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "long-context", "transformer", "attention", "flash-attention", "longformer",
        "bigbird", "hyena", "mlx", "cuda", "pytorch", "nlp", "language-model",
        "question-answering", "ai", "ml", "research", "evaluation", "benchmarks"
    ],
    project_urls={
        "Homepage": "https://github.com/nikjois/OpenContext",
        "Documentation": "https://nikjois.github.io/OpenContext",
        "Repository": "https://github.com/nikjois/OpenContext",
        "Bug Tracker": "https://github.com/nikjois/OpenContext/issues",
        "Changelog": "https://github.com/nikjois/OpenContext/blob/main/CHANGELOG.md",
    },
    package_data={
        "openlongcontext": ["py.typed"],
    },
)