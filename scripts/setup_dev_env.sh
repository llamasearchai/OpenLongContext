#!/bin/bash

# Setup development environment for OpenLongContext
set -e

echo "Setting up OpenLongContext development environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo "Error: Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version check passed: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (adjust based on your system)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev,research,agents,all]"

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pip install pre-commit
pre-commit install || echo "Pre-commit not configured"

# Create necessary directories
echo "Creating project directories..."
mkdir -p results
mkdir -p logs
mkdir -p checkpoints
mkdir -p data/raw
mkdir -p data/processed

# Set up Python path
echo "Setting up Python path..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify installations
echo -e "\nVerifying installations..."
python3 -c "import numpy; print(f'✓ NumPy {numpy.__version__}')"
python3 -c "import scipy; print(f'✓ SciPy {scipy.__version__}')"
python3 -c "import sklearn; print(f'✓ Scikit-learn {sklearn.__version__}')"
python3 -c "import matplotlib; print(f'✓ Matplotlib {matplotlib.__version__}')"
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
python3 -c "import fastapi; print(f'✓ FastAPI {fastapi.__version__}')"
python3 -c "import omegaconf; print(f'✓ OmegaConf {omegaconf.__version__}')"
python3 -c "import openai; print(f'✓ OpenAI {openai.__version__}')"

echo -e "\n[SUCCESS] Development environment setup complete!"
echo "To activate the environment in the future, run: source venv/bin/activate"