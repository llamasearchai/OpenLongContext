# Troubleshooting Guide for OpenLongContext

## Import Errors in IDE/Type Checkers

If you're seeing import errors in your IDE (VS Code, PyCharm, etc.) or from type checkers (Pyright, mypy), but the code runs fine, this is a common Python development issue. Here's how to resolve it:

### Solution 1: Configure Your IDE

#### VS Code with Pylance/Pyright

1. The project includes `.vscode/settings.json` with proper configuration
2. Make sure to open VS Code at the project root (`/Users/o2/Desktop/OpenContext`)
3. Select the correct Python interpreter:
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from `./venv/bin/python` if using virtual environment

#### PyCharm

1. Mark the project root as "Sources Root":
   - Right-click on `openlongcontext` folder
   - Select "Mark Directory as" → "Sources Root"
2. Configure the Python interpreter:
   - Go to Settings → Project → Python Interpreter
   - Add the virtual environment interpreter

### Solution 2: Install in Development Mode

```bash
# From the project root
pip install -e .
```

This creates a link to your development directory in the Python environment.

### Solution 3: Set PYTHONPATH

```bash
# Add to your shell configuration (.bashrc, .zshrc, etc.)
export PYTHONPATH="${PYTHONPATH}:/Users/o2/Desktop/OpenContext"

# Or run before executing scripts
PYTHONPATH=/Users/o2/Desktop/OpenContext python your_script.py
```

### Solution 4: Use Virtual Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
pip install -e .
```

### Solution 5: Verify Installation

Run the verification script:

```bash
python3 verify_imports.py
```

This will check all imports and report any actual missing dependencies.

## Common Issues and Solutions

### Issue: "Import could not be resolved" in Pyright/Pylance

**Cause**: The type checker can't find the Python packages in your environment.

**Solutions**:
1. The project includes `pyrightconfig.json` which disables these warnings
2. Ensure VS Code is using the correct Python interpreter
3. Restart VS Code after installing packages

### Issue: ImportError at Runtime

**Cause**: Packages are actually not installed.

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue: "No module named 'openlongcontext'"

**Cause**: The package is not in Python's path.

**Solutions**:
1. Install in development mode: `pip install -e .`
2. Run from project root directory
3. Add project to PYTHONPATH

### Issue: Conflicting Package Versions

**Cause**: Different versions of packages installed globally vs in virtual environment.

**Solution**:
Always use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Package Dependencies

All required packages are listed in:
- `requirements.txt` - Basic list format
- `setup.py` - With version constraints
- `pyproject.toml` - Modern Python packaging format

The main dependencies are:
- numpy, scipy, scikit-learn - Scientific computing
- torch, transformers - Deep learning
- fastapi, pydantic - API framework
- omegaconf - Configuration management
- matplotlib - Plotting (optional for some features)

## Development Setup Script

For a complete setup, run:

```bash
bash scripts/setup_dev_env.sh
```

This script will:
1. Check Python version
2. Create virtual environment
3. Install all dependencies
4. Set up pre-commit hooks
5. Create necessary directories
6. Verify all imports

## IDE Configuration Files

The project includes configuration for:
- `.vscode/settings.json` - VS Code settings
- `pyrightconfig.json` - Pyright type checker
- `setup.cfg` - Tool configurations (mypy, pytest, etc.)
- `.python-version` - Python version specification

## Still Having Issues?

1. Check that you're using Python 3.9 or higher:
   ```bash
   python3 --version
   ```

2. Clear Python cache:
   ```bash
   find . -type d -name __pycache__ -exec rm -r {} +
   find . -type f -name "*.pyc" -delete
   ```

3. Reinstall packages:
   ```bash
   pip uninstall -y -r requirements.txt
   pip install -r requirements.txt
   ```

4. For persistent issues, please open an issue on GitHub with:
   - Your Python version
   - Operating system
   - IDE and version
   - Complete error message
   - Output of `pip list`