#!/usr/bin/env python3
"""Verify all imports are working correctly."""

import sys
import importlib
from typing import List, Tuple

def check_import(module_name: str) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"✓ {module_name} ({version})"
    except ImportError as e:
        return False, f"✗ {module_name}: {str(e)}"

def main():
    """Check all required imports."""
    print("Verifying imports for OpenLongContext...\n")
    
    # Core dependencies
    core_modules = [
        "numpy",
        "scipy",
        "scipy.optimize",
        "scipy.stats",
        "sklearn",
        "sklearn.gaussian_process",
        "matplotlib",
        "matplotlib.pyplot",
        "torch",
        "torch.nn",
        "transformers",
        "omegaconf",
        "openai",
        "fastapi",
        "pydantic",
        "pytest",
    ]
    
    # Check each module
    failed = []
    for module in core_modules:
        success, message = check_import(module)
        print(message)
        if not success:
            failed.append(module)
    
    # Check OpenLongContext modules
    print("\nChecking OpenLongContext modules:")
    local_modules = [
        "openlongcontext",
        "openlongcontext.ablation.bayesian_optimization",
        "openlongcontext.ablation.hyperparameter_sweep",
        "openlongcontext.ablation.experiment_registry",
        "openlongcontext.evaluation.ablation",
        "openlongcontext.api.routes",
        "openlongcontext.agents.openai_agent",
        "openlongcontext.models.longformer",
    ]
    
    for module in local_modules:
        success, message = check_import(module)
        print(message)
        if not success:
            failed.append(module)
    
    # Summary
    print(f"\n{'='*50}")
    if failed:
        print(f"ERROR: {len(failed)} imports failed:")
        for module in failed:
            print(f"   - {module}")
        print("\nTo fix, run: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("SUCCESS: All imports successful!")
        print("\nYour environment is properly configured.")
        sys.exit(0)

if __name__ == "__main__":
    main()