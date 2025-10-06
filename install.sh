#!/bin/bash
set -e

# FUSION Installation Script
# This script handles the complex PyTorch Geometric installation process

echo "🚀 FUSION Installation Script"
echo "============================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [[ "$python_version" != "3.11" ]]; then
    echo "❌ Error: Python 3.11 is required, but found $python_version"
    echo "Please install Python 3.11 and try again."
    exit 1
fi

echo "✅ Python 3.11 detected"

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Error: No virtual environment detected"
    echo "Please create and activate a virtual environment:"
    echo "  python3.11 -m venv venv"
    echo "  source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment active: $VIRTUAL_ENV"

# Detect platform
platform=$(python3 -c "import platform; print(platform.system().lower())")
arch=$(python3 -c "import platform; print(platform.machine())")

echo "🖥️  Platform: $platform ($arch)"

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install core dependencies first (this includes PyTorch)
echo "🔧 Installing core dependencies..."
pip install -e .

# Install PyTorch Geometric dependencies with platform-specific handling
echo "🧠 Installing PyTorch Geometric dependencies..."

if [[ "$platform" == "darwin" ]]; then
    # macOS installation
    echo "🍎 Detected macOS - using special compilation flags"
    if [[ "$arch" == "arm64" ]]; then
        # Apple Silicon
        MACOSX_DEPLOYMENT_TARGET=11.0 pip install --no-build-isolation torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
    else
        # Intel Mac
        MACOSX_DEPLOYMENT_TARGET=10.15 pip install --no-build-isolation torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
    fi
else
    # Linux/Windows installation
    echo "🐧 Detected Linux/Windows - using standard installation"
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
fi

# Install PyTorch Geometric
echo "📐 Installing PyTorch Geometric..."
pip install torch-geometric==2.6.1

# Install RL dependencies
echo "🤖 Installing reinforcement learning dependencies..."
pip install -e .[rl]

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install -e .[dev]

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "🎣 Setting up pre-commit hooks..."
    pre-commit install
else
    echo "ℹ️  pre-commit not found, skipping hooks setup"
fi

# Verify installation
echo "🔍 Verifying installation..."
python -c "
try:
    import torch
    import torch_geometric
    import networkx
    import numpy
    import pandas
    import stable_baselines3
    import gymnasium
    print('✅ Core packages installed successfully!')
    print(f'   PyTorch: {torch.__version__}')
    print(f'   PyTorch Geometric: {torch_geometric.__version__}')
    print(f'   NetworkX: {networkx.__version__}')
    print(f'   Stable Baselines3: {stable_baselines3.__version__}')
    print(f'   Gymnasium: {gymnasium.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test that fusion module can be imported
echo "🧪 Testing FUSION module import..."
python -c "
try:
    import fusion
    from fusion.analysis import NetworkAnalyzer
    print('✅ FUSION modules imported successfully!')
except ImportError as e:
    print(f'❌ FUSION import error: {e}')
    exit(1)
"

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Run tests:           pytest -v"
echo "2. Check code quality:  ruff check fusion/"
echo "3. Format code:         ruff format fusion/"
echo "4. Type checking:       mypy fusion/"
echo "5. Quick start guide:   cat DEVELOPMENT_QUICKSTART.md"
echo ""
echo "Available commands:"
echo "- fusion-sim    (run simulation)"
echo "- fusion-train  (train models)"
echo "- fusion-gui    (launch GUI)"
echo ""
echo "Happy coding with FUSION! 🚀"
