#!/bin/bash
# Cross-platform installation script for macOS and Linux

# Ensure pip is up to date
pip install --upgrade pip

# Step 1: Install PyTorch first
pip install torch==2.2.2

# Step 2: Install PyG packages with platform-specific logic
if [[ "$(uname)" == "Darwin" ]]; then
  # macOS: Compile from source with specific flags
  echo "Detected macOS. Compiling PyG packages from source..."
  MACOSX_DEPLOYMENT_TARGET=10.15 pip install --no-build-isolation torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
else
  # Linux: Install from pre-built wheels
  echo "Detected Linux. Installing PyG packages..."
  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
fi

# Step 3: Install all other requirements
echo "Installing remaining packages..."
pip install -r requirements.txt
