#!/bin/bash
# setup.sh — One-click environment setup for the TinyML workshop.
#
# Usage:
#   bash setup.sh
#
# After completion, activate the environment and run the scripts:
#   source venv/bin/activate
#   python train.py

set -e

VENV_DIR="venv"
PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

echo "TinyML Workshop — Environment Setup"
echo "------------------------------------"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt "$PYTHON_MIN_MAJOR" ] || \
   { [ "$PYTHON_MAJOR" -eq "$PYTHON_MIN_MAJOR" ] && [ "$PYTHON_MINOR" -lt "$PYTHON_MIN_MINOR" ]; }; then
    echo "ERROR: Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+ is required."
    echo "       You are running Python ${PYTHON_VERSION}."
    echo "       Download Python from https://www.python.org/downloads/"
    exit 1
fi
echo "✓ Python ${PYTHON_VERSION}"

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "✓ Virtual environment '${VENV_DIR}' already exists, skipping creation."
else
    echo "Creating virtual environment '${VENV_DIR}'..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created."
fi

# Activate the virtual environment
source "${VENV_DIR}/bin/activate"
echo "✓ Virtual environment activated."

# Upgrade pip via python -m pip to avoid permission issues
echo "Upgrading pip..."
python -m pip install --quiet --upgrade pip
echo "✓ pip upgraded."

# Install PyTorch from the official index (CPU build, latest compatible version)
echo "Installing PyTorch and torchvision (CPU)..."
pip install --quiet torch torchvision --index-url "$PYTORCH_INDEX"
echo "✓ PyTorch installed."

# Install remaining dependencies
echo "Installing remaining dependencies..."
pip install --quiet -r requirements.txt
echo "✓ All dependencies installed."

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "Activate the environment with:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "Then run the workshop scripts in order:"
echo "    python train.py"
echo "    python ptq.py"
echo "    python qat.py"
echo "    python pruning.py"
echo "    python export_onnx.py"
echo "    python inference.py"
echo ""
