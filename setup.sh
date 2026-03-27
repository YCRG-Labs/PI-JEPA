#!/usr/bin/env bash
# setup.sh — Environment setup for PI-JEPA
# Creates a virtual environment and installs all dependencies.

set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"

echo "=== PI-JEPA Environment Setup ==="

# ── Create virtual environment ──────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# ── Activate ────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
else
    echo "Error: Could not find activation script in $VENV_DIR"
    exit 1
fi

echo "Activated virtual environment"

# ── Upgrade pip ─────────────────────────────────────────────────────────────
pip install --upgrade pip

# ── Install PyTorch (with CUDA if available) ────────────────────────────────
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision
fi

# ── Install dependencies ────────────────────────────────────────────────────
echo "Installing dependencies..."

pip install \
    numpy \
    scipy \
    pyyaml \
    h5py \
    pandas \
    matplotlib \
    neuraloperator \
    pytest \
    hypothesis

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate the environment with:"
echo "  source $VENV_DIR/bin/activate      # Linux / macOS"
echo "  source $VENV_DIR/Scripts/activate  # Windows (Git Bash)"
echo ""
echo "Run the full analysis pipeline:"
echo "  python scripts/run_self_supervised_pipeline.py --config configs/darcy.yaml --output outputs"
echo ""
echo "This will:"
echo "  1. Self-supervised pretraining (500 epochs)"
echo "  2. Data efficiency evaluation vs baselines (FNO, GeoFNO, DeepONet)"
echo ""
echo "Results saved to: outputs/data_efficiency/benchmark_comparison.json"
