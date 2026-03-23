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

# ── Install dependencies ────────────────────────────────────────────────────
echo "Installing dependencies ..."

pip install \
    torch \
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
echo "  source $VENV_DIR/Scripts/activate   # Windows (Git Bash)"
echo ""
echo "Quick start:"
echo "  python scripts/run_experiments.py --sanity-check"
