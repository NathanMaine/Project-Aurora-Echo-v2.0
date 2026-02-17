#!/usr/bin/env bash
# =============================================================================
# Aurora Echo Demo — One-shot setup for P50 / OpenRouter
# =============================================================================
# Usage:
#   cd demo
#   chmod +x setup.sh
#   ./setup.sh
#
# After setup, start the server with:
#   ./run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/venv"

echo "========================================="
echo " Aurora Echo Demo Setup"
echo "========================================="
echo ""

# 1. Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[OK] Python $PYTHON_VERSION found"

# 2. Check NVIDIA GPU (optional)
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "[OK] GPU detected: $GPU_NAME ($GPU_MEM)"
else
    echo "[!!] No NVIDIA GPU detected — ASR will run on CPU (slower but works)"
fi

# 3. Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "[OK] Virtual environment activated"

# 4. Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q
echo "[OK] Dependencies installed"

# 5. Set up .env
if [ ! -f "$PROJECT_DIR/.env" ]; then
    if [ -f "$SCRIPT_DIR/.env.example" ]; then
        cp "$SCRIPT_DIR/.env.example" "$PROJECT_DIR/.env"
        echo ""
        echo "[!!] Created .env from demo template"
        echo "     IMPORTANT: Edit $PROJECT_DIR/.env and set your OPENAI_API_KEY"
        echo "     (OpenRouter key works — get one at https://openrouter.ai/keys)"
    fi
else
    echo "[OK] .env already exists"
fi

# 6. Check if API key is set
if grep -q "paste-your-key-here" "$PROJECT_DIR/.env" 2>/dev/null; then
    echo ""
    echo "========================================="
    echo " SETUP COMPLETE — but you need an API key"
    echo "========================================="
    echo ""
    echo " 1. Get an OpenRouter API key: https://openrouter.ai/keys"
    echo " 2. Edit $PROJECT_DIR/.env"
    echo " 3. Replace 'sk-or-v1-paste-your-key-here' with your actual key"
    echo " 4. Run: cd demo && ./run.sh"
    echo ""
else
    echo ""
    echo "========================================="
    echo " SETUP COMPLETE"
    echo "========================================="
    echo ""
    echo " Start the server:  cd demo && ./run.sh"
    echo " Then open:         http://localhost:8000"
    echo ""
fi
