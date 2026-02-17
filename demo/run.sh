#!/usr/bin/env bash
# =============================================================================
# Aurora Echo Demo — Start the server
# =============================================================================
# Usage:
#   cd demo
#   ./run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/venv"

# Activate venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Verify API key
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "sk-or-v1-paste-your-key-here" ]; then
    echo "ERROR: OPENAI_API_KEY not set in $PROJECT_DIR/.env"
    echo "Get an OpenRouter key at https://openrouter.ai/keys"
    exit 1
fi

echo "Starting Aurora Echo..."
echo "  LLM Provider: ${LLM_PROVIDER_ORDER:-openai}"
echo "  LLM Model:    ${OPENAI_MODEL_ID:-gpt-4o-mini}"
echo "  LLM Base URL: ${OPENAI_BASE_URL:-https://api.openai.com}"
echo "  Whisper:      ${WHISPER_MODEL_SIZE:-medium}"
echo ""
echo "  Open http://localhost:8000 in your browser"
echo ""

cd "$PROJECT_DIR"
exec uvicorn app:app --host 0.0.0.0 --port 8000
