#!/bin/bash

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR="$PROJECT_DIR/.venv"

echo "=========================================="
echo "  Starting Ollama API Proxy"
echo "=========================================="
echo "Project: $PROJECT_DIR"
echo ""

if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "[2/3] Installing/Updating dependencies..."
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo "[3/3] Starting service..."
echo "Service will run on: http://localhost:11433"
echo "Press CTRL+C to stop"
echo ""

"$VENV_DIR/bin/python" "$PROJECT_DIR/main.py"