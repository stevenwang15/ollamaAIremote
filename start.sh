#!/bin/bash
# Create the venv if needed, install dependencies, run the proxy.
set -e

PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
VENV_DIR="$PROJECT_DIR/.venv"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found — install Python 3.9+ first." >&2
    exit 1
fi

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
if ! "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"; then
    # Offline restarts should still work with the existing venv.
    echo "warning: dependency install failed; running with the installed ones" >&2
fi

# Mirror the proxy's LISTEN_PORT resolution (shell env, then .env, then
# default) so the banner shows the port that will actually be bound.
PORT=${LISTEN_PORT:-}
if [ -z "$PORT" ] && [ -f "$PROJECT_DIR/.env" ]; then
    PORT=$(sed -n 's/^LISTEN_PORT *= *//p' "$PROJECT_DIR/.env" | tail -n 1 | tr -d ' "')
fi
PORT=${PORT:-11433}

echo "[3/3] Starting service..."
echo "Service will run on: http://localhost:$PORT"
echo "Press CTRL+C to stop"
echo ""

"$VENV_DIR/bin/python" "$PROJECT_DIR/main.py"
