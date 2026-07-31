#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/Testing"
TARGET_SCRIPT="${1:-companion_target_locker.py}"

cd "$PROJECT_DIR"

echo "[1/4] Checking Python3..."
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Python is not installed. Please install Python 3 first." >&2
    exit 1
fi

echo "[2/4] Creating virtual environment if needed..."
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    rm -rf venv
    "$PYTHON_BIN" -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "[3/4] Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python numpy

echo "[4/4] Running $TARGET_SCRIPT ..."
python "$TARGET_SCRIPT"
