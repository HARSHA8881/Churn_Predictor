#!/bin/bash
# Run the Churn Prediction app using the local virtual environment.
# Usage: bash run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_STREAMLIT="$SCRIPT_DIR/venv/bin/streamlit"

if [ ! -f "$VENV_STREAMLIT" ]; then
    echo "❌ venv not found. Please run:"
    echo "   python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "🚀 Starting Customer Churn Predictor..."
"$VENV_STREAMLIT" run "$SCRIPT_DIR/app.py"
