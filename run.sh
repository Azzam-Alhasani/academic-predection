#!/bin/bash
# run.sh — Easy launcher for Academic Performance Monitoring & Prediction System
# Usage: bash run.sh  (or: chmod +x run.sh && ./run.sh)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Academic Performance Monitoring & Prediction System"
echo " Starting up..."
echo "============================================================"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.10+."
    exit 1
fi

# Create venv if missing
if [ ! -f "venv/bin/activate" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created."
fi

source venv/bin/activate

echo "[INFO] Installing dependencies..."
pip install -r requirements.txt --quiet
echo "[OK] Dependencies ready."

# Generate sample data if missing
if [ ! -f "app/data/sample_students.csv" ]; then
    echo "[INFO] Generating sample student data..."
    python app/data/generate_sample_data.py
    echo "[OK] Sample data generated."
fi

echo ""
echo "============================================================"
echo " Server starting at http://127.0.0.1:5000"
echo " Press Ctrl+C to stop"
echo "============================================================"
echo ""

python run.py
