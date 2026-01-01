#!/bin/bash

# Flask Web Interface Launcher
# ============================
# Simple, reliable startup script for the Flask RAG interface

echo ""
echo "======================================================================"
echo "🌐 FLASK WEB INTERFACE LAUNCHER"
echo "======================================================================"
echo ""

# Detect which environment manager is being used
if command -v conda &> /dev/null; then
    # Check if nvidia_rag environment exists
    if conda env list | grep -q "nvidia_rag"; then
        echo "📦 Activating conda environment: nvidia_rag"
        eval "$(conda shell.bash hook)"
        conda activate nvidia_rag
    else
        echo "⚠️  Conda environment 'nvidia_rag' not found."
        echo "   Please run ./setup.sh first"
        exit 1
    fi
elif [ -d ".venv" ]; then
    echo "📦 Activating Python virtual environment: .venv"
    source .venv/bin/activate
else
    echo "⚠️  No conda environment or .venv found."
    echo "   Please run ./setup.sh first"
    exit 1
fi

echo ""
echo "======================================================================"
echo "🔍 Checking GPU availability..."
echo "======================================================================"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "✅ GPU detected and available"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    else
        echo "💻 Running in CPU mode"
    fi
else
    echo "💻 Running in CPU mode"
fi

echo ""
echo "======================================================================"
echo "🚀 Starting Flask Web Interface"
echo "======================================================================"
echo ""
echo "Flask is starting... This may take a moment."
echo ""
echo "📍 Once started, access the interface at:"
echo "   • http://localhost:7860"
echo "   • http://127.0.0.1:7860"
echo ""
echo "⚠️  IMPORTANT: Use 'localhost' or '127.0.0.1', NOT '0.0.0.0'"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================================================"
echo ""

# Run the Flask app
python app_flask.py
