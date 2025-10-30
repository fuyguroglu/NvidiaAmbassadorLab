#!/bin/bash

# Simple launcher script for the RAG web interface
# Works on Linux, macOS, and WSL

echo "========================================="
echo "  RAG System - Web Interface Launcher"
echo "========================================="
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nvidia_rag

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'nvidia_rag'"
    echo "Please make sure the environment is created:"
    echo "  conda create -n nvidia_rag python=3.11 -y"
    exit 1
fi

echo "✅ Environment activated!"
echo ""

# Check GPU status
echo "🔍 Checking system capabilities..."
if command -v nvidia-smi &> /dev/null; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "🎮 GPU detected and PyTorch CUDA enabled!"
    else
        echo "⚠️  GPU detected but PyTorch CUDA not available"
        echo "   Consider reinstalling with: pip install -r requirements-gpu.txt"
    fi
else
    echo "💻 Running in CPU mode"
fi
echo ""

# Run the web interface
echo "Starting web interface..."
echo ""
echo "📍 Access the interface at:"
echo "   • http://localhost:7860"
echo "   • http://127.0.0.1:7860"
echo ""
echo "⚠️  Don't use http://0.0.0.0:7860 - use localhost instead!"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="
echo ""

python app_simple.py
