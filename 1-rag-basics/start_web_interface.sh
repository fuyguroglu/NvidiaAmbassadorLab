#!/bin/bash

# Launcher script for the RAG web interface
# Works with conda OR Python venv - automatically detects what's available

echo "========================================="
echo "  RAG System - Web Interface Launcher"
echo "========================================="
echo ""

# Detect and activate environment
if command -v conda &> /dev/null && conda env list | grep -q "^nvidia_rag "; then
    # Conda environment exists
    echo "Activating conda environment 'nvidia_rag'..."
    eval "$(conda shell.bash hook)"
    conda activate nvidia_rag

    if [ $? -ne 0 ]; then
        echo "❌ Failed to activate conda environment 'nvidia_rag'"
        echo "Please run setup first: ./setup.sh"
        exit 1
    fi
elif [ -d ".venv" ]; then
    # Python venv exists
    echo "Activating Python virtual environment..."
    source .venv/bin/activate

    if [ $? -ne 0 ]; then
        echo "❌ Failed to activate virtual environment"
        echo "Please run setup first: ./setup.sh"
        exit 1
    fi
else
    echo "❌ No environment found!"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup.sh"
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
        echo "   Running in CPU mode"
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
