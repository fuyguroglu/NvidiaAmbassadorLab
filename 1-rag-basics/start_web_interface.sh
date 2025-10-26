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

# Run the web interface
echo "Starting web interface..."
echo "The interface will open in your browser at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="
echo ""

python app_simple.py
