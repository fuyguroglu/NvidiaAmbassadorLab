#!/bin/bash

# Smart Setup Script for RAG System (Linux/WSL)
# Automatically detects GPU and installs appropriate dependencies

set -e  # Exit on error

echo "=========================================="
echo "  RAG System - Smart Setup Script"
echo "  NVIDIA Ambassador Lab"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found!"
    echo ""
    echo "Please install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Accept conda TOS if needed (silently)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Check if environment already exists
if conda env list | grep -q "nvidia_rag"; then
    echo "⚠️  Environment 'nvidia_rag' already exists"
    echo ""
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n nvidia_rag -y
        echo "✅ Removed"
        echo ""
    else
        echo "Keeping existing environment. Activating..."
        eval "$(conda shell.bash hook)"
        conda activate nvidia_rag
        echo "✅ Activated nvidia_rag environment"
        echo ""
        # Continue to installation check instead of exiting
    fi
else
    # Create conda environment
    echo "📦 Creating conda environment 'nvidia_rag' with Python 3.11..."
    conda create -n nvidia_rag python=3.11 -y

    echo "✅ Environment created"
    echo ""

    # Activate environment
    echo "🔧 Activating environment..."
    eval "$(conda shell.bash hook)"
    conda activate nvidia_rag

    echo "✅ Environment activated"
    echo ""
fi

# Run GPU detection and store result
echo "🔍 Detecting GPU capabilities..."
echo ""

# Check for GPU silently
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_DETECTED=0
    echo "✅ NVIDIA GPU detected!"
else
    GPU_DETECTED=1
    echo "💻 No GPU detected (CPU-only mode)"
fi
echo ""

# Determine recommendation
if [ $GPU_DETECTED -eq 0 ]; then
    RECOMMENDED="GPU version with CUDA support"
    RECOMMENDED_FILE="requirements-gpu.txt"
else
    RECOMMENDED="CPU-only version (smaller, no CUDA)"
    RECOMMENDED_FILE="requirements-cpu.txt"
fi

# Ask user for confirmation
echo ""
echo "=========================================="
echo "💡 Recommendation: Install $RECOMMENDED"
echo "=========================================="
echo ""
read -p "Follow this recommendation? (Y/n): " -n 1 -r
echo ""

# Default to Yes if user just presses enter
if [[ -z $REPLY ]] || [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔄 Installing recommended version..."
    pip install -r "$RECOMMENDED_FILE"
    INSTALL_SUCCESS=$?
else
    echo ""
    echo "=========================================="
    echo "Installation Options:"
    echo "=========================================="
    echo ""
    echo "1. Force GPU version (with CUDA)"
    echo "2. Force CPU version (no CUDA)"
    echo "3. Skip installation (manual setup)"
    echo ""
    read -p "Choose option (1-3): " -n 1 -r
    echo ""
    echo ""

    case $REPLY in
        1)
            echo "🎮 Installing GPU version with CUDA support..."
            pip install -r requirements-gpu.txt
            INSTALL_SUCCESS=$?
            ;;
        2)
            echo "💻 Installing CPU-only version..."
            pip install -r requirements-cpu.txt
            INSTALL_SUCCESS=$?
            ;;
        3)
            echo "⏭️  Skipping installation"
            echo ""
            echo "To install manually:"
            echo "  conda activate nvidia_rag"
            echo "  pip install -r requirements-cpu.txt  # or requirements-gpu.txt"
            exit 0
            ;;
        *)
            echo "❌ Invalid option"
            exit 1
            ;;
    esac
fi

# Check if installation was successful
if [ $INSTALL_SUCCESS -ne 0 ]; then
    echo ""
    echo "❌ Installation failed!"
    echo ""
    echo "Please check the error messages above and try again."
    echo "You can also install manually:"
    echo "  conda activate nvidia_rag"
    echo "  pip install -r $RECOMMENDED_FILE"
    exit 1
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Add documents to the 'data/' folder"
echo "2. Run the system:"
echo "   conda activate nvidia_rag"
echo "   ./start_web_interface.sh"
echo ""
echo "Or test with:"
echo "   python test_system.py"
echo ""
echo "Happy learning! 🚀"
echo ""
