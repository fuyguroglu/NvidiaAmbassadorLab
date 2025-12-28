#!/bin/bash

# Smart Setup Script for RAG System (Linux/WSL/macOS)
# Works with conda OR Python venv - automatically detects what's available

set -e  # Exit on error

echo "=========================================="
echo "  RAG System - Smart Setup Script"
echo "  NVIDIA Ambassador Lab"
echo "=========================================="
echo ""

# Detect available Python environment tools
USE_CONDA=false
if command -v conda &> /dev/null; then
    USE_CONDA=true
    echo "✅ Conda found: $(conda --version)"
    ENV_NAME="nvidia_rag"
    ACTIVATE_CMD="conda activate nvidia_rag"
else
    echo "ℹ️  Conda not found - will use Python venv instead"

    # Check if python3 is available
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 not found!"
        echo ""
        echo "Please install Python 3.8 or higher first:"
        echo ""
        echo "  Ubuntu/Debian:"
        echo "    sudo apt update && sudo apt install python3 python3-pip python3-venv"
        echo ""
        echo "  Fedora/RHEL:"
        echo "    sudo dnf install python3 python3-pip"
        echo ""
        echo "  macOS:"
        echo "    brew install python3"
        echo "    (or download from https://www.python.org/downloads/)"
        echo ""
        exit 1
    fi

    echo "✅ Python found: $(python3 --version)"
    ENV_NAME=".venv"
    ACTIVATE_CMD="source .venv/bin/activate"
fi
echo ""

# Setup environment based on available tool
if [ "$USE_CONDA" = true ]; then
    # CONDA SETUP
    echo "🐍 Using Conda for environment management"
    echo ""

    # Accept conda TOS if needed (silently)
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

    # Check if environment already exists
    if conda env list | grep -q "^nvidia_rag "; then
        echo "⚠️  Environment 'nvidia_rag' already exists"
        echo ""
        read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n nvidia_rag -y
            echo "✅ Removed"
            echo ""
            # Create new environment
            echo "📦 Creating conda environment 'nvidia_rag' with Python 3.11..."
            conda create -n nvidia_rag python=3.11 -y
            echo "✅ Environment created"
            echo ""
        else
            echo "Keeping existing environment."
            echo ""
        fi
    else
        # Create conda environment
        echo "📦 Creating conda environment 'nvidia_rag' with Python 3.11..."
        conda create -n nvidia_rag python=3.11 -y
        echo "✅ Environment created"
        echo ""
    fi

    # Activate environment
    echo "🔧 Activating environment..."
    eval "$(conda shell.bash hook)"
    conda activate nvidia_rag
    echo "✅ Environment activated"
    echo ""

else
    # VENV SETUP
    echo "🐍 Using Python venv for environment management"
    echo ""

    # Check if venv already exists
    if [ -d ".venv" ]; then
        echo "⚠️  Virtual environment '.venv' already exists"
        echo ""
        read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            rm -rf .venv
            echo "✅ Removed"
            echo ""
            # Create new environment
            echo "📦 Creating Python virtual environment..."
            python3 -m venv .venv
            echo "✅ Environment created"
            echo ""
        else
            echo "Keeping existing environment."
            echo ""
        fi
    else
        # Create venv
        echo "📦 Creating Python virtual environment..."
        python3 -m venv .venv
        echo "✅ Environment created"
        echo ""
    fi

    # Activate environment
    echo "🔧 Activating environment..."
    source .venv/bin/activate
    echo "✅ Environment activated"
    echo ""

    # Upgrade pip
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip
    echo ""
fi

# Run GPU detection
echo "🔍 Detecting GPU capabilities..."
echo ""

# Check for GPU silently
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_DETECTED=true
    echo "✅ NVIDIA GPU detected!"
else
    GPU_DETECTED=false
    echo "💻 No GPU detected (CPU-only mode)"
fi
echo ""

# Determine recommendation and install command
if [ "$GPU_DETECTED" = true ]; then
    RECOMMENDED="GPU version with CUDA support"
    INSTALL_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt"
else
    RECOMMENDED="CPU-only version (smaller, no CUDA)"
    INSTALL_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt"
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
    eval "$INSTALL_CMD"
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
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            pip install -r requirements.txt
            INSTALL_SUCCESS=$?
            ;;
        2)
            echo "💻 Installing CPU-only version..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            pip install -r requirements.txt
            INSTALL_SUCCESS=$?
            ;;
        3)
            echo "⏭️  Skipping installation"
            echo ""
            echo "To install manually:"
            echo "  $ACTIVATE_CMD"
            echo "  pip install -r requirements.txt"
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
    echo "  $ACTIVATE_CMD"
    echo "  pip install -r requirements.txt"
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
echo "   $ACTIVATE_CMD"
echo "   ./start_web_interface.sh"
echo ""
echo "Or test with:"
echo "   python test_system.py"
echo ""
echo "Happy learning! 🚀"
echo ""
