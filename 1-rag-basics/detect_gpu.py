#!/usr/bin/env python3
"""
GPU Detection Script
Detects NVIDIA GPU availability and recommends appropriate PyTorch installation
"""

import subprocess
import sys
import platform


def check_nvidia_smi():
    """Check if nvidia-smi command is available and returns GPU info"""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0, result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""


def check_cuda_available():
    """Check if CUDA is available through PyTorch (if already installed)"""
    try:
        import torch
        return torch.cuda.is_available(), torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:
        return None, None  # PyTorch not installed yet


def get_system_info():
    """Get basic system information"""
    return {
        'os': platform.system(),
        'os_version': platform.version(),
        'architecture': platform.machine(),
        'python_version': platform.python_version()
    }


def print_banner():
    """Print a nice banner"""
    print("=" * 60)
    print("  GPU Detection & Installation Recommendation Tool")
    print("  NVIDIA Ambassador Lab - RAG System Setup")
    print("=" * 60)
    print()


def main():
    print_banner()

    # Get system info
    sys_info = get_system_info()
    print(f"🖥️  System Information:")
    print(f"   OS: {sys_info['os']} ({sys_info['architecture']})")
    print(f"   Python: {sys_info['python_version']}")
    print()

    # Check for NVIDIA GPU
    print("🔍 Checking for NVIDIA GPU...")
    has_nvidia_smi, nvidia_output = check_nvidia_smi()

    if has_nvidia_smi:
        print("✅ NVIDIA GPU detected!")
        print()
        # Extract GPU name from nvidia-smi output
        for line in nvidia_output.split('\n'):
            if 'NVIDIA' in line or 'GeForce' in line or 'Tesla' in line or 'Quadro' in line:
                print(f"   {line.strip()}")
        print()
    else:
        print("❌ No NVIDIA GPU detected")
        print()

    # Check if PyTorch is already installed
    cuda_available, cuda_version = check_cuda_available()

    if cuda_available is not None:
        print("📦 PyTorch Status:")
        if cuda_available:
            print(f"   ✅ PyTorch installed with CUDA {cuda_version}")
        else:
            print("   ⚠️  PyTorch installed (CPU-only version)")
        print()

    # Provide recommendations
    print("=" * 60)
    print("📋 RECOMMENDATION")
    print("=" * 60)
    print()

    if has_nvidia_smi:
        print("✨ GPU DETECTED - Recommended Setup:")
        print()
        print("   Install PyTorch with CUDA support for better performance:")
        print()
        if sys_info['os'] == 'Windows':
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        else:
            print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("   Then install other requirements:")
        print("   pip install -r requirements-base.txt")
        print()
        print("   💡 Benefits:")
        print("      - 5-10x faster model inference")
        print("      - Can run larger models (Phi-2, Phi-3)")
        print("      - Better for experimentation")
        print()
    else:
        print("💻 NO GPU DETECTED - Recommended Setup:")
        print()
        print("   Install CPU-only version (smaller download, no CUDA):")
        print()
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print()
        print("   Then install other requirements:")
        print("   pip install -r requirements-base.txt")
        print()
        print("   💡 What you can still do:")
        print("      - Use retrieval-only mode (works great!)")
        print("      - Use TinyLlama (slower but works)")
        print("      - Use cloud API backends (Groq, OpenAI)")
        print("      - Learn RAG concepts perfectly fine")
        print()
        print("   ⚠️  Note: Local LLM inference will be slower on CPU")
        print()

    print("=" * 60)
    print()

    # Return status code for scripting
    return 0 if has_nvidia_smi else 1


if __name__ == "__main__":
    sys.exit(main())
