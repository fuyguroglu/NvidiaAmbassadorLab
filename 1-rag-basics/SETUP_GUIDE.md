# RAG Basics - Student Setup Guide

Welcome! This guide will help you set up and run your first RAG (Retrieval-Augmented Generation) system on **Windows** or **Linux**, regardless of your computer's specifications.

## What You'll Learn

- How to set up a Python environment for AI projects
- How RAG (Retrieval-Augmented Generation) works
- How to work with different model backends based on your hardware

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation by Operating System](#installation-by-operating-system)
3. [Quick Start](#quick-start)
4. [Choosing the Right Backend](#choosing-the-right-backend)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python**: Version 3.9 or higher (we recommend 3.11)
- **Conda** or **Miniconda**: For managing Python environments
- **RAM**: At least 4GB (more is better!)
- **Disk Space**:
  - **CPU-only**: ~3-5GB (smaller PyTorch, no CUDA)
  - **GPU version**: ~6-10GB (includes CUDA support)
- **Internet**: For downloading models and packages
- **GPU** (Optional): NVIDIA GPU for faster performance (not required!)

---

## 🚀 Quick Start (Recommended)

**We now have an automatic setup script that detects your GPU and installs the right version!**

### Windows Quick Setup
```cmd
REM Download/clone the project, then:
cd path\to\NvidiaAmbassadorLab\1-rag-basics
setup.bat
```

### Linux/WSL Quick Setup
```bash
# Download/clone the project, then:
cd path/to/NvidiaAmbassadorLab/1-rag-basics
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. ✅ Detect if you have an NVIDIA GPU
2. ✅ Create the conda environment
3. ✅ Install the appropriate PyTorch version (CPU or GPU)
4. ✅ Install all other dependencies
5. ✅ Guide you through the setup

**That's it!** Skip to [Quick Start](#quick-start-after-installation) section.

---

## Manual Installation (Advanced)

If you prefer manual control or the automatic script doesn't work:

## Installation by Operating System

### 🪟 Windows Installation

#### Step 1: Install Miniconda (if not already installed)

1. **Download Miniconda**:
   - Go to: https://docs.conda.io/en/latest/miniconda.html
   - Download the Windows installer (Miniconda3 Windows 64-bit)
   - **Recommended**: `Miniconda3-latest-Windows-x86_64.exe`

2. **Run the Installer**:
   - Double-click the downloaded `.exe` file
   - Click "Next" through the setup
   - **Important**: Check "Add Miniconda3 to my PATH environment variable" (makes life easier!)
   - Complete the installation

3. **Verify Installation**:
   - Open **Command Prompt** (search "cmd" in Start menu)
   - Type: `conda --version`
   - You should see something like `conda 23.x.x`

#### Step 2: Download the Project

**Option A: Using Git** (if you have Git installed)
```cmd
git clone <repository-url>
cd NvidiaAmbassadorLab\1-rag-basics
```

**Option B: Download ZIP** (easier for beginners)
1. Download the project ZIP file
2. Right-click → "Extract All"
3. Open **Command Prompt** and navigate:
```cmd
cd C:\Users\YourName\Downloads\NvidiaAmbassadorLab\1-rag-basics
```

**Windows Tip**: You can also hold Shift and right-click in the folder, then select "Open PowerShell window here" or "Open Command Prompt here"

#### Step 3: Create Conda Environment

Open **Command Prompt** or **PowerShell** in the project folder:

```cmd
REM Create environment
conda create -n nvidia_rag python=3.11 -y

REM Activate environment
conda activate nvidia_rag
```

**Windows Note**: If you see "CommandNotFoundError", you may need to use:
```cmd
conda init cmd.exe
REM Then close and reopen Command Prompt
```

#### Step 4: Check GPU and Install Dependencies

```cmd
REM Make sure you're in the project folder and environment is activated

REM First, check if you have a GPU
python detect_gpu.py

REM Then install based on your hardware:

REM Option A: Auto-detect (Recommended)
REM If you have NVIDIA GPU, this installs GPU version, otherwise CPU version
nvidia-smi
if %errorlevel% equ 0 (
    pip install -r requirements-gpu.txt
) else (
    pip install -r requirements-cpu.txt
)

REM Option B: Manual choice
REM For GPU (if you have NVIDIA GPU - ~2-3GB download):
pip install -r requirements-gpu.txt

REM For CPU only (no GPU - ~200MB download):
pip install -r requirements-cpu.txt
```

**Windows-Specific Notes**:
- **GPU version**: 10-20 minutes download (includes CUDA)
- **CPU version**: 5-10 minutes download (much smaller)
- Windows Defender might scan files - this is normal
- If you get SSL errors, try: `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements-cpu.txt`

#### Step 5: Add Your Documents

```cmd
REM Copy documents to data folder
copy "C:\path\to\your\document.pdf" data\

REM Or just drag-and-drop files into the data\ folder using File Explorer!
```

#### Windows-Specific Tips:

1. **Use PowerShell for better experience**:
   - PowerShell has better text rendering than Command Prompt
   - Right-click Start menu → "Windows PowerShell"

2. **File Paths**:
   - Windows uses backslashes: `data\document.pdf`
   - Python accepts forward slashes too: `data/document.pdf`

3. **Long Path Names**:
   - If you get "path too long" errors, move the project closer to C:\ drive
   - Example: `C:\Projects\RAG\` instead of `C:\Users\Username\Documents\School\Projects\...`

4. **Antivirus Software**:
   - Windows Defender might slow down pip installations
   - Consider adding Python folder to exclusions (optional)

5. **WSL (Advanced)**:
   - If you're familiar with Linux, consider using WSL2 (Windows Subsystem for Linux)
   - Provides a Linux environment on Windows
   - See WSL section below

---

### 🐧 Linux Installation

#### Step 1: Install Miniconda (if not already installed)

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make it executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Run installer
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts, say 'yes' to initialize conda
# Close and reopen terminal

# Verify
conda --version
```

**Alternative: Using Package Manager** (Ubuntu/Debian)
```bash
# Install from apt (easier but might be older version)
sudo apt update
sudo apt install -y conda

# Or use the official installer method above for latest version
```

#### Step 2: Download the Project

```bash
# Using git
git clone <repository-url>
cd NvidiaAmbassadorLab/1-rag-basics

# Or download and extract
wget <project-zip-url>
unzip <project-name>.zip
cd NvidiaAmbassadorLab/1-rag-basics
```

#### Step 3: Create Conda Environment

```bash
# Create environment
conda create -n nvidia_rag python=3.11 -y

# Activate environment
conda activate nvidia_rag
```

**Linux Tip**: Add to your `~/.bashrc` or `~/.zshrc` for auto-activation:
```bash
# Optional: Auto-activate when entering directory
echo "cd ~/path/to/NvidiaAmbassadorLab/1-rag-basics && conda activate nvidia_rag" >> ~/.bashrc
```

#### Step 4: Check GPU and Install Dependencies

```bash
# Make sure conda environment is activated

# First, check if you have a GPU
python detect_gpu.py

# Then install based on your hardware:

# Option A: Auto-detect (Recommended)
if command -v nvidia-smi &> /dev/null; then
    echo "Installing GPU version..."
    pip install -r requirements-gpu.txt
else
    echo "Installing CPU version..."
    pip install -r requirements-cpu.txt
fi

# Option B: Manual choice
# For GPU (if you have NVIDIA GPU - ~2-3GB download):
pip install -r requirements-gpu.txt

# For CPU only (no GPU - ~200MB download):
pip install -r requirements-cpu.txt
```

**Linux-Specific Notes**:
- **GPU version**: 10-20 minutes download (includes CUDA)
- **CPU version**: 5-10 minutes download (much smaller)
- If you get permission errors, DO NOT use `sudo pip`
- Make sure conda environment is activated
- Some packages might need system libraries:
```bash
# Ubuntu/Debian
sudo apt install -y build-essential python3-dev

# Fedora/RHEL
sudo dnf install gcc gcc-c++ python3-devel

# Arch
sudo pacman -S base-devel
```

#### Step 5: Add Your Documents

```bash
# Copy documents to data folder
cp ~/Documents/your-document.pdf data/

# Or use file manager - most Linux distros have drag-and-drop support
```

#### Linux-Specific Tips:

1. **Use Terminal Multiplexer**:
   ```bash
   # Install tmux for persistent sessions
   sudo apt install tmux  # Ubuntu/Debian

   # Use it
   tmux new -s rag
   # Now your session persists even if terminal closes
   ```

2. **Check GPU Support**:
   ```bash
   # Check if you have NVIDIA GPU
   nvidia-smi

   # Check CUDA availability in Python
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Permissions**:
   - Make sure you own the project directory
   - If needed: `sudo chown -R $USER:$USER ~/path/to/project`

4. **Performance Tip**:
   ```bash
   # Monitor system resources while running
   htop  # Install with: sudo apt install htop
   ```

---

### 🔧 WSL (Windows Subsystem for Linux)

If you're on Windows 10/11 and want a Linux experience:

#### Install WSL2:

```powershell
# In PowerShell (as Administrator)
wsl --install
# Restart computer

# After restart, set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu
wsl --install -d Ubuntu
```

#### Setup in WSL:

```bash
# Once in Ubuntu terminal
# Follow the Linux installation instructions above!

# Access Windows files
cd /mnt/c/Users/YourName/Documents/

# WSL Tips:
# - Files in Linux filesystem are MUCH faster
# - Keep project files in Linux home: ~/Projects/
# - Use VSCode with WSL extension for best experience
```

**WSL Benefits**:
- Native Linux tools and performance
- Better compatibility with many Python packages
- Access to both Windows and Linux filesystems

---

## 🎮 GPU vs CPU: What's the Difference?

### Do I Need a GPU?

**Short answer: No!** This project works perfectly fine without a GPU.

### What's the Difference?

| Feature | With GPU (CUDA) | Without GPU (CPU-only) |
|---------|----------------|------------------------|
| **Installation Size** | ~2-3GB (includes CUDA) | ~200MB (smaller) |
| **Setup Time** | 10-20 minutes | 5-10 minutes |
| **Local LLM Speed** | 5-10x faster | Slower but works |
| **Retrieval-Only Mode** | Same speed | Same speed ✅ |
| **Cloud API Models** | Same speed | Same speed ✅ |
| **Recommended Models** | TinyLlama, Phi-2, Phi-3 | TinyLlama, Cloud APIs |
| **Power Usage** | Higher | Lower |
| **Learning RAG Concepts** | Same | Same ✅ |

### Our Recommendation:

- **Have NVIDIA GPU?** → Use GPU version for better performance with local models
- **No GPU or laptop?** → Use CPU version + Cloud APIs (Groq free tier is great!)
- **Just learning?** → Either works perfectly! Retrieval-only mode is identical on both

### How We Help You Choose:

Our setup scripts automatically detect your GPU and recommend the right version:
```bash
# Automatic detection
./setup.sh        # Linux/WSL
setup.bat         # Windows

# Or check manually
python detect_gpu.py
```

---

## Quick Start (After Installation)

### Step 1: Verify Setup

**Windows (Command Prompt/PowerShell)**:
```cmd
conda activate nvidia_rag
python --version
python -c "import torch; print('PyTorch installed!')"
```

**Linux/WSL**:
```bash
conda activate nvidia_rag
python --version
python -c "import torch; print('PyTorch installed!')"
```

### Step 2: Check Available Backends

**All Platforms**:
```bash
python config.py
```

This shows all available model options and their requirements.

### Step 3: Run Your First Query!

**All Platforms**:
```bash
python rag_flexible.py
```

When prompted, press **Enter** to use `retrieval_only` mode (works everywhere!)

---

## Choosing the Right Backend

The system supports multiple backends. Choose based on your hardware:

### 🟢 Option 1: Retrieval-Only (RECOMMENDED FOR BEGINNERS)

**Best for**: ALL computers, learning how RAG retrieval works
**RAM needed**: < 1GB
**Setup**: None needed!

```python
backend = "retrieval_only"
```

**What it does**: Shows you the relevant document chunks retrieved for your question.

---

### 🟡 Option 2: TinyLlama (LIGHTWEIGHT LOCAL MODEL)

**Best for**: Laptops with 4GB+ RAM
**RAM needed**: ~3-4GB
**Setup**: None needed! Model downloads automatically (first run takes time)

```python
backend = "tinyllama"
```

**Download time**: ~5-10 minutes (2.2GB)
**Run time**: ~10-30 seconds per answer (CPU)

---

### 🟠 Option 3: Phi-2 or Phi-3 Mini (STANDARD LOCAL MODEL)

**Best for**: Computers with 6-8GB+ RAM
**RAM needed**: ~5-6GB
**Setup**: None needed! Model downloads automatically

```python
backend = "phi2"  # or "phi3_mini"
```

**Download time**: ~15-20 minutes (5GB)
**Run time**: ~20-60 seconds per answer (CPU)

---

### 🔵 Option 4: Cloud API Models (FASTEST & MOST RELIABLE)

**Best for**: Anyone with an API key, production use
**RAM needed**: < 1GB (runs in the cloud!)
**Setup**: Requires API key

#### Groq (RECOMMENDED - Free Tier Available!)

**Windows**:
```cmd
REM Set environment variable for current session
set GROQ_API_KEY=your-key-here

REM Or permanently (System Properties → Environment Variables)
setx GROQ_API_KEY "your-key-here"
```

**Linux/WSL**:
```bash
# Set for current session
export GROQ_API_KEY="your-key-here"

# Or add to ~/.bashrc for persistence
echo 'export GROQ_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

Then run:
```python
backend = "groq"
```

**Sign up**: https://console.groq.com (Free tier available!)

---

## Usage Examples

### Example 1: Basic Usage

```python
from rag_flexible import FlexibleRAG

# Initialize with your chosen backend
rag = FlexibleRAG(
    backend="retrieval_only",  # Change this based on your hardware!
    data_dir="./data",
    chunk_size=500,
    k_retrieve=3
)

# Setup (loads and indexes documents)
if rag.setup():
    # Ask questions!
    result = rag.query("What is this document about?")
    print(result)
```

### Example 2: Interactive Mode

```bash
# Run the demo script
python rag_flexible.py

# Follow the prompts to select your backend
```

---

## Troubleshooting

### Windows-Specific Issues

#### Issue: "conda is not recognized"
**Solution**:
```cmd
REM Option 1: Use Anaconda Prompt (search in Start menu)

REM Option 2: Add to PATH manually
REM 1. Search "Environment Variables" in Start menu
REM 2. Edit System Environment Variables
REM 3. Add: C:\Users\YourName\miniconda3\Scripts
```

#### Issue: "Permission Denied" when installing
**Solution**:
```cmd
REM Run Command Prompt as Administrator
REM Right-click → "Run as Administrator"
```

#### Issue: Installation very slow
**Solution**:
- Windows Defender is scanning files
- Add Python/conda directory to exclusions (optional)
- Or just wait - it's one-time setup!

#### Issue: Long path names error
**Solution**:
```cmd
REM Enable long paths in Windows 10/11
REM Run in PowerShell as Administrator:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

---

### Linux-Specific Issues

#### Issue: "Permission denied" errors
**Solution**:
```bash
# Make sure you own the directory
sudo chown -R $USER:$USER ~/path/to/project

# Don't use sudo with pip!
```

#### Issue: Missing system libraries
**Solution**:
```bash
# Ubuntu/Debian
sudo apt install -y build-essential python3-dev libssl-dev

# Fedora
sudo dnf install gcc gcc-c++ python3-devel openssl-devel
```

#### Issue: GPU not detected
**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# If not installed, install NVIDIA drivers
# Ubuntu:
sudo ubuntu-drivers autoinstall

# Then install CUDA toolkit if needed
```

---

### General Issues (All Platforms)

#### Issue: "Out of Memory" Error
**Solution**: Try a smaller model or retrieval-only mode:
```python
backend = "retrieval_only"  # Works on any computer!
```

#### Issue: Model download is slow
**Solution**:
- This is normal! Models are 1-5GB
- Downloads only happen once
- Be patient, go grab coffee ☕

#### Issue: "No documents found"
**Solution**:
```bash
# Check data folder
ls data/        # Linux
dir data\       # Windows

# Make sure you have .pdf or .txt files there
```

#### Issue: API key not found
**Solution**:
```bash
# Windows (check)
echo %GROQ_API_KEY%

# Linux (check)
echo $GROQ_API_KEY

# If empty, set it (see backend setup above)
```

---

## Platform-Specific Performance Tips

### Windows:
- Close unnecessary programs to free RAM
- Use PowerShell instead of CMD for better output
- Consider WSL2 for better performance
- Disable antivirus scanning for conda/Python folders (optional)

### Linux:
- Use `htop` to monitor resource usage
- Close browser/heavy apps when running models
- Consider using a lighter desktop environment
- Check if you have GPU with `nvidia-smi`

---

## Next Steps

1. ✅ **Verify installation**: Run `python config.py`
2. ✅ **Try retrieval-only mode**: Understand how RAG works
3. ✅ **Add your documents**: Put PDFs in `data/` folder
4. ✅ **Experiment**: Try different backends
5. ✅ **Learn**: Read `rag_flexible.py` to see how it works

---

## Getting Help

### If you're stuck:

1. **Check error message carefully** - often tells you exactly what's wrong
2. **Google the error** - Someone else probably had the same issue
3. **Check this guide** - troubleshooting section covers common issues
4. **Ask classmates** - They might have solved it already
5. **Ask instructor** - They're here to help!

### Useful Commands for Debugging:

**Windows**:
```cmd
REM Check Python version
python --version

REM Check installed packages
pip list

REM Check conda environments
conda env list

REM Check environment variables
set
```

**Linux**:
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Check conda environments
conda env list

# Check environment variables
env | grep API
```

---

## FAQ

**Q: Which OS is better for this project?**
A: Both work fine! Linux is slightly easier for some Python packages. Windows with WSL2 gives you best of both worlds.

**Q: Do I need admin/sudo privileges?**
A: For initial conda install, yes. For everything else, no!

**Q: Can I use Python virtualenv instead of conda?**
A: Yes, but conda is recommended as it handles system dependencies better.

**Q: How much disk space do I need?**
A: ~5-10GB total (conda + packages + models).

**Q: Can I run this on a Chromebook?**
A: If it supports Linux apps (Crostini), yes! Follow Linux instructions.

---

## Summary: Quick Command Reference

### Windows (Command Prompt/PowerShell)
```cmd
REM Setup
conda create -n nvidia_rag python=3.11 -y
conda activate nvidia_rag
pip install -r requirements.txt

REM Run
python config.py                 REM View backends
python rag_flexible.py          REM Run system

REM Set API key (if needed)
set GROQ_API_KEY=your-key-here
```

### Linux/WSL (Bash)
```bash
# Setup
conda create -n nvidia_rag python=3.11 -y
conda activate nvidia_rag
pip install -r requirements.txt

# Run
python config.py                 # View backends
python rag_flexible.py          # Run system

# Set API key (if needed)
export GROQ_API_KEY="your-key-here"
```

---

**Happy Learning! 🚀**

*Now you're ready to dive into RAG systems regardless of your operating system!*
