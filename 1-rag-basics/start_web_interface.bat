@echo off
REM Simple launcher script for the RAG web interface
REM Works on Windows

echo =========================================
echo   RAG System - Web Interface Launcher
echo =========================================
echo.

REM Activate conda environment
echo Activating conda environment...
call conda activate nvidia_rag

if %errorlevel% neq 0 (
    echo Failed to activate conda environment 'nvidia_rag'
    echo Please make sure the environment is created:
    echo   conda create -n nvidia_rag python=3.11 -y
    pause
    exit /b 1
)

echo Environment activated!
echo.

REM Check GPU status
echo Checking system capabilities...
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
    if %errorlevel% equ 0 (
        echo GPU detected and PyTorch CUDA enabled!
    ) else (
        echo ! GPU detected but PyTorch CUDA not available
        echo   Consider reinstalling with: pip install -r requirements-gpu.txt
    )
) else (
    echo Running in CPU mode
)
echo.

REM Run the web interface
echo Starting web interface...
echo.
echo Access the interface at:
echo   * http://localhost:7860
echo   * http://127.0.0.1:7860
echo.
echo ! Don't use http://0.0.0.0:7860 - use localhost instead!
echo.
echo Press Ctrl+C to stop the server
echo =========================================
echo.

python app_simple.py

pause
