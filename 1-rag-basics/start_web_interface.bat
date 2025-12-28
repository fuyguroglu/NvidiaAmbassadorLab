@echo off
REM Launcher script for the RAG web interface
REM Works with conda OR Python venv - automatically detects what's available

echo =========================================
echo   RAG System - Web Interface Launcher
echo =========================================
echo.

REM Detect and activate environment
where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    conda env list | findstr "^nvidia_rag " >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        REM Conda environment exists
        echo Activating conda environment 'nvidia_rag'...
        call conda activate nvidia_rag

        if %errorlevel% neq 0 (
            echo [91mFailed to activate conda environment 'nvidia_rag'[0m
            echo Please run setup first: setup.bat
            pause
            exit /b 1
        )
        goto :activated
    )
)

REM Check for venv
if exist ".venv" (
    echo Activating Python virtual environment...
    call .venv\Scripts\activate

    if %errorlevel% neq 0 (
        echo [91mFailed to activate virtual environment[0m
        echo Please run setup first: setup.bat
        pause
        exit /b 1
    )
    goto :activated
)

REM No environment found
echo [91mNo environment found![0m
echo.
echo Please run setup first:
echo   setup.bat
pause
exit /b 1

:activated
echo [92mEnvironment activated![0m
echo.

REM Check GPU status
echo Checking system capabilities...
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
    if %errorlevel% equ 0 (
        echo [92mGPU detected and PyTorch CUDA enabled![0m
    ) else (
        echo [93m! GPU detected but PyTorch CUDA not available[0m
        echo   Running in CPU mode[0m
    )
) else (
    echo [96mRunning in CPU mode[0m
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
