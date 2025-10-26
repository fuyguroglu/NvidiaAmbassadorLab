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

REM Run the web interface
echo Starting web interface...
echo The interface will open in your browser at: http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo =========================================
echo.

python app_simple.py

pause
