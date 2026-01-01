@echo off
REM Flask Web Interface Launcher (Windows)
REM ========================================
REM Simple, reliable startup script for the Flask RAG interface

echo.
echo ======================================================================
echo 🌐 FLASK WEB INTERFACE LAUNCHER
echo ======================================================================
echo.

REM Detect which environment manager is being used
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo 📦 Activating conda environment: nvidia_rag
    call conda activate nvidia_rag
    if %errorlevel% neq 0 (
        echo ⚠️  Failed to activate conda environment 'nvidia_rag'
        echo    Please run setup.bat first
        pause
        exit /b 1
    )
) else if exist ".venv\" (
    echo 📦 Activating Python virtual environment: .venv
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️  No conda environment or .venv found.
    echo    Please run setup.bat first
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo 🔍 Checking GPU availability...
echo ======================================================================

REM Check for GPU
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    nvidia-smi >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ GPU detected and available
        nvidia-smi --query-gpu=name --format=csv,noheader 2>nul | findstr /r "." >nul && nvidia-smi --query-gpu=name --format=csv,noheader 2>nul
    ) else (
        echo 💻 Running in CPU mode
    )
) else (
    echo 💻 Running in CPU mode
)

echo.
echo ======================================================================
echo 🚀 Starting Flask Web Interface
echo ======================================================================
echo.
echo Flask is starting... This may take a moment.
echo.
echo 📍 Once started, access the interface at:
echo    • http://localhost:7860
echo    • http://127.0.0.1:7860
echo.
echo ⚠️  IMPORTANT: Use 'localhost' or '127.0.0.1', NOT '0.0.0.0'
echo.
echo Press Ctrl+C to stop the server
echo ======================================================================
echo.

REM Run the Flask app
python app_flask.py

pause
