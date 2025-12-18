@echo off
REM Smart Setup Script for RAG System (Windows)
REM Automatically detects GPU and installs appropriate dependencies

echo ==========================================
echo   RAG System - Smart Setup Script
echo   NVIDIA Ambassador Lab
echo ==========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [91mX Conda not found![0m
    echo.
    echo Please install Miniconda first:
    echo   Download from: https://docs.conda.io/en/latest/miniconda.html
    echo   Run: Miniconda3-latest-Windows-x86_64.exe
    echo.
    pause
    exit /b 1
)

echo [92m/ Conda found[0m
conda --version
echo.

REM Accept conda TOS if needed (silently)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>nul
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>nul

REM Check if environment already exists
conda env list | findstr "nvidia_rag" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [93m! Environment 'nvidia_rag' already exists[0m
    echo.
    set /p "recreate=Do you want to remove and recreate it? (y/n): "
    if /i "%recreate%"=="y" (
        echo Removing existing environment...
        conda env remove -n nvidia_rag -y
        echo [92m/ Removed[0m
        echo.
    ) else (
        echo Keeping existing environment. Activating...
        call conda activate nvidia_rag
        echo [92m/ Activated nvidia_rag environment[0m
        echo.
        REM Continue to installation check instead of exiting
    )
) else (
        REM Create conda environment
    echo [96mCreating conda environment 'nvidia_rag' with Python 3.11...[0m
    conda create -n nvidia_rag python=3.11 -y

    echo [92m/ Environment created[0m
    echo.

    REM Activate environment
    echo [96mActivating environment...[0m
    call conda activate nvidia_rag

    echo [92m/ Environment activated[0m
    echo.
)

REM Run GPU detection and store result
echo [96mDetecting GPU capabilities...[0m
echo.

REM Check which requirements file to use
if exist "requirements-cpu.txt" (
    REM Check for GPU silently
    nvidia-smi >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo [92mNVIDIA GPU detected![0m
        set "RECOMMENDED=GPU version with CUDA support"
        set "RECOMMENDED_FILE=requirements-gpu.txt"
    ) else (
        echo [96mNo GPU detected (CPU-only mode)[0m
        set "RECOMMENDED=CPU-only version (smaller, no CUDA)"
        set "RECOMMENDED_FILE=requirements-cpu.txt"
    )
) else (
    echo [96mUsing standard requirements.txt[0m
    set "RECOMMENDED=standard installation"
    set "RECOMMENDED_FILE=requirements.txt"
)
echo.

REM Ask user for confirmation
echo.
echo ==========================================
echo Recommendation: Install %RECOMMENDED%
echo ==========================================
echo.
set /p "confirm=Follow this recommendation? (Y/n): "

REM Default to Yes if empty
if "%confirm%"=="" set "confirm=Y"

if /i "%confirm%"=="Y" (
    echo.
    echo [96mInstalling recommended version...[0m
    pip install -r %RECOMMENDED_FILE%
    set INSTALL_SUCCESS=%ERRORLEVEL%
) else (
    echo.
    echo ==========================================
    echo Installation Options:
    echo ==========================================
    echo.
    echo 1. Force GPU version (with CUDA)
    echo 2. Force CPU version (no CUDA)
    echo 3. Skip installation (manual setup)
    echo.
    set /p "choice=Choose option (1-3): "
    echo.

    if "%choice%"=="1" (
        echo [96mInstalling GPU version with CUDA support...[0m
        if exist "requirements-gpu.txt" (
            pip install -r requirements-gpu.txt
        ) else (
            pip install -r requirements.txt
        )
        set INSTALL_SUCCESS=%ERRORLEVEL%
    ) else if "%choice%"=="2" (
        echo [96mInstalling CPU-only version...[0m
        if exist "requirements-cpu.txt" (
            pip install -r requirements-cpu.txt
        ) else (
            pip install -r requirements.txt
        )
        set INSTALL_SUCCESS=%ERRORLEVEL%
    ) else if "%choice%"=="3" (
        echo [96mSkipping installation[0m
        echo.
        echo To install manually:
        echo   conda activate nvidia_rag
        echo   pip install -r %RECOMMENDED_FILE%
        pause
        exit /b 0
    ) else (
        echo [91mX Invalid option[0m
        pause
        exit /b 1
    )
)

REM Check if installation was successful
if %INSTALL_SUCCESS% NEQ 0 (
    echo.
    echo [91mX Installation failed![0m
    echo.
    echo Please check the error messages above and try again.
    echo You can also install manually:
    echo   conda activate nvidia_rag
    echo   pip install -r %RECOMMENDED_FILE%
    pause
    exit /b 1
)

echo.
echo [92m/ Installation complete![0m
echo.
echo ==========================================
echo Next Steps:
echo ==========================================
echo.
echo 1. Add documents to the 'data' folder
echo 2. Run the system:
echo    conda activate nvidia_rag
echo    start_web_interface.bat
echo.
echo Or test with:
echo    python test_system.py
echo.
echo Happy learning! [92m*[0m
echo.
pause
