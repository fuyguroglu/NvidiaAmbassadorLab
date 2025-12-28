@echo off
REM Smart Setup Script for RAG System (Windows)
REM Works with conda OR Python venv - automatically detects what's available

echo ==========================================
echo   RAG System - Smart Setup Script
echo   NVIDIA Ambassador Lab
echo ==========================================
echo.

REM Detect available Python environment tools
set USE_CONDA=false
where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set USE_CONDA=true
    echo [92m/ Conda found[0m
    for /f "delims=" %%i in ('conda --version') do echo %%i
    set ENV_NAME=nvidia_rag
    set ACTIVATE_CMD=conda activate nvidia_rag
) else (
    echo [96m! Conda not found - will use Python venv instead[0m

    REM Check if python is available
    where python >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [91mX Python not found![0m
        echo.
        echo Please install Python 3.8 or higher first:
        echo.
        echo   Option 1 - Microsoft Store:
        echo     Search for "Python 3.11" in Microsoft Store
        echo.
        echo   Option 2 - Official installer:
        echo     Download from https://www.python.org/downloads/
        echo     Make sure to check "Add Python to PATH" during installation
        echo.
        echo   Option 3 - winget:
        echo     winget install Python.Python.3.11
        echo.
        pause
        exit /b 1
    )

    echo [92m/ Python found[0m
    python --version
    set ENV_NAME=.venv
    set ACTIVATE_CMD=.venv\Scripts\activate
)
echo.

REM Setup environment based on available tool
if "%USE_CONDA%"=="true" (
    REM CONDA SETUP
    echo [96mUsing Conda for environment management[0m
    echo.

    REM Accept conda TOS if needed silently
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>nul
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>nul

    REM Check if environment already exists
    conda env list | findstr "^nvidia_rag " >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo [93m! Environment 'nvidia_rag' already exists[0m
        echo.
        set /p "recreate=Do you want to remove and recreate it? (y/n): "
        if /i "%recreate%"=="y" (
            echo Removing existing environment...
            conda env remove -n nvidia_rag -y
            echo [92m/ Removed[0m
            echo.
            REM Create new environment
            echo [96mCreating conda environment 'nvidia_rag' with Python 3.11...[0m
            conda create -n nvidia_rag python=3.11 -y
            echo [92m/ Environment created[0m
            echo.
        ) else (
            echo Keeping existing environment.
            echo.
        )
    ) else (
        REM Create conda environment
        echo [96mCreating conda environment 'nvidia_rag' with Python 3.11...[0m
        conda create -n nvidia_rag python=3.11 -y
        echo [92m/ Environment created[0m
        echo.
    )

    REM Activate environment
    echo [96mActivating environment...[0m
    call conda activate nvidia_rag
    echo [92m/ Environment activated[0m
    echo.

) else (
    REM VENV SETUP
    echo [96mUsing Python venv for environment management[0m
    echo.

    REM Check if venv already exists
    if exist ".venv" (
        echo [93m! Virtual environment '.venv' already exists[0m
        echo.
        set /p "recreate=Do you want to remove and recreate it? (y/n): "
        if /i "%recreate%"=="y" (
            echo Removing existing environment...
            rmdir /s /q .venv
            echo [92m/ Removed[0m
            echo.
            REM Create new environment
            echo [96mCreating Python virtual environment...[0m
            python -m venv .venv
            echo [92m/ Environment created[0m
            echo.
        ) else (
            echo Keeping existing environment.
            echo.
        )
    ) else (
        REM Create venv
        echo [96mCreating Python virtual environment...[0m
        python -m venv .venv
        echo [92m/ Environment created[0m
        echo.
    )

    REM Activate environment
    echo [96mActivating environment...[0m
    call .venv\Scripts\activate
    echo [92m/ Environment activated[0m
    echo.

    REM Upgrade pip
    echo [96mUpgrading pip...[0m
    python -m pip install --upgrade pip
    echo.
)

REM Run GPU detection
echo [96mDetecting GPU capabilities...[0m
echo.

REM Check for GPU silently
nvidia-smi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set GPU_DETECTED=true
    echo [92mNVIDIA GPU detected![0m
) else (
    set GPU_DETECTED=false
    echo [96mNo GPU detected (CPU-only mode)[0m
)
echo.

REM Determine recommendation
if "%GPU_DETECTED%"=="true" (
    set "RECOMMENDED=GPU version with CUDA support"
    set "INSTALL_CMD=pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt"
) else (
    set "RECOMMENDED=CPU-only version (smaller, no CUDA)"
    set "INSTALL_CMD=pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt"
)

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
    call %INSTALL_CMD%
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
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install -r requirements.txt
        set INSTALL_SUCCESS=%ERRORLEVEL%
    ) else if "%choice%"=="2" (
        echo [96mInstalling CPU-only version...[0m
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install -r requirements.txt
        set INSTALL_SUCCESS=%ERRORLEVEL%
    ) else if "%choice%"=="3" (
        echo [96mSkipping installation[0m
        echo.
        echo To install manually:
        echo   %ACTIVATE_CMD%
        echo   pip install -r requirements.txt
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
    echo   %ACTIVATE_CMD%
    echo   pip install -r requirements.txt
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
echo    %ACTIVATE_CMD%
echo    start_web_interface.bat
echo.
echo Or test with:
echo    python test_system.py
echo.
echo Happy learning! [92m*[0m
echo.
pause
