@echo off
REM setup.bat — One-click environment setup for the TinyML workshop.
REM
REM Usage:
REM   setup.bat
REM
REM After completion, activate the environment and run the scripts:
REM   venv\Scripts\activate
REM   python train.py

setlocal enabledelayedexpansion

set VENV_DIR=venv
set TORCH_VERSION=2.5.1
set TORCHVISION_VERSION=0.20.1
set PYTORCH_INDEX=https://download.pytorch.org/whl/cpu

echo TinyML Workshop -- Environment Setup
echo ------------------------------------

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was not found.
    echo        Download Python from https://www.python.org/downloads/
    echo        Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)

REM Check Python version is 3.10+
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)
if %PYTHON_MAJOR% LSS 3 (
    echo ERROR: Python 3.10+ is required. You have Python %PYTHON_VERSION%.
    exit /b 1
)
if %PYTHON_MAJOR% EQU 3 if %PYTHON_MINOR% LSS 10 (
    echo ERROR: Python 3.10+ is required. You have Python %PYTHON_VERSION%.
    exit /b 1
)
echo [OK] Python %PYTHON_VERSION%

REM Create virtual environment
if exist "%VENV_DIR%\" (
    echo [OK] Virtual environment '%VENV_DIR%' already exists, skipping creation.
) else (
    echo Creating virtual environment '%VENV_DIR%'...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM Activate the virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
echo [OK] Virtual environment activated.

REM Upgrade pip
echo Upgrading pip...
pip install --quiet --upgrade pip

REM Install PyTorch from the official index (CPU build)
echo Installing PyTorch %TORCH_VERSION% and torchvision %TORCHVISION_VERSION% (CPU^)...
pip install --quiet ^
    "torch==%TORCH_VERSION%" ^
    "torchvision==%TORCHVISION_VERSION%" ^
    --index-url %PYTORCH_INDEX%
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch.
    exit /b 1
)
echo [OK] PyTorch installed.

REM Install remaining dependencies
echo Installing remaining dependencies...
pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    exit /b 1
)
echo [OK] All dependencies installed.

echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo Activate the environment with:
echo     venv\Scripts\activate
echo.
echo Then run the workshop scripts in order:
echo     python train.py
echo     python ptq.py
echo     python qat.py
echo     python pruning.py
echo     python export_onnx.py
echo     python inference.py
echo.

endlocal
