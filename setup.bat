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
set PYTORCH_INDEX=https://download.pytorch.org/whl/cpu

echo TinyML Workshop -- Environment Setup
echo ------------------------------------

REM Use the Python Launcher (py) which respects installed versions on Windows.
REM This avoids picking up an old Python from PATH.
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: The Python Launcher was not found.
    echo        Download Python from https://www.python.org/downloads/
    echo        Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)

REM Check Python version is 3.10+
for /f "tokens=2 delims= " %%v in ('py -3 --version 2^>^&1') do set PYTHON_VERSION=%%v
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
    py -3 -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM Activate the virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
echo [OK] Virtual environment activated.

REM Upgrade pip via python -m pip to avoid permission issues
echo Upgrading pip...
python.exe -m pip install --quiet --upgrade pip
echo [OK] pip upgraded.

REM Install PyTorch from the official index (CPU build, latest compatible version)
echo Installing PyTorch and torchvision (CPU)...
pip install --quiet torch torchvision --index-url %PYTORCH_INDEX%
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
