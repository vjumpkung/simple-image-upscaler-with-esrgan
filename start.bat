@echo off
setlocal enabledelayedexpansion

:: Check if venv exists, create if it doesn't
if not exist venv (
    echo Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
call venv\Scripts\activate

:: Check for NVIDIA GPU
echo Install requirements
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    pip install --quiet torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
    pip install --quiet -r requirements.txt
) else (
    pip install --quiet -r requirements.txt
)

:: Start the ESRGAN upscaler
echo Starting ESRGAN upscaler...
python esrgan-upscaler.py

:: Deactivate virtual environment
deactivate

pause