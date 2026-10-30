@echo off
title Academic Performance Monitoring & Prediction System
echo ============================================================
echo  Academic Performance Monitoring ^& Prediction System
echo  Starting up...
echo ============================================================
echo.

cd /d "%~dp0"

REM Check Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install / upgrade dependencies
echo [INFO] Installing dependencies from requirements.txt...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies ready.

REM Generate sample data if not present
if not exist "app\data\sample_students.csv" (
    echo [INFO] Generating sample student data...
    python app\data\generate_sample_data.py
    echo [OK] Sample data generated.
)

echo.
echo ============================================================
echo  Server starting at http://127.0.0.1:5000
echo  Press Ctrl+C to stop
echo ============================================================
echo.

REM Open browser after a short delay (runs in background)
start "" timeout /t 2 >nul ^& start "" "http://127.0.0.1:5000"

python run.py
pause
