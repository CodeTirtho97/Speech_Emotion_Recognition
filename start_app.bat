@echo off
echo ============================================================
echo   SPEECH EMOTION RECOGNITION - APPLICATION LAUNCHER
echo ============================================================
echo.

REM Check if models exist
if not exist "models\best_model.pkl" (
    echo [ERROR] Models not found!
    echo Please train the model first by running: python train_model.py
    echo.
    pause
    exit /b 1
)

REM Determine which Python to use
set PYTHON_CMD=python
if exist "C:\Users\User\anaconda3\python.exe" (
    set PYTHON_CMD=C:\Users\User\anaconda3\python.exe
    echo Using Anaconda Python
) else (
    echo Using system Python
)
echo.

echo [1/2] Starting Backend API Server...
echo ============================================================
start "Backend API" cmd /k "cd /d %~dp0 && %PYTHON_CMD% app.py"

timeout /t 3 /nobreak >nul

echo.
echo [2/2] Starting Frontend Server...
echo ============================================================
start "Frontend Server" cmd /k "cd /d %~dp0\frontend && %PYTHON_CMD% -m http.server 8000"

timeout /t 2 /nobreak >nul

echo.
echo ============================================================
echo   APPLICATION STARTED SUCCESSFULLY!
echo ============================================================
echo.
echo Backend API:  http://localhost:5000
echo Frontend UI:  http://localhost:8000
echo.
echo Opening browser...
timeout /t 2 /nobreak >nul
start http://localhost:8000
echo.
echo Press any key to stop all servers...
pause >nul

echo.
echo Stopping servers...
taskkill /FI "WindowTitle eq Backend API*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq Frontend Server*" /T /F >nul 2>&1
echo Done!
