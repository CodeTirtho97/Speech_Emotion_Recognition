@echo off
REM Frontend Server Launcher
REM This script attempts to use the best available Python

echo Starting Frontend Server...
echo.

cd frontend

REM Try Anaconda Python first
if exist "C:\Users\User\anaconda3\python.exe" (
    echo Using Anaconda Python...
    "C:\Users\User\anaconda3\python.exe" -m http.server 8000
    goto :end
)

REM Try system Python
echo Using system Python...
python -m http.server 8000

:end
