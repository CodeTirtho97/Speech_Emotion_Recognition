@echo off
REM Backend Server Launcher
REM This script attempts to use the best available Python

echo Starting Backend API Server...
echo.

REM Try Anaconda Python first (has all dependencies)
if exist "C:\Users\User\anaconda3\python.exe" (
    echo Using Anaconda Python...
    "C:\Users\User\anaconda3\python.exe" app.py
    goto :end
)

REM Try system Python
echo Using system Python...
python app.py

:end
