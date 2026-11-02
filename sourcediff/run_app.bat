@echo off
title SourceDiff Web Application
cd /d "%~dp0"
echo ===================================================
echo   Starting SourceDiff Web Application
echo   URL: http://127.0.0.1:8000
echo ===================================================
echo.
D:\deepfake\.venv311\Scripts\python.exe -m uvicorn sourcediff.api:app --host 127.0.0.1 --port 8000
pause
