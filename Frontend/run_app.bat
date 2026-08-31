@echo off
title Workforce Analytics 2026 Launcher
echo ========================================================
echo   Starting Workforce Analytics Dashboard (2026 Edition)
echo ========================================================
echo.
echo Launching production WSGI server...
cd /d "%~dp0"
start "" http://127.0.0.1:5000
python app.py
pause
