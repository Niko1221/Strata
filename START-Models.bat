@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" tools\start_models.py %*
if errorlevel 1 pause
