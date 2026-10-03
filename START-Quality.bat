@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "tools\personal_control.py" start quality
if errorlevel 1 pause
