@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0dist\Strata\Strata.exe" (
  echo Strata app is not built. Run .venv\Scripts\python.exe tools\build_desktop.py
  pause
  exit /b 1
)
start "" "%~dp0dist\Strata\Strata.exe" --root "%~dp0."
