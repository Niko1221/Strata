@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "tools\strata_models.py" --open %*
) else (
  py -3 "tools\strata_models.py" --open %*
)
if errorlevel 1 pause