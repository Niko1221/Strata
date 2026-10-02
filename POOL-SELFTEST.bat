@echo off
rem Strata's pool: test the pool on this one PC (the model alone, then split between two engines on this GPU).
rem Close Strata first: the GPU must be free.
cd /d "%~dp0"
set CFG=
for /f "delims=" %%f in ('dir /b /o-d strata-*.json 2^>NUL ^| findstr /v /i "pool shared"') do if not defined CFG set CFG=%%f
if not defined CFG (
  echo Run START-HERE.bat first: it sets up the model.
  pause
  exit /b 1
)
echo Testing with %CFG% ...
".venv\Scripts\python.exe" tools\pool_selftest.py %CFG% %*
pause
