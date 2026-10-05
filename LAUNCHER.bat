@echo off
rem Strata's launcher: one window to pick which model to run and with which settings - the sizes setup offers and
rem what they need, the models this PC already has, presets you can save and change, setup's calibration, and the
rem speed of each model (published, measured here, and live while it runs).
rem It changes nothing by itself: a download runs setup exactly as SETUP.bat does, and a start runs the same
rem serve\server.py as run-<model>.bat. Closing this window leaves a started model running.
setlocal
title Strata launcher
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe" & goto run

call :findpy
if defined PY goto run
echo.
echo  The launcher needs Python 3.10 or newer (64-bit).
echo  Run START-HERE.bat once - it installs Python and the model - then LAUNCHER.bat.
pause
exit /b 1

:run
"%PY%" -m launcher %*
if errorlevel 1 pause
exit /b

:findpy
rem the py launcher first, then python on PATH (not the Microsoft Store stub), then the usual per-user folders
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3" & goto :eof
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python" & goto :eof
for %%V in (313 312 311 310) do if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"" & goto :eof
goto :eof
