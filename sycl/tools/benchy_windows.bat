@echo off
rem benchy v1 on Windows: the standard benchmark of Strata's SYCL engine on the Arc Pro B70, run natively from the
rem private Python environment that sycl\START-HERE.bat --backend sycl creates (sycl\tools\benchy_windows.py; the Linux original is
rem sycl\benchy.sh, which needs Docker and sudo and does not run here).
rem
rem   sycl\tools\benchy_windows.bat                         every model set up here, benchy v1's six prompt sizes
rem   sycl\tools\benchy_windows.bat --sizes 20,2185,8000    some sizes only
rem   sycl\tools\benchy_windows.bat --runs 3                each size three times: the median and the spread
rem   sycl\tools\benchy_windows.bat --configs strata-coder-iq1_m.json --out DIR --force --warm
rem
rem Stop the served model first (close its window): the engine is exclusive on the card, and benchy refuses to start
rem while a strata.exe is running or more than 2 GiB of VRAM is in use (--force overrides).  Every run starts warm
rem (Windows has no page-cache drop; --warm is accepted and ignored).  Results: sycl\benchy-results\v1-<date>-windows\
rem (matrix.md and config-*.txt are what to post, docs\INTEL_PERFORMANCE.md "Submitting numbers").
rem
rem Why the .venv: the bench imports sycl\serve\server_intel.py for the engine's environment (which imports
rem serve\server.py and its jinja2) and psutil; both live in .venv, which setup.py fills from requirements.txt.
setlocal
title Strata benchy
cd /d "%~dp0\..\.."
if exist "strata-*.json" goto venv
echo.
echo  No model is set up yet: there is no strata-*.json next to the checkout.
echo  Run sycl\START-HERE.bat --backend sycl first - it builds nothing, it downloads and prepares the model and writes the config.
pause
exit /b 2

:venv
if exist ".venv\Scripts\python.exe" goto run

call :findpy
if defined PY goto mkvenv
echo.
echo  Python 3.10 or newer is not installed.
echo  Install 64-bit Python 3.12 from https://www.python.org/downloads/ ("Add python.exe to PATH"),
echo  then run sycl\START-HERE.bat --backend sycl once and this script again.
pause
exit /b 1

:mkvenv
rem a private environment inside this folder, so nothing is installed into the system Python (as START-HERE.bat does);
rem a fresh one has none of the packages the bench imports, so setup's own list goes in (setup.py's pip step reads it)
%PY% -m venv .venv
if not exist ".venv\Scripts\python.exe" (
    echo  Could not create the Python environment in .venv
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo  Could not install requirements.txt into .venv
    pause
    exit /b 1
)

:run
".venv\Scripts\python.exe" sycl\tools\benchy_windows.py %*
set "RC=%errorlevel%"
if not "%RC%"=="0" pause
exit /b %RC%

:findpy
rem the py launcher first, then python on PATH (not the Microsoft Store stub), then the usual per-user folders
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3" & goto :eof
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python" & goto :eof
for %%V in (313 312 311 310) do if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"" & goto :eof
goto :eof
