@echo off
rem Optional minimal terminal launcher for Strata (see tools\strata_tui.py).
cd /d "%~dp0"
where py.exe >nul 2>nul
if not errorlevel 1 (
    py -3 "tools\strata_tui.py"
    exit /b %errorlevel%
)
python "tools\strata_tui.py"
