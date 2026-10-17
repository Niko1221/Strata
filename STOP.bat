@echo off
rem STOP.bat - stop Strata, and only Strata.  See stop-strata.ps1 for what it matches.
rem   STOP.bat                    stop it
rem   STOP.bat -WhatIf             list what would be stopped, stop nothing
rem   STOP.bat -Root D:\Strata     the install somewhere else
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-strata.ps1" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    pause
)
endlocal & exit /b %RC%
