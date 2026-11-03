@echo off
rem Strata Dev pruefen (dev-check.ps1, docs\DEV_BUILD.md). Argumente werden weitergereicht, z. B. dev-check.bat -SkipPytest
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev-check.ps1" %*
pause
