@echo off
rem Strata Dev bauen und starten (dev-build.ps1, docs\DEV_BUILD.md). Argumente werden weitergereicht, z. B. dev-build.bat -Install
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev-build.ps1" %*
if errorlevel 1 pause
