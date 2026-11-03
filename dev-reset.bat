@echo off
rem Strata Dev zuruecksetzen: loescht nur %LOCALAPPDATA%\StrataDev nach Rueckfrage (dev-reset.ps1, docs\DEV_BUILD.md).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev-reset.ps1" %*
pause
