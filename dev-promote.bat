@echo off
rem Strata Dev: dev-promote.ps1 (docs\DEV_BUILD.md). Ohne Argumente nur ansehen; uebernommen/zurueckgesetzt wird erst mit -Yes.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev-promote.ps1" %*
pause
