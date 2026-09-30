@echo off
title Strata IQ3_S
rem Qwen3.8-Flash-Next, ISTA-DASLab's GSQ-RCO IQ3_S, on http://127.0.0.1:8081/v1 with iq3_s.json (README.md).
rem The config's --main-gpu runs the engine: another program on that card stalls its per-layer spin-waits.
rem Its --second-gpu holds more experts and may drive the display.
rem A PYTHONPATH set for another Python would mix that Python's packages into the venv's.
set "PYTHONPATH="
cd /d "%~dp0.."
".venv\Scripts\python.exe" serve\server.py --engine strata --config examples\iq3_s.json --port 8081
pause
