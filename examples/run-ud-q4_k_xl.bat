@echo off
title Strata UD-Q4_K_XL
rem Qwen3.8-Flash-Next, unsloth's UD-Q4_K_XL, on http://127.0.0.1:8082/v1 with ud-q4_k_xl.json (README.md).
rem The config's --main-gpu runs the engine: another program on that card stalls its per-layer spin-waits.
rem Its --second-gpu holds more experts and may drive the display.
rem A PYTHONPATH set for another Python would mix that Python's packages into the venv's.
set "PYTHONPATH="
cd /d "%~dp0.."
".venv\Scripts\python.exe" serve\server.py --engine strata --config examples\ud-q4_k_xl.json --port 8082
pause
