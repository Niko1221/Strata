#!/bin/sh
cd "/home/asx/sources/Strata"
exec "/home/asx/sources/Strata/.venv/bin/python" "/home/asx/sources/Strata/serve/server.py" "--engine" "strata" "--config" "/home/asx/sources/Strata/strata-unsloth-ud-q4_k_xl.json" "--port" "8080" 
