#!/bin/sh
# Repeats for the two runs behind this report. Precondition for both: a quiet box -
# no other GPU client on the 7900 XTX, no browser or game, production serving stopped.

# Run 1, the legacy suite behind results.json in this folder (thinking off, 3 runs per cell).
# Precondition: the same quiet box; the tool starts the engine itself on port 28987.
"D:/Programs/Strata/.venv/Scripts/python.exe" tools/strata_bench.py --config D:/Programs/Strata/strata-iq3_xxs.json --long --runs 3 --port 28987 --out bench/results-v021

# Run 2, standard-v1 behind ../2026-10-20-standard-rx-7900-xtx-5b4dbf77 (about 20 to 30 minutes after load).
# Precondition: the same quiet box and a freshly loaded engine; a second run against the same
# engine counts as cache reuse and is marked nonconforming.
"D:/Programs/Strata/bin/strata.cmd" bench run --profile strata-iq3_xxs.json --suite standard-v1 --track default --track-notes '0.2.1 release, tuned config (issue 2274 flags), engine 0.1.45'
