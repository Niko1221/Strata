#!/bin/bash
S=/home/<user>/.hermes/skills/software-development/self-hosted-llm-endpoint-triage/scripts/mysrv-bench-timings.mjs
L=/home/<user>/.hermes/cache/scratch/llm_gw/benchC.log
rm -f "$L"
node "$S" concurrent 2 >> "$L" 2>&1
node "$S" solo 2 >> "$L" 2>&1
node "$S" long128 1 >> "$L" 2>&1
echo "# DONE" >> "$L"
