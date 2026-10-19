#!/bin/bash
# The six runs of this report, as they were made (Git Bash on Windows 11, from the Strata folder, engine 0.1.45).
# Configs: configs/*.json (<Strata> and <Strata-data> stand for the install and data folders).  Each run needs the
# Strata server stopped: the tool starts its own engine.
set -e
Q=bench/results/2026-10-18-community-rtx-5070-ti-qwen3-8-flash-next-iq3-s/quality
# quant_quality.py puts the config's lib_dirs into LD_LIBRARY_PATH, which Windows does not read: the CUDA DLLs go on
# PATH by hand, or the engine exits before READY
export PATH="$PWD/.venv/Lib/site-packages/nvidia/cu13/bin/x86_64:$PATH"
PY=.venv/Scripts/python.exe

# Q8_0 reference: Unsloth's Q8_0 (6 shards, sha256-checked by setup.py's own downloader), a pack from
#   python tools/iq_pack.py --gguf <...>-Q8_0-00001-of-00006.gguf --out <Strata-data>/packs/unsloth-q8_0 --compat-bf16
# (setup's step 6 for the unsloth family), the RAM budget setup computes for 63.5 GiB (39 GiB), experts on the CPU
$PY tools/quant_quality.py run --config $Q/configs/q8_0.json --out $Q/data/q8_0.tsv

# IQ3_S, experts on the CPU (the yardstick's default, as in docs/QUANTS.md)
$PY tools/quant_quality.py run --config $Q/configs/iq3_s.json --out $Q/data/iq3s_cpu.tsv
# the same with the GGUF's own IQ4_NL n-gram table instead of ple-fp8.gguf
$PY tools/quant_quality.py run --config $Q/configs/iq3_s-iq4nl-table.json --out $Q/data/iq3s_cpu_iq4nl.tsv

# IQ3_S, the expert cache sized as the live server's (--gpu-experts auto: 4,396 slots), tail skip off and on
$PY tools/quant_quality.py run --config $Q/configs/iq3_s.json --out $Q/data/skip0.tsv --gpu-experts auto --env STRATA_ROUTE_TAIL_SKIP=0
$PY tools/quant_quality.py run --config $Q/configs/iq3_s.json --out $Q/data/skip7.tsv --gpu-experts auto --env STRATA_ROUTE_TAIL_SKIP=7
# the same as skip0 with int8 K/V (the live config's), through the wrapper that replaces the tool's fixed --kv fp16
STRATA_QQ_KV=int8 $PY $Q/qq_kv.py run --config $Q/configs/iq3_s.json --out $Q/data/skip0_int8.tsv --gpu-experts auto --env STRATA_ROUTE_TAIL_SKIP=0

# (a second skip0 run gave byte-identical rows; not kept)
# the per-position table the analysis reads, then the full IQ3_S runs can go (only q8_0 is kept in full)
$PY $Q/make_positions.py --runs $Q/data
gzip -9 -f $Q/data/q8_0.tsv
rm $Q/data/iq3s_cpu.tsv $Q/data/iq3s_cpu_iq4nl.tsv $Q/data/skip0.tsv $Q/data/skip7.tsv $Q/data/skip0_int8.tsv
$PY $Q/analyze.py > $Q/results.txt
