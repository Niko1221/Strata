# Operator notes: what the runner could not see

Added by the operator next to the unedited runner output (`README.md`, `results.json`, `outputs.jsonl`, `config.json`,
`provenance/`). `engine-log-excerpt.txt` is the engine's own log from the start of this engine to the last sanity request,
with home paths, user names, LAN addresses and the API key replaced by placeholders. Hashes of every file: `SHA256SUMS`.

## Two GPUs, not one
The runner records one GPU (`hardware.gpus` has device 0 only) because it measured a server that was already running.
The engine ran a **layer split across two RTX 4070 12 GB cards** (`--layer-split 25`: layers 0-24 on CUDA0, the rest,
the output head and the MTP draft layer on CUDA1), launched with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1`.
The engine log excerpt shows both devices (`layer split across 2 GPUs: CUDA0, then CUDA1 (split 25)`).

| | CUDA0 | CUDA1 |
|---|---|---|
| card | Palit GeForce RTX 4070 12 GB | Palit GeForce RTX 4070 12 GB |
| PCIe | Gen4 x8 (CPU lanes, top slot) | Gen4 x8 (CPU lanes, second slot) |
| power limit | 200 W (the card's maximum) | 200 W |
| clock offsets (NVML) | memory +2200 (= +1100 MHz in Afterburner terms), core +75 | same |

**The cards are overclocked** (memory +2200 / core +75 via NVML offsets, applied at boot). This is not a stock-clock
result. Fans follow a custom curve (40% at 50 °C to 100% at 75 °C); during the run the cards peaked at 64 °C and 63 °C.

Board ASUS PRIME X570-PRO, Ryzen 9 5950X, 4x 32 GB DDR4-3200 (dual rank, running at 3200 MT/s), model files on a
Crucial T500 NVMe. NVIDIA driver 595.91.07 (open kernel module), CUDA 13.2, Ubuntu 24.04.5, kernel 7.0.0-38.

## Configuration (track custom)
Strata v0.2.6 CUDA source build (`4b9e05a2`, engine 0.1.51), `CMAKE_CUDA_ARCHITECTURES=89`, the original Flash-Next
IQ3_S GGUF (sha256 in `results.json`) with a native-experts pack and the MTP draft weights (`mtp-q2_0`) built by the Strata setup tools.
Engine arguments (from the log): `--expert-cache auto --prefill auto --spec 3 --spec-min-p 0.70 --mtp ... --max-context 131072
--kv int8 --ple-io ram --kv-resident 32768 --pcie-frac 0.00 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
--conversation-cache-mib 32768 --conversation-cache-slots 16 --conversation-cache-min-tokens 2048
--conversation-cache-min-free-mib 8192 --layer-split 25`.
Environment: `STRATA_STAGE_TRIM=1 STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_CPU_SHARE_MAX=3072 STRATA_SPLIT_MTP_BATCH=1
STRATA_ADAPT_LAG=2 STRATA_ROUTE_TAIL_SKIP=0`. `STRATA_PREFILL_CPU_SHARE=auto` had to be set explicitly: in this layer split
it was not active by default.

## Background during the run (disclosed, not hidden)
The server is this machine's normal Strata service. It was restarted just before the run (freshly loaded; the runner
confirms 0 requests before the run) and had been idle for 57 minutes before that. **Two small external requests arrived
during the run:** a 28-token and a 25-token health check from another machine (3 and 4 tokens generated, ~0.5 s each,
lines 115 and 124 of `engine-log-excerpt.txt`). The server has one slot, so they ran between benchmark requests, never
alongside one: one just before `performance-prompt-4k-r3`, one just before `performance-prompt-16k-r3`. The likely effect
is that those two requests waited ~0.5 s before starting, which would explain the **maximum** first-token times of the 4K
(3.97 s vs 3.48 s for the other two) and 16K (7.25 s vs 6.80 s) cases. Medians, prompt rates and decode rates are not
affected. The run was not repeated, because a repeat needs another restart of a service other people use.

## Same machine, earlier card (for context)
The same machine with a single RX 6800 XT ran this protocol on 2026-10-21 (PR #3646): prompt 952 (4K) / 1,100 (16K) tok/s,
decode 43-48 tok/s, 12/12 sanity.

## How this was produced
The owner of this machine directed the work; an AI agent (Anthropic's Claude Code) ran the benchmarks and wrote these notes
and the PR. Every number above is from the files in this folder or from the machine at the time of the run.
