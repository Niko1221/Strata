# Community benchmark on one AMD Radeon AI PRO R9700

Measured on 2026-10-31 by sdroege. Two models on Strata 0.2.9 (engine 0.1.55): Unsloth UD-IQ4_XS and
UD-Q4_K_XL, each with the GDN chunked prompt path on and off, plus a Strata 0.2.8 (engine 0.1.54) arm for
each model and a hipBLASLt table comparison on 0.2.9. Main limitation: this is the quick
`tools/strata_bench.py --long` suite, not the standard `strata bench run` protocol, and the engine is a
local source build.

## Hardware and software

- GPU and VRAM: AMD Radeon AI PRO R9700, 32 GiB; PCIe 5.0 x16 (32 GT/s)
- CPU: AMD Ryzen 9 7950X, 16 cores / 32 threads; 121.5 GiB RAM; NVMe Samsung SSD 980 PRO 1 TB plus a
  SATA WDC WD60EFAX
- OS: Fedora 44, kernel 7.2.12-200.fc44.x86_64
- ROCm 10.1 in `/opt/rocm`, hipBLASLt 100401
- Strata v0.2.9 source, engine 0.1.55, local build for gfx1201 (see the limitations)
- Power limit 210 W; VDDGFX offset -25 mV; no other GPU workload running

## Model and configuration

- Unsloth UD-IQ4_XS (3 shards) and UD-Q4_K_XL (4 shards), native GGUF plus a pack, the install's own
  expert profile
- Context 131072, `--kv int8`, `--kv-resident 32768`, conversation cache 8192 MiB / 8000 tokens,
  `--spec 4 --spec-min-p 0.5`, MTP, `--vision`, `--vram-reserve-mib 576`; the Q4 config also has
  `--resident-budget-gib 55`
- Env: `STRATA_HIPBLASLT_TUNING` (the local 100401 table), `STRATA_ROUTE_TAIL_SKIP=0`,
  `STRATA_PLE_AUTO_RAM=0`; the GDN off arms add `STRATA_GDN_CHUNKED=0`
- Sampling in the config: temperature 1.0, top_p 0.95, top_k 20; the suite sends temperature 0 and
  thinking off

Launch command (IQ4; the Q4 config differs only in the pack, the native file and the budget):

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --conversation-cache-mib 8192 --conversation-cache-min-tokens 8000 --vision --vram-reserve-mib 576
```

## Method

- `python tools/strata_bench.py --config <config> --runs 3 --long --out <dir>`
- The 0.2.9 arms start their own server from the config; the 0.2.8 arms ran against a server that was
  already running. Model loading is not timed.
- One warm-up chat and a 1K prompt are discarded first. Every prompt carries a unique tag, so the
  conversation cache cannot help the fresh-prompt tests.
- 3 fresh prompts each at 4,019 / 16,058 / 32,049 tokens with a 24-token cap; 9 short chats with a
  100-token cap; 3 agent sessions of 5 turns with a 64-token cap per turn.
- Prompt and decode speeds are the engine's own timings; TTFT is measured by the client over HTTP.
- No correctness check was run.

## Results

Prompt throughput is the median of the three fresh prompts, with the range over those runs. Decode is the
median over all requests of the test, with its range.

### Unsloth UD-IQ4_XS

| Run | 4K prompt tok/s | 16K prompt tok/s | 32K prompt tok/s | TTFT 4K / 16K / 32K s | Chat decode tok/s | Agent decode tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2.8, engine 0.1.54 | 2333 (2226-2338) | 2922 (2918-2961) | 2848 (2839-2861) | 1.75 / 5.52 / 11.29 | 77.5 (61.8-90.0) | 86.0 (70.6-98.4) |
| 0.2.9, GDN off | 2350 (2261-2363) | 2936 (2932-2969) | 2866 (2858-2877) | 1.73 / 5.50 / 11.22 | 78.5 (59.6-91.2) | 85.8 (62.2-102.0) |
| 0.2.9, GDN on (default) | 2498 (2382-2502) | 3162 (3159-3201) | 3067 (3059-3085) | 1.63 / 5.11 / 10.48 | 76.5 (62.0-91.1) | 85.5 (52.0-97.3) |

### Unsloth UD-Q4_K_XL, `--resident-budget-gib 55`

| Run | 4K prompt tok/s | 16K prompt tok/s | 32K prompt tok/s | TTFT 4K / 16K / 32K s | Chat decode tok/s | Agent decode tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2.8, engine 0.1.54 | 1830 (1768-1834) | 2739 (2722-2763) | 2693 (2669-2704) | 2.22 / 5.89 / 11.94 | 50.6 (42.6-57.6) | 64.6 (38.7-76.9) |
| 0.2.9, GDN off | 1815 (1786-1822) | 2751 (2731-2764) | 2707 (2661-2724) | 2.24 / 5.86 / 11.87 | 47.7 (44.2-57.1) | 62.9 (35.4-80.8) |
| 0.2.9, GDN on (default) | 1939 (1856-1940) | 2934 (2930-2958) | 2890 (2856-2893) | 2.10 / 5.50 / 11.13 | 50.2 (45.2-59.1) | 62.4 (35.8-72.9) |

## What changed

**0.2.8 to 0.2.9 with the GDN path off: nothing measurable.** IQ4 2333/2922/2848 to 2350/2936/2866
(+0.7/+0.5/+0.6%), Q4 1830/2739/2693 to 1815/2751/2707 (-0.8/+0.5/+0.5%). The engine update alone is
inside the run-to-run noise.

**GDN chunked prompt path on (`STRATA_GDN_CHUNKED=1`, the 0.2.9 default on Linux HIP gfx12):**
IQ4 +6.3/+7.7/+7.0% at 4K/16K/32K, Q4 +6.8/+6.6/+6.8%, against the same engine with it off. TTFT
follows (IQ4 4K 1.73 to 1.63 s, Q4 4K 2.24 to 2.10 s). Decode is unchanged within noise in both models:
the chat medians move by a few percent in opposite directions, the distributions overlap almost
completely, and the agent medians are 85.8 to 85.5 (IQ4) and 62.9 to 62.4 (Q4).

The commit that turned the default on (bee469c4) reports +3.5% at 4K and +5.5% at 16K on an R9700. This
machine sees more, and at 32K too.

**hipBLASLt table.** On 0.2.9, the local 32-row gfx1201/100401 table against no table at all: IQ4
2498/3162/3067 with the table and 2481/3116/3034 without (+0.7/+1.5/+1.1%), Q4 1939/2934/2890 and
1948/2924/2879 (within noise). A freshly calibrated table for the same shapes was worse on IQ4 at 4K and
16K (2387/3098/3049, -4.4/-2.0/-0.6%), so it was not adopted. The regenerated table is in `tables/` for
reproduction; it is not the table these runs used.

## Correctness and limitations

- Speed only. No recall, coding or tool-call check was run.
- Not the standard protocol. These are `tools/strata_bench.py --long` runs, which this guide accepts for
  older reports but which are not `strata bench run` results.
- The engine is a local source build of v0.2.9 (engine 0.1.55, gfx1201). The working tree had three
  local commits on top of v0.2.9, but none of them changes the engine: two touch only `serve/server.py`
  (a vision timing log line and a fix for a crash when an engine start fails) and one touches only
  `setup.py` and a test. The engine sources (`src/`, `include/`, `CMakeLists.txt`) are exactly v0.2.9,
  so the binary is what a v0.2.9 source build produces. The two serve changes cannot move these numbers
  either: one adds a log line after an image is read, the other only runs when an engine start fails.
  The recorded commit `ff629c93` is not an upstream commit, but the build contains no relevant change
  over 0.2.9.
- The hipBLASLt 100401 table is local. Upstream ships no table for 100401.
- The 0.2.8 arms ran against an already running server; the 0.2.9 arms started their own.
- The Q4 0.2.8 arm used a config whose model name ends in `-budget`; its settings are the same as the
  shipped Q4 config.
- Absolute numbers depend on the -25 mV VDDGFX offset and the 210 W power limit.
- The agent decode spread is wide in every run (IQ4 52 to 102 tok/s, Q4 35 to 81 tok/s); treat those
  medians as rough.

## Files

- `runs/<name>/results.json`: every request with tokens and timings, plus the machine facts
- `runs/<name>/engine-log-excerpt.txt`: the engine's start-up lines (0.2.9 arms only)
- `tables/gfx1201-hipblaslt-100401-2026-10-31.txt`: the regenerated table, not adopted
