# Community benchmark on NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 5 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti, 88 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz (14 cores, 28 threads); installed RAM: 121.4 GiB; storage: nvme0n1 Lexar SSD NM620 2TB nvme
- OS: Linux 7.0.0-42-generic; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.55, commit c62d5e3a; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=119.42, pcie=gen3 x16, pstate=P2, temp_c=63
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=151.62, pcie=gen3 x16, pstate=P2, temp_c=76
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=139.39, pcie=gen3 x16, pstate=P2, temp_c=69
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=146.13, pcie=gen3 x16, pstate=P2, temp_c=63
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-dsv41 --pack deepseek-v4.1-flash-ist-das-2.00bpw --max-context 131072 --eos 1 --vision on --dspark auto --dspark-conf 0.5 --tp 4 --experts 5 --threads on
```

## Method

Run by `python tools/strata_bench.py --long --runs 5` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 5 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 5 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 5 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 15 | not measured | 34.0 (25.6-48.5) | 0.42 (0.35-0.58) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 5 | 540.8 (464.8-560.0) | not measured | 7.65 (7.38-8.89) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 5 | 913.5 (874.2-920.9) | not measured | 18.18 (18.03-18.99) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 5 | 1040.5 (987.7-1054.4) | not measured | 31.92 (31.50-33.62) |
| agent session, follow-up turns | 12675 (12671-12686) at the last turn | see below | up to 64 | 5 | see below | 29.9 (22.0-37.4) | 11.89 (8.20-15.31) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.6-69.7); earlier-turn tokens read again per session: 2 (1-2) (earlier turns that the cache did not hold; the 11797 (11796-11798) new tokens of the files are read once); first turn (nothing cached) first token: 5.67 s (5.54-6.95).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 86.8 | 6.954 | 23.2 |
| 1 | 3335 | 636 | 2699 | 199.4 | 13.568 | 26.3 |
| 2 | 4850 | 3397 | 1453 | 144.4 | 10.094 | 22.0 |
| 3 | 7960 | 4913 | 3047 | 199.6 | 15.306 | 31.9 |
| 4 | 10044 | 8023 | 2021 | 175.1 | 11.583 | 28.4 |
| 5 | 12675 | 10098 | 2577 | 195.8 | 13.208 | 29.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 747 s.

## Comparison against the first defaults-only run on this machine

Everything below was measured on the same box (4 x RTX 2080 Ti 22 GB, driver 595.91.07, 121 GiB RAM, Xeon E5-2680 v4).
The engine binary is commit `c62d5e3a` plus the uncommitted item changes this run exists to measure; the same code is in the
packaged 4-architecture build (`sm_75`, `sm_86`, `sm_89`, `sm_120`).

**Settings of this run** (the tool cannot record `DS4_*` names in `machine.config.env`, so they are written here):

```text
env:   DS4_ASYNC_T=4  DS4_DEFAULTS=075  DS4_ATTN_SPLIT=0
args:  --pack deepseek-v4.1-flash-ist-das-2.00bpw --max-context 131072 --eos 1 --vision on
       --dspark auto --dspark-conf 0.5 --tp 4 --experts 5 --threads on
```

`DS4_DEFAULTS=075` is the measured 4 x 2080 Ti profile: busy experts as 64-row FP16 GEMMs (`DS4_FAT_ROWS=64`,
`DS4_FAT_PIECE=1024`) and a 2048-row attention slab. The engine log of this run prints both
(`CUDA0..3 busy experts (64 rows or more) as FP16 GEMMs, 1024 rows a piece`). `--threads on` is one host thread per device.
The split-KV default was measured on this box and **reverted** (inside the noise); `DS4_ATTN_SPLIT=0` states that.

| Configuration (same machine) | chat decode | agent decode | turn TTFT | 4K prompt | 16K prompt | 32K prompt |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-03, first run, defaults (`--tp 2`, no `--experts`, no env), 3 short runs | 17.0 | 25.1 | 13.883 s | 361.8 | 595.9 | - |
| 2026-11-03, best later run (`--tp 5`, 5 long runs) | 30.3 | 27.7 | 13.161 s | 423.7 | 670.4 | 765.4 |
| 2026-11-07/08, this code, baseline settings (FAT/slab off, no `--threads`), 5 long runs | 31.0 | 28.3 | 13.107 s | 426.0 | 673.2 | 764.8 |
| **this run** (all measured wins together), 5 long runs | **34.0** | **29.9** | **11.895 s** | **540.8** | **913.5** | **1040.5** |

- Against the first defaults-only run: chat **+100.0%**, agent **+19.1%**, 4K **+49.5%**, 16K **+53.3%**, TTFT **-14.3%**.
- Against the same code with the wins switched off: chat **+9.7%**, agent **+5.7%**, 4K **+26.9%**, 16K **+35.7%**,
  32K **+36.0%**, TTFT **-9.2%**.
- The two wins do not stack to the sum of their solo figures: `--threads on` alone measured chat 35.8-37.9 and
  `DS4_DEFAULTS=075` alone measured 540-542 / 913-915 / 1037-1038 prompt. Together the decode lands at 34.0, i.e. the
  FAT/slab path takes back a few percent of the decode gain while keeping the prompt gain.

Limits: the first run used `--tp 2` and 3 short runs, so part of the +100% chat is the flag change and the run shape, not
code. This box's run-to-run spread is about ±5% decode (the identical baseline config read 31.0 and 30.3 chat, 28.3 and 26.8
agent, in two runs of the same evening), which is why the against-baseline figures above are quoted with it in mind. Prompt
and decode changes of this size are outside that spread. Speed only: no correctness or quality check was run in this suite —
the 075 profile changes numerics (FAT is an FP16 GEMM path); its measured effect on `data/quant_quality/v1` is in
`~/StrataTuneNotes.md`.

### Re-run with the adopted configuration (2026-11-08, same machine)

The first run above used the legacy `DS4_ASYNC_T=4`. A later evaluation found that the drafter's verify windows are T=6, so
`T <= DS4_ASYNC_T` was false at 4 and true at 8: the verify rounds ran synchronously. The configuration now carries
`DS4_ASYNC_T=8` (the engine's own default) and was re-measured with the same suite:

| Configuration | chat decode | agent decode | turn TTFT | 4K prompt | 16K prompt | 32K prompt |
| --- | --- | --- | --- | --- | --- | --- |
| first run (`DS4_ASYNC_T=4`) | 34.0 (25.6-48.5) | 29.9 (22.0-37.4) | 11.895 s | 540.8 | 913.5 | 1040.5 |
| **adopted config (`DS4_ASYNC_T=8`)** | **39.1 (26.4-51.8)** | **31.4 (25.5-39.3)** | 11.931 s | 539.4 | 905.4 | 1035.0 |

Decode +15% chat and +5% agent, prompts unchanged; this is the configuration to reproduce with `config-adopted.json` in this folder (`config.json` describes the first run, which used `DS4_ASYNC_T=4`).
Four `=8` arms and four baselines were run in alternating order for this (run directories `bench/results/2026-11-08-t-async8*` and
`-t-final*`).
