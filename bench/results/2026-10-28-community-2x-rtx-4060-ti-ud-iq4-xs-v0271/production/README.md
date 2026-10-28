# Community benchmark on NVIDIA GeForce RTX 4060 Ti + NVIDIA GeForce RTX 4060 Ti

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4060 Ti + NVIDIA GeForce RTX 4060 Ti, 32 GiB VRAM
- CPU: AMD Ryzen Threadripper PRO 3975WX 32-Cores (32 cores, 64 threads); installed RAM: 125.6 GiB; storage: sda     ST4000VN008-2DR166   usb; sdb     WDC WD20EFAX-68FB5N0 sata; zram0; nvme0n1 CT500P310SSD8        nvme
- OS: Linux 7.1.5-76070105-generic; GPU driver: 610.57.04; CUDA (driver supports up to): 13.3
- Strata: project 0.1.53, commit 46c152a1; engine 0.1.53
- GPU detail (after the run, near idle): power_limit_w=165.00, power_draw_w=28.90, pcie=gen4 x8, pstate=P2, temp_c=51
- GPU detail (after the run, near idle): power_limit_w=165.00, power_draw_w=76.90, pcie=gen4 x8, pstate=P2, temp_c=59
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt-iten --max-context 262144 --kv int8 --remote-expert-opt --pcie-frac 0.00 --spec-min-p 0.50 --kv-resident 32768 --vision --vram-reserve-mib 700 --conversation-cache-mib 8192 --conversation-cache-slots 4
```

Environment of the config: `STRATA_PROMPT_ATTN_IMMA=1`, `STRATA_HC_UPMIX=1`, `STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG=2`, `STRATA_CPU_IQ_ALL_EXACT1=1`, `STRATA_SIDE_MIXER=2`, `STRATA_ATTN_DAILY=1`, `STRATA_GR_UP75=1`, `STRATA_GU_SWIGLU=1`, `STRATA_QSA_ROWS=1`, `STRATA_B6_MMVQ_ROWS=1`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 58.2 (53.7-79.1) | 0.44 (0.39-0.53) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 910.9 (870.0-914.9) | not measured | 4.45 (4.43-4.65) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1787.7 (1781.5-1793.6) | not measured | 9.03 (9.00-9.06) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2074.3 (2073.8-2077.6) | not measured | 15.51 (15.49-15.51) |
| agent session, follow-up turns | 12897 (12891-12917) at the last turn | see below | up to 64 | 3 | see below | 66.5 (57.9-71.1) | 4.41 (4.09-4.65) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.2-69.4); earlier-turn tokens read again per session: 74 (3-74) (earlier turns that the cache did not hold; the 12062 (12062-12064) new tokens of the files are read once); first turn (nothing cached) first token: 3.08 s (3.01-3.08).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 204.3 | 3.013 | 58.1 |
| 1 | 3355 | 628 | 2727 | 628.1 | 4.38 | 64.9 |
| 2 | 4888 | 3348 | 1540 | 374.6 | 4.148 | 61.1 |
| 3 | 8132 | 4951 | 3181 | 692.2 | 4.634 | 66.5 |
| 4 | 10189 | 8184 | 2005 | 480.6 | 4.21 | 57.9 |
| 5 | 12917 | 10234 | 2683 | 605.1 | 4.473 | 67.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 204 s.
