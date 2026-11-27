# Community benchmark on Tesla V100-SXM2-16GB + Tesla V100-SXM2-16GB

Measured on 2026-11-27 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla V100-SXM2-16GB + Tesla V100-SXM2-16GB, 32 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz (22 cores, 44 threads); installed RAM: 125.7 GiB; storage: sda     SAMSUNG HD321KJ     sata; sdb     SAMSUNG HD502HJ     sata; zram0; nvme0n1 KINGSTON SNV3S4000G nvme
- OS: Linux 7.3.6-300.fc45.x86_64; GPU driver: 580.196.07; CUDA (driver supports up to): 13.0
- Strata: project 0.1.69, commit not a git checkout; engine hy4-serve-0
- GPU detail (after the run, near idle): power_limit_w=300.00, power_draw_w=43.18, pcie=gen3 x16, pstate=P0, temp_c=47
- GPU detail (after the run, near idle): power_limit_w=300.00, power_draw_w=172.28, pcie=gen3 x16, pstate=P0, temp_c=61
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: hy4-flash-sq-96-Q8
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
component:strata-hy4 --native Hy4-Flash-SQ-96.strata --engram Hy4-Flash-Engram-Q8.strata --max-context 131072 --session-reuse 1 --vision Hy4-Flash-Vision.strata --layer-split auto --vram-experts-gb 16.18 --ram-experts-gb 66.75
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 52 (49-57) | 0 | 100 (100-100) | 9 | not measured | 18.8 (15.6-21.0) | 0.78 (0.75-0.84) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 322.2 (321.5-323.2) | not measured | 12.81 (12.77-12.84) |
| 16K prompt, fresh | 16527 (16527-16527) | 0 (0-0) | 24 cap | 3 | 316.9 (312.8-317.2) | not measured | 52.19 (52.13-350.50) |
| agent session, follow-up turns | 14318 (14316-14319) at the last turn | see below | up to 64 | 3 | see below | 15.6 (14.5-17.8) | 15.63 (7.85-333.01) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.8 % (60.3-70.8); earlier-turn tokens read again per session: 1 (0-4863) (earlier turns that the cache did not hold; the 13524 (13524-13524) new tokens of the files are read once); first turn (nothing cached) first token: 6.49 s (6.44-6.49). Corpus: `repo_docs` (8ba45cf16bd6) [README.md, docs/HOW_IT_WORKS.md, docs/DETAILS.md, docs/INSTALL.md, docs/MODELS.md].

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 523 | 0 | 523 | 80.6 | 6.492 | 14.0 |
| 1 | 4802 | 553 | 4249 | 193.3 | 22.001 | 15.7 |
| 2 | 6315 | 4864 | 1451 | 182.2 | 7.979 | 15.6 |
| 3 | 9397 | 6376 | 3021 | 187.8 | 16.116 | 15.8 |
| 4 | 11477 | 9460 | 2017 | 236.7 | 8.546 | 17.8 |
| 5 | 14319 | 11532 | 2787 | 178.7 | 15.628 | 14.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## GPU load notes

Both engines log `--layer-split active across 2 GPUs` splitting only the EXPERT arena (SQ-64: GPU1=9 / GPU0=33 owned layers; SQ-96: 8 / 34). The trunk (attention for all 42 layers, embed, dfly2 draft) stays on the primary GPU (engine `--gpus 1,0` -> primary = physical GPU 1), so during the suite GPU 1 ran ~95% while GPU 0 sat at ~4%: it only serves the 16.18 GB of VRAM-resident experts, while 39.86 GB (SQ-64) / 66.75 GB (SQ-96) of experts are CPU-resident. The two V100s sit behind a PCIe host bridge (no NVLink; engine probe: `p2p=0/1, bounce=1`). The prefill numbers are therefore effectively single-GPU trunk throughputs; `--layer-split` does not partition attention in Hy4.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 1493 s.
