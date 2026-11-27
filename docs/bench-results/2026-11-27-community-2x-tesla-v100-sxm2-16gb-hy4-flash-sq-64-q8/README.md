# Community benchmark on Tesla V100-SXM2-16GB + Tesla V100-SXM2-16GB

Measured on 2026-11-27 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla V100-SXM2-16GB + Tesla V100-SXM2-16GB, 32 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz (22 cores, 44 threads); installed RAM: 125.7 GiB; storage: sda     SAMSUNG HD321KJ     sata; sdb     SAMSUNG HD502HJ     sata; zram0; nvme0n1 KINGSTON SNV3S4000G nvme
- OS: Linux 7.3.6-300.fc45.x86_64; GPU driver: 580.196.07; CUDA (driver supports up to): 13.0
- Strata: project 0.1.69, commit not a git checkout; engine hy4-serve-0
- GPU detail (after the run, near idle): power_limit_w=300.00, power_draw_w=44.21, pcie=gen3 x16, pstate=P0, temp_c=47
- GPU detail (after the run, near idle): power_limit_w=300.00, power_draw_w=76.62, pcie=gen3 x16, pstate=P0, temp_c=67
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: hy4-flash-sq-64-Q8
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
component:strata-hy4 --native Hy4-Flash-SQ-64.strata --engram Hy4-Flash-Engram-Q8.strata --max-context 131072 --session-reuse 1 --vision Hy4-Flash-Vision.strata --layer-split auto --vram-experts-gb 16.18 --ram-experts-gb 39.86
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
| short chat | 52 (49-57) | 0 | 100 (22-100) | 9 | not measured | 21.9 (18.7-30.1) | 0.57 (0.56-0.63) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 451.0 (447.9-451.0) | not measured | 9.16 (9.15-9.22) |
| 16K prompt, fresh | 16527 (16527-16527) | 0 (0-0) | 24 cap | 3 | 436.0 (435.7-436.4) | not measured | 37.94 (37.90-37.96) |
| agent session, follow-up turns | 14318 (14315-14320) at the last turn | see below | up to 64 | 3 | see below | 22.0 (19.8-24.9) | 10.45 (5.30-229.21) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.8 % (45.9-70.8); earlier-turn tokens read again per session: 2 (1-11531) (earlier turns that the cache did not hold; the 13524 (13524-13525) new tokens of the files are read once); first turn (nothing cached) first token: 4.06 s (4.03-194.08). Corpus: `repo_docs` (8ba45cf16bd6) [README.md, docs/HOW_IT_WORKS.md, docs/DETAILS.md, docs/INSTALL.md, docs/MODELS.md].

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 523 | 0 | 523 | 128.7 | 194.081 | 26.2 |
| 1 | 4796 | 547 | 4249 | 289.6 | 14.693 | 22.5 |
| 2 | 6310 | 4858 | 1452 | 271.3 | 5.368 | 23.6 |
| 3 | 9390 | 6369 | 3021 | 280.6 | 10.789 | 19.8 |
| 4 | 11470 | 9454 | 2016 | 342.4 | 5.91 | 21.2 |
| 5 | 14315 | 11527 | 2788 | 267.4 | 10.453 | 21.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## GPU load notes

Both engines log `--layer-split active across 2 GPUs` splitting only the EXPERT arena (SQ-64: GPU1=9 / GPU0=33 owned layers; SQ-96: 8 / 34). The trunk (attention for all 42 layers, embed, dfly2 draft) stays on the primary GPU (engine `--gpus 1,0` -> primary = physical GPU 1), so during the suite GPU 1 ran ~95% while GPU 0 sat at ~4%: it only serves the 16.18 GB of VRAM-resident experts, while 39.86 GB (SQ-64) / 66.75 GB (SQ-96) of experts are CPU-resident. The two V100s sit behind a PCIe host bridge (no NVLink; engine probe: `p2p=0/1, bounce=1`). The prefill numbers are therefore effectively single-GPU trunk throughputs; `--layer-split` does not partition attention in Hy4.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 1345 s.
