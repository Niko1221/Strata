# Community benchmark on NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti

Measured on 2026-10-20 by caolonghao with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

The re-measurement of this machine after the 0.2.0-beta tracker reset: the same pair ran the same model on 0.1.40.2 for the earlier report (2026-10-08; a `benchmark.py` harness, see the comparison at the end).

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti, 44 GiB VRAM (the server's GPUs 2 and 3) -
  256-bit TU102 cards with 22 GB memory mods (as in #1225), sm_75, PCIe Gen3 x16 (the engine's probe measured
  13.1 GB/s host->device).
  The pair carries an NVLink bridge (the driver reports both links up, 25.8 GB/s each), but the engine does not
  use it: its split search prices the cross-card path from a host->device PCIe probe (13.1 GB/s), as on the
  TITAN RTX pair in 2026-10-08-community-2x-titan-rtx-0.1.40.3. Power limit **250 W** each (max 280 W);
  #1225 ran the same mods at a 100 W cap.
- CPU: Intel(R) Xeon(R) Silver 4316 CPU @ 2.30GHz (40 cores, 80 threads); installed RAM: 125.5 GiB; storage: shared NVMe (other load present)
- OS: Linux 6.8.0-138-generic; GPU driver: 580.178.04; driver reports CUDA 13.0, engine built with the **CUDA 12.8 toolkit** (gcc 11.4)
- Strata: upstream `main` at `f57c0adc` (engine 0.1.45.3, 9 commits past `v0.2.0-beta`), **source build**
  with `-DCMAKE_CUDA_ARCHITECTURES=75`; this PR branch adds only this report folder
- Background workloads and power limits: GPUs otherwise idle, the suite ran alone on a freshly loaded server

## Model and configuration

- Model and quantization: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at setup's pinned revision `ed59f92`,
  **IQ3_XXS** (two shards, 75.8 GB); setup's default pack for the qwen family, its own MTP draft layer and
  expert profile; no custom draft vocabulary; vision off
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged - `--expert-cache auto`, `--prefill auto`, MTP
  `--spec 4 --spec-min-p 0.5`, `--kv int8 --kv-resident 32768`, `--remote-expert-opt`, `--layer-split auto`,
  CPU expert pool at its default worker count (18); no calibration, no experimental speed projection, no
  low-RAM mode; thinking off for the suite; temperature 0

```text
strata --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --remote-expert-opt
```

## Method

Run by `python tools/strata_bench.py --url http://127.0.0.1:8080 --config strata-iq3_xxs.json --long` (the recorded `--long` command in short) against the HTTP API of a Strata server started beforehand by its run script; model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 77.9 (73.4-106.2) | 0.39 (0.36-0.44) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1032.5 (1028.4-1032.6) | not measured | 3.92 (3.92-3.94) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1490.5 (1484.1-1494.8) | not measured | 10.81 (10.78-10.86) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1745.1 (1739.7-1745.7) | not measured | 18.42 (18.42-18.48) |
| agent session, follow-up turns | 14601 (14552-14617) at the last turn | see below | up to 64 | 3 | see below | 77.5 (53.2-91.5) | 3.39 (2.19-3.89) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.3-69.4); earlier-turn tokens read again per session: 3 (3-74) (earlier turns that the cache did not hold; the 13744 (13744-13745) new tokens of the files are read once); first turn (nothing cached) first token: 1.45 s (1.41-1.48).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 439.5 | 1.411 | 59.3 |
| 1 | 4042 | 624 | 3418 | 918.7 | 3.749 | 82.5 |
| 2 | 5575 | 4105 | 1470 | 665.4 | 2.24 | 83.6 |
| 3 | 9008 | 5568 | 3440 | 891.8 | 3.889 | 70.7 |
| 4 | 11747 | 9064 | 2683 | 833.5 | 3.251 | 89.9 |
| 5 | 14617 | 11810 | 2807 | 838.8 | 3.379 | 66.9 |

Per-request data: `results.json`. Engine log lines from the start (the server was started by its run script, so the excerpt is cut from the engine log by hand): `engine-log-excerpt.txt` - the auto split chose the same placement as 0.1.40.2 (CUDA0 layers 0-25, CUDA1 26-47 + the head, 8,410 of 11,264 profiled pairs on CUDA1).

## Against the same pair on 0.1.40.2 (2026-10-08)

The earlier report used the community `benchmark.py` (fresh nonce prompts, 256-token cap); this one uses `tools/strata_bench.py` (fresh tagged prompts, 24-token cap on the prompt cells). Both take prompt throughput from the engine's own timings with `reused: 0`, so the prompt cells compare engine to engine; the decode cells do not (different output caps and cases) and are left out.

| Prompt tokens | 0.1.40.2 prompt tok/s | 0.1.45.3 prompt tok/s |
| ---: | ---: | ---: |
| 4,096 | 981.3 (960.1-986.2) | 1,032.5 (1,028.4-1,032.6) |
| 32,768 | 1,684.4 (1,677.7-1,703.8) | 1,745.1 (1,739.7-1,745.7) |

Same model, pack, settings and machine; +5.2% and +3.6%. Against **#1225** (the same 22 GB mods at a 100 W cap, 0.1.40 at `82f46a8`, benchmark.py harness): its 4K decode 52.8 and 32K prompt 477 tok/s, against this pair's 0.1.40.2 84.1 and 1,684.4 - the +59% / +253% comparison of the first submission, unchanged and still 0.1.40.2 data.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; storage background load is unmonitored; power limits are read from `nvidia-smi`, not an external meter. `results.json` records commit `4f7612f1` - a report-only branch whose tree is `f57c0adc` plus `bench/results` files. One earlier suite attempt on the same hardware was discarded before any cell was kept: started in `--url` mode without `--config`, the tool had labelled the machine from the wrong GPU; no number in this report comes from it. The suite also ran in `--url` mode because the tool's own server-start path crashes (upstream issue #3188). Total time of the suite: 186 s.

Prepared with an AI assistant (Claude Code) from the raw measurements; the numbers are the engine's and the client's own output, unedited.
