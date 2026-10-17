# Community benchmark on Quadro RTX 6000 + Quadro RTX 6000

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Quadro RTX 6000 + Quadro RTX 6000, 48 GiB VRAM
- CPU: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz (40 cores, 80 threads); OS-visible RAM: 247.2 GiB; model shards and packs on NVMe storage
- OS: Linux 7.0.0-38-generic; Ubuntu 26.04.1 LTS; GPU driver: 595.91.07; source build with CUDA toolkit 12.4. The benchmark auto-detected CUDA 13.2 from the driver, which is its supported CUDA capability, not the toolkit used to compile this engine
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads: Hermes gateway paused for the benchmark and restored afterward; no intentional concurrent inference or synthetic memory/CPU stress tests. Other OS processes remained running. GPU power limits: 260 W per card.
- NUMA: node 0 approximately 157 GiB, node 1 approximately 90 GiB; both GPUs on node 0. Server launched under `numactl --interleave=0,1`; arena and PLE approximately 50/50 across nodes. 39 CPU expert-pool workers plus the host thread.
- Source-built CUDA engine for sm_75 (Release, CUDA 12.4), with the existing ggml checkout; CPU vision helper retained from the earlier install. `STRATA_ROUTE_TAIL_SKIP=7` active by release default, so answers can differ from 0.1.40.3.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
numactl --interleave=0,1 strata --serve --layer-split auto --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --ple-io ram --vision --vram-reserve-mib 700 --remote-expert-opt
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.7 (58.0-76.4) | 0.55 (0.47-0.66) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 840.8 (839.9-841.1) | not measured | 4.81 (4.81-4.82) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1542.0 (1540.7-1548.6) | not measured | 10.46 (10.41-10.47) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1796.7 (1791.6-1797.4) | not measured | 17.89 (17.89-17.94) |
| agent session, follow-up turns | 14488 (14458-14499) at the last turn | see below | up to 64 | 3 | see below | 70.5 (58.9-87.5) | 5.19 (4.68-5.33) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 4 (2-73) (earlier turns that the cache did not hold; the 13632 (13632-13634) new tokens of the files are read once); first turn (nothing cached) first token: 3.41 s (3.35-3.42).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 183.5 | 3.346 | 60.5 |
| 1 | 3704 | 621 | 3083 | 602.5 | 5.153 | 62.1 |
| 2 | 5237 | 3767 | 1470 | 316.1 | 4.687 | 60.1 |
| 3 | 8681 | 5300 | 3381 | 638.8 | 5.334 | 62.7 |
| 4 | 11558 | 8698 | 2860 | 555.6 | 5.192 | 58.9 |
| 5 | 14458 | 11614 | 2844 | 549.7 | 5.223 | 63.2 |

Per-request data: `results.json`. The server was already running, so the tool did not capture a new startup excerpt. Sanitized engine timing excerpts for the custom comparisons are included under `upgrade/` and `numa/`.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 229 s.

## Additional controlled comparisons

See [EXPERIMENTS.md](EXPERIMENTS.md) for a two-launch-per-version 0.1.40.3 versus 0.1.43 A/B and a separate NUMA placement comparison on 0.1.40.3 after a RAM upgrade. Those use a custom corpus and must not be combined with the standard-suite medians above. Per-request data, engine timing lines, placement summaries, sanitized configuration, and the custom corpus script are included. The largest custom input was 43,440 tokens; no 128K or 262K input was benchmarked.
