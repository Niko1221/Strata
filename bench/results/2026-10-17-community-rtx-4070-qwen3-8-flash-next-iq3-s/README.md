# Community benchmark on NVIDIA GeForce RTX 4070

Measured on 2026-10-17 by @leographia with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070, 12 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 93.7 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 617.42; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads: no heavy background workloads, per contributor. GPU power limits: default, per contributor.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --pcie-frac 0.35 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not reported | 66.1 (58.7-75.7) | 0.50 (0.46-0.66) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1825.2 (1809.1-1831.8) | not reported | 2.25 (2.22-2.27) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1794.1 (1793.4-1795.8) | not reported | 9.01 (9.00-9.01) |
| agent session, follow-up turns | 14595 (14575-14598) at the last turn | see below | up to 64 | 3 | see below | 69.7 (58.2-79.9) | 2.30 (2.16-2.38) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 73 (4-74) (earlier turns that the cache did not hold; the 13718 (13717-13719) new tokens of the files are read once); first turn (nothing cached) first token: 1.86 s (1.82-1.86).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 338.3 | 1.824 | 49.6 |
| 1 | 3743 | 623 | 3120 | 1403.4 | 2.267 | 68.4 |
| 2 | 5276 | 3806 | 1470 | 689.9 | 2.156 | 67.6 |
| 3 | 8709 | 5339 | 3370 | 1468.1 | 2.346 | 71.0 |
| 4 | 11635 | 8772 | 2863 | 1276.8 | 2.297 | 79.9 |
| 5 | 14595 | 11696 | 2899 | 1294.0 | 2.287 | 73.2 |

Per-request data: [results.json](results.json), preserved byte-for-byte from the benchmark output. No startup-log excerpt was generated because the suite used an already-running server.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and default power limits are contributor-reported rather than instrumented. Total time of the suite: 113 s.

## Submission notes

- Native Windows engine, Strata 0.1.45 at commit `477ac13cbe8e582c20484871ce459d20d82df09d`. Run the repository's `python tools/strata_bench.py` at that commit to reproduce the suite and its fixtures.
- This is the contributor's tuned configuration, not an unmodified-default performance claim. PCIe share 0.35, spec-min-p 0.70, MTP, adaptive expert cache and experimental speed projection are enabled; exact flags appear above and in the JSON.
- Engine snapshot: 1810 cached experts / 3563 MiB, 388 MiB VRAM free, 7 CPU pool workers. These are reported snapshots, not peak memory measurements.
- Model repository/revision and binary-build provenance beyond the reported version/commit were not recorded. Storage and PCIe link characteristics were not captured by this suite.
- The generated summary's unreported cells are labeled "not reported" here. Raw JSON retains decode timings for the 4K/16K requests, but their 24-token caps target prefill rather than representative decode speed.
- All 33 measured requests succeeded; zero failures. Three sessions contain six agent turns each, with follow-up medians over 15 turns. Tools are simulated exchanges, not external tool execution.
- Concurrency, long-context recall, vision correctness and general answer quality were not tested. Context 262144 is the configured limit; the largest fresh measured prompt was 16058 tokens.
- This submission contains one saved suite; it does not include earlier A/B comparisons or claim an isolated setting advantage.
- Original results.json SHA256: `31749e12b0908b3b45171da879549bb1b2c3a0f333992aa7bb1d109736b76d8c`.