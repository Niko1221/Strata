# Community benchmark on AMD Radeon RX 7900 XTX

Measured on 2026-10-22 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GiB VRAM
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 cores, 32 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.32015.7029; CUDA: not read
- Strata: 0.2.2 (source zip of tag v0.2.2 set up with `setup.py --update`, not a git checkout); engine 0.1.47, the release's `strata-engine-0.1.47-windows-x86_64-hip-rdna3` package (SHA-256 checked against SHA256SUMS.txt)
- Background workloads and power limits: not recorded for this run (a re-run on 2026-10-22; see the folder README for why)

## Model and configuration

- Model and quantization: Swift 1.5 Qwen3.8-Flash-Next GSQ-RCO IQ3_S (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000N-of-00002.gguf`, the files setup downloads), MTP draft layer on
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack swift-iq3_s --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8
```

Environment: `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100500.txt` (from the config) and `STRATA_HC_Q8=1` (set by the launcher, as in daily use)

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- the server was started fresh for this model (`serve/server.py` with the install's config, port 8091) right before the suite; this model was re-run on 2026-10-22, the other two ran on 2026-10-21
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 70.9 (58.3-73.7) | 0.47 (0.40-0.50) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 956.1 (952.3-961.3) | not measured | 4.25 (4.21-4.25) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1207.0 (1192.4-1210.3) | not measured | 13.34 (13.31-13.51) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1172.4 (1171.1-1173.9) | not measured | 27.39 (27.36-27.42) |
| agent session, follow-up turns | 14593 (14592-14594) at the last turn | see below | up to 64 | 3 | see below | 69.3 (24.4-104.9) | 4.22 (4.12-4.32) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.5-69.6); earlier-turn tokens read again per session: 2 (2-73) (earlier turns that the cache did not hold; the 13730 (13730-13730) new tokens of the files are read once); first turn (nothing cached) first token: 3.17 s (3.13-3.19).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 193.1 | 3.186 | 49.8 |
| 1 | 4092 | 624 | 3468 | 826.5 | 4.244 | 63.7 |
| 2 | 5625 | 4155 | 1470 | 360.6 | 4.116 | 67.1 |
| 3 | 9058 | 5689 | 3369 | 811.1 | 4.198 | 73.9 |
| 4 | 11830 | 9122 | 2708 | 658.7 | 4.159 | 89.8 |
| 5 | 14594 | 11877 | 2717 | 661.5 | 4.15 | 69.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Needle recall (`tools/needle_bench.py --lengths 4k,32k,100k`, code word at 10/50/90 % depth, same server right after the suite): 9 of 9 found, see `needles.json`. Total time of the suite: 250 s.
