# Community benchmark on NVIDIA RTX PRO 4000 Blackwell

Measured on 2026-10-21 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 4000 Blackwell, 24 GiB VRAM
- CPU: Intel(R) Core(TM) i9-14900T (24 cores, 32 threads); installed RAM: 125.6 GiB; storage: not measured
- OS: Linux 6.18.57-gentoo-x86_64; GPU driver: 615.78.08; CUDA: not read
- Strata: project 0.1.46, commit c093ac26; engine 0.1.46
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-nvfp4-nvfp4-qad
- Context: 524288 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack pack --native dense.gguf --native-dense-gguf dense.gguf --ple-gguf ple-nvfp4.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 524288 --rope-scaling yarn --rope-scale 2 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.20 --spec-min-p 0.70 --pool-workers 7 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 46.1 (42.0-53.0) | 0.75 (0.68-0.87) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1455.8 (1328.6-1461.1) | not measured | 2.79 (2.78-3.06) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2415.6 (2373.7-2420.6) | not measured | 6.68 (6.67-6.83) |
| agent session, follow-up turns | 14708 (14694-14709) at the last turn | see below | up to 64 | 3 | see below | 56.4 (45.2-72.7) | 2.97 (2.78-3.33) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.5); earlier-turn tokens read again per session: 73 (4-74) (earlier turns that the cache did not hold; the 13828 (13827-13828) new tokens of the files are read once); first turn (nothing cached) first token: 1.92 s (1.91-1.94).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 318.9 | 1.941 | 34.3 |
| 1 | 4093 | 629 | 3464 | 1051.6 | 3.327 | 45.2 |
| 2 | 5626 | 4156 | 1470 | 508.6 | 2.926 | 46.8 |
| 3 | 9059 | 5689 | 3370 | 1022.9 | 3.333 | 56.0 |
| 4 | 11815 | 9107 | 2708 | 864.7 | 3.171 | 68.2 |
| 5 | 14694 | 11875 | 2819 | 917.6 | 3.108 | 51.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 132 s.
