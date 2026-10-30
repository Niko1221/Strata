# Community benchmark on AMD Radeon RX 6600M

Measured on 2026-10-30 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 6600M, 8 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i5-13600 (14 cores, 20 threads); installed RAM: 95.7 GiB; storage: GeIL P3P 1TB SSD/NVMe
- OS: Windows 11 (10.0.26100); GPU driver: 32.0.21045.1000; CUDA (driver supports up to): not read
- Strata: project 0.1.54, commit 6a2c7432; engine 0.1.53
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 8192 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
component:strata-engine --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 8192 --pool-workers 9
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 8.1 (7.5-10.7) | 2.51 (2.26-3.03) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 46.0 (45.7-50.2) | not measured | 87.77 (80.55-88.26) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Skipped and failed

- skipped 16K: the context is 8192 tokens
- skipped agent: context 8192 is below ~10K

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Notes from the reporter

- The two version numbers differ by design. The server is the tree this report was submitted with (project 0.2.8,
  commit `6a2c7432`, on top of `870aedd9`); the engine binary is the one built by hand for this card from commit
  `46c152a1` (`strata 0.1.53`, ROCm 10.2.0a20260930, gfx1032). `results.json` has `engine_build: null` because the
  configuration names the engine as a Terminal component (`component:strata-engine`), not a path with a
  `BUILD.json` beside it.
- The card, the build and the other measurements on this machine: [docs/RX6600M.md](../../../docs/RX6600M.md).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 452 s.
