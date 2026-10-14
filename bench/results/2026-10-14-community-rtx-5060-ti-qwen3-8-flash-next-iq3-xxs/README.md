# Community benchmark on NVIDIA GeForce RTX 5060 Ti

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 5700X 8-Core Processor (8 cores, 16 threads); installed RAM: 64 GB DDR4 (62.7 GiB visible); board: ASUS TUF B550M-PLUS; storage: Acer N3500CN 256 GB NVMe (PCIe 3.0) holding the OS, GGUFs and packs
- OS: Ubuntu 24.04.5 Server (headless), Linux 6.8.0-146-generic; GPU driver: 580.178.04 (open); CUDA: 13.0
- Strata: project 0.1.44.1 from the release source zip (commit not a git checkout), engine built locally by `setup.sh` (sm_120); engine 0.1.44.1
- Background workloads and power limits: none intended (dedicated headless box); stock power limits

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf`, ModelScope)
- Vision encoder: mmproj BF16 on the CPU, 768 image tokens (loaded, not exercised by the suite)
- Context: 262144 tokens; parallel slots: 1
- Settings: not the install's defaults (the suite ran against an already running server, so the tool could not see them). Engine arguments after `setup.sh --calibrate` plus hand-added options:
  `--expert-cache auto --prefill auto:16384 --spec 4 --mtp <mtp> --max-context 262144 --kv int8 --vision --vram-reserve-mib 700 --pcie-frac 0.20 --spec-min-p 0.50 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97 --kv-resident 32768 --conversation-cache-mib 6144 --conversation-cache-slots 4`
- Environment: `STRATA_ADAPT_OVERLAP=1` (opt-in since 0.1.43.1; on this machine it measured faster than off with our own decode test)
- Thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 74.6 (64.7-93.2) | 0.45 (0.42-0.52) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1321.3 (1185.7-1327.5) | not measured | 3.06 (3.05-3.41) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1834.9 (1732.8-1834.9) | not measured | 8.78 (8.78-9.29) |
| agent session, follow-up turns | 14597 (14565-14607) at the last turn | see below | up to 64 | 3 | see below | 83.7 (59.2-102.2) | 3.25 (2.96-3.74) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 73 (2-73) (earlier turns that the cache did not hold; the 13729 (13727-13731) new tokens of the files are read once); first turn (nothing cached) first token: 1.46 s (1.46-1.46).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 422.0 | 1.464 | 67.8 |
| 1 | 3770 | 649 | 3121 | 872.5 | 3.599 | 84.5 |
| 2 | 5303 | 3833 | 1470 | 461.3 | 3.21 | 77.3 |
| 3 | 8747 | 5296 | 3451 | 929.6 | 3.738 | 88.8 |
| 4 | 11654 | 8792 | 2862 | 804.7 | 3.584 | 83.7 |
| 5 | 14597 | 11701 | 2896 | 853.6 | 3.422 | 65.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 123 s.
