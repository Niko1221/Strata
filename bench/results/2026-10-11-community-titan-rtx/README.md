# Community benchmark on NVIDIA TITAN RTX

Measured on 2026-10-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA TITAN RTX, 24 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz (10 cores, 20 threads); installed RAM: 125.7 GiB; storage: NVMe (model and packs on /dev/nvme0n1p2)
- OS: Linux 6.1.183-1-MANJARO; GPU driver: 580.178.04; CUDA: 13.0
- Strata: project 0.1.43, commit d7953d8f (a working tree that merges upstream/main; the tree's diff to upstream/main touches setup/serve/tests only, no engine sources); engine 0.1.43
- Background workloads and power limits: GNOME desktop on the same GPU (~60 MiB VRAM), otherwise idle; power limit 280 W (default); nvidia-smi sampled during a separate decode run: 1965 MHz, 251-272 W, GPU util 91-100%, memory util 30-52%

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.33 --spec-min-p 0.70
```

Environment of the config: `STRATA_BF16_TC=1`, `STRATA_SELECT_SIMT=1`

These two switches were A/B-ed on this machine the same day with the same engine and the same suite (one restart per arm): 16K prompt 1,319 -> 1,615 tok/s (+22%), 4K 1,088 -> 1,141 (+4.9%), agent-turn first token 4.09 -> 3.96 s, decode within the run-to-run spread.

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.3 (59.2-69.6) | 0.53 (0.41-0.64) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1140.5 (1126.6-1144.9) | not measured | 3.55 (3.54-3.60) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1615.4 (1610.2-1622.7) | not measured | 9.98 (9.94-10.01) |
| agent session, follow-up turns | 14453 (14435-14489) at the last turn | see below | up to 64 | 3 | see below | 67.5 (55.2-87.9) | 3.96 (3.64-4.11) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.5-68.8); earlier-turn tokens read again per session: 74 (4-145) (earlier turns that the cache did not hold; the 13634 (13633-13636) new tokens of the files are read once); first turn (nothing cached) first token: 1.67 s (1.66-1.67).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 372.8 | 1.66 | 50.3 |
| 1 | 3712 | 631 | 3081 | 768.1 | 4.042 | 65.0 |
| 2 | 5245 | 3775 | 1470 | 407.5 | 3.639 | 65.7 |
| 3 | 8689 | 5308 | 3381 | 829.2 | 4.113 | 63.2 |
| 4 | 11594 | 8734 | 2860 | 720.2 | 4.01 | 77.4 |
| 5 | 14453 | 11607 | 2846 | 727.2 | 3.955 | 57.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 141 s.
