# Community benchmark on NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 4080 SUPER

Measured on 2026-10-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

**Run 2 of 2: after `START-HERE.bat --calibrate`.** The same machine was measured earlier the same day with the settings of a fresh install: [`2026-10-19-community-rtx-5060-ti-plus-rtx-4080-super-qwen3-8-flash-next-iq2-xs`](../2026-10-19-community-rtx-5060-ti-plus-rtx-4080-super-qwen3-8-flash-next-iq2-xs/README.md) (run 1). Same Strata commit (`477ac13c`), same model files, same suite. The only engine arguments that differ (from `engine_info` in both `results.json`) are `--pcie-frac` 0.55 to 0.90, `--spec-min-p` 0.50 to 0.70 and `--prefill-pipe` 0 to 3072; calibration kept the default expert-pool worker count.

| Median (range in each README) | Run 1, before | Run 2, after |
| --- | ---: | ---: |
| short chat, decode tok/s | 45.2 | 80.4 |
| short chat, first token s | 0.73 | 0.36 |
| 4K prompt, prompt tok/s | 750.1 | 1150.2 |
| 16K prompt, prompt tok/s | 705.7 | 1707.5 |
| 16K prompt, first token s | 22.79 | 9.44 |
| agent session follow-ups, decode tok/s | 44.3 | 72.6 |
| agent session follow-ups, first token s | 4.30 | 3.14 |
| suite total time s | 218 | 120 |

Not controlled: this is not an interleaved A/B. `--calibrate` restarted the engine several times before run 2, so the operating system's file cache and the background load were probably not the same in the two runs, and part of the difference may not come from the three settings. Run 1 also had a wider spread (decode 28.0 to 61.5 tok/s, 16K prompt 556 to 1015 tok/s) than run 2 (70.7 to 109.3 and 1701 to 1720). No repeat of either run was made.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 4080 SUPER, 32 GiB VRAM
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 cores, 32 threads); installed RAM: 61.6 GiB; storage: several NVMe SSDs (models not recorded)
- PCIe: setup's check read Gen 5 x8 for the RTX 5060 Ti at idle (the RTX 4080 SUPER's link was not read, and neither was re-read under load). The owner reports both GPUs run at x8 because the board's lanes are shared with the NVMe drives.
- OS: Windows 11 (10.0.26300); GPU driver: 616.56; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the configuration after `START-HERE.bat --calibrate --no-browser --yes` (it tuned `--prefill-pipe 3072`, `--pcie-frac 0.90`, `--spec-min-p 0.70`; its own synthetic measurement read 115.6 tok/s, not comparable to the suite below); context 262144 chosen by hand (setup's default for 16 GB cards is 65536); layer split `auto` across both GPUs; thinking off for the suite; temperature 0; experimental speed projection off

```text
strata.exe --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --pcie-frac 0.90 --spec-min-p 0.70 --prefill-pipe 3072
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 80.4 (70.7-109.3) | 0.36 (0.27-0.41) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1150.2 (1084.0-1270.7) | not measured | 3.52 (3.18-3.73) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1707.5 (1701.3-1719.6) | not measured | 9.44 (9.37-9.47) |
| agent session, follow-up turns | 14561 (14556-14568) at the last turn | see below | up to 64 | 3 | see below | 72.6 (55.8-83.3) | 3.14 (2.15-3.65) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.6-68.8); earlier-turn tokens read again per session: 5 (4-74) (earlier turns that the cache did not hold; the 13719 (13717-13719) new tokens of the files are read once); first turn (nothing cached) first token: 1.45 s (1.45-1.47).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 423.6 | 1.47 | 61.1 |
| 1 | 3757 | 637 | 3120 | 893.5 | 3.517 | 74.2 |
| 2 | 5290 | 3820 | 1470 | 645.1 | 2.302 | 83.3 |
| 3 | 8723 | 5283 | 3440 | 949.0 | 3.651 | 71.9 |
| 4 | 11629 | 8767 | 2862 | 886.4 | 3.258 | 77.6 |
| 5 | 14556 | 11657 | 2899 | 902.4 | 3.24 | 55.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 120 s.
