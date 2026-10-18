# Community benchmark on NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 4080 SUPER

Measured on 2026-10-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

**Run 1 of 2: before calibration.** This run uses the settings of a fresh install (`--calibrate` had not been run). The same machine was measured again the same day after `--calibrate`: [`2026-10-19-community-rtx-5060-ti-plus-rtx-4080-super-qwen3-8-flash-next-iq2-xs-2`](../2026-10-19-community-rtx-5060-ti-plus-rtx-4080-super-qwen3-8-flash-next-iq2-xs-2/README.md) (run 2).

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
- Settings: the configuration written by `START-HERE.bat --setup --yes --family qwen --model IQ2_XS --context 262144 --gpus 1,0` (setup's own default context for 16 GB cards is 65536, raised here by hand); no calibration; layer split `auto` across both GPUs; thinking off for the suite; temperature 0

```text
strata.exe --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 45.2 (28.0-61.5) | 0.73 (0.64-1.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 750.1 (732.4-777.2) | not measured | 5.38 (5.20-5.52) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 705.7 (556.3-1014.7) | not measured | 22.79 (15.87-28.90) |
| agent session, follow-up turns | 14558 (14528-14567) at the last turn | see below | up to 64 | 3 | see below | 44.3 (18.1-54.0) | 4.30 (2.69-6.80) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.8); earlier-turn tokens read again per session: 4 (2-4) (earlier turns that the cache did not hold; the 13718 (13717-13718) new tokens of the files are read once); first turn (nothing cached) first token: 1.71 s (1.68-2.10).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 361.9 | 1.706 | 36.1 |
| 1 | 3759 | 638 | 3121 | 676.9 | 4.642 | 51.6 |
| 2 | 5292 | 3822 | 1470 | 529.3 | 2.806 | 51.6 |
| 3 | 8725 | 5355 | 3370 | 708.2 | 4.796 | 36.5 |
| 4 | 11632 | 8770 | 2862 | 692.3 | 4.165 | 47.8 |
| 5 | 14558 | 11659 | 2899 | 680.4 | 4.303 | 45.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 218 s.
