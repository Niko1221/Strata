# Community benchmark on AMD Radeon RX 7900 XTX (gfx1100, eGPU)

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX (gfx1100, "AMD Radeon" as the ROCm device name), 24 GiB VRAM
- GPU connection: the 7900 XTX is **not socketed in this mini-PC** - it sits in an AOOSTAR AG03 eGPU dock (Sapphire Pulse card) on a separate box, reached over an **Oculink cable**. The engine matches the link at `pcie_frac 0.19`, i.e. roughly 1/4 of a desktop x16 PCIe 4.0 bandwidth, so prompt/decode transfers are Oculink-bound; expect slower rates than a socketed 7900 XTX report
- CPU: AMD Ryzen 7 H 255 w/ Radeon 780M Graphics (8 cores, 16 threads), ASRock NUC Box K12 mini-PC; installed RAM: 64 GB; storage: 1 TB and 4 TB M.2 SSDs; OS-visible RAM: 60.6 GiB
- OS: Fedora Linux, 7.2.10-200.fc44.x86_64; ROCm: gfx1100 device, hipBLASLt 10.0.2 (the tuning file `gfx1100-hipblaslt-100200.txt` matches the compiler target); the ROCm release and HIP versions were not read at run time; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded; this eGPU box also runs image and video generation through ComfyUI, which was not active during this run; monitors are connected through the iGPU

## Model and configuration

- Model and quantization: swift-1.5-iq2_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --pack swift-iq2_xs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100200.txt`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 86.1 (67.1-105.3) | 0.35 (0.30-0.39) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1098.4 (1092.2-1099.2) | not measured | 3.68 (3.68-3.70) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1389.7 (1383.6-1398.5) | not measured | 11.58 (11.51-11.63) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1411.9 (1405.1-1417.2) | not measured | 22.74 (22.65-22.84) |
| agent session, follow-up turns | 14570 (14539-14576) at the last turn | see below | up to 64 | 3 | see below | 85.4 (68.1-122.5) | 3.65 (3.04-3.95) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 52 (4-74) (earlier turns that the cache did not hold; the 13729 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 2.31 s (2.23-2.32).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 275.8 | 2.231 | 61.5 |
| 1 | 3744 | 623 | 3121 | 840.1 | 3.738 | 68.1 |
| 2 | 5277 | 3807 | 1470 | 487.2 | 3.042 | 79.6 |
| 3 | 8721 | 5340 | 3381 | 862.9 | 3.946 | 84.4 |
| 4 | 11637 | 8775 | 2862 | 801.6 | 3.6 | 98.2 |
| 5 | 14576 | 11677 | 2899 | 808.4 | 3.62 | 75.0 |

Per-request data: `results.json`. The server was already running when the suite started, so there is no `engine-log-excerpt.txt` in this folder. Note the matched PCIe link: the engine reports `pcie_frac 0.19`, so the Oculink path gives roughly a quarter of a desktop x16 bandwidth - expect slower prompt moves than a socketed 7900 XTX report.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 206 s.
