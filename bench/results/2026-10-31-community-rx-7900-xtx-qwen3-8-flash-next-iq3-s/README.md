# Community benchmark on AMD Radeon RX 7900 XTX

Measured on 2026-10-31 by @str0g with `tools/strata_bench.py` (3 runs per test, medians with ranges, `results.json` schema 1). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run. This is the quick suite, not the `strata bench run` protocol: the longest arm here is 16K rather than 32K or 128K, and nothing was compared against another build or setting on this machine.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX (gfx1100, device id 0x744c), 24 GiB VRAM (24,560 MiB read by the benchmark). The Ryzen iGPU (about 2 GiB of VRAM) is in the machine and was not used (`gpu 0`). PCIe link speed and width: not read.
- CPU: AMD Ryzen 7 9800X3D (8 cores, 16 threads); installed RAM: 91.7 GiB; storage: a 2 TB Crucial CT2000MX500SSD1 on SATA and two 2 TB NVMe drives (IRP-SSDPR-P44A-4K0-80, WDS200T1X0E-00AFY0). The model sits on NVMe; the benchmark does not record which drive.
- OS: Arch Linux, kernel 7.2.11-arch1-1. GPU driver: the in-kernel `amdgpu` of that kernel (`rocm-smi --showdriverversion` answers `7.2.11-arch1-1`); `amd-smi` is not installed here, which is why the benchmark left the driver line blank. ROCm 7.2.4 (`rocm-core 7.2.4-1`, `/opt/rocm/.info/version`), with `/opt/rocm/lib` on the loader path and its hipBLASLt. CUDA: not applicable, this is a HIP build. The driver and ROCm lines were filled in by hand on the same machine after the run.
- Strata: project 0.1.55, commit 5a934634; the running server reported engine 0.1.55. The engine binary is a source build, not a release binary: `engine/strata` from a source checkout, built 2026-10-31 12:00 by `setup.py`'s HIP path, with a `BUILD.json` stamp of `source: local-hip`, `archs: ["gfx1100"]`, `src: e038e1ce6c4b7854`, `version: 0.1.54`. The 0.1.54 in the stamp and the 0.1.55 reported at run time disagree; not investigated.
- GPU power limit and draw: not recorded, before or after the run. Background workloads: nothing was started on purpose beyond the desktop session, and no check for other GPU programs was taken during the suite.

## Model and configuration

- Model and quantization: Qwen3.8-Flash-Next, ISTA-DASLab's expert-pruned GSQ-RCO release in `IQ3_S` ([ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)), the set the Terminal pins as `qwen/IQ3_S` in `data/gguf_fingerprints.json`: 3.78 bpw overall, 3.331 bpw of experts, 83,606,638,080 bytes in two shards. Repository revision: not recorded by the benchmark; the files came from the install, and their hashes are not in this report.
- GGUF filenames: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (the `--native`) and `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (used as `--ple-gguf`).
- Vision: on, encoder on the GPU, `mmproj-Qwen3.8-Flash-Next-BF16.gguf`. Expert profile: the install's own `expert-profile.bin`, not rebuilt for this run. MTP: the install's `mtp/rt`. No other custom pack or profile.
- Context: 204,800 tokens; KV `int8` with a 32,768-token resident window; expert cache `auto`; prefill `auto`; `--vram-reserve-mib 700`; one parallel slot; no layer split and no low-RAM mode.
- As the server reported it at start: 8,216 expert slots holding 15,954 MiB of VRAM (all on this one card), prefill cap 8,192 tokens in 1 stage, conversation cache 8,192 MiB in 4 slots, RAM expert arena 47,962 MiB, PCIe share cap 0.55, 7 pool workers, 872 MiB VRAM free with the model loaded.
- Drafting and sampling: `--spec 4 --spec-min-p 0.5 --mtp rt` (the server's own info line lists `spec 6`, `mtp_max 4` and `lookup 3`; read as reported, not investigated). Temperature 0, otherwise the server's defaults. Thinking off, which is the suite's default and keeps the answers short.
- Calibration and experimental speed projection: none set. The suite sends one calibration request to size its word lists; that is the tool's prompt, not a model setting.

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 204800 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700
```

The command is the install's own configuration, unchanged (HIP backend), shortened to file names; local paths, the host name and the port are left out.

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: a system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)

One engine session for the whole suite, 149 s of request time. The engine fills its GPU expert cache while it starts (see `engine-log-excerpt.txt`), so every measured request ran with a filled cache; the cache was not reset between runs. The 4K and 16K arms are fresh prompts each time (`reused_tokens` is 0 in `results.json`); the agent arm is the only one meant to reuse a prefix. The `--clients` and `--long` arms were not run, so there is no concurrency number and no 32K prompt. The prompts are the tool's own word lists and project-file text, reproducible from `tools/strata_bench.py`; the unique tag differs per run, so token counts move by a few tokens.

Timing boundaries: `prompt_tps` and `decode_tps` are the engine's own figures (`prompt_per_second`, `predicted_per_second` in the answer), and `prompt_ms` and `decode_ms` are kept per request in `results.json`. `ttft_s` and `total_s` are client-side, measured over HTTP on the loopback, and exclude model loading. Thinking was off, so the first token is answer text and not reasoning. The 4K and 16K arms reached their 24-token cap in every run and the chat arm its 100-token cap in all 9; the agent turns generated 24 to 64 tokens against a 64-token cap. No request failed and none was cancelled.

Memory, read once from the engine at start (a startup snapshot, not a peak during inference): 872 MiB VRAM free with the model loaded, 15,954 MiB in the expert cache, 47,962 MiB of RAM expert arena; the engine says it loads about 55 GB of experts into RAM and locks part of them for the GPU. Per-process peak RSS was not recorded. All 33 measured requests returned tokens (`failures` and `skipped` are empty in `results.json`), and the engine stayed up for the whole suite. The engine printed its usual note about keeping more VRAM free when the card also drives the desktop (`engine-log-excerpt.txt`); this run kept 700 MiB reserved, with 872 MiB left free with the model loaded.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 68.6 (57.1-78.4) | 0.47 (0.40-0.51) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 933.2 (933.0-936.1) | not measured | 4.33 (4.32-4.33) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 997.0 (995.2-999.5) | not measured | 16.13 (16.09-16.16) |
| agent session, follow-up turns | 12932 (12923-12946) at the last turn | see below | up to 64 | 3 | see below | 81.7 (73.0-97.4) | 3.26 (2.09-3.78) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token. Total latency over the whole suite was 149 s of request time, and per-request `total_s` is in `results.json` (chat 1.74 to 2.23 s, 4K 4.69 to 4.74 s, 16K 16.40 to 16.51 s, agent turns 2.25 to 4.56 s). Failures: none. The tool skipped no arm (`skipped` is empty); the 32K arm (`--long`) and concurrency (`--clients`) were simply not requested.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.3-69.5); earlier-turn tokens read again per session: 4 (3-74) (earlier turns that the cache did not hold; the 12065 (12063-12065) new tokens of the files are read once); first turn (nothing cached) first token: 1.93 s (1.88-1.94).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 327.1 | 1.884 | 59.8 |
| 1 | 3364 | 634 | 2730 | 838.4 | 3.28 | 79.2 |
| 2 | 4897 | 3427 | 1470 | 708.9 | 2.099 | 73.0 |
| 3 | 8141 | 4961 | 3180 | 849.1 | 3.769 | 80.1 |
| 4 | 10210 | 8205 | 2005 | 767.0 | 2.637 | 77.6 |
| 5 | 12946 | 10263 | 2683 | 818.2 | 3.302 | 77.4 |

Per-request data (tokens, both timings, `prompt_ms`, `decode_ms`, draft offers and accepts): `results.json`. Engine start-up lines: `engine-log-excerpt.txt`. Draft acceptance over the run sits between 9 and 71 accepted drafts of 19 to 113 offered, per request, which is part of why the decode range is wide (57.1 to 78.4 tok/s on the short chats).

## Correctness and limitations

- No correctness check was run: no needle/recall test, no coding task, no tool-call result, no image question. This report says how fast the machine answered, not whether the answers were right.
- No comparison: one build, one configuration, one engine session. Nothing was run against an older or newer engine, and no setting was varied, so no speed-up or regression is claimed here.
- Not tested: prompts beyond 16,058 tokens (the context is 204,800 and was never filled), concurrency (`--batch` was not started), thinking on, vision, a second GPU, and behaviour under load from another program on the card.
- The suite is short (24 to 100 new tokens per measured request), so decode over a long answer was not measured; the 100-token chats are the longest decode samples here.
- The conversation uses the model's own replies, so token counts differ a little between runs. Power limits, clock state and background load were not recorded, and the driver and ROCm lines were added by hand after the run.
