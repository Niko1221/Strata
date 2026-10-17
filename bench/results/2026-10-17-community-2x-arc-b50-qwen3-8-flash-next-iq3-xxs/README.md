# Community benchmark on 2x Intel Arc Pro B50

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges), plus the upstream needle test. One bare-metal host, two Arc Pro B50s sharing the layers of a single Flash-Next IQ3_XXS session. Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed plus recall, not a general quality evaluation.

## Hardware and software

- GPU: **2x Intel Arc Pro B50**, Battlemage G21 (`8086:e212`), nominal 16.2 GB VRAM each, `xe` kernel driver, as `level_zero:0` and `level_zero:1`. The automatic collector lists them as "Intel Arc B50" and cannot read total VRAM on SYCL; per-card VRAM figures are in the startup excerpt below.
- CPU: AMD Ryzen 9 5950X, 16 cores / 32 threads. RAM: 60.7 GiB installed. Storage: single local NVMe (LVM) holding model, packs and logs.
- OS: Ubuntu 26.04.1 LTS, kernel `7.0.0-38-generic`. GPU stack: `libze-intel-gpu1` / `libze1` / `intel-opencl-icd` 26.05.37020.3-1, Level Zero loader 1.28.2, Intel oneAPI 2026.1.1, native Release build. **JIT, not AOT** — this host has no `intel-ocloc`, so no `bmg-g21` ahead-of-time build.
- GPU firmware: `21.1182` on both cards (FWCODE), OptionROM `23.1066.0.0`.
- Strata: project/engine 0.1.45, commit `477ac13cbe8e582c20484871ce459d20d82df09d`. Source build, no container or VM for the engine itself.
- Background workloads and power limits: none recorded; this was a single-purpose host, not a shared server. Idle power and peak VRAM were not captured for this run.
- Layer split: `--layer-split 25` — CUDA0 runs layers 0-24, CUDA1 layers 25-47 plus the head, one hand-off per window. Both cards are required; there is no single-card path.

## Model and configuration

- Model and quantization: **qwen3.8-flash-next-iq3_xxs** (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`); both GGUF shard filenames are in the command below. Shard 2 (the n-gram lookup table) is byte-identical across quantizations and is stored once, shared by hard link with the IQ2_XS build on this host.
- Context: **262144 tokens** (the model's native window), parallel slots: 1
- Vision: off. Custom expert profile: none — the shipped `expert-profile-sycl.bin`. MTP draft layer from the packaged `rt` bundle. Calibration and experimental speed projection: off.
- Settings: the install's own configuration, unchanged (sycl backend); thinking off for the suite; temperature 0

**Memory plan at 262,144 context** (from the engine's own startup lines; `--kv-resident 32768` keeps the whole KV in pinned host RAM and only the attended window in VRAM):

```text
strata generate: KV streaming: 32768 of 262144 cells per QSA layer in VRAM, the K/V in 1.55 GiB of pinned RAM
strata generate: 8035 slots, 11.70 GiB of VRAM (CUDA0 expert cache)
strata generate: 6136 experts of the later stages are not in VRAM (served by the host)
strata generate: 7.11 GiB mirrored in pinned host memory (STRATA_MIRROR_MIB=12288)
strata serve: 1969 MiB of VRAM free with everything loaded
```

`STRATA_VERIFY_NO_HOST=0` is **required** here: the second layer-split stage has 6,136 experts the host serves, and with the flag on the engine refuses to start. Raising the context from 65,536 to 262,144 cost only 103 host-served experts and 1 MiB of free VRAM, because the KV never leaves pinned RAM.

```text
strata --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile-sycl.bin --expert-cache auto --prefill 1024 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --stream-experts --vram-reserve-mib 2048
```

Environment of the config: `SYCL_CACHE_PERSISTENT=0`, `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_STAGER_THREADS=12`, `ONEAPI_DEVICE_SELECTOR=level_zero:*`, `STRATA_STAGE_TRIM=1`, `STRATA_VERIFY_NO_HOST=0`, `STRATA_MIRROR_MIB=12288`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not benchmarked (short input) | 30.8 (28.9-40.2) | 0.89 (0.80-1.01) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 607.0 (603.7-609.7) | not benchmarked (short output) | 6.67 (6.64-6.71) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 678.1 (677.0-680.3) | not benchmarked (short output) | 23.74 (23.66-23.78) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 662.0 (661.4-662.6) | not benchmarked (short output) | 48.48 (48.43-48.53) |
| agent session, follow-up turns | 14575 (14541-14613) at the last turn | see below | up to 64 | 3 | see below | 33.3 (17.2-40.7) | 8.71 (5.30-10.15) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

**Prompt throughput falls off with prompt length on this configuration**, which the suite's prompt lengths do not reach:

| Actual prompt tokens | Prompt tok/s | Context configured |
| ---: | ---: | ---: |
| 4,019 | 607 | 262,144 |
| 16,058 | 678 | 262,144 |
| 32,049 | 662 | 262,144 |
| 258,039 | **379** | 262,144 |

The 258K figure is from a separate single measurement with a fresh prompt (`cache_n=0`), not from this suite. The cause is sparse-attention block lookups growing with the accumulated context, not the KV cache, which stays resident. Decode is unaffected (~27-33 tok/s at every length).

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.9); earlier-turn tokens read again per session: 4 (3-74) (earlier turns that the cache did not hold; the 13717 (13717-13718) new tokens of the files are read once); first turn (nothing cached) first token: 3.15 s (3.15-3.17).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 195.6 | 3.166 | 23.5 |
| 1 | 3744 | 624 | 3120 | 339.1 | 9.257 | 32.1 |
| 2 | 5277 | 3807 | 1470 | 276.0 | 5.379 | 32.8 |
| 3 | 8710 | 5341 | 3369 | 333.8 | 10.149 | 32.3 |
| 4 | 11628 | 8766 | 2862 | 336.2 | 8.569 | 33.3 |
| 5 | 14541 | 11642 | 2899 | 332.0 | 8.786 | 17.2 |

Per-request data: `results.json`. The memory-plan excerpt quoted above comes from the live engine log; host, configuration, version and SHA-256 facts are in `host-and-run-facts.json`. See `REPRODUCE.md` for commands and what was and was not anonymised.

## Correctness and limitations

**Needle-in-a-haystack recall** (`tools/needle_bench.py`, one request per cell, greedy, thinking off, synthetic haystack built from this repository's own text):

| Length | Depth | Actual prompt tokens | Found | Client seconds |
| --- | ---: | ---: | --- | ---: |
| 32k | 10 % | 32792 | yes | 79 |
| 32k | 50 % | 32792 | yes | 79 |
| 32k | 90 % | 32793 | yes | 79 |
| 128k | 10 % | 127285 | yes | 328 |
| 128k | 50 % | 127283 | yes | 328 |
| 128k | 90 % | 127283 | yes | 172 |
| 262k | 10 / 50 / 90 % | — | **request refused, HTTP 400** | — |

**The 262k rows are the tool's 2%-under-length sizing colliding with a full 262,144 window**, not a failure to serve long prompts: the same configuration was separately measured with a **fresh 258,039-token prompt, which completed in 682.6 s at 379.5 tok/s and returned all three exact anchor values from the document**; an over-length request is rejected with HTTP 400 instead of crashing. Needle requests share one haystack per length, so their timings may include prefix-cache benefits; the tool does not record reused tokens.

**Also verified before the suite:** a streamed arithmetic and tool round-trip smoke test — 6 times 7 returned 42, a forced synthetic `read_file` call and supplied result produced the exact final answer. It accessed no real files.

Limitations:
- Speed and recall only; no reasoning, coding or general quality claim.
- Short-decode runs are capped at 100 tokens and depend on draft acceptance, so the pooled chat median does not generalise.
- The conversation uses the model's own replies, so token counts differ slightly between runs.
- Peak VRAM, GPU power draw and thermals were not captured for this run; average power in a separate 45-minute long-prompt soak was ~64 W per card with both cards plateauing at 60-67 °C.
- Total time of the speed suite: 447 s. The needle test ran separately.
