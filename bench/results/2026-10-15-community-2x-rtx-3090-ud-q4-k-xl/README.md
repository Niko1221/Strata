# Community benchmark on 2x NVIDIA GeForce RTX 3090 (layer split)

Measured on 2026-10-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP.

**What was tested.** A production Strata service, benchmarked against its own running HTTP API (`--url`), so the server was live for real traffic for the whole suite and nothing was restarted. This is the engine's release tree, not a branch: the configuration carries `--no-ple-prefetch`, a flag this deployment turned on after measuring a decode regression with the default per-row PLE prefetch hint on Linux — see [#2281](https://github.com/Niko1221/Strata/issues/2281) for the A/B and the mechanism. The numbers below are the configuration as it runs in production, not an experiment.

## Hardware and software

- GPU and VRAM: 2x NVIDIA GeForce RTX 3090, 24576 MiB each (sm_86), **GPUs 0 and 1** selected, power limit 350 W each; PCIe 3.0 x16 (measured 7.9 GB/s host-to-device, which is what the expert tier streams over). A third card, a Quadro RTX 4000 8 GB on GPU 2, runs a document-parsing service on another container and is not used by this run.
- CPU: Intel Xeon Platinum 8163 @ 2.50 GHz, 2 sockets x 24 cores = 48 cores / 96 threads, 2 NUMA nodes; installed RAM: 160.0 GiB
- Storage: ZFS pool on NVMe SSD (a 1.6 TiB dataset holds the model); the model is read once at load and the expert tier is served from RAM, so storage speed does not enter these numbers
- OS: Linux 7.0.14-23-pve (Proxmox host; the service runs inside an LXC container with `LimitMEMLOCK=infinity`); GPU driver 615.71.09; CUDA 13.4 (`libcudart.so.13`)
- Strata: project 0.1.44.1, engine 0.1.44.1 (self-reported by `strata --version`); source build from the release tree, not a git checkout, so no commit hash is recorded by the runner
- Background workloads and power limits: no other GPU workload on GPUs 0-1 during the suite; the 350 W default limit was in force; the host also runs the parsing service on GPU 2, which does not touch these cards

## Model and configuration

- Model and quantization: `unsloth/Qwen3.8-Flash-Next` UD-Q4_K_XL, 4 GGUF shards totalling 103.69 GiB (`Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf`). The Hugging Face revision is not recorded next to the files, so it is not stated here.
- Expert pack: 1.4 GiB (`packs/unsloth-ud-q4_k_xl`); MTP draft: `mtp/rt`; learned expert profile from production calibration
- Context: 262,144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --serve \
  --pack .../packs/unsloth-ud-q4_k_xl \
  --native .../models/unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --expert-profile .../data/expert-profile.bin \
  --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 \
  --mtp .../mtp/rt --max-context 262144 --kv k8v4 --kv-resident 32768 \
  --ple-io ram --no-ple-prefetch
```

At start-up the engine logged `PLE table locked in RAM in 2.2 s`, `resident RAM: 45.18 GiB of experts in RAM, 15631 exchanged with the VRAM tier`, and `caches hold 11774 of 24576 (~97.9%)`.

## Method

Run by `python tools/strata_bench.py --url http://127.0.0.1:8080 --runs 3 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- 32K prompt (`--long`): same generator, 3 fresh prompts
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)

Every speed figure below is the engine's own: all 15 requests carry `decode_tps_source = "engine"` and `prompt_tps_source = "engine"` in `results.json`. Total suite time 239 s.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 84.9 (76.2-106.8) | 0.39 (0.33-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 741.1 (740.5-741.7) | not measured | 5.46 (5.44-6.49) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1440.1 (1413.6-1440.7) | not measured | 11.20 (11.20-11.41) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1521.2 (1508.9-1521.4) | not measured | 21.12 (21.12-21.29) |
| agent session, follow-up turns | 14580 (14492-14603) at the last turn | see below | up to 64 | 3 | see below | 93.1 (74.3-121.5) | 5.57 (5.10-5.73) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.5-68.7); earlier-turn tokens read again per session: 4 (3-144) (earlier turns that the cache did not hold; the 13729 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 3.36 s (3.32-3.38).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 185.3 | 3.324 | 59.5 |
| 1 | 3741 | 620 | 3121 | 561.7 | 5.606 | 74.3 |
| 2 | 5274 | 3804 | 1470 | 289.4 | 5.11 | 78.7 |
| 3 | 8705 | 5325 | 3380 | 596.8 | 5.699 | 90.7 |
| 4 | 11631 | 8768 | 2863 | 518.7 | 5.558 | 99.9 |
| 5 | 14580 | 11681 | 2899 | 523.6 | 5.576 | 85.5 |

Per-request data: `results.json`. Engine start-up lines: `engine-log-excerpt.txt` (the benchmark did not start this server, so the excerpt was taken from the service log by hand).

## Correctness and limitations

Checks run against the same server on the same day, outside the speed suite (speed suite itself ran no correctness check):

- OpenAI `/v1/chat/completions`: a factual short question returned the expected answer, `finish_reason: stop`.
- Anthropic `/v1/messages`: works, and returns a `thinking` block plus a `text` block, as a reasoning model should.
- A message quoting a special control token as literal text (the `#2026` class of bug) was not truncated; `finish_reason: stop`.
- An ~8K-token prompt completed: 6342 prompt tokens, 101 generated, 95.9 decode tok/s, 1070.7 prompt tok/s.
- Service health: active, 0 restarts, no engine errors in the log.

Limitations:

- **Not a controlled comparison.** This is one configuration in production. The only reference point quoted from our own records is the same model on the same host one version back (v0.1.43, default PLE prefetch): a 420-token cold prompt went from 72.4 to 85.8 tok/s and the decode expert-cache hit rate from 94.6 % to 99.2 %. Those are single cold-start measurements, not a suite.
- **The agent session continues from the model's own replies**, so its token counts differ between runs and between engines; the 5.57 s follow-up-turn TTFT is not directly comparable to another engine's unless the per-turn token counts are matched. Decomposing it: this arm read a median 2899 tokens on the last turn at an effective 520 tok/s of incremental prefill.
- **The 4K prompt throughput (741 tok/s) is well below the 16K and 32K figures (1440 / 1521)**; the short-prompt end is where this configuration is weakest, and we have not isolated why.
- **`--no-ple-prefetch` is load-bearing here.** With the default, the same build measured 5.1 % slower decode and 19 % slower prefill in a 16-run A/B on a test host; the mechanism and the A/B are in #2281. A reader on a different OS should not assume the flag is neutral.
- Machine details come from the server, `nvidia-smi` and the OS. The suite's own hardware probe recorded the GPUs correctly; the PCIe measurement and the GPU-2 note were added by hand.
