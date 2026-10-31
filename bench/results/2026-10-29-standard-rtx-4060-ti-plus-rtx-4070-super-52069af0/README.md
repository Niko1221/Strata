# Strata standard benchmark: NVIDIA GeForce RTX 4060 Ti, NVIDIA GeForce RTX 4070 SUPER

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261029T163739Z-52069af0`, started 2026-10-29T16:37:39Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4060 Ti; NVIDIA GeForce RTX 4070 SUPER
- CPU: AMD Ryzen 7 5700X 8-Core Processor; RAM: 63.9 GiB
- Storage of the model files: ssd Fanxiang S500PRO 2TB (nvme)
- OS: Windows 10.0.26300; drivers: not measured; runtime: not measured
- Strata commit: `55a557bb6d77facc73d291c3f9894289496c770c`; engine 0.1.52; release build
- Background work: Windows 11 desktop session: Edge, Notepad++, NVIDIA overlay, GameViewer and MiPcContinuity running; the Strata desktop app is connected to the same server; strata-vision runs on the CPU with 8 threads

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (desktop app setup (Strata terminal 0.2.4.1, engine 0.1.52 release binary, preset 'IQ3_XXS 256K'): context 262144, --kv int8 --kv-resident 32768, --expert-cache auto, --prefill auto, --spec 4 --spec-min-p 0.5, --mtp rt, layer split auto K=35 across both cards, vision on the CPU, conversation cache 8192 MiB / 2 slots)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 34 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 83.8 (82.6-85.8) n=3 | 2.09 (2.02-2.11) n=3 | 3.94 (2.91-4.33) n=3 | 8.0 (5.5-8.3) n=3 | 2/3 |
| prompt-4k | 588.3 (581.1-589.7) n=3 | 74.6 (68.9-85.9) n=3 | 6.98 (6.97-7.08) n=3 | 11.66 (9.27-14.05) n=2 | 13.8 (12.9-14.5) n=3 | 3/3 |
| prompt-16k | 1061.7 (1055.1-1062.4) n=3 | 71.7 (68.1-77.5) n=3 | 15.47 (15.45-15.56) n=3 | 22.20 (22.20-22.20) n=1 | 22.6 (22.1-23.0) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 1 | 8 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **12 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | pass | the expected JSON value |
| I2 | pass | the expected JSON value |
| I3 | pass | the expected JSON value |
| I4 | pass | the exact expected text |
| R1 | pass | the exact expected text |
| R2 | pass | the exact expected text |
| R3 | pass | the exact expected text |
| R4 | pass | the expected JSON value |
| T1 | pass | the expected tool call |
| T2 | pass | the expected tool call |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 56 request(s) before this run.

## Facts the runner could not read (added by the operator after the run)

`results.json` keeps every value below as `null` with a reason; none of it was filled into the JSON.

- Card memory and power limits, read on this machine with `nvidia-smi`: NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB, 220 W limit, PCI 00000000:0a:00.0, the main card; NVIDIA GeForce RTX 4060 Ti, 16,380 MiB, 165 W limit, PCI 00000000:09:00.0, the helper card and the one that drives the display. Both under WDDM.
- Driver and runtime: NVIDIA driver 610.88, CUDA user-mode driver 13.3. The engine is the release package for CUDA 13 (`0.1.52-windows-x86_64-cuda-cuda13`, build id `a42cc3d7df70+cuda13.0`, file sha256 `dbd0f9b692e9790c…`).
- Which install ran the suite: the desktop app's own control install (`strata 0.2.7`, build 0.1.53), which has no `.git` and carries a `COMMIT` file `55a557bb6d77…`. That is where `software.strata_commit`, `software.runner_commit`, `protocol.git_commit` and `build_kind: release` come from, and its `bench/prompts/standard-v1/MANIFEST.sha256` hashes to the recorded corpus `e75eff8b9feea04f…`. The track note I passed says "terminal 0.2.4.1", which is a second, older install (`0.1.49.1`, COMMIT `c33fa7ba6d0d…`) that is on the PATH but did not run this suite.
- The engine options a running server does not expose, from the start-up line of the engine session that served this run: `--serve --pack <pack>/iq3_xxs --native <IQ3_XXS shard 1> --ple-gguf <IQ3_XXS shard 2> --expert-profile data/expert-profile-iq3_xss.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <draft dir> --max-context 262144 --kv int8 --vision --conversation-cache-mib 8192 --conversation-cache-slots 2 --conversation-cache-min-free-mib 1024 --vram-reserve-later-mib 1024 --kv-resident 32768 --vram-reserve-mib 1152 --layer-split auto`.
- **One correction to the track note above.** It says `layer split auto K=35`; that number came from an earlier engine session on the same model, not from the session that served this run. That session chose **K=14**: `layer split auto: K=14 - predicted 37.4 ms per decode window; the caches hold 9802 of 24576 profiled pairs (~95.8% of the routed mass)`, a near tie with K=17 (37.6 ms). The 4060 Ti runs layers 0-13, the 4070 SUPER runs layers 14-47 and the head. The command line and `configuration.track_notes` in `results.json` are kept exactly as they were typed.
- The same start-up lines also give: PCIe probe 6.7 GB/s host to device (`pcie_frac 0.19`); no AVX-512, so the expert kernels run on AVX-2; expert arena 33,352 MiB locked, 8 GiB of it pinned under WDDM; 39.97 GiB read at 1.73 GiB/s; expert cache 6,987 slots (9.79 GiB), 2,507 of them on the 4070 SUPER; KV streaming 32,768 of 262,144 cells per QSA layer in VRAM plus 0.77 GiB of pinned RAM; the draft layer 835 MiB; 7 CPU pool workers, one per physical core; PLE read probe 89,383 rows/s.
- The suite went to the server the desktop app had been serving since 2026-10-30 00:12:37 local (2026-10-29T16:12Z); the runner did not start it. Windows and the programs listed in `hardware.background_workloads` were using the 4060 Ti for the display during the run, so neither card was idle. The optional concurrency stage was not selected.
- The probe that returned no card memory, no driver and no CUDA version is the runner's, not a limit of this machine: `nvidia-smi --query-gpu=index,name,memory.total,power.limit,driver_version --format=csv,noheader,nounits` answers normally here (16,380 and 12,282 MiB, 165.00 and 220.00 W, 610.88). I did not investigate why it came back empty for the runner.
