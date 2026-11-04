# Strata standard benchmark: NVIDIA GeForce RTX 3070 Laptop GPU

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261104T092300Z-d8aeab0c`, started 2026-11-04T09:23:00Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8.0 GiB dedicated)
- CPU: AMD Ryzen 7 5800H with Radeon Graphics; RAM: 63.4 GiB
- Storage of the model files: ssd SAMSUNG MZVLQ1T0HALB-00000 (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 617.14; runtime: cuda 13.4
- Strata commit: `06698c674673008395ba6e971580538d6b1ea6e0`; engine 0.1.56; release build
- Background work: Hermes agent session sharing the same loopback server (single slot, concurrency 1) plus the strata-vision.exe helper; nvidia-smi at start: 7801 MiB used of 8192 MiB, engine reports 0 MiB VRAM free with everything loaded

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (engine 0.1.56 official portable release build, build stamp commit 06698c674673008395ba6e971580538d6b1ea6e0 in engine/BUILD.json (source f-0291-rc1, CUDA 13.0, archs sm_75;sm_86;sm_89;sm_120, ptx, portable); host checkout Strata 0.2.6; hand-tuned launch: --pack <models>\packs\iq3_xxs (native IQ3_XXS-00001-of-00002 + Ple <models>\models\IQ3_XXS\...-00002-of-00002) + custom --expert-profile <install>\data\expert-profile.bin --expert-cache 2133; --prefill auto; --spec 4 suffix drafts (no drafter/MTP, mtp:false); --max-context 131072 --kv k8v4 --kv-resident 16384 --no-kv-grow; --pcie-frac 0.18; --conversation-cache-mib 8096; --vram-reserve-mib 64; --vision (strata-vision.exe, mmproj BF16); env STRATA_ROUTE_TAIL_SKIP=0 STRATA_ROUTE_PRIOR=1.5; engine already answered requests before the run)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 7 min 6 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 35.7 (33.6-37.0) n=3 | 1.53 (1.48-61.07) n=3 | 5.46 (4.82-62.91) n=3 | 12.4 (11.0-71.0) n=3 | 0/3 |
| prompt-4k | 503.3 (494.9-505.7) n=3 | 36.1 (35.1-36.2) n=3 | 8.19 (8.14-8.33) n=3 | 15.14 (15.14-15.14) n=1 | 22.5 (22.2-22.7) n=3 | 3/3 |
| prompt-16k | 549.0 (547.4-550.5) n=3 | 36.1 (34.9-36.6) n=3 | 29.93 (29.83-30.01) n=3 | 35.78 (35.78-35.78) n=1 | 44.0 (43.9-44.6) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 5 request(s) before this run.


---

## Submitter addendum (written by the person who ran it, not by `strata bench run`)

**What kind of run this is.** The engine was launched the way I actually run this machine every day: a hand-tuned `IQ3_XXS` Flash-Next + Ple pack (native `IQ3_XXS-00001-of-00002` + Ple `IQ3_XXS-00002-of-00002` experts) with a locally built `expert-profile.bin` (24576 ranked pairs) and a pinned expert cache of 2133 MiB, `--max-context 131072 --kv k8v4 --kv-resident 16384 --no-kv-grow`, `--prefill auto`, `--pcie-frac 0.18`, `--conversation-cache-mib 8096`, `--vram-reserve-mib 64`, `--spec 4` suffix drafts (no drafter/MTP downloaded). So this is a `custom` track reading: every runtime number here comes from my tuned set, not from release defaults. The launch line and the resolved status are in `provenance/launch-command.txt` and `provenance/engine-status.json`.

**Not a cold-model reading.** The engine had already answered 5 requests before the run and the model was 7 min 6 s into its loaded life; the expert cache and conversation cache were warm and evolving (the log shows `decode expert cache hit rate: 79.8-84.1%` for the benchmark requests). Every number is steady-state, no cold-model or cold-cache case was measured and none should be inferred from this folder.

**One contaminated sample.** `performance-short-r1` is not usable as a timing sample: the TTFT of 61.07 s and the total of 71.0 s were paid in the queue, not by the model. The agent session that shares this loopback server issued a 93582-token request (93191 of them reused, 2069 generated in 56.4 s) just before it: in `engine-log-excerpt.txt` that request's own line is 121291 and its report block runs through 121295, and `performance-short-r1`'s prompt line follows at 121296; the log shows the 93191-token conversation cache restored (1.29 GB) and parked again (95650 tokens) in between, which is what the short request waited behind. The engine's own timings for that same request are unaffected — 1640 ms of prompt and 9952 ms of decode, i.e. 47.0 tok/s prompt / 33.6 tok/s decode — and the two later short repetitions are clean (TTFT 1.48-1.53 s, total 11.0-12.0 s). Read the short case from `r2` and `r3`.

**Capped requests.** Six of the nine performance requests reached the 512-token output cap (reasoning included), so their `first_answer_s` is not available and their total time is the time to the cap; `prompt-4k` and `prompt-16k` have one usable first-answer sample each. Prompt speed should be read from 4K and 16K: 503.3 and 549.0 tok/s median, and decode is 36.1 tok/s median at both lengths, which is what an 8 GiB laptop card with this pack does.

**Corpus.** I compared the local `bench/prompts/standard-v1` tree against the published `MANIFEST.sha256` file by file: 35 of 35 entries match their recorded SHA-256, `BUILD.json` and the runner report record the same corpus id `e75eff8b9feea04f...`. No corpus deviation. The check output is kept in `provenance/corpus-check.txt`, the manifest and build stamp in `provenance/corpus-manifest.sha256` / `provenance/corpus-build.json`.

**What the engine log covers.** `engine-log-excerpt.txt` is the engine's own launch header plus the run window (log lines 121288-121404), which contains the warm-up, the nine performance requests and the twelve sanity checks of this run, interleaved with the agent requests that share the server. Request bodies of other conversations are not part of this extract.

**Not run.** The optional concurrency stage was not run: the protocol's `standard-v1` workload is single-slot on a single server and my server has one slot, so a concurrency stage here would measure the slot queue, not the machine. `--concurrency` was never passed to the runner. No request was retried, nothing was excluded from the summaries beyond what `results.json` reports.

**Other local context.** The 8 GiB card was full during the run (`nvidia-smi` at start: 7801 MiB used of 8192 MiB, engine reports 0 MiB free with everything loaded) and the `strata-vision.exe` helper was live; my agent session was on the same server through the single slot. The commit stamp was taken from `engine/BUILD.json` of the portable Windows CUDA 13 release build (`engine 0.1.56`, source `f-0291-rc1`, archs `sm_75;sm_86;sm_89;sm_120`), host checkout 0.2.6. Model file hashes: `provenance/MODEL_FILES.sha256`.

Measured on 2026-11-04 by SK-Hardwired.

**Artifacts and hashes.** `SHA256SUMS.txt` holds the SHA-256 of every file in this folder except itself, computed over the bytes as committed (LF endings). If your checkout converts line endings (Windows `core.autocrlf`), verify against the committed blob rather than the working file, e.g. `git show <rev>:bench/results/2026-11-04-standard-rtx-3070-laptop-gpu-d8aeab0c/results.json | sha256sum`.

