# Strata standard benchmark: NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3060

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261104T121115Z-0fed5a2f`, started 2026-11-04T12:11:15Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090; NVIDIA GeForce RTX 3060
- CPU: AMD Ryzen 7 5700X 8-Core Processor; RAM: 31.9 GiB
- Storage of the model files: ssd Samsung SSD 970 EVO 500GB (nvme)
- OS: Windows 10.0.26200; drivers: not measured; runtime: not measured
- Strata commit: `ffa80d73133ea5c548aefe57889a16e8085a938f`; engine 0.1.56; source build
- Background work: Qwen and whisper stopped; desktop on the 3060; nothing else on GPUs

## Model and configuration

- Model: qwen3.8-flash-next-iq2_xs; quantization IQ2_XS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` `92cee27ae5bbadcd...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **default** (Windows 11, RTX 3090 24GB + RTX 3060 12GB layer split auto, 32GB DDR4-3200, Ryzen 7 5700X, IQ2_XS full model on NVMe, --resident-experts, --pipeline-windows 2, draft vocab fr, engine 0.1.56 (app 0.2.10))
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 21 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 109.2 (96.3-117.1) n=3 | 0.67 (0.62-0.78) n=3 | 1.65 (1.44-5.34) n=3 | 3.2 (3.0-6.1) n=3 | 1/3 |
| prompt-4k | 866.5 (704.0-941.0) n=3 | 102.2 (91.9-122.1) n=3 | 4.76 (4.38-5.85) n=3 | 7.72 (6.32-9.12) n=2 | 9.7 (8.6-11.4) n=3 | 3/3 |
| prompt-16k | 1489.8 (1339.4-1591.4) n=3 | 108.9 (102.5-112.4) n=3 | 11.04 (10.34-12.26) n=3 | 13.53 (13.53-13.53) n=1 | 15.7 (14.9-17.2) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Submitter notes (written by hand, NOT measured by the runner)

These notes complete the `not measured` fields above and add practical findings. Everything the runner measured is in the sections above and in `results.json`. The only change made to the files the runner produced is the replacement of local file system paths by `<local-path>` (one place in `results.json`, the `--out` argument, and one in this README); line endings are unchanged (LF). `SHA256SUMS` lists the hashes of the files as submitted. Claims in this section that cannot be checked from the files of this folder are marked **(not verifiable here)**.

### Remarks on lines written by the runner (left unchanged above)

- **OS line:** the machine runs Windows 11 25H2 (build 26200). The runner prints "Windows 10.0.26200" because Windows 11 keeps the 10.0 version number.
- **"revision not measured" (Model line):** the repository and the revision are given in the "Model" section below.

### Hardware details the runner could not read

| Item | Value |
| --- | --- |
| NVIDIA driver | 617.42 |
| GPU 0 | GeForce RTX 3090 24 GB, power limit 350 W (stock 420 W) |
| GPU 1 | GeForce RTX 3060 12 GB, power limit 170 W (stock) |
| PCIe | both cards run at PCIe 3.0 x8 under load (AMD X470 chipset, the two CPU slots split x8 + x8; nvidia-smi reports max x16) |
| RAM | 32 GB DDR4-3200 (31.9 GiB visible) |
| Storage | model files on an NVMe SSD (Samsung 970 EVO 500 GB) |
| OS | Windows 11 25H2 (build 26200) |
| Display | the Windows desktop runs on the RTX 3060 |

### Model

- Original Qwen3.8-Flash-Next, IQ2_XS, ISTA-DASLab files (`Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{1,2}-of-00002.gguf`), downloaded 2026-10-07 from the Hugging Face repository `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (folder `IQ2_XS/`). The revision was not recorded at download time, but on 2026-11-04 I compared the SHA-256 of my two local files with the LFS hashes the Hugging Face API lists for that folder: both match (`92cee27ae5bb…` and `316b46f3a2dd…`, full values in the README above). The repository's last modification (2026-09-29) predates my download, and its head commit on 2026-11-04 was `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. So the files are byte-identical to the ones published there **(checked against the live API, not verifiable from this folder)**.
- The pack (`experts.bin`) and the MTP runtime were produced by setup from those files.

### Launch configuration (see `server-config.sanitized.json`)

`--resident-experts --pipeline-windows 2 --spec 4 --spec-min-p 0.5 --kv int8 --max-context 65536 --expert-cache auto --prefill auto --mtp-draft-vocab draft_vocab_fr.bin`, `--gpus 0,1`, `layer_split: auto` (Strata put the 3060 first and the 3090 last), `reasoning_effort: none` in the server config for the real-use runs. The standard benchmark sends its own `reasoning_effort: xhigh`, so thinking was on for the numbers above: that is why most performance requests reach the 512-token cap (decode speed is still the engine's own timing).

### Observations outside the standard protocol (my own logs, **not verifiable here**)

These come from my own runs and logs, which are not part of this folder. The only speeds that can be checked from this folder are the ones measured by the standard run above.

- With reasoning off in the server config, the decode speeds printed by the server during chat and agent-style use were typically about 105-130 tok/s.
- 27 French-language agentic runs (9 scenarios x 3: writing, multi-turn file edits, sandbox traps) through a local coding-agent harness: 25 passed. The 2 failures were model behaviour, not the engine: an unrequested file move after an ambiguous "yes, go", and swapped old/new arguments in an edit tool call followed by a tool loop that the harness stopped. French output was clean (no English, no calques) and tool arguments stayed in French.
- A raw streamed tool call returns `finish_reason: "tool_calls"` and a final `data: [DONE]` as expected (checked by hand).

### Pitfalls found on this setup (Windows, 32 GB, two cards)

1. **`UPDATE.bat` does not replace an engine that is already new enough.** From 0.1.52 it pulled the 0.2.10 code but kept engine 0.1.52 (the setup only updates an engine older than its `MIN_ENGINE`, 0.1.46). To get the 0.1.56 engine we set `version` to `0.1.45` in `engine\BUILD.json` and ran `UPDATE.bat` again; it fetched 0.1.56 from the signed feed and kept the old engine in `engine\.previous`. A documented way to ask for the latest engine would help.
2. `strata bench run --url` against an already running server works, but must be the first thing sent to a freshly loaded engine, or the run is marked as reusing the cache.
