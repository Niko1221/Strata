# Strata standard benchmark: Quadro RTX 5000, Quadro RTX 5000

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261030T110159Z-2871c04a`, started 2026-10-30T11:01:59Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPUs: 2x Quadro RTX 5000 (16.0 GiB dedicated each, sm_75, no NVLink), layer split at layer 24 (CUDA0 runs layers 0-23, CUDA1 layers 24-47). Power limit 200 W per card. Measured host-to-device PCIe bandwidth to the second card: 12.3 GB/s (the engine's start-up probe, `engine.log`); link width and generation: not measured.
- CPU: Intel(R) Xeon(R) W-2133 CPU @ 3.60GHz; RAM: 94.0 GiB
- Storage of the model files: nvme SAMSUNG MZVLB512HAJQ-000H2 (nvme)
- OS: Ubuntu 24.04.4 LTS 7.0.0-38-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0, cuda-driver-supports 13.0
- Strata commit: `870aedd9e1ebe1c91f94da93b59af17729f6c5c0` (tag v0.2.8); engine 0.1.54; source build with `CMAKE_CUDA_ARCHITECTURES=75`, `STRATA_PORTABLE=OFF`, CUDA 12.9 toolkit. The binary sha256 starts `b70b04e504efb913` (133,290,480 bytes) and links `libcudart.so.12` / `libcublas.so.12` from that toolkit (`ldd`); the runner's `runtime: cuda 13.0` is what it read from the driver. The install's `engine-component.json` is the release manifest for the official 0.1.54 CUDA 13 build and does not describe this binary. The vision encoder runs on the CPU by design (`vision.gpu: false`), which is why the start of `engine.log` shows `ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected` from that separate process; the engine itself used both GPUs.
- Background work: headless Z4, no other GPU processes; Pi traffic to 8087 paused

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`IQ3_S/`), downloaded from `resolve/main` on 2026-10-08. The revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` was passed with `--model-revision` on the evidence that the repository has not changed since 2026-09-29 and its `main` LFS object ids for both shards equal the sha256 of the two files used here; the download itself did not record a commit. Original weights' revision: not measured.
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (release 0.2.8 (tag 870aedd9) with a local sm_75 CUDA 12.9 engine build 0.1.54; 2x Quadro RTX 5000 layer split 24, pipeline-windows 2, trim-stage-weights, int8 KV, MTP spec 4, vision on, 1 slot, STRATA_SM75_PERF=1 STRATA_EXP_MODE=8 STRATA_BF16_TC=1, 200 W cap)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json` (effective arguments of the run profile, recorded by the runner because it started the engine), the sanitized launch configuration in [provenance/run-config.json](provenance/run-config.json), and the engine's own log [engine.log](engine.log). The runner redacted the value of `--conversation-cache-min-tokens` as `<redacted>` (its filter treats names containing "token" as secrets); the value is 4000.
- Vision encoder `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 bytes, sha256 `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`), 6 CPU threads.
- Expert profile `data/expert-profile.bin`, 196,632 bytes, sha256 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf` (the same profile listed in the single-card Quadro RTX 5000 report, issue #3854). The `iq3_s` pack was built by the setup from shard 1 with `tools/iq_pack.py` (native experts, per the pack index header); the pack itself is not attached.
- MTP draft pack `--mtp <strata-data>/mtp/rt`, built by the setup from the model's own MTP tensors: `dense.bin` 116,099,072 B `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a`, `dense.txt` 1,851 B `991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd`, `draft_vocab.bin` 425,196 B `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`, `experts.bin` 707,788,800 B `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f`.
- KV: `--kv int8`, no KV streaming window flag set. Low-RAM mode: not set. CPU workers: not set (engine defaults). Prefill: `--prefill auto`. Expert cache `--expert-cache auto`: 5761 slots (10.36 GiB) on CUDA0 and 4488 slots (9.00 GiB) on CUDA1. Env `STRATA_SM75_PERF=1`, `STRATA_EXP_MODE=8`, `STRATA_BF16_TC=1`. No calibration, no experimental speed projection.

## Method

Command (run from the Strata checkout at the commit above, with the production server on this machine stopped first so the runner could own the port):

```text
strata.py bench run --profile <install>/strata-iq3_s.json --suite standard-v1 --track custom --track-notes "release 0.2.8 (tag 870aedd9) with a local sm_75 CUDA 12.9 engine build 0.1.54; 2x Quadro RTX 5000 layer split 24, pipeline-windows 2, trim-stage-weights, int8 KV, MTP spec 4, vision on, 1 slot, STRATA_SM75_PERF=1 STRATA_EXP_MODE=8 STRATA_BF16_TC=1, 200 W cap" --quantization IQ3_S --model-revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 --build-options "source build, CUDA 12.9, CMAKE_CUDA_ARCHITECTURES=75, STRATA_PORTABLE=OFF" --background "headless Z4, no other GPU processes; Pi traffic to 8087 paused"
```

The runner started the engine from that profile, waited for the model, ran the protocol and stopped the engine (`started_engine: true`, no request before the run). `engine.log` is the runner's capture of the engine's output; it holds exactly the 22 scheduled requests (22 `prompt N tokens` lines). Home paths in it are the runner's own `<user>` / `<strata>` placeholders. Artifact hashes written by the runner are in `results.json`; the files added after the run:

```text
2a679d24f9bd068a  provenance/run-config.json
b48b4e70c12bcba7  provenance/draft-acceptance-counts.txt
```

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 45 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 70.3 (66.9-72.3) n=3 | 1.15 (0.76-1.15) n=3 | 2.41 (2.05-2.72) n=3 | 5.2 (5.1-5.6) n=3 | 0/3 |
| prompt-4k | 1007.7 (1006.1-1016.4) n=3 | 62.3 (61.2-66.3) n=3 | 4.09 (4.06-4.10) n=3 | 9.86 (9.86-9.86) n=1 | 12.3 (11.8-12.4) n=3 | 3/3 |
| prompt-16k | 1542.5 (1534.5-1555.7) n=3 | 63.6 (58.7-64.5) n=3 | 10.66 (10.57-10.72) n=3 | 16.94 (16.94-16.94) n=1 | 18.7 (18.6-19.3) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

The runner's table leaves out the `short` prompt rate; `results.json` keeps it at 69.0 (68.8-104.4) n=3 tok/s.

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
- Repeat of an earlier run, disclosed for completeness: the same build and configuration was run once before against the already-running production server (`--url`, run `20261030T020746Z-94efe6ea`, status complete, 12/12 sanity). Medians there: prompt 1012.3 / 1547.6 tok/s and decode 61.5 / 61.8 tok/s at 4K / 16K; here 1007.7 / 1542.5 and 62.3 / 63.6. That run is not in this PR because `--url` on a split over identical cards listed one GPU instead of two (issue #4473, PR #4477); it is available if a reviewer wants it.
- Decode falls as the prompt grows: 70.3 tok/s on the ~80-token case, 62.3 and 63.6 tok/s at 4K and 16K. Speculative-decoding acceptance across this run's nine performance requests is 61.7% (1872 of 3033, from `engine.log`); the same server's everyday requests of 40,000+ input tokens accept 65.9% (27981 of 42446) and the earlier run's nine accepted 61.6% (2066 of 3356), all three sets in [provenance/draft-acceptance-counts.txt](provenance/draft-acceptance-counts.txt) and `engine.log` (numbers only, no prompt text). No claim is made about why: fixture prompts, the 512-token cap and the larger KV window all move it.
- 6 of 9 performance requests reached the 512-token cap while still reasoning, so their totals are time-to-cap.
- Clocks were not captured per request; the 200 W limit is recorded.
- The speeds describe a source build for sm_75, not an official 0.2.8 release binary.

## Report fields the runner could not fill

`results.json` carries each of these as `null` with the reason quoted here. Nothing else at report level is missing.

| Field | Runner's reason |
| --- | --- |
| `protocol.model_card_revision` | the profile's model-card revision is not pinned yet (protocol draft) |
| `run.optional_duration_s` | the optional concurrency stage was not selected (1 slot) |
| `model.source_revision` | not given: pass --source-revision |
| `model.tokenizer_revision` | not given: pass --tokenizer-revision (the tokenizer is the pack's own) |
| `model.custom_pack` | no custom pack or expert profile was declared by the operator (--custom-pack); the `iq3_s` pack, the expert profile and the MTP pack are named with hashes above |
