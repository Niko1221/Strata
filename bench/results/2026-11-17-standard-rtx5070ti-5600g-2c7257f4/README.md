# RTX 5070 Ti / Ryzen 5600G: Strata 0.2.15 draft standard benchmark

Measured 2026-11-17T00:56:17Z UTC by @firetussin (Sonny's Labs). This is one half of a before/after CPU and platform comparison, using official release binaries in a **custom container**. Protocol `strata-qwen38-thinking-v1-draft.1` is **draft**; runner status **complete**, custom track. These are pilot measurements, not maintainer-approved standard results.

## Hardware and software

- GPU: NVIDIA GeForce RTX 5070 Ti, 15.92 GiB dedicated; recorded power limit 300 W.
- CPU: AMD Ryzen 5 5600G with Radeon Graphics; 10 logical CPUs available to the LXC and Docker workload. Installed RAM: 64 GB as recorded for this machine in the earlier report; current runner-observed usable RAM: 58.75 GiB. LXC allocation: 54 GiB; no Docker memory limit.
- OS reported by runner: Ubuntu 24.04.5 LTS, kernel 7.0.14-29-pve; workload runs in Docker inside Proxmox LXC. Host distribution details and model-storage throughput: not measured in this run.
- NVIDIA driver 620.70.06. The runner reports CUDA 13.5 from nvidia-smi; the selected engine package is a CUDA 13.0 build. These are different version sources.
- Strata 0.2.15 / Control 0.1.61; engine 0.1.60. Control and runner commit `c77572806996349b49995e4f98d41b5294d36a21`. Local container packaging; official release binary contents, no engine source modifications.
- Background desktop/host services were not controlled exhaustively. API inference counter delta was exactly 22, matching the suite; no extra inference was detected. Clocks, temperatures, transfer bandwidth, PCIe saturation and peak inference RAM/VRAM: not measured during these runs.

## Model and configuration

- Original Qwen3.8-Flash-Next IQ3_XXS, launched with family qwen. Exact immutable quantization/source/tokenizer repository revisions were not captured; see null values and reasons in results.json. Model and helper filenames and full SHA-256 hashes are retained there.
- CPU vision enabled; no images are included in this workload. Vision encoder identity/hash was not captured in the standard result. Release-supplied expert profile and draft vocabulary hashes are in results.json; no custom expert-profile training or pack tuning was performed.
- Context 131,072 tokens; INT8 KV; resident KV window 32,768; expert cache auto; prefill auto; capacity one. Effective drafting: DFlash2, spec 6, confidence 0.7. Original operator notes mention requested MTP spec4/min-p0.5; config.json records the effective settings used by both runs.
- Frozen standard-v1 sampling: thinking and preserve_thinking on; reasoning_effort xhigh; temperature 1; top_p 0.95; top_k 20; min_p 0; presence penalty 0; repetition penalty 1. No tuning or experimental speed projection.
- [Sanitized effective configuration](config.json), [packaging and launch recipe](../2026-11-17-standard-rtx5070ti-5700x-592c4aa5/container-recipe.md), [official package identities](provenance/packages.json), [operator disclosure](provenance/operator.json).

```sh
/opt/strata/terminal/bin/strata bench run --profile /data/install/strata-iq3_xxs.json --no-start --suite standard-v1 --track custom --track-notes 'Official 0.2.15 Control 0.1.61 + feed-selected CUDA engine 0.1.60 packages, local container (published image unavailable); current IQ3_XXS 128K INT8/32768 resident CPU vision MTP spec4 min-p0.5; no benchmark defaults altered; same public endpoint retained; request counters audited for concurrent client traffic.' --credential-env STRATA_BENCH_KEY --quantization IQ3_XXS --out bench-results/20261117T005617Z-2c7257f4 --output json
```

## Method

The unchanged upstream standard-v1 runner sent 22 requests over loopback: one 128-token warm-up, nine performance requests (short/4K/16K in three repetitions, cap 512), and 12 sanity cases (cap 2,048). Optional concurrency was not selected. No retries. Required duration including warm-up: **125.239 s**, excluding model loading and hashing.
The production model was already loaded and had answered 3 request(s) before the runner's baseline. The suite's corpus was not previously run in that loaded engine. There was no restart immediately before the suite and no OS cache flush; expert and filesystem caches remained warm. This differs from the draft protocol's strict freshly-loaded-engine procedure; read this as a warm-service pilot, not proof of protocol activation readiness. All performance KV reused-token counts were zero. Preserve this disclosure rather than claiming cold-engine measurements.
- Frozen fixtures: `bench/prompts/standard-v1` at the recorded commit; corpus hash e75eff8b9feea04f745520b392c4c17d97aefc1ba185e222aa6740efa8f68407. [Raw requests/outputs](outputs.jsonl) retain usage, engine timers, finish reasons, and client event offsets. Prompt and decode rates use engine time; TTFT is dispatch to the first nonempty reasoning/answer/tool delta. Reasoning tokens count toward decode and output caps. Total time is client request latency, including prompt and decoding; loading is excluded.

## Results

| Case | Actual input tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short | 77 | 0 | 512 | 3 | 125.98 (121.78–133.38), n=3 | 86.85 (84.73–87.35), n=3 | 0.63 (0.59–0.65), n=3 |
| prompt-4k | 4093 | 0 | 512 | 3 | 1301.76 (1300.19–1305.46), n=3 | 77.52 (77.15–81.83), n=3 | 3.17 (3.16–3.17), n=3 |
| prompt-16k | 16387 | 0 | 512 | 3 | 3552.12 (3546.82–3555.20), n=3 | 80.23 (77.03–85.94), n=3 | 4.64 (4.64–4.65), n=3 |

Short-prompt prefill is dominated by fixed request overhead; it is included for completeness, not sustained-bandwidth comparison. [Original runner report](runner-report.md) includes total latency, first-answer latency and every capped/early-EOS outcome. Cap truncation is retained; no continuation was requested.

## Correctness and limitations

- **12/12 sanity checks passed**: instruction 4/4, retrieval 4/4, tool call 4/4. No failed or skipped required request. Performance outcomes: 0 early-completed, 9 cap-truncated. See runner-report.md for first-answer exclusions.
- This suite measures up to approximately 16K actual input tokens despite 128K configured context. No full-context correctness claim is made. Custom long-context timing experiments are not part of this submission.
- One suite on each platform, three samples per performance case. CPU cores/cache/topology, available RAM and PCIe capability changed together; automatic planner choices and output/draft acceptance can influence rates. No isolated CPU-core, memory, or PCIe causal attribution is supported. No PCIe saturation measurement was made.
- Protocol/model-card/tokenizer pinning limitations remain explicitly recorded. Both native runner validation and independent raw-record reconstruction were checked; see the verification script and checksum manifests in the comparison report.

## Comparison

See the [5700X comparison report](../2026-11-17-standard-rtx5070ti-5700x-592c4aa5/README.md#before-and-after) and the [other CPU run](../2026-11-17-standard-rtx5070ti-5700x-592c4aa5/README.md).
