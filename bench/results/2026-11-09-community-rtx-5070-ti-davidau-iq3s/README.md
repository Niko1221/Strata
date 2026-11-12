# RTX 5070 Ti on Windows: DavidAU IQ3_S, Strata 0.2.9 vs 0.2.12

This community report compares the installed releases on the same machine and model. The standard-v1 workload showed higher decode medians after the update, while the community suite showed lower chat and agent decode medians. Prompt throughput was nearly unchanged. These are observations from one before/after pair, not a general speedup claim.

Measured on 9 November 2026 in Asia/Krasnoyarsk (UTC+7). The standard runner records UTC dates on 8 November; the original run IDs and timestamps are retained. No submitter identity is inferred from the machine account.

## Submission classification

**Community report; custom configuration; draft-workload evidence, not an approved standard benchmark.** The model is a DavidAU derivative, not the original Qwen weights covered by the original-Qwen profile. Both standard-v1 files retain the runner-generated profile ID and status. Their `complete` status describes execution completeness and does not establish profile conformance. Do not pool these results with original-Qwen standard results. The draft protocol ID is `strata-qwen38-thinking-v1-draft.1`.

## Hardware and software

| Item | Recorded value |
|---|---|
| GPU | One NVIDIA GeForce RTX 5070 Ti, 16 GiB nominal VRAM (15.9 GiB reported) |
| CPU / RAM | AMD Ryzen 7 7700X, 8 cores / 16 threads; 63.1 GiB RAM reported |
| Model storage | Samsung SSD 980 500GB, NVMe SSD |
| OS | Windows 11 Pro, build 26200; runner OS version 10.0.26200 |
| NVIDIA driver | 621.14, unchanged |
| Driver-reported CUDA capability | 13.5; this is not a measured toolkit version |
| Engine package family | Official Windows x86-64 CUDA13 release binaries |
| Before | App/Terminal 0.2.9; Control/engine 0.1.55 |
| After | App/Terminal 0.2.12; Control/engine 0.1.58 |
| Before runner commit | `c62d5e3a15b40ea51c96aae9e8d36928965f53ab` |
| After runner commit | `9780a8c27d84f408240ed4b07062ab7275f02f09` |
| Background workload | Not recorded; not fully controlled |

The engine package build IDs recorded during the update were `99866a47a2b8+cuda13.0` and `8a42a9457db6+cuda13.0`. Runner commits and engine build IDs are different identifiers. Community snapshots report a 300 W power limit and PCIe gen5 x16 near the end of the runs; these are snapshots, not a time series or a controlled power experiment.

## Model and configuration

- Repository: [DavidAU/Qwen3.8-Flash-Next-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF](https://huggingface.co/DavidAU/Qwen3.8-Flash-Next-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF); recorded download revision `8d09a8a84cc067203d2903db831c8aa8713a5f3f`.
- Quantization: IQ3_S, two GGUF shards. The API ID is the generic `qwen3.8-flash-next-iq3_s`; model identity here comes from the recorded weight paths and SHA-256, not that ID alone.
- Context: 131072; KV: int8; capacity: two slots; measured benchmark requests are sequential. MTP on; vision off; no retuning during the comparison.
- The same prepared pack, expert profile and MTP runtime paths were used. Their content hashes and full preparation provenance were not captured in this comparison. The MTP component was reused from the existing Flash-Next installation, not freshly prepared from DavidAU in this experiment.
- Original source-weight/tokenizer/model-card revisions: not recorded. The quantization-repository revision above comes from the local download record and was not auto-detected by the runner. Raw JSON null values are preserved.

| GGUF | SHA-256 (identical before/after) |
|---|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | `8a14ab07ca7b54619e6b250cd1d83f97f79ab7eef3caf8ee9e6543bc9ac6a193` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

The effective engine command was observed with the same arguments in both versions; only the executable path/version changed. A portable transcription is in [configuration.md](configuration.md). The runner could not expose the full engine options: the raw config.json files intentionally retain null values. These observations supplement them and are not a fresh server inspection.

Community JSON also preserves the server's runtime engine_info snapshots. Automatic allocation was slightly different despite identical command-line arguments: expert_slots 1417 → 1416, expert_cache_mib 2797 → 2795. Both snapshots report prefill_cap 4864, pool_workers 7, conversation_cache_mib 1536, internal spec 6, mtp_max 4, batch_slots 2 and drafting mtp. Internal spec is a server-reported field and is not relabelled as the command-line --spec 4 argument. This was not a fixed-allocation or bit-identical runtime comparison.

## Method

One before arm, then an update and one after arm. Each arm ran standard-v1 followed by community --long. The before engine was already serving; the after engine was restarted. No order-swapped rounds were performed. The benchmark runners also changed with the release; this is an installed-release comparison, not an isolated engine-binary experiment.

### Draft standard-v1 workload

`strata bench run` sent one warm-up, nine performance requests (three repetitions each of short, 4K and 16K), then twelve sanity requests. No optional concurrency stage was run. A two-slot server is not a two-client benchmark. Output limits: 128 for warm-up, 512 for performance, 2048 for sanity; reasoning counts toward the cap. Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1, reasoning_effort xhigh; thinking enabled and preserved. Seeds and limits match across the arms.

Both corpora hash to `e75eff8b9feea04f745520b392c4c17d97aefc1ba185e222aa6740efa8f68407`; effective template SHA-256 is `6844d811f4e732d6e970dd286b051ea37dca3465c196051c91ecfd357c23c399`. The corpus-build metadata describes the bundle's build-time template and may name a different template hash; it is retained separately from the effective served-template hash, not treated as proof of upstream-template equivalence.

Rates below use the runner's engine timings. TTFT is measured by the client to the first qualifying reasoning/answer/tool delta. Warm-up and sanity are excluded from speed summaries. Required-run duration including warm-up: 123.594 s before, 120.985 s after; model loading and hashing are excluded. Both suites include the shared-prefix/expert-cache state described by their generated reports. Reaching a performance cap is recorded as truncated, not a transport failure.

| Metric (tokens/s) | Engine 0.1.55 median (min–max), n | Engine 0.1.58 median (min–max), n | Median change |
|---|---|---|---:|
| Short decode | 70.48 (68.09–72.26), n=3 | 73.55 (72.89–76.24), n=3 | +4.35% |
| 4K decode | 66.67 (62.72–68.99), n=3 | 69.07 (66.58–75.43), n=3 | +3.60% |
| 16K decode | 65.95 (64.37–69.58), n=3 | 70.76 (69.78–71.71), n=3 | +7.30% |
| 4K prompt | 2635.37 (2617.18–2638.26), n=3 | 2636.56 (2629.11–2645.08), n=3 | +0.05% |
| 16K prompt | 2772.05 (2770.84–2773.88), n=3 | 2782.03 (2780.52–2784.06), n=3 | +0.36% |

Both arms attempted all 21 measured requests plus warm-up, with no failed requests, and passed 12/12 sanity checks (4/4 instruction, 4/4 retrieval, 4/4 tool-call). Before: 2 completed and 7 capped performance requests; after: 3 completed and 6 capped. Missing first-answer times remain null and are not counted as zero. See [before](before-standard/README.md) and [after](after-standard/README.md) for TTFT, total latency, every cap and exclusion.

### Community speed suite

`tools/strata_bench.py --long`, three runs per test, temperature 0, thinking off. Short chats: nine measured requests, up to 100 generated tokens. Fresh prompts: 4019/16058/32049 actual tokens, 24-token output cap, three requests each, zero prefix reuse. Agent: three six-turn sessions; 15 follow-up turns contribute to decode summaries, with up to 64 generated tokens per turn. The conversation contains the model's own previous replies, so later inputs differ slightly between arms. No correctness score is derived from this speed-only suite.

| Metric (tokens/s) | Engine 0.1.55 median (min–max), n | Engine 0.1.58 median (min–max), n | Median change |
|---|---|---|---:|
| Chat decode | 67.10 (60.90–82.40), n=9 | 65.60 (63.30–86.90), n=9 | -2.24% |
| Agent follow-up decode | 77.10 (58.90–86.80), n=15 | 70.60 (59.30–87.50), n=15 | -8.43% |
| 4K prompt | 2587.70 (2509.70–2589.90), n=3 | 2590.20 (2504.40–2591.10), n=3 | +0.10% |
| 16K prompt | 2723.40 (2720.80–2726.80), n=3 | 2734.40 (2729.50–2736.50), n=3 | +0.40% |
| 32K prompt | 2921.50 (2920.10–2927.00), n=3 | 2929.20 (2927.10–2932.00), n=3 | +0.26% |

Agent follow-up cache hit: 71.09% before, 70.78% after. Median re-read tokens per session: 3 before, 74 after. All 15 measured follow-up turns in each arm returned metrics. The generated histories differ, so these cache/decode observations do not by themselves identify an engine regression. See the full [before](before-community/README.md) and [after](after-community/README.md) reports for ranges, timings and per-turn data.

## Failures, missing evidence and limits

- No HTTP/stream request failure in either completed benchmark arm. Performance truncations are included, not hidden. Small sanity checks do not establish overall model quality.
- Only one paired series, sequential before/after, no randomized order or confidence intervals. Temperature 1 standard outputs can differ despite fixed seeds. Background load, initial warm state and generated output differences can affect speed. Do not infer universal gains or regressions from these medians.
- Context capacity was 128K, but actual input was only tested to about 32K in community and 16K in standard-v1. Full 128K performance/recall was not tested. No llama.cpp result is included.
- The new release twice reported ready and answered its launch probe, but its background process was absent after the initiating SSH command ended. The measured after arm kept that SSH session open. Cause is unconfirmed. A later one-off Windows process launch survived session closure and answered a separate probe; this is an operational observation, not a benchmark sample or a proposed engine fix.
- Early pre-measurement attempts used an incorrect URL/profile identifier or a missing output parent. They did not produce measured suites and are not substituted for failed benchmark requests. After initialization was corrected, the completed runs below were retained without selecting best repetitions.
- The installed `bench compare` command returned `unsupported benchmark report` for these standard files; comparison values here are computed directly from saved summaries.
- Full startup logs, prepared-component hashes, a continuous memory/power trace and full raw SSE stream captures are unavailable locally. Standard outputs.jsonl contains request payloads, outputs, usage/timings and client event offsets; community JSON contains per-request metrics but not full generated conversation text. No missing logs or measurements have been reconstructed.

## Evidence and redaction

Each subfolder contains the generated report and per-request JSON; the standard folders also include outputs.jsonl, config.json and template/corpus provenance. Numeric measurements, generated outputs, run IDs and timestamps were preserved. The private output-directory prefix in standard run.command was replaced with `<RESULTS>/`; unrelated drive inventory was omitted from the community README and machine.storage fields. Source artifacts remain unchanged in the local project.

[sanitization.json](sanitization.json) records changed files and original/published hashes. [SHA256SUMS](SHA256SUMS) covers every other file in this report directory after redaction. It is not a maintainer approval manifest. All available standard artifact hashes were checked against the published copies.

References: [community submission guide](https://github.com/Niko1221/Strata/blob/main/docs/COMMUNITY_BENCHMARKS.md), [draft protocol](https://github.com/Niko1221/Strata/blob/main/docs/BENCHMARK_PROTOCOL.md), [0.2.12 release](https://github.com/Niko1221/Strata/releases/tag/v0.2.12).
