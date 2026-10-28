# Community benchmark on NVIDIA GeForce RTX 3070: Flash-Next IQ3_S

Measured on 2026-10-28 with the unmodified upstream `tools/strata_bench.py`, suite 1, three repetitions. Prepared with AI assistance from real host measurements. Contributed by @stekiclaw.

This is the community quick speed suite described in `docs/COMMUNITY_BENCHMARKS.md`, not the thinking-enabled draft standard benchmark and not a maintainer-approved result. The original generated report is retained as `README.generated.md`, and unmodified per-request measurements are in `results.json`. This report supplements provenance and limitations without changing measurements.

## Hardware and software

- GPU: NVIDIA GeForce RTX 3070, 8 GiB dedicated VRAM; selected GPU 0.
- CPU: Intel Core i9-10900K, 10 cores / 20 threads.
- Installed RAM: 61.3 GiB.
- Model storage: Samsung PM9A1 NVMe, device model PM9A1 NVMe Samsung 1024GB.
- PCIe: Gen 3 x16 observed; queried maximum Gen 3 x16.
- GPU power limit: 220 W.
- OS: Linux kernel 7.0.0-38-generic.
- NVIDIA driver: 595.99.02; driver-reported supported CUDA version: 13.2. This is not evidence of the CUDA toolkit used to compile the engine; that build field was unavailable to this community runner.
- Strata CLI: 0.2.7, build 0.1.52, established separately with the installed CLI's `--version`.
- Runtime COMMIT file: `a42cc3d7df70d1940de82cd0882f03e1f5dfc741`; engine 0.1.52. Existing release installation; no source build or runtime modification was performed for this test.
- The legacy community runner did not recognize the Terminal bundle's project version or COMMIT file, leaving those fields null in the original JSON. The separately recorded COMMIT above does not alter that raw JSON.
- Community guide inspected at upstream revision `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`.
- Runner SHA-256: `f6267a67dd27a938af8f756b1bbbf49b232d9168f986889d09b85b5df4859f7d`.
- Background work: existing OS services and an auxiliary resource sampler. A preflight GPU process query showed only the owned Strata engine; no additional GPU workload was observed in that snapshot. This is not proof that all background activity was absent throughout the run.

## Model and configuration

- Served model ID: `qwen3.8-flash-next-iq3_s`; quantization: IQ3_S.
- Exact quantization repository/revision, original model revision and tokenizer revision: not established for this report. Do not infer these from filenames.
- Native file: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`.
- PLE file: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`.
- Prepared pack: `iq3_s`; expert profile: `expert-profile.bin`. Pack source identity/preparation revision is not established here.
- Effective context: 65,536 tokens; one parallel slot.
- KV: int8, resident window 32,768; VRAM reserve 1,024 MiB.
- Expert cache and prefill launch settings: auto. Observed engine information reported 813 expert slots / 1,606 MiB expert cache, 9 pool workers, prefill cap 1,536, PCIe fraction 0.34 and a 1,024 MiB conversation cache with four slots.
- MTP disabled in the serving profile; vision disabled. This does not mean all speculation is disabled: native draft counters are nonzero in some agent requests. No MTP-on/off or speculation-on/off comparison was performed.
- Sampling: temperature 0, thinking disabled (`reasoning_effort=none`, `chat_template_kwargs.enable_thinking=false`). Other sampling values were not explicitly fixed by this legacy runner.
- No persistent serving configuration changes were made for this test. Prior calibration/tuning provenance, low-RAM policy and experimental speed-projection state were not independently established and must not be assumed off.

Sanitized engine launch specification (the component selector is resolved by the installed Strata runtime, not a standalone executable path):

```text
component:strata-engine --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --max-context 65536 --kv int8 --vram-reserve-mib 1024 --kv-resident 32768
```

Sanitized benchmark invocation; `<runtime>` and `<resident-profile>` refer to the existing installation and saved profile, with private absolute paths omitted:

```text
<runtime>/python/bin/python3.12 -u <runtime>/tools/strata_bench.py --config <resident-profile> --url http://127.0.0.1:8080 --runs 3 --out <output-parent>
```

The complete sanitized configuration and observed engine information are in `results.json` under `machine.config` and `machine.engine_info`. Launch flags and effective engine information are reported separately; they should not be assumed identical merely because they use similar names.

## Method

The upstream suite ran on the inference host against its already-loaded loopback server, without restarting it for this community run. The expert cache was already warm from prior use. This is not a cold-model benchmark, nor a rerun of the draft standard protocol. No model download or engine upgrade occurred.

- Warm-up: one short chat, two one-token token-count calibration requests (200 and 800 words), and one approximately 1K-token prompt. These are excluded from measured request totals.
- Short chat: three upstream questions, each repeated three times, output cap 100 tokens (9 measured requests).
- 4K and 16K: three generated word-list prompts per target, output cap 24 tokens (6 measured requests). Actual inputs were 4,019 and 16,058 tokens respectively; labels describe targets, not exact input counts.
- Agent session: three sessions of six requests each, comprising an initial turn followed by five file-addition turns; four tool schemas are exposed, but the prompt asks for prose rather than tool execution. Each added file contributes about 1,500 words. The model's own answer is appended to subsequent turns; cap 64 tokens per turn (18 measured requests).
- Agent file construction uses project documents if available, otherwise generated filler. The exact per-run request bodies and full generated replies were not persisted by this legacy runner, so byte-identical replay of agent conversations is not claimed.
- Fresh chat and prompt requests have unique tags. All measured short/4K/16K rows reported zero reused tokens. Agent turns intentionally reuse conversation prefixes; expert-cache warming and conversation-prefix reuse are distinct effects.
- Required/default measured requests: 33, all returned tokens with no recorded error; no tests were skipped. Optional `--long` (32K) and `--clients` tests were not selected.
- Prompt and decode throughput use native engine timings, not generated tokens divided by total request time. TTFT is client monotonic time to the first nonempty content/reasoning piece, ignoring empty chunks. Thinking was disabled.
- Total suite time: 296.707 seconds, approximately 4 min 57 s; includes warm-up, calibration and machine probes, excludes model loading. Per-request wall times and engine prompt/decode durations are retained in `results.json`.

## Results

Values are median (minimum–maximum). These are measurements of this one configuration, not an A/B speedup claim.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Samples | Prompt tok/s | Decode tok/s | TTFT seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Short chat | 46 (43–51) | 0 | 100 in every request | 9 | not summarized | 46.0 (45.6–47.6) | 0.676 (0.560–0.726) |
| 4K prompt, fresh | 4,019 | 0 | 24 in every request | 3 | 411.2 (410.9–411.4) | not summarized by upstream report | 9.802 (9.800–9.807) |
| 16K prompt, fresh | 16,058 | 0 | 24 in every request | 3 | 367.9 (367.5–368.1) | not summarized by upstream report | 43.682 (43.661–43.727) |
| Agent follow-up turns | final turn 9,087 (9,085–9,094) | see below | up to 64 per turn | 3 sessions / 15 follow-ups | see raw rows | 58.5 (41.8–96.4) | 5.188 (5.016–5.240) |

### Agent session / conversation cache

- Follow-up token cache hit: 71.102% (71.097–71.107), across three sessions.
- Earlier-turn tokens read again: 5 per session in every session.
- New file tokens read: 8,207 (8,207–8,208) per session.
- Initial-turn TTFT: 2.510 s (2.392–2.516).
- Short, repetitive agent answers and varying accepted drafts can raise decode rates; the 96.4 tok/s maximum is not sustained general-purpose chat throughput.

First session in full; all three sessions are retained in `results.json`:

| Turn | Prompt tokens | Reused | Read | Prompt tok/s | TTFT s | Decode tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 609 | 0 | 609 | 258.3 | 2.392 | 42.2 |
| 1 | 2,274 | 632 | 1,642 | 328.7 | 5.024 | 44.6 |
| 2 | 3,980 | 2,337 | 1,643 | 324.1 | 5.104 | 44.5 |
| 3 | 5,686 | 4,043 | 1,643 | 317.7 | 5.205 | 45.3 |
| 4 | 7,392 | 5,749 | 1,643 | 318.2 | 5.195 | 67.3 |
| 5 | 9,094 | 7,452 | 1,642 | 316.0 | 5.225 | 58.0 |

### Resource observations: partial coverage only

The first auxiliary sampler failed to start because of command-string quoting. It was repaired without restarting the benchmark. The replacement recorded 53 samples over approximately 105.3 seconds near the end of the suite; these are not whole-suite resource peaks and do not reconstruct earlier activity.

- Sampling target: 2 seconds; scope: whole host, not Strata process RSS.
- CPU busy average in the sampled interval: 20.5%.
- Minimum available host RAM in that interval: 8.05 GiB.
- Sampled maximum GPU memory: 7,189 MiB; sampled maximum GPU temperature: 70°C.
- No swap-out counter increase in that interval; this does not mean swap was unused or no swap activity occurred earlier.
- The runner's post-run GPU snapshot reported 69°C and 172.18 W. This is an endpoint snapshot, not a peak or a rigorously verified idle reading.

## Correctness and limitations

- This community run is a speed/cache suite, not an accuracy evaluation. It performs no actual tool execution or complete external tool round trip.
- A separate thinking-enabled standard run is outside this contribution; its scores are not used to claim correctness for this thinking-off configuration.
- Short chat requests all produced the 100-token cap; long-prompt requests all produced 24 tokens. Hitting a cap is not proof of a complete answer.
- The legacy runner defines success as no recorded error plus at least one returned token; it does not retain terminal finish reasons or explicitly require `[DONE]`. Therefore “33 successful rows” is its transport/output classification, not independently verified natural-stop completion for all answers.
- No recorded request failures; no skipped default tests. Auxiliary sampler startup failed separately as disclosed above.
- Largest fresh input measured here: 16,058 tokens. A 65,536-token serving window is not a full-window capacity or recall pass.
- Optional 32K prompt, concurrency, long-context needle tests, broad reasoning/coding quality and baseline-versus-tuned comparisons were not run.
- Engine timing lines/startup logs are not attached by the community runner when using an existing server. Native timing fields remain in each measured JSON row. A separately sanitized log excerpt can be added to a publication bundle if available; its absence must not be disguised by fabricated logs.
- Model/source/tokenizer revisions and pack preparation provenance remain unknown; report them as unknown unless independent evidence is found.
- Original JSON and generated README are preserved. Supplemental version information comes from independent CLI/COMMIT probes, not from editing their original null fields.

## Artifacts and reproduction

- `results.json`: byte-identical output of the official community runner, all 33 measured request rows.
- `README.generated.md`: original generated report; its unrecognized version/COMMIT fields are explained above.
- `config.json`: sanitized effective profile copied from the original result.
- `benchmark-console.txt`: actual benchmark console output, private host paths replaced. This is not an engine startup log.
- `resources.partial.jsonl`: 53 whole-host samples from the replacement sampler only.
- `provenance.json`: independently observed version, runner identity and upstream guide revision.
- `SHA256SUMS`: hashes of the publication artifacts.

Use the unmodified upstream `tools/strata_bench.py` from the recorded runtime commit with its sibling imports (`ab_engine.py`, `agent_turns_bench.py`, `serve_load.py`) and an installed compatible Strata runtime/model. The command above reproduces the procedure, not byte-identical model responses. No custom engine code, model weights or packs are included.
