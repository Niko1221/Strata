# Repeat-run operator notes

Measured by stekiclaw on 2026-10-28. Run `20261028T164913Z-f6edd148` is a fresh-engine repeat on the same machine, not independent hardware reproduction. The earlier run `20261028T051058Z-091dbac0` is retained separately in PR #4457. No selection of a best run and no claimed speedup.

## Protocol and outcome

Protocol `strata-qwen38-thinking-v1-draft.1` remains draft / pilot validation pending. Runner status: `complete`. Required duration: 453.642000 seconds, including the official warmup but excluding loading, model hashing and host probing. Wrapper wall time: 530.098475 seconds. Individual installation/load/hash durations were not separately instrumented in this repeat. Optional concurrency was not selected. All scheduled outcomes remain in results.json/outputs.jsonl. Performance requests reaching their prescribed output cap: 4/9. Sanity: 12/12. A complete status does not imply all answers passed or that maintainers approved the result.

## Effective configuration

Unmodified official Strata 0.2.7 macOS ARM64 Terminal release, commit a42cc3d7df70d1940de82cd0882f03e1f5dfc741, engine 0.1.52-metal. Existing setup-generated original Flash-Next Q2_0 configuration: 32,768 context, one slot, expert-cache 64, prefill auto, MTP `--spec 2`, `--mmap-experts`, `STRATA_METAL_STREAM=0`. Vision is ENABLED and uses the GPU helper; no image requests were sent. Effective config.json and engine.log are authoritative. This repeat does not carry the earlier run's incorrect 'vision off' operator note. No tuning, engine modifications or configuration changes were made.

The runner uses the protocol's fixed prompts, seeds, sampling, output caps, timeout, scoring and order. Sampling fields in evidence describe the values requested; separate backend sampler-parity testing was not run. Historical issue #3006 is not proof of the measured release's behavior. Missing source-weight, tokenizer and model-card revisions are preserved as null with reasons, not inferred from current repository heads.

## Hardware and background workload

Apple M1 Max, 10 CPU cores, 32 GPU cores, 64 GiB unified memory, macOS 26.6.2 build 25G83, internal APPLE SSD AP1024R NVMe, AC power. CPU/GPU share one pool; do not add shared GPU memory to RAM. The automatic storage probe may remain null; allowlisted host snapshots supply storage identity separately.

Daily-driver desktop, not an isolated host. Chrome, Safari, Adobe background services, WindowServer, Tailscale, oMLX and OS activity remained present. The benchmark sent no oMLX inference. oMLX health reported zero loaded models before the run. Background CPU/GPU utilization was not controlled. Applications were not killed. Both Strata profile/source configuration and oMLX top-level JSON hashes were unchanged after the run; oMLX post-run health was healthy.

## Resource observations

See resource-observations.json for actual timestamps, interval coverage, highest observed individual process RSS and system swap. Private telemetry sampled loading, hashing and inference approximately every five seconds. RSS is not an exact peak or a dedicated GPU allocation counter; do not sum shared mappings across processes. Swap belongs to the entire daily-driver system, not exclusively to Strata. GPU power, clocks, temperatures and exact peak allocation were not measured. pmset snapshots report only what the OS exposed, not proof of a constant temperature or absence of throttling.

## Interpretation and limits

Per-case summaries and all samples are in README.md/results.json. TTFT includes first reasoning or generated tool output; first-answer latency is separate. Null first-answer samples remain missing, never zero. Cap-truncated performance requests remain in throughput summaries; cap-truncated sanity answers fail. Prompt throughput uses freshly processed engine token counts and engine prompt duration; decode throughput uses accepted reasoning plus answer counts and engine decode duration, not total request time. Warm expert and OS page caches are not reset between requests; fresh-engine does not mean cold OS caches or zero-cache measurements.

This is one configuration with three repetitions per performance case, not a broad quality/coding evaluation. Not run: optional concurrency, long-context checks beyond the protocol, image tests, sustained soak, additional models/quants, same-machine A/B optimization comparison, CUDA/HIP/SYCL builds or hardware tests. No source changes are proposed. Approval, protocol activation and independent replication remain separate from local validation and PR submission.

## Startup diagnostics

The full original sanitized engine.log is retained. Keyword-selected diagnostics (not measured bandwidth or capability assertions):

```text
strata generate: PCIe probe failed -> pcie_frac default 0.55
```

## Evidence and privacy

The official artifacts are copied byte-for-byte and their SHA-256 values rechecked. host-start.json and host-end.json use an explicit allowlist; resource-observations.json contains aggregate numeric observations, not private process lists. No model weights, generated packs, private host/process captures, host identifiers or credentials are submitted. Independent verifier output and official validator output accompany the report after verification. AI assistance was used to prepare and check the report, not to generate measurements.
