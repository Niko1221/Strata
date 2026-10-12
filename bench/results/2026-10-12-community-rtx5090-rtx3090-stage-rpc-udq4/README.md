# Community benchmark on RTX 5090 + RTX 3090, two-host stage RPC

Measured on 2026-10-12 by [@petria](https://github.com/petria), using original
Unsloth Qwen3.8-Flash-Next UD-Q4_K_XL and Strata 0.1.43 with custom, uncommitted
stage-RPC changes. This is a **results-only report of an experimental build**,
not a report of stock Strata network support or a change adding RPC to upstream.
No stock/single-GPU A/B comparison or coding-quality test was run.

## Hardware and software

The cards are in **two separate Linux guests**, not one local multi-GPU machine.
The coordinator executes layers 24–47, the output head, MTP and sampling; the
worker executes layers 0–23. Each guest keeps its own model and expert storage;
RAM is not pooled. Stage activations and control messages use a private link
with mutual TLS.

| Item | Coordinator | Worker / desktop |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 5090 | NVIDIA GeForce RTX 3090 |
| Nominal VRAM | 32 GiB | 24 GiB |
| VRAM reported before the suite | 32607 MiB | 24576 MiB |
| CPU model | Ryzen 9 9950X3D, 16-core model | Ryzen 9 5900X, 12-core model |
| Guest-visible logical CPUs | 32 | 24 |
| Guest-visible RAM | 42.6 GiB | 98.2 GiB |
| OS kernel | Linux 7.0.0-38-generic | Linux 7.0.0-38-generic |
| NVIDIA driver | 615.71.09 | 580.178.04 |
| CUDA compiler | 13.3.73 | 13.3.73 |
| CUDA architecture | sm_120 | sm_86 |
| Configured GPU power limit | 600 W | 420 W |
| Idle PCIe snapshot | Gen1 x16 | Gen1 x16 |

PCIe generations downshift at idle; the telemetry reports maximum generation 4,
but no separate sustained-load PCIe link test was performed. Storage is a
guest-visible ext4 virtual block device on each node; physical backing devices
and disk throughput were not measured for this run. The desktop and browsers
remained active on the worker; their memory is included in whole-card readings.

The measured engines were source-built, Release, CUDA enabled,
`STRATA_ENABLE_STAGE_RPC=ON`, `STRATA_MMQ_KQUANTS=ON`, native experts enabled,
portable mode off. Base commit:
[`7695859aa13afb248bdb4d36f9b2e95707db079d`](https://github.com/Niko1221/Strata/commit/7695859aa13afb248bdb4d36f9b2e95707db079d).
The custom source changes are not published by this report: base commit alone
cannot reproduce the RPC deployment. Binary hashes, matching stage fingerprint,
tool hashes and artifact hashes are in [provenance.json](provenance.json).

## Model and configuration

- Model: [unsloth/Qwen3.8-Flash-Next-GGUF, UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL),
  revision `38bb39ee97821de2c9009abb7e93950eec396e66`; original model, not Swift.
  Four shards, 111334654784 bytes together. This is a mixed-format quantization,
  not uniform Q4_K. Shard 1 supplies model metadata; shard 2 supplies the PLE table.
- Native experts and `--compat-bf16` dense pack as described by
  [UNSLOTH_Q4.md](../../../docs/UNSLOTH_Q4.md). Base-Qwen `mtp/rt` draft pack and
  `data/expert-profile.bin`. Pack, tokenizer, draft-vocabulary and model hashes
  are recorded; large files and packs are not included.
- Context 196608 tokens, INT8 KV; 2048-token pipelined prefill, 24/24 stage split.
  One active sequence; no concurrency test.
- `--spec 4 --spec-min-p 0.5`: MTP maximum 4, suffix lookup enabled. Runtime
  `spec=6` is combined verification capacity, not MTP depth 6.
- Resident experts; automatic GPU expert cache. Worker reserve 4096 MiB;
  coordinator reserve 1000 MiB. RAM headroom 8 GiB worker / 6 GiB coordinator.
  CPU pools 11 worker / 31 coordinator.
- Worker GPU expert cache: 4110 slots / 12292 MiB; expert RAM complement
  24407 MiB. Coordinator: 7244 slots / 21650 MiB; RAM complement 15099 MiB.
- Vision enabled with `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, image-token cap 1024;
  no images were sent by this benchmark.
- No experimental speed projection requested. No new model calibration was run;
  the suite's word-count calibration is not model calibration.
- Benchmark requests: temperature 0, `reasoning_effort=none`,
  `chat_template_kwargs.enable_thinking=false`. Server defaults remained medium
  reasoning, temperature 1, top-p .95, top-k 20, min-p 0, repetition penalty 1.

Both sanitized native launch argument lists and server settings are in
[launch.json](launch.json). Paths, private addresses and TLS file locations are
placeholders; no certificate or private-key contents are included. The API client
ran on the coordinator using loopback. For reproduction, keep the API on loopback
or use authentication; do not expose an unauthenticated API publicly.

## Method

The existing engine was idle at the start; it was **not restarted**, so the expert
cache was already warm. No cache reset or model loading is included. Actual command
on the coordinator, with only the output directory made portable:

```sh
python3 tools/strata_bench.py \
  --config strata-rpc.json --url http://127.0.0.1:8090 \
  --runs 3 --long --out /path/to/results
```

This calls the native streamed `/v1/chat/completions` API, not a separate proxy,
Codex CLI or the Responses API. The upstream suite warms up with one short chat,
two word-count calibration requests and a 1K prompt; those four requests are
excluded from per-test summaries. There are 36 measured requests: 9 short chats,
9 fresh long prompts, and 18 turns across three agent sessions.

- Chat questions and generated word lists come from `tools/ab_engine.py`.
  The output caps are 100 tokens for chat and 24 for fresh-prompt tests.
- Each agent session contains a system prompt and four tool definitions, then
  six turns total. Five follow-ups each add about 1500 words from public project
  documents and retain the model's previous answers. Output cap: 64 tokens per
  turn. The instruction explicitly asks **not to call tools**: this measures
  prefix reuse, not tool execution. Document excerpt hashes are in provenance.
- Fresh prompts have unique tags. All fresh 4K/16K/32K requests reported zero
  reused prompt tokens; warmed expert cache and prompt-prefix reuse are different.
- Prompt/decode rates and their millisecond durations are the engine's values in
  the HTTP response. TTFT is measured by the client from request send to its first
  nonempty generated content/reasoning delta, ignoring keep-alives/empty deltas;
  thinking was disabled, so these are answer-token first-token measurements.
- A local wrapper samples the native both-node telemetry and worker-process
  `nvidia-smi` memory about every two seconds. It collected 92 samples with no
  sampling errors. The sampler adds some overhead and does not guarantee the
  absolute memory peak.

Suite time: **182.681 seconds**, including warm-up/calibration, excluding model
loading and later artifact copying. Per-request client latency and engine timing
values are in [results.json](results.json); the console log is
[benchmark.log](benchmark.log). No startup engine-log excerpt was produced because
the harness attached to an already-running server.

## Results

Values are medians with minimum–maximum ranges. The agent decode and TTFT numbers
pool 15 follow-up turns; its final context statistics cover three sessions.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Short chat | 46 (43–51) | 0 | 100 | 9 | See per-request JSON | 71.3 (60.0–78.8) | 0.415 (0.379–0.463) |
| Fresh 4K prompt | 4019 | 0 | 24 | 3 | 1226.4 (1224.2–1226.4) | 67.0 (59.8–74.1) | 3.302 (3.295–3.302) |
| Fresh 16K prompt | 16058 | 0 | 24 | 3 | 1420.4 (1419.3–1424.0) | 64.8 (60.5–67.5) | 11.329 (11.302–11.338) |
| Fresh 32K prompt | 32049 | 0 | 24 | 3 | 1446.1 (1445.1–1454.4) | 69.0 (68.7–72.8) | 22.197 (22.069–22.211) |
| Agent follow-ups | Final turn 14484 (14437–14504) | See below | 62 (15–64) | 3 sessions / 15 follow-ups | 1070.2 (787.7–1277.6) | 88.7 (61.5–124.9) | 2.290 (1.863–3.204) |

These are deliberately short continuations, particularly the 24-token prefill
fixtures; their decode numbers are not a long-generation performance claim.

### Conversation-prefix reuse and MTP

- Agent prefix reuse: 68.781% (68.580–68.803%) across the follow-ups per session.
- Earlier-turn context reread: 4 tokens/session median (3–74).
- New document tokens freshly read: 13634/session median (13632–13635).
- First agent turn TTFT: 1.035 s (1.023–1.051).
- Whole-interval counters, including the four unmeasured warm-up requests:
  40 requests, 291876 input tokens, 90041 reused tokens, 1946 output tokens.
  Prompt processing 156.8597 s; decoding 25.4151 s. Weighted decode is
  76.57 tokens/s, calculated from output / decode time, **not** total wall time.
- MTP draft-token acceptance over that whole interval: 1177 / 1868 = 63.01%.

### Memory and hardware during inference

| Sampled peak | Coordinator | Worker host |
| --- | ---: | ---: |
| Whole-GPU VRAM | 31.20 GiB | 21.50 GiB |
| GPU utilization | 100% | 100% |
| GPU power | 228.2 W | 304.5 W |
| System-used RAM | 22.50 GiB | 40.31 GiB |
| Private-link NIC receive | 78.91 MiB/s | 0.19 MiB/s |
| Private-link NIC transmit | 0.15 MiB/s | 107.08 MiB/s |

The worker's **Strata process** held 18050 MiB (**17.63 GiB**) in all 92 samples,
below its 20-GiB desktop budget. Whole-GPU VRAM includes the desktop and other
processes; it must not be confused with Strata-process memory. The reserve is an
application startup cache budget, not a driver-enforced hard quota.

Network peaks are independently sampled NIC totals, not synchronized RPC-only
rates or a network bandwidth test; do not add both ends. System-used RAM is not
engine RSS. Worker overall cache-counter deltas were 444185 hits / 594887 lookups
(74.67%); these counts are not the 68.78% conversation-prefix reuse metric and
are not labeled decode-only cache hit rate.

Source machine facts and sampled hardware are in [hardware.json](hardware.json),
[telemetry.jsonl](telemetry.jsonl) and [telemetry-summary.json](telemetry-summary.json).
The upstream harness's coordinator-only metadata is preserved in `results.json`;
its `cpu_cores=32` is a logical-CPU fallback, not a physical-core count. This report
uses the corrected interpretation rather than changing the raw numeric record.

## Correctness and limitations

- **No failed or skipped suite tests.** Each measured request returned tokens;
  this is an API/speed check, not evidence of coding quality or task completion.
- Both services' MainPIDs, invocation IDs, active states and configuration hashes
  were unchanged after the run. The server was healthy and idle afterward.
- No recall/needle, coding-task, actual tool-call, image, concurrency, near-full
  192K-context or long-output test was performed by this suite.
- No stock-Strata, single-GPU, cold-cache or same-day A/B baseline was measured.
  This report makes no comparative speedup claim.
- Prior traffic warmed the expert cache. Conversation replies and MTP acceptance
  vary, even with fixed input and temperature zero. Desktop load was not eliminated.
- Independent exact reproduction is limited by the unpublished custom RPC source.
  [REPRODUCE.md](REPRODUCE.md) states what the provided scripts can reproduce and
  what requires the original custom build. This results-only PR contains no engine
  patches, binary executables, model files, credentials or private user prompts.
