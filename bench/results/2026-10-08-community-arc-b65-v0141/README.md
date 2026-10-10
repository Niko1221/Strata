# Arc Pro B65: Strata v0.1.41 on PCIe Gen4

Measured on 2026-10-08 by timnevits. **Official upstream release**, commit
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, with **no local engine patches**.
Codex, an OpenAI AI agent, ran the tests and drafted and submitted this report
with my approval. The measurements come from my B65 hardware.

Both memory profiles pass their bounded correctness checks. With the same 7K inputs, the 8K/prefill-512 profile delivers **43.52 decode tok/s**, versus **42.39** with the 262K/prefill-512 profile (-2.6%). The 8K/prefill-4096 profile reduces median 7K TTFT from **18.20 to 7.62 seconds**, with decode tradeoffs below.

## Hardware and configuration

One Intel Arc Pro B65, 32 GiB VRAM (`8086:e222`), **Gen4 x16**, 200 W cap;
i5-12600K, 128 GB DDR4-3200, Samsung 980 NVMe. Other model owners were stopped.
Ubuntu 26.04.1, kernel 7.0.0-38, xe/NEO 26.22.38646.7, Level Zero 1.28.6.
Native Release/JIT build with existing oneAPI 2026.1.0 and oneMKL, precise FP,
correctly rounded divide/sqrt and 32-lane subgroups. The documented build option
`STRATA_SYCL_SPIN_MAX=20000` preserves the Linux xe wait bound: upstream
driver detection picks this host's UHD display GPU (i915) before the B65. The tagged engine labels
itself `0.1.41`; the commit and binary hash identify this build.
Exact metadata: [system.json](system.json), [build.json](build.json).

Original **full 512-expert Flash-Next IQ2_XS**, not Coder:
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`. Original Q2_0 MTP pack from
`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`.
[artifacts.json](artifacts.json) identifies both shards, retained dense pack,
tokenizer/template, draft and expert ranking. Ranking matches the release
byte for byte. The model has a native 262144-token window. The tables compare **8192 and
262152 native context allocations (8K/262K profiles)**, using matched short and 7K inputs. The API reserves eight verification slots,
leaving 8184 and 262144 usable combined tokens respectively. No model, quantization, driver, compiler or ranking changes
between configurations.

One request, INT8 resident KV, four workers, automatic hot
expert cache and streaming experts. Every missing expert is in a protected
pinned host mirror (16 GiB cap). `STRATA_VERIFY_NO_HOST=1` keeps computation
on the GPU, including direct reads from that mirror. Native verify/commit
graphs, MTP window 4/max 4/min-p 0.5 (up to three drafts). Thinking off,
greedy/seed 17; prompt, conversation and suffix caches, short-read and expert
adaptation off. `STRATA_DBG_NAN` and QFUSE absent. Allocation alias checking
keeps its upstream default. The two 8K profiles reserve 3072 MiB and differ only in prefill batch size:
[512](profile-8k-512.json), [4096](profile-8k-4096.json). The
[262K profile](profile-262k-512.json) uses prefill 512, reserve 4096 MiB and
explicit `--no-prefill-borrow`, preserving the smaller window's implicit
no-borrow policy. It reserves more KV and has fewer hot expert slots; this is
a comparison of the complete memory profiles, not context as a single variable.
[environment.json](environment.json) is shared.

Hot expert slots, cache sizes and mirror sizes below are identical across each
profile's three main runs. Free VRAM is the observed range across those runs.
Expert slots count per-layer experts, rather than the model's 512 experts
per MoE layer. They help explain the memory/performance tradeoff.

| Profile | Hot expert slots | Hot cache MiB | Pinned missing-expert mirror GiB | Free VRAM at startup MiB (range) |
| --- | ---: | ---: | ---: | ---: |
| 8K / 512 | 17881 | 24588 | 9.01 | 3185–3197 |
| 8K / 4096 | 16160 | 22207 | 11.33 | 4229–4237 |
| 262K / 512 | 14388 | 19761 | 13.72 | 4180–4185 |

## Three-run results

Main suite: **three fresh processes per profile**, five fixed tasks at each
of 512 and 7000 input tokens, 640 generated tokens. An unmeasured full-shape
warmup precedes each input tier. Tasks cover code, prose, inventory and incident
planning; padding precedes the instruction. Each profile has 48 total requests,
30 measured. For each table row, take the median of five tasks within each run,
then the median and range of those three run medians. Individual cells are
available in the raw data.

Public-fixture suite: three fresh processes per profile, 2185-token request
first, then 20-token request, 256 outputs each, followed by an isolation canary.
No separate warmup; any first-request cost stays in its measurement. This is
9 total requests/6 measured per profile. Its prompts and output lengths differ
from the main suite.

| Native context allocation | Prefill | Input/workload | Output | Runs | Prompt tok/s | Decode tok/s | TTFT seconds |
| ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| 8192 | 512 | 512, five tasks | 640 | 3 | 446.69 (427.20–446.89) | 45.02 (44.15–45.04) | 1.180 (1.179–1.232) |
| 8192 | 512 | 7000, five tasks | 640 | 3 | 385.29 (366.67–386.01) | 43.52 (42.72–43.52) | 18.203 (18.170–19.126) |
| 8192 | 512 | 20, public fixture | 256 | 3 | 66.84 (66.80–68.38) | 52.14 (52.14–54.49) | 0.334 (0.326–0.334) |
| 8192 | 512 | 2185, public fixture | 256 | 3 | 396.62 (396.36–407.84) | 49.16 (49.14–49.72) | 5.570 (5.419–5.573) |
| 8192 | 4096 | 512, five tasks | 640 | 3 | 412.97 (410.19–420.71) | 44.17 (44.15–44.67) | 1.274 (1.251–1.282) |
| 8192 | 4096 | 7000, five tasks | 640 | 3 | 923.28 (922.98–933.83) | 42.31 (42.30–42.95) | 7.617 (7.531–7.620) |
| 8192 | 4096 | 20, public fixture | 256 | 3 | 62.75 (62.52–62.91) | 52.79 (52.29–53.51) | 0.354 (0.353–0.355) |
| 8192 | 4096 | 2185, public fixture | 256 | 3 | 845.49 (843.08–851.62) | 47.11 (47.11–47.42) | 2.648 (2.628–2.654) |
| 262152 | 512 | 512, five tasks | 640 | 3 | 398.23 (392.10–398.54) | 43.53 (43.29–43.54) | 1.320 (1.319–1.340) |
| 262152 | 512 | 7000, five tasks | 640 | 3 | 347.25 (341.26–347.27) | 42.39 (42.19–42.42) | 20.195 (20.194–20.548) |
| 262152 | 512 | 20, public fixture | 256 | 3 | 59.93 (59.92–60.11) | 51.01 (51.01–51.56) | 0.370 (0.369–0.370) |
| 262152 | 512 | 2185, public fixture | 256 | 3 | 374.33 (374.31–376.47) | 48.79 (48.79–48.98) | 5.905 (5.872–5.905) |

Values are median (range). Loading/startup is timed separately and excluded
from request timings; the OS weight-file cache stays warm. Fixed expert
placement, **zero prompt reuse**, no adaptation. Prompt/decode rates use the
engine's separate token counts and durations. TTFT measures request start to
first native emitted token. Raw native timing lines, counts, draft acceptance,
input/output hashes and elapsed times: [results.json](results.json),
[CSV](results.csv), [summary](summary.json), [lifecycle](lifecycle.json).
Generated content is discarded; only numeric/hash receipts persist.

Prefill 4096 cuts the 7K TTFT median by 58.2%. 5/5 long benchmark cells have different output hashes from prefill 512, so their decode differences can include changed continuations and draft acceptance. The slowest 7K decode-cell change is -17.0%. Normal serving keeps prefill 512. With prefill 512, the larger memory profile's 7K TTFT is 20.19s versus 18.20s; 10/10 matched main cells have identical output hashes across memory profiles. See per-workload data before generalizing the aggregate.

## Validation and limits

Both 8K profiles pass **21 native reference checks** and **24 API/lifecycle
checks**, including exact target-only/one-draft/MTP4 agreement on the tested
256-output workloads, tools, queued isolation, near-8K recall, 640 outputs,
cancellation during long prefill and restart. Same-profile benchmark repeats
match exactly and all successful native processes exit 0 after graceful QUIT.
The 262K profile separately passes five native checks (target-only/MTP4 at
32K and full, one draft at 32K), with exact 128-token agreement, and 12 API
checks including progressive/full recall, a repeated full request and overflow
rejection. Full-window timing is capacity evidence, separate from the three-run
matched-input tables. These are bounded runtime checks, not broad quality or
roleplay qualification.
[Full-window receipts](full-window-checks.json) keep native and API/SSE timings
separate.

The full registered kernel suite is **26 pass / 4 fail**. Available actual IQ
fixtures and native GGUF checks over eight layers pass. The remaining
registration/fixture, unsupported QFUSE and grouped-S2 bitwise failures are
explained in [QUALIFICATION.md](QUALIFICATION.md), with numeric/reference
receipts in [qualification.json](qualification.json),
[native-reference.json](native-reference.json) and [api-checks.json](api-checks.json).
No claim that the whole kernel suite passes.

Independent GPU, PCIe, memory and thermal guards remain active. Successful runs
have no resets, PCIe errors, swap-out or OOM; peaks are recorded in qualification.json.
Failed qualification attempts are excluded from throughput. Both previously
local mirror safety fixes are upstream: incomplete coverage refuses startup,
and pinned mirror memory is freed before QUIT exit.

The qualified installed profile uses 262144 usable API tokens, prefill 512 and a 4096 MiB reserve. Routed pre/post-reboot full-window, tool, authorization and queued-isolation checks, graceful unload/restart and retained-model recovery pass. The actual Pi client was not rechecked. Fresh full-window prefill takes 13.2–13.6 minutes in these individual checks; capacity does not imply low latency or broad long-context quality. This report is pinned to official v0.1.41.

[BUILD.md](BUILD.md) gives exact preparation and portable reproduction steps.
[Portable fixture audit](portable-fixture-audit.json) verifies that the published
drivers produce the measured main inputs and public token fixtures, without
sending another inference request.
