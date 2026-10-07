# Dual RTX 5090: Strata 0.1.40 performance report

Strata 0.1.40 with Flash-Next IQ3_S on our dual RTX 5090 system delivered **223.2 tok/s median native decode across 330 completed production requests**, with **218.3 tok/s time-weighted decode**. Production observations cover prompt-length buckets through **224K–256K**, where two requests decoded at **180.4 and 215.5 tok/s**.

Separately, controlled synthetic tests compared layer splitting with peer-expert mode through 250,053 prompt tokens. Production observations and synthetic benchmarks are reported separately.

### Production workload observation

Source: native per-request engine statistics captured on **October 7, 2026 at 04:30:09 CDT**, with a requested lookback of 30 minutes. Deployment: Flash-Next IQ3_S, Strata 0.1.40, 262,144-token context, INT8 KV, and 32,768 resident KV tokens.

These measurements came from production coding-agent activity, not fixed-output benchmark requests. They measure engine performance, not coding correctness or completed-task quality.

| Metric | Observed value |
|---|---:|
| Completed requests represented | 330 |
| Native decode mean / median | 227.1 / 223.2 tok/s |
| Native decode time-weighted rate | 218.3 tok/s |
| Native decode p95 / minimum / maximum | 285.3 / 154.6 / 326.7 tok/s |
| Prompt / reused / generated tokens | 41,502,360 / 40,538,507 / 209,287 |
| Prefix reuse | 97.7% |
| Engine request duration median / p95 | 1.35 / 18.07 seconds |
| Mean per-request expert-cache hit rate | 100.0% |

Native decode excludes prefill, queue and proxy time. The time-weighted decode rate excludes idle intervals and is not whole-window throughput. All 330 represented requests finished with stop/length outcomes; these outcomes do not establish answer correctness.

### Production decode by prompt length

Buckets use each request’s prompt-token count, not batch-wide token counts. K denotes 1,024 tokens.

| Prompt length | Requests | Mean decode | Median decode | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| 0–32K | 16 | 231.5 | 231.5 | 175.8 | 265.6 |
| 32–64K | 49 | 220.9 | 218.2 | 161.6 | 324.7 |
| 64–96K | 40 | 225.3 | 220.9 | 192.8 | 298.7 |
| 96–128K | 64 | 235.3 | 234.7 | 154.6 | 311.3 |
| 128–160K | 63 | 226.9 | 220.9 | 165.6 | 323.7 |
| 160–192K | 67 | 233.9 | 236.8 | 167.1 | 326.7 |
| 192–224K | 29 | 205.8 | 198.1 | 162.1 | 273.0 |
| 224–256K | 2 | 197.9 | 197.9 | 180.4 | 215.5 |

All rates are tokens per second.

The 192K–224K bucket contains 29 requests with a median of **198.1 tok/s**. The 224K–256K bucket contains only two requests, so its result is preliminary. Bucket boundaries do not establish the exact maximum observed prompt length.

### Production prefill

| Metric | Observed value |
|---|---:|
| Per-request prefill median / p95 | 1,515.3 / 3,883.9 tok/s |
| Time-weighted prefill | 3,778.9 tok/s |
| Newly processed prompt tokens | 963,853 |
| Total native prompt-processing time | 255.06 seconds |
| Median new tokens per request | 944 |
| Requests processing fewer than 1,024 new tokens | 180 of 330 |
| Large prefills: at least 16,384 new tokens | 4 requests |
| Large-prefill median / p95 | 10,193.0 / 10,922.0 tok/s |

Prefill rates divide uncached prompt tokens by native prompt-processing time. Most requests reused substantial prefixes; short follow-ups include fixed processing overhead and are not peak-prefill benchmarks. The four large-prefill observations provide a limited sample of sustained prompt-processing throughput.

### Production measurement limits

The engine retained 500 request records, and retention was full. The requested lookback may therefore omit evicted records. Native history resets on restart and includes direct and proxy requests reaching the engine.

This is an aggregate production snapshot. Private prompts, generated content and task details are excluded. Per-request sampling settings and speculative acceptance were not captured in this report, so differences from the controlled tests below cannot be attributed to a particular cause.

Production and synthetic measurements use native decode timing, but their workloads differ. The synthetic results below should not be interpreted as a production performance ceiling or as a matched comparison with this snapshot.

### Synthetic test setup

- **Hardware:** 2× RTX 5090 32 GB, PCIe Gen5-capable x8 links; Ryzen 9 9950X; 128 GB RAM.
- **Software:** Ubuntu 26.04, NVIDIA 610.43.02 with consumer-GPU P2P enabled; CUDA 13.0 build targeting sm_120.
- **Strata:** 0.1.40, commit `1735d6471df29b42c26170efaac1f1446a58640f`.
- **Model:** Qwen3.8-Flash-Next GSQ-RCO IQ3_S.
- **Settings:** 262,144 context, INT8 KV, 32,768 resident KV tokens, automatic prefill, speculation 4, MTP.
- **Sampling:** temperature 1.0, top-p 0.95, top-k 20, min-p 0, repetition penalty 1.0, seed 42, thinking enabled.
- **Clocks:** +250 MHz core offset, NVML memory offset +1500, 575 W limit per GPU. These are not stock-clock results.

### Synthetic layer split versus peer experts

Three rounds per configuration, synthetic prose prompts, 2,048 generated tokens per request. Figures below are medians. First requests had zero reused prompt tokens.

| Actual prompt tokens | Mode | Uncached prefill tok/s | Native decode tok/s | Total request seconds |
|---:|---|---:|---:|---:|
| 65,588 | Layer split | 10,823 | 180.9 | 17.61 |
| 65,588 | Peer experts | 4,507 | 174.9 | 26.49 |
| 250,053 | Layer split | 11,722 | 175.5 | 33.85 |
| 250,053 | Peer experts | 4,311 | 166.1 | 71.17 |

Repeated-prefix requests were much closer: at 250K, median total latency was **13.32 seconds with layer splitting versus 13.50 seconds with peer experts**.

Peer access initialized successfully. This comparison changes the execution arrangement; it is **not** a controlled P2P-enabled versus P2P-disabled comparison.

### A source-level lead

In the tested revision:

- `src/prefill/prefill.cpp`, around line 144: `fused_ring()` returns false when `core::peer_portable()` is true.
- Around line 2514: peer mode follows an MMQ-only dispatch; the fused-path conditions require `no_peer`.
- `src/program/generate.cpp`, around line 5926: peer prompt processing uses `set_peer()`, while the layer-split branch configures `set_stage_helper()`.

These differences suggest investigating **actual kernel selection, chunk/ring sizing, expert placement, transfer time and synchronization waits**. They do not prove which difference causes the slowdown—or that the faster configuration used fused kernels on every layer.

A useful next step would be to instrument one 64K uncached/repeated pair in each mode before running a larger matrix. Simply removing the peer guard would be inappropriate: the surrounding code ties kernel selection to buffer sizing.

### Synthetic Q4 screening result

We also ran a short 64K screen with 1,024 generated tokens:

| Recipe | Uncached prefill tok/s | Uncached decode tok/s | Repeated-prefix decode tok/s |
|---|---:|---:|---:|
| Upstream IQ3_S | 10,670 | 170.9 | 175.0 |
| Upstream UD-Q4_K_XL | 7,516 | 142.1 | 152.6 |
| Custom dual-GPU Q4 fork | 6,955 | 142.0 | 140.1 |

The custom candidate was `kim-haneol/Strata`, branch `linux-2x5080`, commit `03556ef3bb08dd9fd0b770e4c2e01caf1ee0bce3`.

This was only one first/repeated pair per recipe, with configuration and memory-layout differences. It did not reveal a Q4 performance improvement on this hardware. Measuring expert residency and quant-specific kernels seems more useful than another broad parameter sweep.

### Synthetic measurement limits

Prefill rates use uncached tokens and native prompt time. Decode rates use native generation time. Total latency includes the complete streamed request but excludes model loading. Different generated text and speculative acceptance can affect results despite identical sampling settings.

Exact model-file hashes and the patched-driver source revision were not captured, limiting exact reproduction. These are performance observations, not model-quality comparisons.

Thanks to the Strata contributors. I hope these measurements help identify a productive dual-5090 optimization target.
