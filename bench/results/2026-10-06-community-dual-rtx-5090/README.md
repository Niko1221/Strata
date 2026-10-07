# Dual RTX 5090: Strata 0.1.40 performance report

Strata 0.1.40 with Flash-Next IQ3_S on our dual RTX 5090 system delivered **243.3 tok/s median native decode and 243.4 tok/s time-weighted decode across 203 completed production requests**.

Separately, controlled synthetic tests found a substantial uncached-prefill performance gap between layer splitting and peer-expert mode. Both datasets are presented below, with their workloads and limitations identified.

### Production workload observation

Source: native per-request engine statistics reported by the local monitoring script on **October 6, 2026 at 19:23:41 CDT**, requested lookback 30 minutes. These are workload observations, not fixed-output benchmark requests.

| Metric | Observed value |
|---|---:|
| Completed requests represented | 203 |
| Native decode mean / median | 243.6 / 243.3 tok/s |
| Native decode time-weighted rate | 243.4 tok/s |
| Native decode p95 / minimum / maximum | 296.0 / 178.0 / 331.8 tok/s |
| Prompt / reused / generated tokens | 12,776,478 / 12,146,604 / 148,857 |
| Prefix reuse | 95.1% |
| Engine request duration median / p95 | 2.40 / 12.47 seconds |
| Mean per-request expert-cache hit rate | 100.0% |
| Large-prefill median / p95 | 9,254.2 / 10,924.5 tok/s; 4 requests with at least 16,384 new tokens |

Native decode excludes prefill, queue and proxy time. The time-weighted decode rate is not whole-window throughput: it excludes idle intervals. All 203 represented requests finished with stop/length outcomes; that is not a count of successfully completed coding tasks or a quality score. The engine's 500-record retention was full, so the requested lookback may omit evicted requests. Native history includes direct and proxy requests reaching the engine; no private task data is included here.

| Prompt length | Requests | Native decode median |
|---|---:|---:|
| 0–32K | 32 | 244.9 tok/s |
| 32–64K | 87 | 235.4 tok/s |
| 64–96K | 55 | 239.4 tok/s |
| 96–128K | 28 | 252.5 tok/s |
| 128–160K | 1 | 254.2 tok/s |

The per-request prefill median was only 1,656.8 tok/s, but **122 of 203 requests processed fewer than 1,024 new tokens**; median new tokens per request was 759. Those cached follow-ups are overhead-dominated and should not be compared with large uncached-prefill benchmarks. Time-weighted prefill was 4,344.1 tok/s across 629,874 new tokens and 145.00 seconds of native prompt time.

### Production versus synthetic decode

In the production 64–96K prompt bucket, median native decode was **239.4 tok/s**, approximately **32% above** the synthetic 65,588-token layer-split median of **180.9 tok/s** below. This is an overlapping context range, not a matched request comparison.

Both rates exclude prefill. Therefore the gap cannot simply be attributed to one metric including prefill/overhead while the other excludes it. High prefix reuse improves request latency but does not, by itself, establish the cause of higher native decode speed.

Different generated content, speculative acceptance, output lengths, cache residency, effective launch settings or workload-dependent execution paths are candidates to investigate—not established explanations. The snapshot does not include speculative acceptance or a full effective-request configuration audit. No matched replay or profiler evidence currently isolates the cause.

A useful additional investigation is to reproduce the gap with a shareable workload, capture effective request settings, output length, native timings and speculative acceptance, and compare actual execution paths before changing kernels. The synthetic results should **not** be presented as a ceiling or representative production decode rate.

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
