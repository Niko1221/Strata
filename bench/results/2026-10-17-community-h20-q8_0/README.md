# H20: Q8_0 at 256K, single-stream and four-slot serving

Measured on 2026-10-17 (Asia/Shanghai), contributed by @LeslieWylie.

One H20 completed a **261,992-token uncached prompt** and returned the verification
phrase placed in its middle. This is a capacity and simple recall check, not a
general long-context quality score. The report also records a three-run short
single-stream test and a temporary four-slot serving test. A separate four-H20
SGLang FP8 service is included as a deployment reference, not a matched-engine or
matched-quantization benchmark.

This is an archival report of **Strata 0.1.44.1 at `296b59d2`**. No engine changes,
default changes, current-main rerun, or before/after performance improvement are
claimed. No new inference measurements were made while preparing this PR.

## Hardware and software

| Item | Recorded value |
| --- | --- |
| Strata GPU | One NVIDIA H20, physical device 3; 97,871 MiB total VRAM |
| CPU | Container exposes `Intel(R) Xeon(R) Processor`, 150 logical CPUs; exact SKU, CPU quota and NUMA binding not captured |
| RAM | `/proc/meminfo` reports 1,782,579,200 KiB = 1,700 GiB host-visible total; not an exclusive reservation |
| Filesystem | NFS for model/experiment files; underlying physical medium and RAM speed unknown |
| OS | Ubuntu 22.04.5 LTS; kernel `5.10.134-013.8.2.kangaroo.al8.x86_64` |
| NVIDIA driver | 580.95.05 |
| Build | Source build, CUDA 12.1 compiler, sm_90, Python 3.11 |
| Strata source | `296b59d2f73c8f04a805df29c2fa97655adace78`, engine 0.1.44.1 |
| llama.cpp dependency | `3cf03257f219afbe7334045ff7c6a06ac68c627d` |
| Successful cuBLAS runtime | Task-private `nvidia-cublas-cu12==12.8.4.1`; see compatibility observation below |

A later read-only inventory reported a 500 W GPU power limit and maximum PCIe
generation 5 / width 16. These are post-run inventory values, not captured
per-request clocks, negotiated link measurements, or evidence of power tuning.
The host was shared: existing services on other GPUs continued running. Their
CPU/memory interference was not eliminated. Full build flags were not captured.

## Model and configuration

- Community `unsloth/Qwen3.8-Flash-Next-GGUF`, **Q8_0**, revision
  `38bb39ee97821de2c9009abb7e93950eec396e66`.
- Six `Qwen3.8-Flash-Next-Q8_0-0000N-of-00006.gguf` shards, 188,225,033,248 bytes
  in total. All six matched the pinned upstream SHA256 checks during preparation.
- Native pack and MTP artifacts were prepared for that upstream build. No custom
  model quantization was performed. The upstream expert profile was 196,632 bytes,
  SHA256 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.
- Maximum context **262,144**, int8 KV, **32,768 resident KV tokens**, automatic
  expert cache and prefill, 4,096 MiB VRAM reserve, MTP `--spec 4 --spec-min-p 0.5`.
- `STRATA_ROUTE_TAIL_SKIP=0`, vision disabled. The pinned engine also automatically
  used an experimental native **Q5_K output head**. The GGUF's Q8_0 label therefore
  does not describe every runtime projection. No output-quality equivalence to
  FP8 or to another Strata configuration was tested.
- Single-slot configuration for capacity and 512-output baseline; temporary
  `parallel=4` plus `--batch-mtp` for the serving matrix. The API reported four
  actual serving slots. Eight clients shared those four slots and queued.

[config-single.json](config-single.json) and
[config-parallel4.json](config-parallel4.json) preserve the settings with
`${STRATA_ROOT}` path placeholders. Replace the placeholders with the local
prepared installation root before using a config; JSON does not expand them.
The configs use localhost port 18083 and physical GPU 3, which must be adjusted
for another machine. Neither the data scripts nor this report starts a server.

Binary hashes of the prepared native pack/MTP artifacts were not captured; this
report does not promise a bitwise build reproduction. The exact model revision,
available profile hash, configurations and synthetic input fixtures are provided.

## Method and timing definitions

Capacity requests were serial, one pass per case, with thinking disabled. They
used the pinned upstream `tools/strata_bench.py` chat/digest helpers. Prefill and
decode speeds in the capacity table come from **engine timings**; TTFT and total
latency come from the streaming client. The same engine process continued across
cases: the expert cache was not reset. Each long recall case had a distinct
marker and reported zero reused prompt tokens. This does not imply a cold expert
cache or a cold filesystem cache.

The serving comparison used identical paired inputs, `temperature=0`, `top_p=1`,
`seed=42`, thinking disabled, localhost streaming, and alternating backend order.
It did not send the two backends' benchmark waves simultaneously. One warm-up
wave per backend/profile was excluded: a single 64-token request for the baseline,
and four concurrent 32-token requests for the four-slot profile. Every request
in a concurrent wave started from a shared client barrier; model loading is
outside the timings.

- Wave throughput = sum of actual completion tokens / (latest end − earliest
  start), including prompt work and queueing. Aggregate tables use wave medians.
- Client decode estimate = `(completion_tokens − 1) / (end − first content event)`.
  This is separate from engine decode timing and includes stream completion overhead.
- TTFT ends at the first nonempty answer-content event; empty deltas are ignored.
- Per-request P50/P95 use linear interpolation across all requests in the group.
- C1 has three measured waves per backend. C2/C4/C8 have two waves each. The 32K
  reference has one request per backend. Small-sample P95 is not a long-term SLA.

SGLang's existing `stream_interval=8` was retained, while Strata streamed tokens
individually. The common client formula still includes these different chunking
behaviors. Strata reported zero cached tokens for all serving requests; SGLang did
not expose cached-token details, so its cache amount is **unknown**, not zero.

The short fixtures cover hash tables, transaction isolation, HTTP connection
pooling and LRU caches. C1 uses the first topic, C2 the first two, C4 all four, and
C8 repeats all four. Both backends receive paired identical inputs, but content
mix changes across concurrency levels. The unique first-line client identifier
distinguishes profile, wave and client. Expert caches were allowed to adapt.

## Capacity and basic generation

Each row has one measured request. The three recall prompts repeat an English
filler sentence around the middle phrase `H20-FLASH-7391`; all returned it exactly.

| Case | Actual prompt tokens | Reused tokens | Output tokens | TTFT s | Engine prefill tok/s | Engine decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arithmetic, 37 × 41 = 1517 | 27 | 0 | 5 | 1.218 | 25.0 | 16.4 |
| Short continuous output | 34 | 0 | 128 | 0.825 | 44.3 | 23.8 |
| 32K middle phrase | 32,762 | 0 | 11 | 9.452 | 3510.2 | 16.1 |
| 128K middle phrase | 131,071 | 0 | 11 | 36.312 | 3624.0 | 8.8 |
| Near-256K middle phrase | 261,992 | 0 | 11 | 77.879 | 3376.0 | 8.3 |
| Near-256K continuous output | 261,785 | 245,760 | 128 | 6.056 | 2748.9 | 15.1 |

The last row processed only **16,025 new prompt tokens**. Its 6.056-second TTFT
does not represent a full 256K prefill. Eleven-token recall outputs are too short
to characterize sustained decode. Total latencies and original outputs/usage are
in [capacity-results.json](capacity-results.json).

One original boundary request was rejected with HTTP 400: 262,009 prompt tokens
+ 128 requested output tokens + server `CTX_SLACK=8` = **262,145**, one above the
window. Trimming 224 filler tokens produced the successful final row. The failed
request remains in the JSON with `passed=false`; it is not counted as successful
throughput. There were six successful capacity requests and this one resolved
budget rejection.

## Short single stream: 365 input / 512 output tokens

Three measured requests per backend after warm-up, Strata in its original
single-slot configuration. Values below are client measurements.

| Metric | One-H20 Strata | Four-H20 SGLang FP8 reference |
| --- | ---: | ---: |
| Decode tok/s median [min, max] | 25.157 [17.965, 25.781] | 119.003 [104.780, 119.959] |
| TTFT s median [min, max] | 0.830 [0.799, 0.908] | 0.291 [0.288, 0.302] |
| End-to-end output tok/s, wave median | 24.217 | 111.732 |
| Completion latency P95 s | 28.531 | 5.109 |

All three runs, including the slower first run, are retained. Strata engine
decode timings were 18.0, 25.8 and 25.2 tok/s; these are not substituted for the
client estimates used in the paired table.

## Serving matrix: 363–366 input / 256 output tokens per request

Strata used the temporary four-slot/batch-MTP configuration here. The 512-output
single-slot baseline is a different workload; the two sections do not establish
a single-stream speedup from enabling four slots.

| Clients | Waves/backend | Strata total tok/s | TP4 total tok/s | TP4 / Strata total | (TP4 / 4) / Strata |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 34.642 | 109.934 | 3.17× | 0.79× |
| 2 | 2 | 39.911 | 202.483 | 5.07× | 1.27× |
| 4 | 2 | 46.492 | 378.479 | 8.14× | 2.04× |
| 8 | 2 | 47.368 | 683.684 | 14.43× | 3.61× |

The final column normalizes by GPU count only; it is not a cost or energy metric.

| Clients | Strata client decode P50 tok/s | TP4 client decode P50 tok/s | Strata TTFT P95 s | TP4 TTFT P95 s | Strata latency P95 s | TP4 latency P95 s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.925 | 122.426 | 0.836 | 0.280 | 9.216 | 2.380 |
| 2 | 22.434 | 117.553 | 1.787 | 0.510 | 12.943 | 2.686 |
| 4 | 13.119 | 111.827 | 3.926 | 0.439 | 22.162 | 2.739 |
| 8 | 12.983 | 100.131 | 25.141 | 0.470 | 43.186 | 3.008 |

C4 to C8 added little throughput in this **four-slot** configuration and increased
queueing. Native eight-slot serving was **not tested**. This is not a search for
the maximum possible single-H20 throughput.

A supplemental 32K pair used **32,826 prompt / 256 output tokens**, one request
per backend, with Strata in its four-slot profile but only one active client:

| Metric | Strata | TP4 reference |
| --- | ---: | ---: |
| TTFT s | 9.484 | 2.278 |
| Client decode tok/s | 36.083 | 102.381 |
| Total latency s | 16.551 | 4.768 |

This is different from the 32,762-token recall case. It is a supplemental sample,
not a repeated capacity measurement.

### Reference-service and shared-host limits

The reference was an existing SGLang FP8 deployment on four other H20s, TP4,
installed package version `0.0.0.dev1+gdaf631719`, no speculative decoding,
`chunked_prefill_size=8192`, `mem_fraction_static=0.85`. Its weight revision was
not recorded. It was not restarted, retuned, or cleared of production caches.
Both engine and quantization differ, and output-quality equivalence was not
measured. These numbers compare the two recorded deployments; they do not isolate
hardware scaling or prove that one engine is intrinsically faster.

Across the comparison, all **70 non-warm-up requests** reached the requested
output cap without API errors or visible reasoning content. This checks the
generation path and output length, not the correctness of every generated guide.
During Strata wave interiors (excluding the first two and last one seconds),
the four reference GPUs produced 412 utilization samples: one was at least 20%,
with a peak of 64%. External traffic was therefore not proven absent.

## Memory observations

- GPU memory sampled every two seconds: capacity peak **93,941 MiB**; serving
  comparison peak **94,229 MiB** on the Strata card. Instantaneous peaks may be missed.
- Single-slot engine expert cache: 17,136 slots / 83.35 GiB. The private four-slot
  profile reduced it to 16,303 slots / 79.29 GiB, with about 50 MiB private MTP
  KV/buffers per slot; batching did not have zero memory cost.
- Single-slot engine RSS snapshot: **133,928,676 KiB (~127.7 GiB)**, 176 process
  threads; startup logs reported 150 participating custom CPU-pool threads. This
  was not a host-memory peak study; an OMP setting of 16 did not bound that pool.
- Pinned host KV was about 3.09 GiB for the streaming setup. Single-GPU operation
  still depends on host RAM and backing model files.
- The four reference GPUs' individually sampled peaks were 94,528 / 94,150 /
  94,156 / 93,674 MiB. Their sum is 367.68 GiB, not necessarily a simultaneous
  peak; usage includes preallocated KV and is not just model weight size.

## cuBLAS compatibility observation

The original CUDA 12.1/cuBLAS 12.1.0.26 runtime failed on the 32K prefill with
SIGFPE. GDB located the failure in a cuBLASLt heuristic, reached through
`cublasGemmEx` and Strata's BF16 GEMM, with a zero divisor at `div r13d`.

Only this task's config `lib_dirs` was changed to load a private
`nvidia-cublas-cu12==12.8.4.1` runtime. The compiler, engine source and system CUDA
were retained. The same 32K input then completed, followed by the larger capacity
cases. Three T=8192 BF16 witness shapes (N/K: 320/10240, 10240/320, 2560/2560)
matched sampled CPU references under that runtime; the preserved successful
log lines report `cublas=120804` and `max_error=0`.

This is a machine/runtime-specific compatibility observation, not an upstream
source fix or a recommendation to replace system libraries. The report does not
establish the minimum working cuBLAS version or general numerical equivalence.

## Files and replay

- [capacity-results.json](capacity-results.json): every capacity request, including
  original usage, timings, output and the retained HTTP 400.
- [serving-waves.json](serving-waves.json): every serving wave, including warm-ups,
  per-request output, cache counters, client timings and available engine timings.
- [serving-summary.json](serving-summary.json) and [serving-metrics.csv](serving-metrics.csv):
  non-warm-up aggregates. `aggregate_tps` is median wave throughput;
  `decode_tps_p50` is median per-request client decode, not engine decode.
- [environment.json](environment.json): hardware, versions, source/model provenance,
  memory observations and fields explicitly left unknown.
- [engine-log-excerpt.txt](engine-log-excerpt.txt): selected existing startup,
  numerical-witness and single-slot server completion lines. Server completion
  summaries use their own boundary and can differ from the JSON's engine timing.
- [prompts.json](prompts.json): 47 exact synthetic inputs stored as literal/repeated
  segments, with SHA256 and byte counts. Contains no user documents or real conversations.
- [measure.py](measure.py): fixture reconstruction and a portable wrapper around
  the **unchanged original serving HTTP request/timing function**. Operational
  process guards, GPU sampling and profile switching are not included. The public
  wrapper has been checked offline only; it has not generated new benchmark data.

Offline verification, which sends no requests:

```sh
python measure.py verify
```

To replay a single fixture against an **already running** prepared server:

```sh
python measure.py request --url http://127.0.0.1:18083 \
  --model qwen3.8-flash-next-unsloth-q8_0 \
  --ids baseline-single-512-w0-client0 --max-tokens 512
```

Use all `client0`…`client7` IDs from the same recorded C8 wave in one `--ids`
argument list to reproduce its barrier concurrency. Use a fresh process/state
according to the configuration being evaluated; fixture replay itself does not
reset caches or switch serving profiles. Set the API key in the environment
variable named by `--api-key-env` rather than writing it into shared commands.

The serving function uses temperature 0/top_p 1/seed 42 and disables thinking,
matching the recorded comparison. For the capacity recall test, reconstruct an
input with `python measure.py prompt context-256k` and request a short answer;
the published capacity timings remain those of the original upstream helper,
not measurements by the new wrapper.

The originally measured Strata service was returned to its single-slot profile
after the comparison. No claim is made about later service uptime, complex
long-document understanding, coding-agent quality, quantization equivalence,
native eight-slot throughput, or performance on newer Strata builds.
