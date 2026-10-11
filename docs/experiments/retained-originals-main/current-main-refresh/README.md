# Current-main retained-originals refresh

Fresh validation against upstream `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`, using candidate source `e55607b036dd5346dceaabadc44bafd8f38d522e`. The port preserves current native asynchronous worker/owner selection and excludes downloads when the outgoing expert already has its canonical RAM original. The feature remains experimental and off by default.

## Scope and qualification

CUDA, HIP and standalone SYCL builds passed. CUDA/HIP exchange ownership tests passed; the retained fixture exercises 55,392 exchanges. Host ASan/UBSan checks passed on both compiler hosts. These backend builds and host tests do not establish HIP/SYCL full-model runtime behavior. Component logs and status receipts are included here.

All fifteen model cases completed and passed their arithmetic function checks. Default-off current-main/candidate outputs matched all 512 IDs. Four lifecycle cases at 8K/32K and widths 1/4 passed twelve STOP/drain/following-function checks at streamed prefix lengths 7, 19 and 65. Following requests used fresh prompts. This is request-level evidence, without an internal state digest or broad quality guarantee.

## Fresh paired observations

RTX PRO 6000 Blackwell 96GB, Ryzen 7950X, 128GB RAM; Qwen3.8-Flash-Next Q8_0, FP16 KV, mmap PLE, resident budget 56 GiB, 8192-token prefill chunks and 1024 generated benchmark tokens. Exact-routing controls set `STRATA_ROUTE_TAIL_SKIP=0`. Native asynchronous adaptation remains enabled. Full settings, environment, device metadata, binaries and material source hashes are in [plan.json](plan.json) and [source-manifest.json](source-manifest.json).

| Fresh input / width | Off decode tok/s | On decode tok/s | Full benchmark IDs | Enabled counter delta, bytes |
|---|---:|---:|---|---:|
| 32K / 1 | 77.133990 | 75.598736 | First difference 16 | 1,911,398,400 |
| 32K / 4 | 131.719427 | 136.538795 | First difference 16 | 2,506,752,000 |
| 128K / 1 | 68.798248 | 81.349254 | All 1024 match | 2,057,625,600 |
| 128K / 4 | 124.559056 | 131.023364 | First difference 797 | 2,699,980,800 |

Differences are zero-based. The 32K width-1 initial function output also differs; the other three paired initial functions match. One fixed-order observation per arm with warm filesystem caches cannot establish a general speedup. Even the exact-output 128K width-1 pair has different adaptive placement work. Three changed-output pairs cannot isolate acceleration from changed work or establish quality equivalence. [All case timing observations](measurements.md) include startup separately.

The counter is cumulative avoided D2H payload. The table subtracts reports surrounding each benchmark; an asynchronous completion near a boundary can include earlier in-flight work. These are neither measured PCIe bus totals nor isolated decode traffic, and do not measure a paired control's traffic. Startup copies and approximately eight additional GiB of pinned RAM must be charged separately. Seed capacity was 1644 experts; retained duplicates can fall as a canonical original moves into primary RAM. No sampled peak-memory series was collected in this campaign.

## Reproduction and provenance

Use the [parent ownership contract and configuration](../README.md), build native experts with tests, and compare retained budget 0 versus 8 with fixed profile-tail seeding. Use the case labels and complete arguments in the plan, replacing path placeholders with local files; compare full token arrays and counters in each result. Run the ownership fixture and host sanitizers before model checks. The [independent completion audit](current-audit.json) includes every case and cancellation count. Historical results in the parent directory retain their original source and settings.

Candidate engine SHA256: `e9fce20667af2b7e53499bb3a34942faf4652574f5f201fddbd9612ce7784b87`.
Baseline engine SHA256: `7cad18d61b63fe8bc8077a36842eb7709de52209c703b67590697a4da17c46e1`.
Tested source archive SHA256: `c482590f4026b2ccd71be60932ae45f46bcda98ced9becd570ac13d96fece8b5`.
