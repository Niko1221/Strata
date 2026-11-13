# DeepSeek-V4.1 Flash SYCL: B70 serial performance

Measured on 2026-11-13. Median (minimum–maximum), three requests per length:

| Actual input / output | Prefill token/s | Decode token/s | Client TTFT, s | Client total, s |
| --- | ---: | ---: | ---: | ---: |
| 1024 / 256 | 197.0 (191.5–198.4) | 13.0 (12.3–14.5) | 5.203 (5.166–5.350) | 24.921 (22.818–26.190) |
| 4096 / 256 | 377.2 (374.3–378.8) | 12.6 (12.5–13.7) | 10.868 (10.823–10.952) | 31.106 (29.629–31.347) |
| 8192 / 256 | 600.4 (598.8–602.7) | 12.4 (12.2–13.0) | 13.661 (13.609–13.696) | 34.316 (33.346–34.609) |

All nine requests returned HTTP 200 and normal SSE completion, generated exactly 256 tokens, and had zero cached prompt tokens. Each length had one excluded 128-output-token warmup. Requests ran serially in 1024/4096/8192 order for seeds 1/2/3. Expert residency and OS file page cache remained warm. Rates use engine timers; client TTFT ends at the first nonempty generated delta, and total time ends at SSE completion. Every request reached its output cap. These are absolute custom measurements, without a comparator or standard-protocol approval.

## Environment and configuration

- Intel Arc Pro B70, 32 GB; one GPU exposed. Xeon w5-3435X, 16 cores / 32 threads. Host MemTotal 503.36 GiB; container RAM limit 192 GiB, 24-CPU quota, 16 engine workers. The RAM limit is not a minimum installed-memory claim. The selected GPU was exclusive; the shared host was not fully isolated.
- Ubuntu 24.04.2, kernel `6.17.0-1007-intel`; `intel/omix:0.4.1-devel-ubuntu24.04`, image `sha256:3e8b0e469c972b312bf4fe114ed95f629bc6a65c706bc266db085468dca3b09f`.
- icpx 2026.1.1, oneMKL 2026.1, Level Zero v2; Intel GPU runtime 26.31.39395.16, IGC 2.40.16, libze 1.32.0. Release SYCL JIT build, no AOT. Engine version remains 0.1.59.1.
- [Unsloth UD-IQ1_M](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/f924b94f19c1fea02c06a65c8475b0a45e2b88b1/UD-IQ1_M), five mixed-format GGUF shards, plus the Q8_0 DSpark drafter. File identities and download revisions: [model-provenance.json](model-provenance.json).
- Context **16384 for every case**, one slot, top-6, reserve 2048 MiB, mmap on, demand store off, vision off, DSpark on/confidence 0. Automatic prefill chunks, SYCL busy-expert threshold 32, attention slab 1024. Greedy sampling, thinking off. No profiler or trace settings.
- [Measured engine source and binary identities](measured-source.json). Implementation commit `730e44a8f3da4396d0a30cfca0c7260aa7054ece`, based on upstream `13eedf552417a6a791a7d13c6597e24300ee3a09`; binary SHA-256 `3c81b234ffe80359123a67f6c0ac8bd3ba162da938a9c7c255af8934aeb1e5ed`.

[All nine raw records and outputs](results.jsonl), [excluded warmups](warmups.json), [engine timings](engine-performance.txt), [summary](summary.json), [frozen requests](config/performance.json) and [configuration](config/service.json). The published records omit repetitive SSE transport events while retaining final text, metrics and outcomes. Hardware load-link bandwidth, power and peak process memory were not captured.

## Reproduce

Mount the checkout at `/work`, existing model directory read-only at `/models`, and a writable build/evidence directory at `/evidence`. Expose only the selected GPU's card/render nodes. Use the image and limits above, with 8 GiB shared memory and host mapping `127.0.0.1:18761:8080`. Inside the container:

```sh
source /opt/intel/oneapi/setvars.sh
cd /work
cmake -S sycl -B /evidence/build -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON
cmake --build /evidence/build --target strata-dsv41 -j4
python3 tools/dsv41/pack_gguf.py --model /models/UD-IQ1_M \
  --dspark-gguf /models/dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \
  --out /evidence/model-pack
apt-get update
apt-get install -y python3-venv
python3 -m venv /evidence/service-env
/evidence/service-env/bin/python -m pip install -r requirements.txt
mkdir -p /evidence/exact-perf
ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0 \
  /evidence/service-env/bin/python -m serve.server --engine strata \
  --config /work/bench/results/2026-11-13-community-b70-deepseek41-sycl-1k4k8k/config/service.json \
  --host 0.0.0.0 --port 8080 --no-api-key
```

On the host, from the checkout root, use the frozen corpus and HTTP client used for collection:

```sh
mkdir -p bench/results/2026-11-13-community-b70-deepseek41-sycl-1k4k8k/runs
python3 bench/results/2026-11-13-community-b70-deepseek41-sycl-1k4k8k/tools/performance.py
```

The runner writes `runs/performance01` and refuses to overwrite it. Configuration paths are normalized; measured argument values and request bodies are preserved. Input counts include the rendered chat template and were verified against API usage.

Only Linux single-B70 serial GGUF text serving is measured here. Component/service checks and their scope are described in the PR. No claim covers concurrent serving, other devices or full-model cross-backend equivalence.
