# PR-source CUDA build and runtime validation — 2026-10-21

Tested code commit: `74976e928fd10736ece4035fc061e19d14d3851b`, based on main `9ae93f7b1fe989d04a8e404d32cddaeeff10505f`. Subsequent report-only commits do not change this tested code. Unlike the earlier release experiments, these measurements run the actual proposed defaults in a freshly built engine and the server from the same source tree.

## Build

Linux, GCC 13.3, CUDA 12.8, sm70; Ryzen 9 9950X, 128 GB RAM, dual V100 PCIe 32 GB at 150 W each. CUDA-only build. HIP/SYCL hardware and toolchains are unavailable and were not validated; the NVIDIA defaults call is guarded out of HIP/Metal execution, and no non-NVIDIA kernel code is changed by this PR.

```sh
cmake -S . -B build-v100 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/opt/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=OFF \
  -DSTRATA_MMQ_TAB=OFF -DSTRATA_BUILD_TESTS=ON
cmake --build build-v100 --target strata nv_defaults_test -j 12
ctest --test-dir build-v100 -R '^nv_defaults_test$' --output-on-failure
```

Build completed (158 steps); CTest passed 1/1. An initial configure without `STRATA_EXPERIMENTAL_SM60=ON` correctly rejected sm70; adding the documented option resolved it. Attached configure/build logs have machine-local paths replaced with `/work/Strata` and `/opt/cuda-12.8`.

Engine SHA-256: `76829f0f59ccba261462d52343016c2d481fcbe26eb1deaefe5af9615b793aa1`.

## Actual automatic dispatch versus manual configuration

Use the same model and fixed configuration as the parent report, including explicit `--batch-groups 2` and later-GPU reserve 650 MiB. PF_WMMA and PF_WMMA2 were both 0. Run the extended report's `measure.py` (same exact synthetic prompts and sampling) on a fresh server per arm:

- Manual: all 13 proposed settings supplied explicitly; `STRATA_SM70_PERF=0` disables automatic selection.
- Auto: remove all 13 individual variables from the process environment; `STRATA_SM70_PERF=1`. Startup reported all 13 applied:

```text
strata: sm_70 (Volta): 13 exact speed switches on by default (STRATA_SM70_PERF=0 turns them off; a switch you set is kept): SIDE_MIXER GR_UP75 B6_MMVQ_ROWS GU_SWIGLU QSA_ROWS ATTN_DAILY SM70_TABLE GR_NORM_FAST ARGMAX_ROWS_SM70 QIL EXPDEC GR_DOWN_TG DEQUANT_WIDE
```

| Metric | Manual | Automatic |
|---|---:|---:|
| Japanese decode, three-run mean (tok/s) | 82.47 | 82.43 |
| Code decode, three-run mean (tok/s) | 122.40 | 122.43 |
| 15,352-token prefill, one request (tok/s) | 1,522.4 | 1,524.3 |

All 10 output hashes matched. Every request had exactly one engine timing record. This establishes equivalent behavior to the manually configured stack on the tested machine, not a fresh speedup estimate against untuned current main. No token-ID/logit identity claim is made.

## Long-context and concurrency

Automatic configuration: two simultaneous requests, each 32,000 input tokens plus 512 output tokens, repeated twice. Same synthetic cafe records and start/end sentinel task as the earlier stress report. All four passed, with zero cached tokens and 32,512 total tokens each.

- Pair completion: 43.572 / 43.597 s.
- Peak core temperatures: GPU0 52 C / GPU1 54 C; HBM 54 / 57 C.
- Minimum observed free VRAM: 4,375 / 235 MiB.
- No request failures or thermal-guard intervention observed. All requests completed without OOM.

This remains limited recall/liveness screening. Other models, single-card/16 GB/SXM2 hardware, broader numerical identity, and extended endurance are unverified. Maintainer review of the scope of automatic defaults is still required. The normal production service was restored after the experiment.
