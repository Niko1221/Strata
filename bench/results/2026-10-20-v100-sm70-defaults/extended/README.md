# Historical Strata optimizations on this V100 system — 2026-10-20

Scope: current Qwen IQ3_S / dual PCIe V100 32 GB / 150 W each / 32K / two slots. Review release documentation through v0.2.1, measure applicable optional paths in the installed v0.2.0-beta engine, separating numerical changes from exact-kernel candidates. This is not exhaustive testing of all models, hardware combinations or every possible parameter value.

Sources: upstream docs/NVIDIA_V100.md, docs/DETAILS.md, src/core/nv_defaults.cpp and CUDA/prefill dispatch code; releases v0.1.39 through v0.2.1. Latest release introduces serial batch-group selection for two NVIDIA devices; older engine can select one group explicitly.

## Candidate inventory

- Already enabled/tested: SIDE_MIXER2, GR_UP751, B6_MMVQ_ROWS1, GU_SWIGLU1, QSA_ROWS1, ATTN_DAILY1. Production presence penalty0 and later-GPU reserve650 kept fixed.
- V100 decode table: SM70_TABLE1 (interleaved matrix-vector kernels, expert mode8, norm/up).
- Decode fusion: FUSE_DECODE1 + FUSE_WAIT2.
- Norm/greedy pick: GR_NORM_FAST1 + ARGMAX_ROWS_SM701.
- INT8 KV gather alternative: Q8_GATHER_WIDE0.
- Resident-expert window kernels: EXPDEC1.
- Down projection grouping: GR_DOWN_TG1.
- Prompt scorer: SCORES_THREAD1.
- Prompt FP16 tensor-core experts: PF_WMMA1 + PF_WMMA21; changes arithmetic vs default MMQ, cannot infer quality preservation from throughput alone.
- Dense-weight dequantization: DEQUANT_WIDE1.
- Selection: SCORES_HALVE1 + TOPK_SPLIT1 + TOPK_SAMPLE1.
- Quantization/interleave fusion: QIL1 with SM70_TABLE1.
- DeltaNet recurrence: GDN_REC_WARP1 exact control; GDN_REC_WARP2 changes summation order.
- Prompt memory planner: EMB_REUSE_ACCOUNT1; may change chunks and rounding.
- Host thread placement: AUX_CPUSauto. STAGE_PIN was initially listed but excluded after finding upstream IQ3_S corruption warning; a later-priority test drop-in forces STAGE_PIN=0 for both host and all-candidate arms.

## Exclusions / conditions

- HIP/RDNA, SYCL/Intel and Metal kernels cannot run on NVIDIA V100.
- sm80+ INT8 tensor-core attention/QSA split/upmix and sm90+ cluster kernels do not apply to sm70. Turing PA_V1_PREFETCH is bypassed by Volta's dedicated prompt attention dispatch; setting it would not test that kernel.
- CPU expert exact-row kernels, adaptive tier swaps, PCIe share balancing, route-tail skipping and expert file I/O prefetch cannot improve this fully resident decode path. Some require a single GPU or exclude batch/layer split. Do not force a different memory topology merely to claim coverage.
- GDN chunked paths documented as slower than the existing path on V100 are not presumed improvements.
- ROUTE_PRIOR/RESIDENT and speed-projection options intentionally alter routing/model behavior; they are quality/approximation experiments, not output-preserving runtime upgrades.
- KV host registration is mainly a WSL pinning workaround; this system is native Linux, context fits the configured KV window.
- MMQ_DIRECT requires an additional STRATA_MMQ_TAB build and extra device-code/cache memory; not available in the installed engine. A separate CUDA12.8/sm70 build was subsequently made and tested (below).
- Batch grouping should be measured with simultaneous requests; a solo tok/s test cannot evaluate it.

## Protocol

Each arm restarts the same server with temporary environment additions to the production six-switch stack. Nine original short requests (three per workload) plus one repeated synthetic document: 15,352 input tokens / 64 output tokens, no prompt reuse. Decode uses engine log tok/s; long-input metric uses engine prefill tok/s. Output texts compared against matching baseline requests. Temporary override removed and production restored in finally. Thermal/fan guards stay active.

## Initial findings

The complete exact-candidate stack is not uniformly faster: decode improves but the exact DeltaNet recurrence lowers long-input prefill. The reduced candidate combines SM70_TABLE1, GR_NORM_FAST1, ARGMAX_ROWS_SM701, QIL1, EXPDEC1, GR_DOWN_TG1 and DEQUANT_WIDE1 on top of the production six-switch stack. Initial measurement: Japanese82.4/code122.4 tok/s, 15,352-token prefill1519.2 tok/s, all10 output texts identical. Final stability checks passed (below).

PF_WMMA1/PF_WMMA21 improved long-input prefill to1625.9 tok/s but changed2/10 output texts. GDN_REC_WARP2 changed3/10 without a meaningful speed gain. Neither is selected for production. GDN_REC_WARP1 preserved outputs but reduced prefill to1425.2 tok/s. The all-candidate trial forced STAGE_PIN0 because of the upstream corruption warning.

## Measured speed matrix

Each complete arm has 10 requests. Japanese/code are three-run means; prefill is one cold 15,352-token synthetic prompt, so small prefill differences are screening observations, not statistically established gains.

| Arm | Japanese tok/s | Code tok/s | Prefill tok/s | Matching output texts |
|---|---:|---:|---:|---:|
| history-base1 | 78.7 | 116.9 | 1511.7 | 10/10 |
| history-base2 | 78.8 | 117.1 | 1513.9 | 10/10 |
| history-down | 79.0 | 117.5 | 1508.8 | 10/10 |
| history-expdec | 78.8 | 117.8 | 1514.5 | 10/10 |
| history-fuse | 78.7 | 116.8 | 1508.5 | 10/10 |
| history-gather | 78.7 | 117.0 | 1513.6 | 10/10 |
| history-normpick | 79.9 | 118.4 | 1512.5 | 10/10 |
| history-scores | 78.7 | 117.0 | 1506.8 | 10/10 |
| history-sm70 | 80.6 | 119.7 | 1511.8 | 10/10 |
| history-wmma | 78.7 | 117.0 | 1625.9 | 8/10 |
| history2-all-exact | 82.5 | 122.6 | 1432.5 | 10/10 |
| history2-base1 | 78.7 | 117.0 | 1509.6 | 10/10 |
| history2-base2 | 78.6 | 116.7 | 1513.6 | 10/10 |
| history2-combined | 81.8 | 121.3 | 1511.0 | 10/10 |
| history2-emb | 78.7 | 116.9 | 1518.3 | 10/10 |
| history2-gdnexact | 78.7 | 117.0 | 1425.2 | 10/10 |
| history2-gdnfast | 78.8 | 117.1 | 1516.0 | 7/10 |
| history2-host | 78.5 | 116.7 | 1510.3 | 10/10 |
| history2-qil | 80.9 | 120.1 | 1509.9 | 10/10 |
| history2-select | 78.7 | 116.9 | 1512.7 | 10/10 |
| history2-wide | 78.8 | 117.1 | 1529.7 | 10/10 |
| history3-base | 78.7 | 116.8 | 1509.1 | 10/10 |
| history3-refined | 82.4 | 122.4 | 1519.2 | 10/10 |
| mmq-base1 | 78.7 | 116.9 | 1508.5 | 10/10 |
| mmq-base2 | 78.8 | 117.1 | 1511.6 | 10/10 |
| mmq-copy | 78.8 | 117.1 | 1510.8 | 10/10 |
| mmq-direct | 78.7 | 116.9 | 1525.8 | 10/10 |

## Final selection and production state

Applied seven additional settings on top of the previous six-switch stack:

```
STRATA_SM70_TABLE=1
STRATA_GR_NORM_FAST=1
STRATA_ARGMAX_ROWS_SM70=1
STRATA_QIL=1
STRATA_EXPDEC=1
STRATA_GR_DOWN_TG=1
STRATA_DEQUANT_WIDE=1
```

Selection is based on combined performance, not proof of an independent gain from each flag. The refined combination measured82.4 Japanese /122.4 code tok/s versus paired baseline78.7/116.8 (about4.7%/4.8%). All10 texts match. Its15,352-token prefill1519.2 vs1509.1 is essentially unchanged given one long sample per arm. Slower/no-gain/answer-changing options were omitted.

Final32K stress: two simultaneous32000-input/512-output requests, twice; all4 passed both sentinels, no reused input. Pair completions43.55/43.58s. Core peaks56/55°C, HBM57/58°C, minimum free VRAM4397/267MiB. No GPU-error matches in tested logs.

### Two-card batch grouping

v0.2.1 release proposes a default of one group for exactly two NVIDIA cards. Tested equivalent `--batch-groups 1` on the same installed engine, same selected optimizations and32K/two-slot workload. Pair completion59.06/58.80s, versus43.55/43.58s with two groups. All4 requests passed sentinels, but only2/4 whole texts matched the two-group arm. One group is not adopted; this workload is dominated by long input and is not a general aggregate-decode benchmark.

The tested deployment explicitly keeps `--batch-groups 2`. This PR does not change the global batch-group default.

### Direct MMQ build experiment

Built the existing v0.2.0-beta sm70/CUDA12.8 source tree with `STRATA_MMQ_TAB=ON`. Compared DIRECT0, DIRECT2 (same layout but copies), DIRECT1, DIRECT0 again. All10 output texts matched the official-engine baseline. DIRECT1 really used resident experts in place (engine log counters); all12288 expert pairs remained cached. Decode unchanged (~78.7/116.9). Prefill1525.8 versus1508.5/1510.8 screening references: only about1%, not a demonstrated substantial end-to-end win. Kept official engine in production; optional build retained for reproducibility, no unrelated model files removed.


## Reproduction and raw evidence

Use the hardware, model, memory reserves and sampling settings in the parent README. Every extended arm adds its listed settings to the initial six-switch baseline. Run `STRATA_API_KEY=... BENCH_OUTPUT_DIR=./results python3 measure.py LABEL PORT ENGINE_LOG` against an otherwise idle local server; restart the engine between arms. The script preserves the exact synthetic prompts and hashes outputs rather than publishing their text. `summary.csv` is derived from the attached per-request JSON; all screening arms, including output-changing and slower candidates, are included. The label `prefill16k` means 15,352 actual prompt tokens, not 16,384.

The measured engine was v0.2.0-beta (official engine 0.1.45.3); reviewing release notes through v0.2.1 does not make these v0.2.1 performance results. The optional MMQ experiment used a separate CUDA 12.8/sm70 build. These measurements do not validate a current-main GPU build of the proposed defaults. Decoded-text equality is not a token-id/logit identity or quality guarantee. Single V100, 16 GB, SXM2, other models and longer endurance remain untested.
