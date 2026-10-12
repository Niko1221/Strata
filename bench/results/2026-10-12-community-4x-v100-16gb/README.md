# Community benchmark on 4× Tesla V100-SXM2-16GB

Measured on 2026-10-12 by a community user. A 4-GPU V100 (sm_70) run of the
Unsloth `UD-Q4_K_XL` model with a bandwidth-proportional layer split, plus an
sm_70 `-DSTRATA_MMQ_KQUANTS=ON` experiment whose result (zero prefill gain) is
the main finding. Two context lengths, and a needle recall check (6/6 found at
32k and 128k).

## Hardware and software

- GPUs: **4× Tesla V100-SXM2-16GB** (sm_70, HBM2 4096-bit), TCC/compute mode
- CPU: 2× Xeon E5-2696 v4 (22C/44T each, 88 threads total), no AVX-512
- RAM: 247 GiB DDR4-2133
- Storage: Samsung PM981 1 TB NVMe
- PCIe (Gen3, asymmetric): GPU0 **x8** (6.15 GB/s H2D), GPU1 x16 (11.89), GPU2 **x8** (5.90), GPU3 x16 (10.29). One card (GPU0) was previously x4 before a re-seat; link width matters a lot for this workload.
- OS: Ubuntu 26.04.1 LTS, kernel 7.0; NVIDIA driver 580.178.04; CUDA 12.9.86
- Strata: **v0.1.42 source build** (`-DCMAKE_CUDA_ARCHITECTURES=70`), plus
  `-DSTRATA_MMQ_KQUANTS=ON` (see Findings — this changes nothing for prefill here)

## Model and configuration

- Model: `Qwen3.8-Flash-Next-UD-Q4_K_XL` (Unsloth), 4 shards, 111.33 GiB total
- Pack: native-experts pack built with `tools/iq_pack.py --compat-bf16`
  (`native_experts.txt` v4, 1079 tensors, 303 native / 460 converted)
- Context: 262144; `--kv int8 --kv-resident 32768 --ple-io ram`
- Expert cache: `auto` (≈4370 slots/13.05 GiB per card, ~70% of experts resident)
- `--layer-split 9,25,34` (9/16/9/14 layers) — tuned per-card to PCIe bandwidth
- `--spec 4 --spec-min-p 0.8 --suffix-draft 8 --pipeline-windows 2 --trim-stage-weights`
- env: `STRATA_SM70_TABLE=1 STRATA_SELECT_SIMT=1 STRATA_SPEC_DEPTH=1 STRATA_KV_PREFETCH=1`

```text
engine-cuda12/strata --serve \
  --pack <pack>/unsloth-ud-q4_k_xl \
  --native <model>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.8 --suffix-draft 8 --mtp <mtp>/rt \
  --max-context 262144 --kv int8 --ple-io ram --kv-resident 32768 \
  --prompt-cache 24 --prompt-cache-every 8192 \
  --conversation-cache-mib 16384 --conversation-cache-slots 6 \
  --trim-stage-weights --pipeline-windows 2 --layer-split 9,25,34
```

## Method

Two fixed prompts, fresh random seeds per run, cold (no prefix reuse), timing
from the engine log (`prompt N tokens = R reused + F read in M ms`):

- **2K**: ~600-word prompt (~2050 tokens), 384-token cap, 5 runs
- **81K**: ~24000-word prompt (~81K tokens), 32-token cap, 3 runs

Prompt throughput = freshly-read tokens / read time; decode = generated tokens /
generate time (both from the engine log, so prompt processing is not folded into
decode). TTFT ≈ prompt read time (first token follows immediately). Reused-token
follow-up rows were filtered out; all reported rows are 0-reused cold reads.

## Results

| Config | Prompt tokens (actual) | Reused | Generated | Runs | Prompt tok/s median [range] | Decode tok/s median [range] | TTFT s median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 2029–2131 | 0 | 324–384 | 5 | **458.5** [451.0–462.3] | 95.7 [89.0–140.5] | ≈4.5 |
| 81K | 81061–81259 | 0 | 17 | 3 | **2989.0** [2961.0–3166.8] | 77.5 [75.7–85.8] | ≈27.2 |

Steady-state prompt throughput from a two-point `T = F + N/R` fit is **R ≈ 3490 tok/s**
with **F ≈ 3.9 s** fixed overhead. Decode is draft-acceptance dependent and varies
with content; the short-context ceiling is ~160 tok/s, long-context ~86 tok/s (KV
beyond `--kv-resident` streams from host RAM).

Per-run JSON: `bench_v100_final-2k.json`, `bench_v100_final-81k.json`; layer-split
sweep (6 arms) in `sweep_q4kxl_results.json`; exact config in `launch-config.json`.

## Findings

1. **`-DSTRATA_MMQ_KQUANTS=ON` gives zero prefill gain on sm_70.** We compiled the
   MMQ Q4_K/Q5_K/Q5_1 kernels (`mul_mat_q_case` host symbols 9→12, `.nv_fatbin`
   +9.47 MB) and confirmed the path was taken (`STRATA_PREFILL_MMQ` on vs off changes
   the prompt-path slot borrow 1635→1606). Prefill was unchanged within noise
   (2K 313.9 vs 315.8; 81K 1949 vs 1962). The reason: on this machine the prefill
   bottleneck is **PCIe streaming of non-resident experts**, not the dequant→FP16
   GPU work — the official "4× less GPU time" was measured on sm_86 (int8 tensor
   cores), and does not transfer to sm_70 (DP4A only).
2. **Bandwidth-proportional layer split matters.** With the asymmetric x8/x16/x8/x16
   links, the naive `11,25,37` put 12 layers on an x8 card (stage time ∝ layers/H2D),
   making it the bottleneck. Rebalancing to `9,25,34` (layers ∝ H2D) lifted 2K
   prefill 397.9 → 449.2 (**+12.9%**) and end-to-end +8.7%, and made repeated 81K
   requests stable (the old split degraded 2744→1401→149 tok/s across three runs;
   the new split holds ~3000).
3. **`STRATA_SM70_TABLE=1`** was kept on throughout (consistent with the earlier
   V100 sm_70 report); no A/B was re-run here.

## Correctness and limitations

- **Needle-in-a-haystack: 6/6 found** via `tools/needle_bench.py` at 32k and 128k,
  depths 10/50/90 (see `needles.json`). Repeated 128k requests were stable
  (no throughput collapse), consistent with the rebalanced layer split.
- TTFT is approximated from engine-log read time, not a client-side first-token timestamp.
- Decode figures are single-session and draft-dependent; treat the 81K decode as indicative only.
- One observed operational quirk: long-context (81K) traffic could leave the host in a
  degraded memory state that survived engine restarts and cleared only on reboot; this did
  not reproduce under the rebalanced `9,25,34` split (not root-caused further).
  **Root-caused on 2026-10-13**: it is not the host but conversation-cache parking —
  see `bench/results/2026-10-13-community-4x-v100-16gb-convcache` (+10.5% 81K prefill
  with `--conversation-cache-mib 0`).
