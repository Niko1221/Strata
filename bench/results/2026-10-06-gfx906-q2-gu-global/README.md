# gfx906 Q2_0 gate/up: token-inner reuse without LDS staging

## Scope

Strata 0.1.40 at `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`.
The candidate is opt-in with `STRATA_EXP_MODE=13`; the default stays 7.
It is compiled only for `STRATA_HIP_GFX906` and dispatched only for Q2_0
gate/up with H=2560 and FF=640.

The existing mode-7 Q2_0 gate/up path uses the R2 AMD kernel. This candidate
keeps its 16-row block geometry, puts up to four tokens inside the weight
loop, and reads activations from global memory. A previous LDS-staged
candidate regressed singleton groups; it is not included here.

The packed two-bit codes are expanded exactly into signed bytes
{-1, 0, 1, 2}. The integer dot chain, float scale expression and per-row
summation order are retained. There is no weight repack, extra persistent
VRAM allocation, precision change, or change to down projection.

## Hardware and model

- Two gfx906 GPUs with 16 GiB each
- Xeon E5-2698B v3, AVX2, 128 GiB RAM
- shefowl/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Hybrid-GGUF,
  revision `55568c1b41d2e381a59a281447fb10935c3c6d0a`
- Variant: IQ3_XXS-Q2_0; routed experts are Q2_0, 512 per layer
- Capacity 204800; INT8 KV, 32768 resident KV tokens; prefill chunk 4096
- Layer split 27; 19078 expert slots, 25150 MiB expert cache
- PLE locked in RAM; MTP maximum 4 plus suffix lookup maximum 3
- Text-only benchmark; no prompt reuse; clocks were not locked

The pinned 0.1.40 tree needs gfx906 build compatibility fixes in
`fused_gr.cu`, the MTP stream-priority API, and `vmm.cpp`. Those fixes were
held constant in both arms and are excluded from this patch. See
[#1083](https://github.com/Niko1221/Strata/pull/1083) for the upstream build work.

### Software and additional run details

- Host: Ubuntu 26.04.1 LTS, kernel 7.0.0-30-generic; container: Ubuntu 24.04.4 LTS
- Source build: HIP 7.14.60850, AMD clang 23.0.0git; Release, STRATA_HIP_GFX906=ON, gfx906, STRATA_PORTABLE=ON
- Driver package version was not recorded. Exact compiler revision is in results.json
- Hybrid GGUF and native pack are on SATA SSD; PLE is locked in RAM
- Shards: Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0-00001-of-00002.gguf and the corresponding 00002-of-00002.gguf
- Local MTP pack: Q2_0 expert weights, Q8_0 dense weights, the default 106299-entry draft vocabulary. Pack/profile SHA256 values are in results.json
- 15 CPU pool workers; no other GPU workload during benchmarks; no calibration or speed projection enabled for this comparison
- A post-run sysfs snapshot reported 190 W caps and 16.0 GT/s x16 GPU-facing links on both cards. These were not logged during ABBA; the full upstream PCIe topology was not measured
- Each fresh engine preloads the profile-ranked VRAM tier. OS page-cache state was not controlled
- All native output tokens, including reasoning, are counted. TTFT, peak host RAM, process RSS and exact peak VRAM were not measured for this report; expert-cache allocation above is not a peak
- No failed model requests occurred in the reported ABBA series

## End-to-end measurements

Each ABBA series starts four fresh engines: mode 7, 13, 13, 7.
Each request generates exactly 1024 tokens, with EOS stopping disabled.
Temperature 1, top-p 0.95, top-k 20; seed 12345 for code, 54321 for Russian.
TG is generated tokens divided by native decode time; PP is input tokens
divided by native prompt time. Loading and API overhead are excluded.
Aggregate rates divide the total tokens by the total time, not a mean of rates.

| Prompt | Baseline TG | Candidate TG | Gain | Output IDs |
|---|---:|---:|---:|---|
| 4096, synthetic Python review | 58.1060 | 58.9643 | +1.477% | identical in all four runs |
| 65536, synthetic Python review | 46.4388 | 47.1017 | +1.428% | identical in all four runs |
| 65536, Russian queue review | 31.4692 | 31.8606 | +1.244% | see qualification below |

Both candidate repetitions beat both baseline repetitions in each series.
PP is effectively unchanged: about 340 tok/s at 4K and 584 tok/s at 64K.
Individual timings, draft counters and output hashes are in `results.json`.

**Russian qualification:** the baseline diverged from its own repeat at
zero-based output index 213. Baseline A1 and candidate B1 diverged at 386;
the two candidate runs agreed. Different outputs and draft acceptance
counts mean this row is observational, not a controlled equal-work gain.
The cause of baseline nondeterminism was not isolated. The two code series
are the equal-output performance evidence.

A clean-source 4K check reproduced all 1024 IDs from the earlier code series:
57.9510 -> 58.8171 tok/s. This extra pair is a verification check, not another
ABBA series.

### Per-arm medians and ranges

There are two measured runs per arm in each ABBA series; the guide suggests at least three for a general community report. This is a bounded optimization comparison. The extra cleaned-source pair is kept separate.

| Workload / mode | PP median [min, max] | TG median [min, max] |
|---|---:|---:|
| code4k / 7 | 339.903 [339.782, 340.024] | 58.106 [57.987, 58.226] |
| code4k / 13 | 339.893 [339.872, 339.914] | 58.965 [58.780, 59.150] |
| code64k / 7 | 584.361 [584.358, 584.364] | 46.439 [46.403, 46.475] |
| code64k / 13 | 584.281 [584.181, 584.382] | 47.102 [47.063, 47.141] |
| ru64k / 7 | 584.137 [584.046, 584.227] | 31.469 [31.468, 31.471] |
| ru64k / 13 | 584.212 [584.191, 584.232] | 31.861 [31.814, 31.907] |

The Russian rows retain the unequal-output qualification above.

## Component and numerical checks

- 280 real-weight cases on both GPUs, including layers 0/23/47,
  groups 1/2/8/32/64 and 1..8 tokens, plus mixed lengths and padded grids
- No measured component regression in that grid; singleton whole-expert
  speedup 1.0310..1.0496x, median 1.0381x
- 282 additional intermediate checks, including all 48 layers, both GPUs
  and seeds 17/1234
- Exact comparison of 70,744,320 gate/up/SwiGLU float values,
  26,529,120 quantized activation bytes and 94,325,760 output float values:
  zero differences
- Cleaned-source spot checks passed on both GPUs; the benchmark also
  passed a 256-expert Coder reference case after making expert counts dynamic
- The two-bit expansion was exhaustively checked for all 65536 packed
  16-bit codewords during development

The microbenchmark keeps its existing three interleaved timing rounds and
minimum-event-time statistic. These finite tests are not a proof for every
input or a broad model-quality evaluation. CUDA, other AMD architectures,
other model geometries, and a fully occupied 200K context were not tested.

## Reproduce the component checks

Build `strata` and `native_expert_bench` with the gfx906 configuration in
`docs/AMD_HIP.md` and the pinned llama.cpp dependency. Resolve the existing
0.1.40 build blockers above separately.

Set `MODEL` to Hybrid GGUF shard 1 and `BUILD` to the build directory.
Run without another GPU workload:

```sh
for gpu in 0 1; do
  for g in 1 2 8 32 64; do
    for t in 1 2 3 4 5 6 7 8; do
      HIP_VISIBLE_DEVICES=$gpu "$BUILD/native_expert_bench" "$MODEL" 0,23,47 "$g" "$t" 7 13 300
    done
  done
  for g in 3 8 31 64; do
    HIP_VISIBLE_DEVICES=$gpu STRATA_BENCH_MIXED=1 STRATA_BENCH_PAD=7 \
      "$BUILD/native_expert_bench" "$MODEL" 0,7,23,35,47 "$g" 8 7 13 300
  done
done
```

For the 282-case intermediate matrix, on each GPU run all layers 0..47
with G=32, T=8, mixed lengths, padding 7, and seeds 17 and 1234.
Then run layers 0/23/47 with G=1/8/64, T=1/2/4/6/8 and seed 17.
Ten iterations were used here because this pass checks values, not timing.

`STRATA_BENCH_MIXED` makes group g contain 1 + (g * 5) % T entries;
`STRATA_BENCH_PAD` adds inactive groups to the launch grid;
`STRATA_BENCH_SEED` overrides the activation seed.
The intermediate inspection applies only to Q2_0.

## Reproduce the model series

`fixtures.py` recreates the synthetic prompts. It requires the repository's
tokenizer dependencies and Hybrid shard 1. No private conversation is included.

```sh
REPORT=bench/results/2026-10-06-gfx906-q2-gu-global
python3 "$REPORT/fixtures.py" --model "$MODEL" --output /tmp/q2-prompts
```

Use `run.py` for one fresh native engine per measurement. The command after
`--` is executed without a shell; it must start the native serve protocol.
For example, set PACK, PLE, MTP and PROFILE to the corresponding local files:

```sh
HIP_VISIBLE_DEVICES=0,1 STRATA_ARENA_MMAP=1 STRATA_STAGE_TRIM=1 \
python3 "$REPORT/run.py" --ids /tmp/q2-prompts/code64k.ids.json \
  --mode 13 --seed 12345 --output /tmp/q2-B1.json -- \
  "$BUILD/strata" --serve --pack "$PACK" --native "$MODEL" \
  --ple-gguf "$PLE" --expert-profile "$PROFILE" --expert-cache auto \
  --prefill 4096 --spec 4 --spec-min-p 0.5 --mtp "$MTP" \
  --max-context 204800 --kv int8 --kv-resident 32768 \
  --vram-reserve-mib 600 --pcie-frac 0 --ple-io ram --layer-split 27 \
  --prompt-cache 0 --prompt-cache-every 0 --eos-ids 2147483647
```

Run modes 7,13,13,7 for each prompt with distinct output paths.
For `ru64k`, use seed 54321. Ensure enough locked-memory allowance for PLE.
The fixture generator was checked against all three measured token-ID arrays: exact matches.
The run helper passed a positive protocol parse and three rejection checks.
The portable helper was added to reproduce the native protocol used by the
measurement driver; it is not an API-server benchmark. Review local stderr
before sharing it, since engine logs may contain local paths.

## Existing work checked before submission

At review time, all 72 open PR titles/descriptions and targeted closed-PR
searches for gfx906, Q2_0, grouped experts and STRATA_EXP_MODE were inspected.
Relevant diffs were compared:

- [#638](https://github.com/Niko1221/Strata/pull/638): existing gfx906 backend and IQ LDS paths, already in the base
- [#1083](https://github.com/Niko1221/Strata/pull/1083): build fixes, excluded here
- [#1084](https://github.com/Niko1221/Strata/pull/1084): DP4A change in the wave32 compat header; this tested gfx906 path uses the separate platform compat header
- [#1127](https://github.com/Niko1221/Strata/pull/1127): signed-zero handling in Q2_0 dequantization for gfx1151, a different function/architecture
- [#1100](https://github.com/Niko1221/Strata/pull/1100) and [#1097](https://github.com/Niko1221/Strata/pull/1097): Volta QPN8/fusions, separate kernels and gating
- [#1049](https://github.com/Niko1221/Strata/pull/1049): fused quantizer finite-value handling, unchanged here

No direct duplicate was found in that review. No speedup from another PR is
included in the A/B claims above.

Developed with an OpenAI coding assistant. Measurements were run on the
hardware described above.
