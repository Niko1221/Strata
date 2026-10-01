# Two V100 GPUs with unequal PCIe links

## Test system

- GPUs: two NVIDIA Tesla V100-PCIE-16GB cards, 16 GiB per card.
- Physical GPU0: PCIe Gen3 x2. Physical GPU1: PCIe Gen3 x16.
- CPU: AMD Ryzen 5 3600, 6 cores and 12 threads. The AMD display GPU was not used.
- RAM: about 46 GiB of usable physical memory. Swap: 8 GiB.
- Model storage: Kingston NVMe SSD, ext4 filesystem.
- OS: Linux 7.0.0-34-generic, x86-64.
- NVIDIA driver: 580.178.04. CUDA compiler: 12.8.93.
- Build: Release, CUDA architecture 70, native experts enabled, `GGML_NATIVE=ON`.
- Base source revision: `e3375c861954b98f8200522c49f464aac1bb8c3c`.
- Model: Qwen3.8-Flash-Next-GSQ-RCO, Q2_0, with the native expert pack and expert profile.
- Context capacity: 262,144 tokens. KV format: int8.
- GPU power limits: 250 W per card. Memory clock: 877 MHz. Observed SM clocks: 1,245–1,380 MHz.
- Observed GPU temperatures: 62–83 degrees C. Clocks were not locked. Thermal state can change the results.

## Code change

The 8 GiB arena-registration limit was intended for Windows WDDM. Before this change, the same limit applied to Linux when two GPUs were used. The startup log reported 12 registered layer slices, or about 7 GiB. The later GPU stage could not copy its experts directly from the full host arena.

The limit now applies only to Windows. Linux first attempts to register the full arena. The existing per-layer fallback remains available if the full registration fails. On this system, the changed engine registered all 31.64 GiB with `cudaHostRegister PORTABLE ok`.

Full registration also selects the existing 384-slot prompt streaming ring. No expert, attention, or quantization kernel was changed.

## Decode cleanup and correctness

The host pool no longer writes zeros into rows owned by a published GPU plan. The GPU writes these zeros before it adds the expert results. Both the normal reader and the device-plan reader ignore stale host values for these rows. The no-plan path and the single-token dispatch keep their required host zeros.

The verify graph also skips the float-to-BF16 conversion when the shared expert uses its native FP32-input gate. The fallback scalar gate still receives its BF16 input. This removes unused work without changing projection arithmetic.

An attempted Q8_1 image-sharing change was rejected. The native projection quantizer is compiled with `--use_fast_math`; the routed-expert quantizer is not. On 32,768 constructed rounding-boundary blocks, the two kernels produced 114,165 different quant bytes. Their scale and sum bytes matched. The original quantizers, buffers, and compiler flags remain unchanged.

Verification:

- Six CUDA parity tests passed, including the new poisoned-host-row regression.
- Real Q2_0 expert checks passed on layers 0, 1, 2, 3, 20, and 47.
- Three 128-token completions matched the pre-cleanup engine exactly in normal mode.
- Three more matched with device planning and split verify groups enabled.
- Unbatched mode showed a case-B difference in an original-versus-original control. The new engine matched all three completions from the second original control. Do not treat this ablation as fully deterministic.

The completion comparisons used a fixed expert set (`--adapt-swaps 0`), a fixed per-request PCIe share of 0.28, greedy sampling, seed 123, and a fresh engine. These are parity controls, not the selected performance settings.

[Machine-readable correctness evidence](../bench/results/2026-09-30-asymmetric-v100/verification.json).


## Selected runtime settings

Use this profile for a balance between prefill and decode on this system:

```json
{
  "gpu": [0, 1],
  "layer_split": "20"
}
```

Keep the existing model paths, expert profile, and credentials. In the engine `args` list:

- Keep `--expert-cache auto`, `--prefill auto`, `--max-context 262144`, and `--kv int8`.
- Keep `--spec-min-p 0.70`.
- Remove `--pcie-frac` and its value. Do not override the PCIe share in a request.
- Do not set a custom prompt ring or lending percentage.

GPU0 runs layers 0–19. GPU1 runs layers 20–47. Each GPU probes its own link. The measured shares were 0.00 on the x2 link and about 0.28 on the x16 link. The measured host-to-device rates were about 1.5 and 13.1 GB/s.

The automatic layer placement remains a decode-cost model. It does not include prompt transfer bandwidth. Therefore, this profile uses an explicit placement. It is not a new default for other hardware or models.

## Paired repeat results

The baseline used the installed engine before the patch, automatic layer placement (24/24), and an explicit PCIe share of 0.28. The optimized arm used the final rebuilt engine, the safe decode cleanup, and the profile above. This comparison measures all retained code and runtime changes together.

Each arm started with a fresh engine. Both arms used seed 42, three repeats per size, and a 256-token output budget. Exact prompt counts matched between arms: 8,818–8,820 and 29,527–29,529 tokens. Every row generated 256 tokens and reused zero prompt tokens. Expert caches could adapt between rows in each arm. Timings came from the engine through `/metrics`, not from client wall time.

| Prompt size | Baseline prefill median | Optimized prefill median | Change | Baseline decode median | Optimized decode median | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ~8K | 438.3 tok/s | 656.9 tok/s | +49.9% | 56.3 tok/s | 69.4 tok/s | +23.3% |
| ~32K | 574.8 tok/s | 1,049.4 tok/s | +82.6% | 55.3 tok/s | 67.9 tok/s | +22.8% |

Prefill improved in all six paired rows. Decode improved in five of the six rows. The decode ranges were 55.0–57.2 versus 63.5–70.1 tok/s at ~8K, and 49.1–59.0 versus 54.9–68.3 tok/s at ~32K. Three repeats are not a statistical significance test. These gains apply to this workload and test sequence, not to every prompt.

Raw data: [baseline repeats](../bench/results/2026-09-30-asymmetric-v100/baseline-repeats/matrix.json), [final optimized repeats](../bench/results/2026-09-30-asymmetric-v100/complete-repeats/matrix.json).

An intermediate arm, before the decode cleanup, measured 632.3 / 56.3 tok/s at ~8K and 742.4 / 53.8 tok/s at ~32K. Its decode result did not show a clear gain. [Intermediate raw data](../bench/results/2026-09-30-asymmetric-v100/final-repeats/matrix.json). Each arm used a separate fresh engine. Clock, thermal, and cache changes prevent attribution of the full difference to one cleanup in isolation.

## Tuning trials

These trials used the same prompt sequence, seed 42, and 256 output tokens. They are single runs, not repeat medians. The original-engine trial used the running service. Each changed-engine trial started with a fresh engine. All rows reused zero prompt tokens. Except for the first two rows, each GPU used its own PCIe probe.

| Trial | ~8K prefill / decode tok/s | ~32K prefill / decode tok/s | Raw data |
| --- | ---: | ---: | --- |
| Original engine, 24/24, fixed PCIe share | 451.5 / 61.6 | 744.6 / 51.6 | [matrix](../bench/results/2026-09-30-asymmetric-v100/seeded-baseline/matrix.json) |
| Full registration, 24/24, fixed PCIe share | 524.5 / 54.6 | 809.6 / 52.5 | [matrix](../bench/results/2026-09-30-asymmetric-v100/fixed/matrix.json) |
| Full registration, 24/24, per-link shares | 524.4 / 63.1 | 795.8 / 53.4 | [matrix](../bench/results/2026-09-30-asymmetric-v100/auto/matrix.json) |
| GPU0 first, 20/28 | 642.4 / 63.4 | 892.3 / 53.8 | [matrix](../bench/results/2026-09-30-asymmetric-v100/k20/matrix.json) |
| GPU0 first, 18/30 | 690.7 / 54.0 | 935.0 / 64.4 | [matrix](../bench/results/2026-09-30-asymmetric-v100/k18/matrix.json) |
| GPU0 first, 16/32 | 756.9 / 56.0 | 1,045.8 / 53.1 | [matrix](../bench/results/2026-09-30-asymmetric-v100/k16/matrix.json) |
| 20/28, 128-slot prompt ring | 683.2 / 64.4 | 815.8 / 52.7 | [matrix](../bench/results/2026-09-30-asymmetric-v100/k20-ring128/matrix.json) |
| 20/28, 16,384-token chunk ceiling | 680.1 / 56.2 | 788.7 / 51.3 | [matrix](../bench/results/2026-09-30-asymmetric-v100/k20-chunk16k/matrix.json) |
| GPU1 first, 28/20 | 507.3 / 62.3 | 727.8 / 61.2 | [matrix](../bench/results/2026-09-30-asymmetric-v100/reverse28/matrix.json) |

The 16/32 placement gave the highest prefill rate in these trials. Reversed GPU order gave a higher ~32K decode rate, but lower prefill. Neither result proves a universal optimum. The 20/28 profile was selected for balanced use.

A separate three-repeat arm, before the decode cleanup, lowered the speculative confidence threshold from 0.70 to 0.40. Decode medians fell from 56.3 to 55.5 tok/s at ~8K and from 53.8 to 52.4 tok/s at ~32K. The selected profile keeps 0.70. [Raw data](../bench/results/2026-09-30-asymmetric-v100/spec04-repeats/matrix.json).

## Full-context smoke

The final engine completed the full runner matrix after the paired repeat arm. Seed 44 supplied fresh prompt prefixes. Each row generated 64 tokens and reused zero prompt tokens. This was one run with warm, adaptive expert caches, not a paired baseline comparison.

| Actual prompt tokens | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 4,142 | 499.9 | 62.8 |
| 4,376 | 512.5 | 65.2 |
| 8,818 | 658.3 | 64.0 |
| 29,528 | 1,049.3 | 62.6 |
| 117,837 | 934.4 | 50.6 |
| 256,076 | 659.5 | 45.7 |

[Final full matrix](../bench/results/2026-09-30-asymmetric-v100/complete-matrix/matrix.json). An earlier long-context smoke, before the decode cleanup and with seed 43, is also available as [raw data](../bench/results/2026-09-30-asymmetric-v100/long-context/matrix.json). These long-context runs differ in prompt prefixes and cache history. Do not use them as an isolated before-and-after decode comparison.

## Reproduction

Build the configured Release tree:

```sh
cmake --build build-perf --parallel 2
```

Point the server at the built engine. Apply the runtime settings above for the optimized arm. Use the original engine and settings for the baseline arm. Restart the engine before each seeded arm to clear the conversation cache.

```sh
.venv/bin/python bench/run_v100_bench.py \
  --model-gguf "$MODEL_GGUF" \
  --only '~8K,32K' --seed 42 --repeats 3 --max-tokens 256 \
  --out "$RESULT_DIR"
```

`MODEL_GGUF` is the first model shard. `RESULT_DIR` is a separate output directory for each arm. The runner reads the local service API key and does not record it. The runner rejects a row if it reuses any prompt tokens. The new seed and repeat options preserve the original 64-token default when no output budget is supplied.

For the full matrix, omit `--only` and `--repeats`. Use `--seed 44 --max-tokens 64` and a new output directory. Run it after the paired repeats without an engine restart to reproduce the warm-cache sequence above.
