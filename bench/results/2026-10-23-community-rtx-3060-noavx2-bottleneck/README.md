# Bottleneck attribution on a no-AVX2 dual-12GB rig: why a custom no-AVX2 engine fork is low-ROI

A measurement study (not a speed report) on hardware the maintainers do not have: it says
*where* decode time goes on this box and *whether* an engine fork could help. Hardware:
Dell Precision T5610, 2x RTX 3060 12 GB (sm_86) layer split, 2x Xeon E5-2697 v2 (Ivy Bridge,
**no AVX2**), 128 GB DDR3-1866 quad-channel, Windows 11. Engine 0.1.47 source build
(`STRATA_ISA_FLOOR=avx`), model Swift-IQ3_XXS-abliterated (SC117), prod configuration.

## Method and findings

**1) Production telemetry (1,492 requests).** decode median 53.8 tok/s (p10 43.5 / p90 69.7),
fresh-ingest ~153 new tok/s, draft acceptance 68.2 %. Expert cache effective: ~93 % hit, only
~0.5 % of routed experts read over PCIe. **corr(decode, pcie_share) = -0.06** in production ->
decode does not track PCIe/miss load there.

**2) GPU load during decode is low.** Sampled `nvidia-smi` utilization during a sustained
localhost decode: both GPUs average ~40-50 % (median ~50 %, max ~100 %, only 3-8 % of samples
above 80 %). The GPUs are NOT compute-saturated -> decode is not GPU-compute-limited, and there
is idle GPU capacity (relevant to overlap / offload levers).

**3) CPU<->GPU overlap contends for bandwidth.** Ran upstream's `src/core/concurrent_main.cpp`
(`strata-concurrent`, source build, dummy arena, valid windows):

| mode | CPU DRAM read | GPU stream |
| --- | ---: | ---: |
| CPU only | 73.5 GB/s | - |
| GPU only | - | 15,589 roles/s |
| BOTH | 59.1 GB/s (-19.6 %) | 13,021 roles/s (-16.5 %) |

The CPU expert-path DRAM reads and the GPU H2D/streaming path share the DDR3/PCIe bandwidth;
running together costs each side ~16-20 %. So a "perfect" CPU+GPU overlap does NOT double the
decode - there is a real bandwidth-contention ceiling on this machine.

**4) The obvious overlap knob does not go higher.** `--pipeline-windows` is hard-capped at 2
(generate.cpp: `std::min(2, ..)`); we already run 2. There is no "3" to try.

## Conclusion

- Decode here is not bus/PCIe-bound in production, not GPU-compute-saturated; the CPU expert
  path and the serialization around it matter, but any CPU-kernel or overlap optimization is
  bounded by the measured ~16-20 % CPU/GPU bandwidth contention on DDR3-1866.
- The realistic upper bound of a custom no-AVX2 engine fork is ~10-20 %, against the recurring
  cost of rebasing a diverging engine - so we do not build one and stay on upstream, which keeps
  improving the AVX1/no-AVX2 path. Part of the value here is negative: a fork is not warranted.

(No per-request suite JSON - this is an attribution study; the numbers above are the measurements.)