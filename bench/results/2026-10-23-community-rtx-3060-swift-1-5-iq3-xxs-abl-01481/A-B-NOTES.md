# A/B addendum: engine 0.1.47 -> 0.1.48.1 (v0.2.3.1), same rig

Measured 2026-10-23 on the same hardware, same configuration, `tools/strata_bench.py --url`
against the model bound to 127.0.0.1 only. Comparison is against the 2026-10-22 abliterated
IQ3_XXS run (engine 0.1.47) of the same folder naming.

Hardware: 2x RTX 3060 12 GB (sm_86) layer split, Xeon E5-2697 v2 **without AVX2** (24 cores/48
threads, 2 sockets), 128 GB RAM, Windows 11. Source build `STRATA_ISA_FLOOR=avx`, CUDA 13.
Model: Swift-IQ3_XXS-abliterated (SC117), 128k context, --kv q4_0/--kv-resident 32768,
--prefill auto:32768, --pipeline-windows 2, --vram-reserve-mib 1344, Trim/MTP_STAGE0 env on.

Headline (median, range in the report folder):

| Metric | 0.1.47 | 0.1.48.1 | Delta |
| --- | ---: | ---: | ---: |
| short chat decode tok/s | 44.9 | 44.1 | -1.8 % (noise) |
| 4K prompt tok/s | 522.9 | 475.8 | -9.0 % |
| 16K prompt tok/s | 781.9 | 745.0 | -4.7 % |
| agent decode tok/s | 47.0 | 50.9 | +8.3 % |
| agent cache hit % | 69.5 | 69.6 | ~0 |

Reading comment for the no-AVX2 CPU path: prompt ingestion is a few percent slower on 0.1.48.1
here (4K -9 %, 16K -4.7 %), decode is flat-to-better (agent +8.3 %, chat ~unchanged), the
conversation-cache reuse on fresh agent sessions improved (re-read 73 -> 1). On this box the
large-document workload (agents send 50-95k-token prompts) weighs reading speed, so 0.1.48.1
was not adopted for production yet; a re-test when a later engine changes the prefill/reading
path is planned.

Vision re-test with the fixed reader (828a2559, #3690) on this rig: a picture that previously
became ~93,700 image tokens now stays bounded (test image = 78 image tokens), answer correct,
4.4 s total. The fix is confirmed working on dual 12 GB NVIDIA.

Per-run data: results.json. Nothing else changed between the two engine versions.