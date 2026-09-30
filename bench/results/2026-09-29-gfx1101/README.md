# gfx1101 deployment characterization (RX 7800 XT)

RX 7800 XT (device ID 0x747e, 60 CU, wave32, 16 GiB, reports gfx1101),
Ryzen 9 5950X, 121 GiB RAM. Model: Qwen3.8-Flash-Next Coder IQ1_M
(`qwen3.8-flash-next-coder-iq1_m`), 64K context, MTP on, KV cache streamed to
RAM (0.9 GB), hipBLASLt prefill with the calibrated
`tools/hip/gfx1101-hipblaslt-100401.txt` table (26 rows; loaded at startup).

Probe: `tools/hip/bench_prefill.py` (warmup, then 4 fresh / 4 follow-up
trials of 128 generated tokens; timings parsed from the engine log).
Fresh prompts are 4,210 or 8,830 tokens; follow-ups reuse the cache and add
113-248 new tokens.

| run | fresh prefill | decode (128 tok) |
|---|---:|---:|
| idle GPU | 898-953 tok/s (avg 918) | 38-44 tok/s (avg 42) |
| concurrent agent load on the host | 846-903 tok/s (avg 887) | 37-44 tok/s (avg 40) |

The two runs are within noise of each other, so the concurrent host load
(CPU-side agent traffic) did not measurably perturb this GPU-bound
deployment. Prefill holds ~900 tok/s from 4K to 9K tokens.

`gfx1101-bench-prefill.json` — loaded run. `gfx1101-bench-prefill-idle.json` — idle run.

Notes:

- This is a deployment characterization (the first on gfx1101), not a
  benchmark against the gfx1100 numbers in
  [AMD_HIP_PERFORMANCE.md](../../docs/AMD_HIP_PERFORMANCE.md); the cards,
  models and engine paths differ.
- The tuning table was re-generated on the idle GPU and is byte-identical to
  the table generated under load, so the load also did not skew the
  solution ranking.
