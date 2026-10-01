# gfx1101 deployment characterization (RX 7800 XT, engine 0.1.30)

RX 7800 XT (device ID 0x747e, 60 CU, wave32, 16 GiB, reports gfx1101),
Ryzen 9 5950X, 121 GiB RAM. Engine 0.1.30 (the tree this branch is rebased
onto), hipBLASLt prefill with the calibrated
`tools/hip/gfx1101-hipblaslt-100401.txt` table (26 rows; loaded at startup).

Probe: `tools/hip/bench_prefill.py` (warmup, then 4 fresh / 4 follow-up
trials of 128 generated tokens; timings parsed from the engine log).
Fresh prompts are 4,210 or 8,830 tokens; follow-ups reuse the cache and add
113-248 new tokens. `reasoning_effort: "none"` (the models are reasoning
models; without the flag the answer lands in `reasoning_content`).

| model | context | expert cache (VRAM) | fresh prefill | decode (128 tok) |
|---|---|---|---:|---:|
| Coder IQ1_M | 65K | 4,873 experts, 9.27 GiB | 898-956 tok/s (avg 920) | 39-43 tok/s (avg 41) |
| Q2_0 (2-bit) | 256K | 7,551 experts, 9.72 GiB | 909-957 tok/s (avg 929) | 46-52 tok/s (avg 50) |

`coder-iq1m-0130-65k.json` — Coder IQ1_M at 65K. `q20-0130-256k.json` — Q2_0
at the model's full 256K window: the KV cache streams to RAM (~3.4 GB,
`--kv-resident`) and the GPU expert cache stays at 9.72 GiB of the 16 GiB.
Expert arenas: 23.4 GiB (Coder) / 31.6 GiB (Q2_0) of the 121 GB.

Notes:

- A Q2_0 run at 65K (engine 0.1.25, before the rebase) gave the same prefill
  (~898-953 tok/s) and decode (38-44 tok/s), so the 0.1.30 rebase changed
  nothing measurable for this deployment.
- An earlier run measured under concurrent host load (CPU-side agent traffic)
  was within noise of the idle runs, and re-running the hipBLASLt tuner under
  that load produced a byte-identical table.
- This is a deployment characterization on gfx1101, not a benchmark against
  the gfx1100 numbers in
  [AMD_HIP_PERFORMANCE.md](../../docs/AMD_HIP_PERFORMANCE.md); the cards and
  models differ.