# DFlash2 VRAM planning audit — 2026-10-19

Hardware: NVIDIA RTX 4070 Ti SUPER, 16376 MiB, Ryzen 9 5900X,
125.7 GiB RAM, Linux, CUDA 13.4, SM89 engine. The target is the local
Qwen3.8-Flash-Next NVFP4 pack with INT8 target KV and the shared Q5_K head.
The official BF16, Q8_0 and Q4_0 artifacts are present in
`../Strata-data/dflash2-official/`. No downloads or user configuration edits
were needed. Engines ran sequentially.

## Full allocation at 128K capacity

All runs set `--max-context 131072 --dflash-window 0 --expert-cache auto
--kv-resident 32768 --vram-reserve-mib 700 --prefill 512`. Target KV is
streamed with 32768 cells resident; DFlash2 has its own full 131072-cell
capacity. `STRATA_DFLASH_KV_GROW=0` forces every DFlash2 pool and its attention
scratch to be allocated at startup rather than growing on demand.

Each engine reached READY, processed a 24-token prompt and generated eight
tokens without an allocation failure. The DFlash2 allocation below includes
weights, private pools and scratch; it excludes the shared target head.

| Artifact | DFlash2 allocation, MiB | Auto expert slots | GPU free after generation, MiB |
| --- | ---: | ---: | ---: |
| BF16 | 4237.2 | 1680 | 321 |
| Q8_0 | 3827.5 | 1833 | 332 |
| Q4_0 | 3608.8 | 1919 | 326 |

The 700 MiB reserve is withheld before later verifier allocations and graph
capture; it is not a promise of 700 MiB remaining after those allocations.
The quantization savings produced more expert slots, confirming that the
auto cache saw the real drafter footprint before it filled the GPU.

## Elastic growth

The Q8_0 run with `STRATA_DFLASH_KV_GROW=1` started with 8192 mapped cells,
945.5 MiB of DFlash2 allocations and 2926 expert slots. An 8448-token prompt
forced a growth to 16384 cells. The runtime transferred memory from 74 cache
slots and reported 2852 of 2926 slots holding experts. Prefill and eight
generated tokens completed successfully; 320 MiB of GPU memory remained.

This checks an actual growth step with a nearly full auto cache, including
target KV streaming and draft KV growth at the same time.

## Setup estimate correction

The previous setup estimate used 128 for the attention scratch stride.
`qsa_decode_attn_scratch_floats` retains the compiled 256-wide stride even
for the official checkpoint's 128-wide heads. The runtime allocation was
already correct; the setup estimate understated scratch by about 256 MiB
at 131072 cells.

Setup now prices 20480 bytes per token for five FP16 K/V pools plus 4128
bytes per token for eight attention scratch rows, the selected artifact's
weights and a 50 MB allowance for fixed buffers and allocator overhead.
Its 128K estimates are 4.481 GB (BF16), 4.051 GB (Q8_0) and 3.822 GB
(Q4_0), each above the observed allocation. The installer regression test
checks these measured allocation bounds. The runtime cache uses actual
free device memory rather than this setup estimate.

## Reproduction and limits

Run `.venv/bin/python bench/results/2026-10-19-dflash2-vram/probe.py` with the
same local model files. `results.json` records commands, executable hash,
environment, INFO responses, generated tokens and GPU memory snapshots;
the four engine logs record allocation, auto-cache sizing and growth.

This is a memory allocation audit, not a throughput comparison or a full
131072-token prompt test. It checks full 128K DFlash2 pool/scratch allocation,
short generation and one real elastic growth step. Other GPU sizes,
unstreamed target KV at 128K, simultaneous external GPU allocations and
HIP/SYCL were not tested.
