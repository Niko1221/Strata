# Seven runs of the draft protocol on 0.2.7, same box (2026-10-29 UTC)

All on the 2x RX 6900 XT box described in `NOTES.md`, Qwen3.8-Flash-Next GSQ-RCO IQ3_S, protocol `strata-qwen38-thinking-v1-draft.1`, every run `complete`, 12 of 12 sanity checks, 0 failed, 0 stalls, the optional concurrency stage included (three rounds of four). Same matrix as the 0.2.2 pilot in #3790, so the two can be read side by side. Medians of the performance cases (prompt tok/s / decode tok/s; TTFT s):

**Two cards (layer split)**

| folder (run id) | cards | track | engine | settings | short decode | prompt-4k | prompt-16k | concurrency, s per round |
|---|---|---|---|---|---|---|---|---|
| `...-2x-rx-6900xt-e8ae7694` | 2 | **default** | same | same, `--layer-split auto`, no pipeline windows | 61.7 | 462 / 54.4; 8.9 | 644 / 56.7; 25.5 | 35-39 |
| `...-2x-rx-6900xt-b696e06d` | 2 | custom | same | same + `--pipeline-windows 2` | 73.8 | 829 / 66.0; 5.0 | 1,356 / 74.0; 12.1 | 28-31 |
| `...-2x-rx-6900xt-3c33d65d` | 2 | custom | source build 3177b96f (9ca7fd21 + #3821) - this box's production engine | same + `--pipeline-windows 2`, `--prefill-pipe 1600` with `STRATA_PREFILL_PIPE_FLOOR=4096`, 256K context, parallel 1 | 76.1 | 1,021 / 72.4; 4.0 | 1,585 / 73.2; 10.4 | 27-31 |

**One card**

| folder (run id) | cards | track | engine | settings | short decode | prompt-4k | prompt-16k | concurrency, s per round |
|---|---|---|---|---|---|---|---|---|
| `...-1x-rx-6900xt-bd3f807f` | 1 | **default** | release package `rx6000` 0.1.52, profile made by `strata launch` | spec 4/0.5, 64K context, kv int8, ple-io ram, vision cpu, parallel 1, no env | 45.8 | 476 / 42.5; 8.6 | 505 / 41.4; 32.5 | 49-55 |
| `...-1x-rx-6900xt-b3628de3` | 1 | custom | release package `rx6000` 0.1.52 | + upstream opt-in switches (`STRATA_HIP_PROMPT_F16`, `SH_STREAM`, `HIP_ADAPT_KERNEL_COPY`, `ROCBLAS_TUNING` table, `IO_THREADS=128`, `PREFILL_CPU_SHARE=auto`, `HC_PERSIST`, `PIPED_CONCURRENT`, `PREFILL_STREAM_MIN=2048`), spec 3/0.7, 128K context, `presence_penalty 0` in the run config | 46.2 | 859 / 47.5; 4.8 | 1,100 / 45.1; 14.9 | 44-48 |
| `...-1x-rx-6900xt-f84db514` | 1 | custom | source build 9ca7fd21 (v0.2.7 + four local opt-in patches) | the switches above plus the patches' own, spec 3/0.7 | 49.0 | 1,023 / 44.9; 4.0 | 1,232 / 49.1; 13.4 | 44-45 |
| `...-1x-rx-6900xt-38556baa` | 1 | custom | same | same + `STRATA_KVA=1` | 50.3 | 1,026 / 43.6; 4.0 | 1,345 / 45.5; 12.2 | 44-48 |

What the matrix says on this hardware:

- **0.2.7 reads the same as 0.2.2 out of the box**: the default track gives 476 / 505 (one card) and 462 / 644 (two cards) against 479 / 507 and 464 / 645 on 0.2.2 (#3790). The release package with the upstream switches is also unchanged (859 / 1,100 and 829 / 1,356 against 877 / 1,107 and 849 / 1,369).
- **Prompt reading is still where the opt-in switches matter on gfx1030**: default to switches is x1.8 / x2.2 on one card and x1.8 / x2.1 on two. The local patches (#3381, #3382, #3073, #3095) add another +19% / +12% on one card; on two cards the production build adds +23% / +17%, of which about +4.5% at 16K comes from #3821's chunk floor (the same build without it read 1,024 / 1,516; the protocol's 16K prompt is cut into 3 x 5,632 instead of 2 x 8,192).
- **`STRATA_KVA=1` on one card** reads the 16K case +9% (1,232 -> 1,345) with 12/12 sanity. The protocol's 4K prompt is 4,084-4,088 tokens, just under KVA's 4,096 minimum, so the 4K case is not approximated (the engine logs `KVA configured but NOT applied`).
- **Decode under the protocol's sampling** stays inside its run-to-run spread between the custom arms, as on 0.2.2; the two-card arms with `--pipeline-windows 2` decode 66-76, the default-track split 54-62.
- The patched build is described in `NOTES.md` and, step by step with its A/B data, in [dual-6900xt-flash-next-tuning](https://github.com/Vop3p/dual-6900xt-flash-next-tuning/blob/main/docs/linux-dual-6900xt.md).
- Each run is one pass of the suite; the arms ran one after another within two hours, not interleaved.

Developed with an AI coding assistant; every number above was measured on the machine described here.
