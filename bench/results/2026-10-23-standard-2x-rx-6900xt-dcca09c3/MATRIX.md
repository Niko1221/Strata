# Six runs of the draft protocol on the same box (2026-10-23 UTC)

All on the 2x RX 6900 XT box described in `NOTES.md`, Qwen3.8-Flash-Next GSQ-RCO IQ3_S, protocol `strata-qwen38-thinking-v1-draft.1`, every run `complete`, 12 of 12 sanity checks, 0 failed, the optional concurrency stage included (three rounds of four). Medians of the performance cases (prompt tok/s / decode tok/s; TTFT s):

| folder (run id) | cards | track | engine | settings | short decode | prompt-4k | prompt-16k | concurrency, s per round |
|---|---|---|---|---|---|---|---|---|
| `...-1x-rx-6900xt-68eeaea6` | 1 | **default** | release package `rx6000` 0.1.47, profile made by `strata launch` | spec 4/0.5, 64K context, kv int8, ple-io ram, vision cpu, parallel 1, no env | 43.2 | 479 / 40.5; 8.6 | 507 / 40.7; 32.4 | 49-57 |
| `...-2x-rx-6900xt-be7e97da` | 2 | **default** | same, main + helper | same, layer split 0-23 / 24-47, no pipeline windows | 61.2 | 464 / 51.8; 8.9 | 645 / 53.3; 25.4 | 35-40 |
| `...-1x-rx-6900xt-af8df11c` | 1 | custom | release package `rx6000` 0.1.47 | + upstream opt-in switches (`STRATA_HIP_PROMPT_F16`, `SH_STREAM`, `HIP_ADAPT_KERNEL_COPY`, `ROCBLAS_TUNING` table, `IO_THREADS=128`, `PREFILL_CPU_SHARE=auto`, `HC_PERSIST`, `PIPED_CONCURRENT`, `PREFILL_STREAM_MIN=2048`), spec 3/0.7, 128K context, `presence_penalty 0` in the run config | 48.5 | 877 / 43.5; 4.7 | 1,107 / 46.0; 14.8 | 44-47 |
| `...-2x-rx-6900xt-54277ffd` | 2 | custom | same | same + `--pipeline-windows 2` | 76.2 | 849 / 66.4; 4.9 | 1,369 / 67.7; 12.0 | 26-34 |
| `...-1x-rx-6900xt-b6d9cf1a` | 1 | custom | source build 34cd8534 (v0.2.2 + five local opt-in patches) | the switches above plus the patches' own, spec 3/0.7 | 45.5 | 996 / 45.0; 4.2 | 1,225 / 43.3; 13.4 | 44-45 |
| `...-2x-rx-6900xt-dcca09c3` | 2 | custom | same (this box's production run config, 256K context, parallel 2, vision on) | same + `--pipeline-windows 2 --batch-mtp` | 66.4 | 1,013 / 62.9; 4.1 | 1,511 / 62.0; 10.9 | 33-34 |

What the matrix says on this hardware:

- **The release package works in the default track** on one and on two RX 6000 cards; the `strata launch` path (download of the package and the picture reader, own model files, 64K profile) ran unattended. One detail for the AMD packaging: the Terminal points the engine at a rocBLAS tuning table named for rocBLAS 5.4.0 (`tools/hip/gfx1030-rocblas-5.4.0.*.txt`) that the package's rocBLAS cannot open, so the engine logs `cannot open rocBLAS tuning file ... rocBLAS's default kernels run`.
- **Prompt reading is where the opt-in switches matter on gfx1030**: default 479 / 507 tok/s (4K / 16K, one card) against 877 / 1,107 with the upstream switches on, and 464 / 645 against 849 / 1,369 on the split. The local patches add another 14 / 11% (one card) and 19 / 10% (two cards) on top.
- **Decode under the protocol's sampling** is 40-46 tok/s on one card and 52-68 on the split in every configuration; the differences between the custom arms are inside the run-to-run spread of sampled decoding (see `NOTES.md`, 64 vs 73 tok/s for the same request), so the protocol's decode column does not separate these engine variants on this box. The default-track split reads 52-53 because it has no pipeline windows (setup does not turn them on); the two-card custom arms with `--pipeline-windows 2` read 62-68.
- Each run is one pass of the suite; the arms were run one after another within 50 minutes, not interleaved.

Developed with an AI coding assistant; every number above was measured on the machine described here.
