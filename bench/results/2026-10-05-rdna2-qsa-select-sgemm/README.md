# RX 6900 XT (gfx1030): the QSA block scores on rocBLAS SGEMM (2026-10-05)

Without matrix cores the prompt path's QSA selection scores every (query, KV block) pair with the warp kernel
(`qsa_block_scores_warp_kernel`, one warp per pair). `STRATA_SELECT_SGEMM=1` runs the same scores as four rocBLAS SGEMMs
per batch (one per indexer head) plus a relu-and-add kernel, with the rocBLAS solution index measured on the card at
the first call per reach bucket. Every number below is the same binary with and without that switch.

## Rig

- 2x RX 6900 XT 16 GB (gfx1030, PCIe 4.0 x8 each), Ryzen 5 5600X, 128 GB DDR4-3200, Ubuntu 26.04, Linux 7.0, ROCm 10.0.0
  (rocBLAS 5.6.0, 244 listed solutions for the shape). `qsa_select_bench` on one card (`HIP_VISIBLE_DEVICES=1`).
- Engine: `origin/main` at `6f32ec0` (0.1.39) with this change, built with
  `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DSTRATA_PREFILL_MMQ=ON -DSTRATA_PARITY_PROMPT_ATTN=ON`.
  No other tuning (the rocBLAS table of #981 is not on `main`, so the prompt speed below is `main`'s, not this rig's best).
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S (native IQ pack), MTP head, layer split over both cards:
  `--kv int8 --kv-resident 32768 --max-context 131072 --adapt-every 100000`, `--ple-io ram`, `STRATA_IO_THREADS=128`.

## `qsa_select_bench` (`qsa_select_bench.txt`)

`./build-hip/qsa_select_bench <ctx> 256 3 262144`: 256 queries, 3 reps, a 262,144-token capacity (so the first call
measures every reach bucket up to 131,072; the per-bucket lines are in the file). "SGEMM" is the bench's tensor-core
column (`qsa_block_scores_tc`), which on this card is this path.

| context (blocks) | warp kernel scores | SGEMM scores | FP64 gate (both scorers) | selections |
| --- | ---: | ---: | --- | --- |
| 131,072 (32,769) | 2.739 ms | 0.811 ms (3.4x) | PASS, max err 3.4e-05 / 5.6e-05 at scale 268 | identical 256/256, cells differing 0.0000% |
| 32,768 (8,193) | 0.679 ms | 0.304 ms (2.2x) | PASS, max err 3.2e-05 / 5.5e-05 at scale 269 | identical 256/256, cells differing 0.0000% |

The measurement that matters most, reach <= 32,768 x 256 queries x K = 128: rocBLAS's own kernel choice 1.56 ms
(1.4 TFLOPS), the fastest listed solution 0.11 ms (19 TFLOPS). The remaining 0.7 ms of the 128K row is the four
relu-and-add passes and the tail kernel; top-k (not touched) is 3.6 ms there.

## End to end (`engine-off.log`, `engine-on.log` excerpts in `serve.txt`)

Two engine starts, one per arm, the same two prompts (a document from the corpus plus "Summarize the document above in
about 300 words"), temperature 0, 256 generated tokens, `STRATA_PREFILL_TIMING=1`. "prompt" is the server's
`strata serve: prompt ... read in T ms` line; "qsa select" is the per-8K-chunk `prefill timing` term for the last chunk of
the prompt.

| prompt | `STRATA_SELECT_SGEMM` unset | `=1` | last-chunk `qsa select`, unset -> `=1` |
| --- | ---: | ---: | --- |
| 50,517 tokens | 756 tok/s (66.8 s) | 753 tok/s (67.1 s) | 282 -> 115 ms |
| 130,681 tokens | 810 tok/s (161.3 s) | 833 tok/s (156.9 s) | 764 -> 271 ms |

Decode is unchanged (the decode path scores one query on the warp kernel; this change does not touch it). The generated
text is the same in both arms.

## Limits

One machine, one card family, one run per arm. The solution is chosen by time alone, once per engine start; two
solutions that tie could swap between starts, which moves near-tie selections, not the class of accuracy (FP32 both ways).
