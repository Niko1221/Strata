# 2026-10-11: FP16 sliding window + attention sinks over a quantized KV cache (`kv_window.hpp`)

Adds a small FP16 overlay to a quantized KV cache: the `sinks` cells at the head of the sequence and the `window`
most recent cells are kept in FP16 and read from there, while the bulk stays in `--kv`'s format. SKVQ's tiering
(arXiv 2405.06219) with the sink tokens of arXiv 2508.04257. Off by default
(`--kv-window 128 --kv-sink 8`, or `STRATA_KV_WINDOW`/`STRATA_KV_SINK`); with neither, every reader takes the path
it took before the change, bit for bit. Design and scope: `docs/kv-window.md`.

## Toolchain (read this first)

Every number below comes from a build with **CUDA 13.0**, the toolkit the release binaries use (here obtained through
pip's `cuda-toolkit` components; any CUDA 13.0 install works - see Reproduce).

A build with the locally installed **CUDA 13.2 is numerically broken for this model on this box** and must not be
used for quality work: the same source, same flags and same model score **0/8** on a ~5K-token recall probe where
the release binary and the 13.0 build score **8/8**, *even with `--kv fp16`* - so the fault is in the model's
weight kernels, not in the KV cache. This is the same miscompile family as the 0.1.40.3 quality bug.

## Representation (deterministic, `ctest -R kv_window_parity`)

K error against an FP16 cache, relative RMS, by position bucket - the overlay covers two of the three:

| bucket | q4_0 | q4_0 + overlay |
| --- | --- | --- |
| sinks (first 8 cells) | 8.940% | **0.000%** |
| window (last 128 cells) | 8.576% | **0.000%** |
| bulk (everything between) | 8.568% | 8.568% |

`kv_window_parity` also proves, bit for bit: the sinks and the window read *exactly* equal to a pure FP16 cache; a
cell outside them is bit-identical to `q4_0`; `kv_window_restore` reproduces the bulk's own dequantization; the
slots are in range and pairwise distinct. 6/6 checks pass, and the same suites that guard the existing formats
still pass on the same toolchain (`kv_q4_parity`, `kv_q8_parity`, `kv_stream_parity` 90% hit / 0 failures,
`kv_hybrid_parity`).

Cost, measured where it is paid:

| | value |
| --- | --- |
| decode attention, 8192 cells, overlay off -> on | 0.080 ms -> 0.080 ms (0.0%; -0.1%..0.0% over runs) |
| extra writes per append | one ~1 KB K/V row per layer per token |
| overlay footprint | 304 KB per layer for 152 slots (8 sinks + 128 window + 16 slack) |
| q4_0 bulk for comparison | 576 KiB per layer per 1024 cells (576 B per cell: K + V, 2 heads); the overlay is 1.65% of a 32K-cell layer |

## End to end (RTX 5080, IQ3_XXS, ctx up to 140000, `--kv-resident 20480` streaming, `--spec 4`)

Probe: a long archive document with **12 five-digit codes** to copy exactly - six stated in the body of the
document (bulk precision decides them) and six "tail" codes in the document's last few paragraphs - plus three
short questions. Score is codes reproduced exactly, out of 12.

**Correction (review of this PR):** in the probe as these numbers were measured, the six tail codes were spread over
the last ~6 paragraphs and followed by the answer instructions and all twelve questions - several hundred tokens
before the generation position, i.e. **outside** the 128-cell window. So the table below does not isolate window
recall; every arm (including exact FP16) scored 12/12, so it shows "no regression", not an overlay effect.
`tools/kv_pr_report.py` now asks each tail code in its own request with the record placed directly before the
question (~50 tokens from generation, inside the window); these numbers predate that change.

| arm | 4K | 16K | 32K | 64K | 131K | loops | prefill t/s | decode t/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| upstream `q4_0` (overlay off) | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 0/3 | 2277-3894 | 142.8-182.5 |
| `q4_0` + window | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 0/3 | 2261-3870 | 141.9-181.1 |
| `fp16` (exact KV, upper bound) | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 0/3 | 2125-3355 | 146.9-180.1 |

Ranges are the spread across context lengths; prefill is the prompt's own rate on a unique document (nothing
cached), decode is the answer's rate as served with the MTP drafts. The same probes at 5K with a needle-recall
variant (facts and queries instead of codes) are also 100% on all three arms.

**Finding: at 4K through 131K, on these probes, `q4_0` KV is indistinguishable from exact FP16 KV, and the overlay
changes nothing measurable** - neither quality nor speed. The error the overlay removes (8.57% relative RMS on K)
does not reach the model's output at 4 bits. Its purpose is therefore stated as an enabler rather than a win: at
2-3 bits of bulk, where that same error grows several-fold, an exact recent window plus exact sinks is the
difference between a usable cache and a degenerate one. The representation table above is the mechanism; the
end-to-end table is the "no regression" half.

## Reproduce

```
# 1. the toolchain (once): a CUDA 13.0 toolkit - e.g. pip's components, whose root is <dir>/nvidia/cu13 - then the
#    build (CU13 = that toolkit root; arch 120 = RTX 50-series, set yours)
python -m pip install --target <dir> "cuda-toolkit[all]==13.0.3.0"
cmake -G Ninja -S . -B build-cu130 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=$CU13/bin/nvcc -DCUDAToolkit_ROOT=$CU13
cmake --build build-cu130 --target strata kv_window_parity kv_q4_parity kv_q8_parity kv_stream_parity kv_hybrid_parity

# 2. the deterministic metrics
cd build-cu130 && ./kv_window_parity      # the bitwise checks + the error table + the kernel timing
./kv_q4_parity && ./kv_q8_parity && ./kv_stream_parity && ./kv_hybrid_parity

# 3. end to end: stop whatever serves on this GPU first (the arms each start their own serve), then
#    (python = the repo's environment; paths and the API key: see tools/kv_bench_env.py)
KV_ENGINE=build-cu130/strata python tools/kv_pr_report.py
KV_ENGINE=build-cu130/strata KV_LENGTHS="64K:1600,131K:3200" python tools/kv_pr_report.py
```

## Notes and known gaps

- **Scope:** the decode attention (fused, the default) and speculative verification read the overlay; the prompt
  attention and the `--no-fast-attn` gather still read the bulk - correct, unimproved (`docs/kv-window.md`).
- **Restored sessions** rebuild the overlay from the snapshot's quantized bulk, so a restored window is at bulk
  precision rather than exact. Correct, not exact, and `qsa_state_zero` clears it for the same reason.
- **Speculative decoding** needs `window + slack >= draft length`: a draft batch is appended before it is read, and
  `kKvWindowSlack` (16 slots) keeps those look-ahead writes off live window slots. A larger `--spec T` than the
  slack would need the slack raised.
- **`--dump-logits` on the native IQ pack** wrote 0-byte files when this README was first written (the speculative
  loop never reached the per-token dump site). Fixed later in this PR: the native speculative loop now writes rows
  through the verifier's logits, and the header is repaired to the rows actually written. The teacher-forced KL /
  top-1 table this write-up lacked is in `bench/results/2026-10-12-kv-precision/README.md`.
