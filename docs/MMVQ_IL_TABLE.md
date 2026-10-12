# The interleaved small-batch matvec (STRATA_MMVQ_IL): how to measure a table for your card

The decode and verify windows run their dense projections (2 to 4 tokens at a time) through `native_mmvq_il`, which reads an
interleaved copy of the q8_1 activations. Every choice it makes gives the same bits as the plain kernels
(`STRATA_MMVQ_IL=0` turns the whole path off); only the speed differs. How many matrix rows one warp takes
(1, 2 or 4, or 0 = use the plain kernel) is a table in `src/kernels/cuda/native_mmvq.cu`, indexed by the
quantisation type (IQ4_XS, Q4_K, Q5_K, Q6_K, and Q3_K where measured), the token count and the matrix's
row count (five classes). One-column entries read the plain q8_1 blocks without a pack launch.

Which table a card gets:

| card | table |
|---|---|
| sm_86 (Ampere consumer) | `kIlRows_86`, measured on an RTX A4000; includes Q3_K and a separate one-column Q3_K table |
| sm_89 (Ada), sm_120 (Blackwell consumer) and other unlisted sm_80+ cards | `kIlRowsShared`, measured on an RTX 3060 and an RTX 5070 together (a cell is non-zero only if the worse of the two cards still gains 3%) |
| sm_75 (Turing) | `kIlRowsSm75`, exact input/output shapes, only with exactly `STRATA_MMVQ_IL=1`; [measurements and community reports](OLDER_GPUS.md#turing-sm_75-opt-in-interleaved-verify-projections) |
| sm_70 (Volta) | `kIlRows_70`, only with `STRATA_SM70_TABLE=1` |
| sm_60 (Pascal GP100) | `kIlRows_60`, unless `STRATA_SM60_TABLE=0` |
| other older cards | the path is off; the plain kernels run |
| a card listed in `kIlArch` | its own table |

## Measuring sm_89 (or any other card)

1. Build with tests: `cmake -S . -B build -DSTRATA_BUILD_TESTS=ON` and `cmake --build build --target mmvq_il_parity`.
2. Close everything else on the GPU, then run `build/mmvq_il_parity --bench --emit-table` (a few minutes).
   It first checks bit-for-bit parity of every shape and every rows choice (it must end with `mmvq_il_parity: OK`),
   prints a `BENCH` line per shape and token count, and last a ready-to-paste block, one line per type.
3. Post the whole output on the issue. A maintainer pastes the block as `kIlRows_89[]` and adds
   `{89, kIlRows_89, sizeof(kIlRows_89) / sizeof(IlRows)}` to `kIlArch`.
4. To try a table without rebuilding, set `STRATA_MMVQ_IL_ROWS="type:ncols:r0,r1,r2,r3,r4;..."`, for example
   `STRATA_MMVQ_IL_ROWS="23:3:0,0,2,4,4;12:2:0,1,1,1,1"` (the five numbers are the row classes below 2048, 4096, 8192,
   12288 and above; each is 0, 1, 2 or 4). Then A/B the decode: same flags, `STRATA_MMVQ_IL_ROWS` set or unset, 10
   interleaved pairs, medians.
   This class override does not apply to sm_75: that card's table needs the exact reduction width, excludes
   unmeasured shapes, Q3_K and one column, and remains strict opt-in. Use `--sm75-bench` for the paired exact-shape
   sweep, including pack cost; do not replace it with an emitted class table without separate validation.

The harness preserves `MMVQ_IL_ONLY` (case-name substring), `--bench --emit-table`, Q3_K and one-column parity.
It also reports the actual launch rows and uses seven alternating timing pairs. `--tables-only` checks the
architecture lookups without CUDA initialization; `--sm75-tables-only` can also check sm_75 isolation while
`STRATA_MMVQ_IL_ROWS` is set. `STRATA_MMVQ_IL_T1=0` and `STRATA_MMVQ_IL_Q3K=0` retain their upstream meanings.
The CPU-only checks are also registered with CTest:
`ctest --test-dir build -R '^mmvq_il_(tables|tables_disable_extensions|sm75_override_isolation)$' --output-on-failure`.

The emit rule: per cell, the fastest rows count if it takes at least 3% off the plain kernel's time, else 0; shapes of one
class that disagree give 0; a class with no measured shape copies its neighbour below.
