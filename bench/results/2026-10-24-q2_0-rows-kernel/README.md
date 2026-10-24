# q2_0 GGUF expert rows: the AVX-512 v2 kernel against the original

`bench/micro/q2g_rows_bench.cpp` calls `q2_0_gguf_rows_multi` (the original) and `q2_0_gguf_rows_multi_v2` on identical buffers and compares the outputs with `memcmp`, without tolerance. Measured on a Ryzen 7 9800X3D (96 MB L3), Windows 11 Pro for Workstations, with `clang++ 23.1.4` and `cl 19.51.36260`, on synthetic seeded Q2_0 weights; no model is loaded. The four complete outputs are in this directory.

## Build

```sh
clang++ -std=c++17 -O3 -Iinclude -c bench/micro/q2g_rows_bench.cpp -o q2g_rows_bench.o
clang++ -std=c++17 -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mf16c -mfma -Iinclude -c src/kernels/cpu/expert.cpp -o expert.o
clang++ -pthread q2g_rows_bench.o expert.o -o q2g_bench.exe
```
```bat
cl /nologo /O2 /std:c++17 /EHsc /DNDEBUG /Iinclude /c bench\micro\q2g_rows_bench.cpp /Fo:q2g_rows_bench.cpp.obj
cl /nologo /O2 /std:c++17 /arch:AVX512 /EHsc /DNDEBUG /Iinclude /c src\kernels\cpu\expert.cpp /Fo:expert.cpp.obj
link /nologo /OUT:q2g_bench_msvc.exe /MAP:q2g_bench_msvc.map q2g_rows_bench.cpp.obj expert.cpp.obj
```

The kernel TU takes the engine's AVX-512 flags, the bench TU does not; the ISA guard scans the linked binary. `--validate-only` runs the sweep alone: 320 cases, every `nt` 1..8 against every `nblocks` 1..40, `rows` 7, plus a second call over the last four rows of each case.

## Columns

- `ratio` is the median v2/original ratio; `ratio_lo..hi` is the lowest and highest per-repetition ratio, not a confidence interval.
- `orig_p95` and `v2_p95` are the largest of 9 warm samples and the second largest of 21 cold ones.
- A warm case runs 9 repetitions; a cold case runs 21, each call preceded by a read of a 128 MiB buffer (the L3 is 96 MB).
- Both kernels are called directly; the dispatch path is not exercised.

## Warm, full groups (`nblocks >= 8`), first run

| tokens | blocks | rows | clang | MSVC |
|---|---|---|---|---|
| 8 | 10 | 640 | 0.877 | 0.986 |
| 1 | 10 | 640 | 0.790 | 0.684 |
| 4 | 10 | 640 | 0.814 | 1.010 |
| 8 | 40 | 1280 | 0.729 | 0.915 |
| 1 | 40 | 1280 | 0.546 | 0.384 |
| 4 | 40 | 1280 | 0.619 | 0.912 |
| 8 | 8 | 2048 | 0.789 | 0.939 |
| 1 | 8 | 2048 | 0.547 | 0.524 |
| 8 | 9 | 1024 | 0.866 | 0.969 |
| 1 | 9 | 1024 | 0.766 | 0.674 |
| 8 | 16 | 512 | 0.749 | 0.925 |
| 1 | 16 | 512 | 0.516 | 0.437 |

## Cold (128 MiB flush) and fewer-than-eight-block rows, first run

| case | tokens | blocks | rows | clang | MSVC |
|---|---|---|---|---|---|
| cold | 8 | 10 | 640 | 0.890 | 1.003 |
| cold | 1 | 10 | 640 | 0.803 | 0.692 |
| cold | 8 | 40 | 1280 | 0.745 | 0.921 |
| cold | 1 | 40 | 1280 | 0.568 | 0.395 |
| < 8 blocks | 8 | 1 | 4096 | 1.000 | 1.008 |
| < 8 blocks | 1 | 1 | 4096 | 1.000 | 1.000 |
| < 8 blocks | 8 | 7 | 1024 | 0.999 | 1.007 |
| < 8 blocks | 1 | 7 | 1024 | 0.999 | 1.000 |

The fewer-than-eight-block rows take the original in both arms; two of the four MSVC cases have a repetition range that does not contain 1.0 (`nt=8, nb=1`: 1.004-1.013, `nt=8, nb=7`: 1.003-1.008). A second run of the same binaries moved some medians by up to 10.5% (clang) and 16% (MSVC).

## Limits

- One machine, one synthetic weight layout.
- The harness calls the two exported kernels directly; no model is loaded and no end-to-end number is measured.
- Cold cases are one call per sample and are wide.

## Files

- `q2g_rows_bench-clang.txt`, `q2g_rows_bench-msvc.txt` - first run of each build.
- `q2g_rows_bench-clang-run2.txt`, `q2g_rows_bench-msvc-run2.txt` - second run of the same binaries.
- `q2g_isa_guard.txt` - `tools/check_isa_guard.py` on the MSVC build, exit 0.
- `q2g_ctest.txt` - the CTest entry passing, and the same entry skipping under `STRATA_FORCE_AVX2=1`.
