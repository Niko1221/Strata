# RTX 4090: the QSA block scores on FP16 tensor cores, `STRATA_SELECT_F16=1` (2026-11-04)

The raw outputs behind the pull request that adds `STRATA_SELECT_F16=1` (an opt-in FP16 tensor-core scorer for the
prompt path, sm_80+). One card, one build, the switch off and on. The default path is the previous code.

## Rig

- GeForce RTX 4090 24 GB (sm_89), driver 595.97, PCIe 4.0 x16. Intel Core i7-13700KF, 128 GB RAM. Windows 11 Pro
  (build 26200). Nothing else used the card during the runs. A shared build slot kept other heavy jobs off the CPU.
- MSVC 19.44, CUDA 13.0.48. `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON
  -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89-real`, the pinned ggml (3cf03257).
- Engine: `main` at ffa80d73 (0.1.56) with the branch `qsa-select-f16`. The same binary ran both arms. Only the
  environment variable differed.
- Model: Qwen3.8-Flash-Next IQ3_S (the Swift GSQ-RCO IQ3_S shards of a community re-upload), the native pack, the
  shipped expert profile and MTP head. Engine arguments: `--max-context 1000000 --rope-scaling yarn --prefill auto:16384
  --kv int8 --kv-resident 32768 --expert-cache auto --resident-budget-gib 43 --ple-io ram --spec 4 --spec-min-p 0.5
  --mtp --vision --vram-reserve-mib 2048 --conversation-cache-mib 16384 --conversation-cache-slots 2`. The n-gram
  table sat in RAM (26.8 GiB).
- Prompt: a 976K-token needle prompt with five planted codes, one request per run, then a 300-token summary on the
  cached context. `STRATA_PREFILL_TIMING=1` printed the phase timeline of each read.

## Files

| File | What |
| --- | --- |
| [bench-tf32-default.txt](bench-tf32-default.txt) | `qsa_select_bench` matrix, the switch off (the 3xTF32 scorer): 40K to 1M cells x 256 queries, 1M cells x 1 to 4,096 queries |
| [bench-f16.txt](bench-f16.txt) | the same matrix with `STRATA_SELECT_F16=1` |
| [bench-f16-split12.txt](bench-f16-split12.txt) | the one-term and two-term variants (`STRATA_SELECT_F16_SPLIT=1|2`, bench only) at 1M cells |
| [ctest-default.txt](ctest-default.txt) | `ctest -R "qsa_|batch_rows|decode_cluster"`, the switch off |
| [ctest-f16.txt](ctest-f16.txt) | the same with `STRATA_SELECT_F16=1` |
| [engine-1m-ab.txt](engine-1m-ab.txt) | the 1M prompt reads: sweep rows and `strata prefill timing` lines, the engine table's runs and the one-read top-k runs |

## Kernel: `qsa_select_bench <cells> <queries> 10`

Scores = the block-score GEMM per batch, in ms. The bench compares each scorer with the warp kernel (the reference) and
with an FP64 copy. "Max err" is the largest error against FP64 as a fraction of the score scale. "Same" is the number of
queries whose selection equals the reference's.

| Cells x queries | TF32 ms | F16 ms | TF32 max err | F16 max err | TF32 same | F16 same |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40,000 x 256 | 0.134 | 0.092 | 1.7e-6 | 1.4e-6 | 256/256 | 256/256 |
| 131,072 x 256 | 0.407 | 0.180 | 1.6e-6 | 1.2e-6 | 256/256 | 256/256 |
| 262,144 x 256 | 0.855 | 0.353 | 1.7e-6 | 1.2e-6 | 255/256 | 255/256 |
| 524,284 x 256 | 1.629 | 0.777 | 1.7e-6 | 1.2e-6 | 256/256 | 256/256 |
| 524,288 x 256 | 1.592 | 0.746 | 1.4e-6 | 1.2e-6 | 256/256 | 256/256 |
| 600,000 x 256 | 1.929 | 0.934 | 1.6e-6 | 1.2e-6 | 256/256 | 255/256 |
| 700,000 x 256 | 2.143 | 1.048 | 1.5e-6 | 1.1e-6 | 256/256 | 256/256 |
| 1,000,000 x 256 | 3.059 | 1.490 | 1.6e-6 | 1.2e-6 | 256/256 | 256/256 |
| 1,000,000 x 1 | 0.194 | 0.187 | 1.5e-6 | 1.0e-6 | 1/1 | 1/1 |
| 1,000,000 x 8 | 0.197 | 0.190 | 1.8e-6 | 1.4e-6 | 8/8 | 8/8 |
| 1,000,000 x 255 | 3.063 | 1.487 | 1.7e-6 | 1.2e-6 | 255/255 | 255/255 |
| 1,000,000 x 257 | 3.244 | 1.654 | 1.6e-6 | 1.2e-6 | 257/257 | 257/257 |
| 1,000,000 x 1,024 | 12.375 | 5.783 | 1.5e-6 | 1.3e-6 | 1023/1024 | 1023/1024 |
| 1,000,000 x 4,096 | 50.341 | 23.185 | 1.4e-6 | 1.1e-6 | 4092/4096 | 4092/4096 |

The FP64 gate passed in every run of both matrices. The one selection that moves at 600,000 cells is 0.0008% of the
cells. The one-term and two-term variants take 1.41 and 1.42 ms at 1M x 256, fail the gate (8.4e-5 and 7.6e-5 of the
scale) and change 38 of 256 selections, so the switch runs the three-term kernel.

## Engine: the 976K-token prompt

One request per arm, the same binary, the n-gram table in RAM. "Read" is the prompt read of the engine's `strata serve`
line. The phases come from the `strata prefill timing` line of that read. [engine-1m-ab.txt](engine-1m-ab.txt) holds
both lines of every run.

| Arm | GPU timeline s | `qsa select` s | Read s | Read tok/s | Needles | Summary decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF32 (the default) | 330.0 | 95.5 (28.9%) | 330.5 | 2,954 | 5/5 | 65.7 |
| `STRATA_SELECT_F16=1` | 298.0 | 61.8 (20.7%) | 298.5 | 3,270 | 5/5 | 63.9 |
| F16 and the one-read top-k, a local merge of both branches, every switch on | 280.9 | 46.9 (16.7%) | 281.3 | 3,470 | 5/5 | 61.4 |

Every phase other than the selection stays within 0.4 s of the other arm (the largest moves: `wait copy` -0.36 s,
`gemm down` +0.31 s, `dequant` +0.29 s). The decode speed of the 300-token summary moves about 4 tok/s from run to run.
The code answers of all arms are identical. The third row is a build of this branch merged with the one-read top-k
change (its own pull request) and both switches on: `STRATA_SELECT_F16=1 STRATA_TOPK_ACTIVE_ANY=1 STRATA_TOPK_SAMPLE=1`.

[engine-1m-ab.txt](engine-1m-ab.txt) also holds the runs of the one-read top-k change on its own branch and binary, the
switches off and on. The two runs that failed before the read ran again: one started with a cut RAM budget,
because another job held the memory, and one hit the engine's start-up time limit. The model store drive was slow during these
runs (60 MB/s, 1 ms per 4 KB read). The read does not touch it: the engine reports 0 blob reads from the file, the
experts, the K/V and the n-gram table sit in RAM and VRAM. Only the start-up took longer (16 to 27 minutes).
