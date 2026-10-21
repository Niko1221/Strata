# kernels in the window

- window: 53.0 s (52.277 s from the trace's start)
- GPU busy: 49194.85 ms (92.82 %), idle: 3805.15 ms
- kernel launches: 264052 (sum of their times: 49203.59 ms)

## copies

| kind | count | total ms | max ms | total MB | avg MB |
|---|---|---|---|---|---|
| H2D | 2484 | 210.665 | 0.132 | 2467.28 | 0.993 |
| D2H | 1004 | 78.232 | 0.109 | 930.82 | 0.927 |
| 8 | 14 | 0.037 | 0.006 | 0.55 | 0.039 |

## kernels

| # | kernel | launches | total ms | avg us | max us | % of window | grid | block | smem KB |
|---|---|---|---|---|---|---|---|---|---|
| 1 | strata::kernels::<unnamed>::wait_flag_ge_kernel | 10800 | 40773.985 | 3775.37 | 1182717.42 | 76.93 | [1, 1, 1] | [1, 1, 1] | 0.0/0.0 |
| 2 | void strata::kernels::<unnamed>::native_gu_multi_kernel< | 7200 | 1849.543 | 256.88 | 1118.7 | 3.49 | [160, 40, 1] | [256, 1, 1] | 0.0/0.0 |
| 3 | void strata::kernels::<unnamed>::native_down_multi_kernel< | 7200 | 1445.766 | 200.8 | 961.28 | 2.73 | [320, 40, 1] | [256, 1, 1] | 2.8/0.0 |
| 4 | void strata::kernels::<unnamed>::native_mmvq_multi_kernel<strata::kern | 9576 | 1125.67 | 117.55 | 370.36 | 2.12 | [5120, 1, 1] | [32, 4, 1] | 3.0/0.0 |
| 5 | void strata::kernels::<unnamed>::native_mmvq_multi_kernel<strata::kern | 4536 | 427.492 | 94.24 | 543.5 | 0.81 | [5120, 1, 1] | [32, 4, 1] | 2.2/0.0 |
| 6 | void strata::kernels::<unnamed>::gr_up_multi_kernel< | 3713 | 328.799 | 88.55 | 108.95 | 0.62 | [160, 1, 1] | [256, 1, 1] | 6.0/0.0 |
| 7 | void strata::kernels::<unnamed>::native_q6_k_mmvq_kernel< | 199 | 299.938 | 1507.22 | 3367.3 | 0.57 | [248320, 1, 1] | [32, 4, 1] | 0.4/0.0 |
| 8 | strata::kernels::<unnamed>::fetch_blobs_kernel | 3600 | 250.717 | 69.64 | 1096.85 | 0.47 | [384, 1, 1] | [256, 1, 1] | 0.0/0.0 |
| 9 | void strata::kernels::<unnamed>::gr_down_staged_kernel< | 3713 | 227.183 | 61.19 | 69.37 | 0.43 | [41, 1, 1] | [256, 1, 1] | 0.0/40.0 |
| 10 | void strata::kernels::<unnamed>::native_mmvq_multi_kernel<strata::kern | 38 | 212.216 | 5584.64 | 5676.63 | 0.4 | [124160, 1, 1] | [32, 4, 1] | 3.0/0.0 |
| 11 | void strata::kernels::<unnamed>::native_mmvq_multi_kernel<strata::kern | 2268 | 166.667 | 73.49 | 219.88 | 0.31 | [5120, 1, 1] | [32, 4, 1] | 1.5/0.0 |
| 12 | void strata::kernels::<unnamed>::gpu_stamp_kernel< | 78450 | 140.582 | 1.79 | 4.16 | 0.27 | [1, 1, 1] | [1, 1, 1] | 0.0/0.0 |
| 13 | void strata::kernels::<unnamed>::native_q5_k_mmvq_kernel< | 2520 | 131.174 | 52.05 | 335.61 | 0.25 | [10240, 1, 1] | [32, 4, 1] | 0.4/0.0 |
| 14 | void strata::kernels::<unnamed>::gr_up_multi_kernel< | 1762 | 129.541 | 73.52 | 132.11 | 0.24 | [160, 1, 1] | [256, 1, 1] | 4.5/0.0 |
| 15 | void strata::kernels::<unnamed>::bf16_f32_mmvf_multi_kernel< | 4636 | 108.085 | 23.31 | 599.49 | 0.2 | [512, 1, 1] | [256, 1, 1] | 0.5/0.0 |
| 16 | void strata::kernels::<unnamed>::gr_down_staged_kernel< | 1762 | 93.805 | 53.24 | 100.09 | 0.18 | [41, 1, 1] | [256, 1, 1] | 0.0/30.0 |
| 17 | void strata::kernels::<unnamed>::attn_chunk_kernel_pre75< | 1089 | 88.31 | 81.09 | 116.92 | 0.17 | [33, 2, 4] | [256, 1, 1] | 15.5/0.0 |
| 18 | void strata::kernels::<unnamed>::native_mmvq_multi_kernel<strata::kern | 18 | 86.161 | 4786.74 | 4855.91 | 0.16 | [124160, 1, 1] | [32, 4, 1] | 2.2/0.0 |
| 19 | void strata::kernels::<unnamed>::gdn_step_norm_multi_kernel< | 2340 | 83.557 | 35.71 | 45.31 | 0.16 | [48, 1, 1] | [128, 4, 1] | 6.0/0.0 |
| 20 | strata::kernels::<unnamed>::gr_norm_split_kernel | 7843 | 78.909 | 10.06 | 19.74 | 0.15 | [4, 4, 1] | [256, 1, 1] | 0.0/0.0 |
| 21 | strata::kernels::<unnamed>::gdn_step_commit_layers_kernel | 65 | 66.97 | 1030.3 | 1057.68 | 0.13 | [48, 4, 36] | [32, 4, 1] | 1.0/0.0 |
| 22 | void strata::kernels::<unnamed>::native_mmvq_multi_kernel<strata::kern | 2053 | 65.782 | 32.04 | 84.28 | 0.12 | [1280, 1, 1] | [32, 4, 1] | 3.0/0.0 |
| 23 | strata::kernels::<unnamed>::route_multi | 3120 | 64.965 | 20.82 | 34.33 | 0.12 | [1, 1, 1] | [32, 8, 1] | 0.0/0.0 |
| 24 | void strata::kernels::<unnamed>::native_small_mmvq_kernel<strata::kern | 1152 | 62.118 | 53.92 | 347.48 | 0.12 | [2560, 1, 1] | [32, 4, 1] | 0.4/0.0 |
| 25 | void strata::kernels::<unnamed>::gr_down_staged_kernel< | 1480 | 57.783 | 39.04 | 57.88 | 0.11 | [41, 1, 1] | [256, 1, 1] | 0.0/10.0 |
