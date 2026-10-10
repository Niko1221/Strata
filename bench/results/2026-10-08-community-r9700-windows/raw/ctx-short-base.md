# bench_decode - ctx-short-base

- run: 20261008-212724  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~4096 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctx-short-base | 1 | yes | 17154 | 0 | 17282 | 128 | 3024 | 42.3 | 95.0 | 74.6 | 2 | 415 | 23.62 |
| ctx-short-base | 2 | yes | 17154 | 17149 | 17282 | 128 | 2293 | 55.8 | 90.0 | 89.5 | 2 | 807 | 17.91 |
| ctx-short-base | 3 | yes | 17154 | 17149 | 17282 | 128 | 2181 | 58.7 | 95.8 | 91.5 | 2 | 1199 | 17.04 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 55.8 | 42.3 | 58.7 | 16.4 |
| decode_tps_warm | 2 | 57.25 | 55.8 | 58.7 | 2.9 |
| prefill_tps | 3 | 96.9 | 68.3 | 922.3 | 854.0 |
| ms_per_token | 3 | 17.9141 | 17.0391 | 23.625 | 6.5859 |
| hit_pct | 3 | 95.0 | 90.0 | 95.8 | 5.8 |
| accept_pct | 3 | 89.5349 | 74.6032 | 91.4634 | 16.8602 |
| prompt_tokens | 3 | 17154.0 | 17154.0 | 17154.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 17282.0 | 17282.0 | 17282.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctx-short-base | 1 | 83 | 1.76 | 1.54 | 36.43 | 30.73 | 24.79 | 2.10 | 0.07 | 0.17 | 0.01 | 1.78 | 0.05 | 0.27 | 2.57 |
| ctx-short-base | 2 | 53 | 2.62 | 2.42 | 43.26 | 35.10 | 25.87 | 5.44 | 0.13 | 0.32 | 0.02 | 4.94 | 0.01 | 0.28 | 3.14 |
| ctx-short-base | 3 | 55 | 2.49 | 2.33 | 39.66 | 33.03 | 26.89 | 2.74 | 0.12 | 0.27 | 0.02 | 2.32 | 0.01 | 0.26 | 3.12 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

