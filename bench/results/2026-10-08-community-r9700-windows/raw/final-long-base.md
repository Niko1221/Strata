# bench_decode - final-long-base

- run: 20261008-211718  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~32768 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-long-base | 1 | yes | 132886 | 0 | 133014 | 128 | 2984 | 42.9 | 93.4 | 77.5 | 6 | 605 | 23.31 |
| final-long-base | 2 | yes | 132886 | 132881 | 133014 | 128 | 2651 | 48.3 | 94.8 | 84.0 | 6 | 1021 | 20.71 |
| final-long-base | 3 | yes | 132886 | 132881 | 133014 | 128 | 2549 | 50.2 | 97.9 | 66.7 | 6 | 1151 | 19.91 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 48.3 | 42.9 | 50.2 | 7.3 |
| decode_tps_warm | 2 | 49.25 | 48.3 | 50.2 | 1.9 |
| prefill_tps | 3 | 94.4 | 92.1 | 1096.6 | 1004.5 |
| ms_per_token | 3 | 20.7109 | 19.9141 | 23.3125 | 3.3984 |
| hit_pct | 3 | 94.8 | 93.4 | 97.9 | 4.5 |
| accept_pct | 3 | 77.5 | 66.6667 | 84.0 | 17.3333 |
| prompt_tokens | 3 | 132886.0 | 132886.0 | 132886.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 133014.0 | 133014.0 | 133014.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-long-base | 1 | 69 | 2.16 | 1.86 | 43.25 | 35.10 | 27.44 | 3.25 | 0.10 | 0.27 | 0.02 | 2.84 | 0.01 | 0.26 | 3.19 |
| final-long-base | 2 | 66 | 2.14 | 1.94 | 40.17 | 33.61 | 27.70 | 2.60 | 0.09 | 0.22 | 0.01 | 2.26 | 0.01 | 0.25 | 3.07 |
| final-long-base | 3 | 67 | 2.39 | 1.91 | 38.04 | 33.66 | 28.82 | 1.47 | 0.10 | 0.17 | 0.01 | 1.13 | 0.05 | 0.26 | 3.31 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

