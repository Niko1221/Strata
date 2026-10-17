# bench_decode - final-long-base-b

- run: 20261008-212401  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~32768 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-long-base-b | 1 | yes | 132886 | 0 | 133014 | 128 | 2988 | 42.8 | 93.5 | 76.5 | 6 | 611 | 23.34 |
| final-long-base-b | 2 | yes | 132886 | 132881 | 133014 | 128 | 2560 | 50.0 | 97.5 | 76.8 | 6 | 774 | 20.00 |
| final-long-base-b | 3 | yes | 132886 | 132881 | 133014 | 128 | 2701 | 47.4 | 97.5 | 67.5 | 6 | 897 | 21.10 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 47.4 | 42.8 | 50.0 | 7.2 |
| decode_tps_warm | 2 | 48.7 | 47.4 | 50.0 | 2.6 |
| prefill_tps | 3 | 94.5 | 93.3 | 1106.1 | 1012.8 |
| ms_per_token | 3 | 21.1016 | 20.0 | 23.3438 | 3.3438 |
| hit_pct | 3 | 97.5 | 93.5 | 97.5 | 4.0 |
| accept_pct | 3 | 76.5432 | 67.5 | 76.8293 | 9.3293 |
| prompt_tokens | 3 | 132886.0 | 132886.0 | 132886.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 133014.0 | 133014.0 | 133014.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-long-base-b | 1 | 69 | 2.17 | 1.86 | 43.30 | 35.07 | 27.42 | 3.18 | 0.10 | 0.28 | 0.02 | 2.76 | 0.01 | 0.30 | 3.18 |
| final-long-base-b | 2 | 68 | 2.21 | 1.88 | 37.65 | 33.01 | 28.22 | 1.51 | 0.09 | 0.18 | 0.01 | 1.20 | 0.01 | 0.28 | 3.11 |
| final-long-base-b | 3 | 74 | 2.08 | 1.73 | 36.50 | 32.50 | 27.69 | 1.48 | 0.09 | 0.16 | 0.01 | 1.17 | 0.04 | 0.29 | 3.04 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

