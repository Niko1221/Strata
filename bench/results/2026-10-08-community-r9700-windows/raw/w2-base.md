# bench_decode - w2-base

- run: 20261008-193132  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-base | 1 | yes | 99696 | 0 | 99824 | 128 | 2920 | 43.8 | 93.2 | 69.3 | 6 | 580 | 22.81 |
| w2-base | 2 | yes | 99696 | 99691 | 99824 | 128 | 2686 | 47.7 | 97.2 | 66.3 | 6 | 874 | 20.98 |
| w2-base | 3 | yes | 99696 | 99691 | 99824 | 128 | 2516 | 50.9 | 97.6 | 75.6 | 6 | 1034 | 19.66 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 47.7 | 43.8 | 50.9 | 7.1 |
| decode_tps_warm | 2 | 49.3 | 47.7 | 50.9 | 3.2 |
| prefill_tps | 3 | 93.1 | 86.8 | 1101.2 | 1014.4 |
| ms_per_token | 3 | 20.9844 | 19.6562 | 22.8125 | 3.1562 |
| hit_pct | 3 | 97.2 | 93.2 | 97.6 | 4.4 |
| accept_pct | 3 | 69.3182 | 66.2921 | 75.641 | 9.3489 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-base | 1 | 67 | 2.31 | 1.91 | 43.58 | 35.18 | 27.21 | 3.42 | 0.10 | 0.29 | 0.02 | 2.99 | 0.01 | 0.26 | 3.38 |
| w2-base | 2 | 69 | 2.29 | 1.86 | 38.93 | 33.25 | 28.13 | 1.74 | 0.10 | 0.18 | 0.01 | 1.43 | 0.01 | 0.26 | 3.26 |
| w2-base | 3 | 69 | 2.13 | 1.86 | 36.47 | 32.21 | 27.43 | 1.43 | 0.10 | 0.15 | 0.01 | 1.15 | 0.01 | 0.27 | 3.14 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

