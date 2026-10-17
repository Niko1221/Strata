# bench_decode - w2-dbstore-2

- run: 20261008-195118  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-dbstore-2 | 1 | yes | 99696 | 0 | 99824 | 128 | 2863 | 44.7 | 93.2 | 69.3 | 6 | 580 | 22.37 |
| w2-dbstore-2 | 2 | yes | 99696 | 99691 | 99824 | 128 | 2649 | 48.3 | 97.2 | 66.3 | 6 | 874 | 20.70 |
| w2-dbstore-2 | 3 | yes | 99696 | 99691 | 99824 | 128 | 2491 | 51.4 | 97.6 | 75.6 | 6 | 1034 | 19.46 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 48.3 | 44.7 | 51.4 | 6.7 |
| decode_tps_warm | 2 | 49.85 | 48.3 | 51.4 | 3.1 |
| prefill_tps | 3 | 93.7 | 89.9 | 1096.8 | 1006.9 |
| ms_per_token | 3 | 20.6953 | 19.4609 | 22.3672 | 2.9062 |
| hit_pct | 3 | 97.2 | 93.2 | 97.6 | 4.4 |
| accept_pct | 3 | 69.3182 | 66.2921 | 75.641 | 9.3489 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-dbstore-2 | 1 | 67 | 2.31 | 1.91 | 42.73 | 35.05 | 27.20 | 3.42 | 0.10 | 0.29 | 0.02 | 2.99 | 0.01 | 0.25 | 3.37 |
| w2-dbstore-2 | 2 | 69 | 2.29 | 1.86 | 38.39 | 32.92 | 27.88 | 1.73 | 0.10 | 0.18 | 0.01 | 1.43 | 0.01 | 0.24 | 3.37 |
| w2-dbstore-2 | 3 | 69 | 2.13 | 1.86 | 36.10 | 32.03 | 27.41 | 1.40 | 0.09 | 0.14 | 0.01 | 1.14 | 0.01 | 0.25 | 3.12 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

