# bench_decode - final-base-b

- run: 20261008-210716  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-base-b | 1 | yes | 99696 | 0 | 99824 | 128 | 2836 | 45.1 | 93.7 | 75.9 | 6 | 522 | 22.16 |
| final-base-b | 2 | yes | 99696 | 99691 | 99824 | 128 | 2709 | 47.2 | 97.1 | 74.0 | 6 | 838 | 21.16 |
| final-base-b | 3 | yes | 99696 | 99691 | 99824 | 128 | 2461 | 52.0 | 97.7 | 76.8 | 6 | 912 | 19.23 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 47.2 | 45.1 | 52.0 | 6.9 |
| decode_tps_warm | 2 | 49.6 | 47.2 | 52.0 | 4.8 |
| prefill_tps | 3 | 93.7 | 92.3 | 1113.4 | 1021.1 |
| ms_per_token | 3 | 21.1641 | 19.2266 | 22.1562 | 2.9297 |
| hit_pct | 3 | 97.1 | 93.7 | 97.7 | 4.0 |
| accept_pct | 3 | 75.9036 | 74.026 | 76.8293 | 2.8033 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-base-b | 1 | 66 | 2.26 | 1.94 | 42.96 | 35.01 | 27.19 | 3.22 | 0.11 | 0.27 | 0.02 | 2.81 | 0.01 | 0.28 | 3.32 |
| final-base-b | 2 | 72 | 2.07 | 1.78 | 37.63 | 32.15 | 27.20 | 1.62 | 0.09 | 0.16 | 0.01 | 1.34 | 0.01 | 0.25 | 3.01 |
| final-base-b | 3 | 68 | 2.21 | 1.88 | 36.19 | 32.40 | 27.61 | 1.45 | 0.10 | 0.16 | 0.01 | 1.14 | 0.02 | 0.26 | 3.11 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

