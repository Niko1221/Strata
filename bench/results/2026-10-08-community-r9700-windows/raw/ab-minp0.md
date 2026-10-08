# bench_decode - ab-minp0

- run: 20261008-215056  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ab-minp0 | 1 | yes | 99696 | 0 | 99824 | 128 | 2702 | 47.4 | 94.2 | 46.9 | 6 | 691 | 21.11 |
| ab-minp0 | 2 | yes | 99696 | 99691 | 99824 | 128 | 2526 | 50.7 | 97.8 | 42.9 | 6 | 999 | 19.73 |
| ab-minp0 | 3 | yes | 99696 | 99691 | 99824 | 128 | 2579 | 49.6 | 98.2 | 40.1 | 6 | 1214 | 20.15 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 49.6 | 47.4 | 50.7 | 3.3 |
| decode_tps_warm | 2 | 50.15 | 49.6 | 50.7 | 1.1 |
| prefill_tps | 3 | 123.6 | 117.3 | 1152.5 | 1035.2 |
| ms_per_token | 3 | 20.1484 | 19.7344 | 21.1094 | 1.375 |
| hit_pct | 3 | 97.8 | 94.2 | 98.2 | 4.0 |
| accept_pct | 3 | 42.8571 | 40.113 | 46.9136 | 6.8006 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ab-minp0 | 1 | 54 | 4.00 | 2.37 | 50.04 | 30.66 | 21.15 | 4.81 | 0.17 | 0.55 | 0.02 | 4.06 | 0.01 | 0.27 | 14.42 |
| ab-minp0 | 2 | 57 | 3.95 | 2.25 | 44.31 | 28.27 | 21.89 | 2.34 | 0.16 | 0.32 | 0.01 | 1.83 | 0.01 | 0.22 | 13.90 |
| ab-minp0 | 3 | 60 | 3.95 | 2.13 | 42.98 | 27.54 | 21.80 | 1.92 | 0.14 | 0.30 | 0.01 | 1.45 | 0.01 | 0.19 | 14.18 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

