# bench_decode - final-shstream0-minp0

- run: 20261008-210432  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-shstream0-minp0 | 1 | yes | 99696 | 0 | 99824 | 128 | 2872 | 44.6 | 94.7 | 40.5 | 6 | 678 | 22.44 |
| final-shstream0-minp0 | 2 | yes | 99696 | 99691 | 99824 | 128 | 2544 | 50.3 | 97.6 | 43.6 | 6 | 993 | 19.88 |
| final-shstream0-minp0 | 3 | yes | 99696 | 99691 | 99824 | 128 | 2531 | 50.6 | 98.4 | 40.9 | 6 | 1162 | 19.77 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 50.3 | 44.6 | 50.6 | 6.0 |
| decode_tps_warm | 2 | 50.45 | 50.3 | 50.6 | 0.3 |
| prefill_tps | 3 | 116.4 | 116.1 | 1112.4 | 996.3 |
| ms_per_token | 3 | 19.875 | 19.7734 | 22.4375 | 2.6641 |
| hit_pct | 3 | 97.6 | 94.7 | 98.4 | 3.7 |
| accept_pct | 3 | 40.9357 | 40.4624 | 43.6364 | 3.1739 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-shstream0-minp0 | 1 | 59 | 3.93 | 2.17 | 48.68 | 30.61 | 21.01 | 4.71 | 0.17 | 0.54 | 0.02 | 3.96 | 0.01 | 0.28 | 14.56 |
| final-shstream0-minp0 | 2 | 56 | 3.95 | 2.29 | 45.42 | 29.55 | 22.63 | 2.40 | 0.16 | 0.34 | 0.02 | 1.88 | 0.01 | 0.25 | 14.26 |
| final-shstream0-minp0 | 3 | 58 | 3.95 | 2.21 | 43.63 | 28.67 | 22.37 | 1.81 | 0.16 | 0.29 | 0.01 | 1.33 | 0.01 | 0.26 | 14.10 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

