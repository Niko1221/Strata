# bench_decode - w2-shstream0-minp0

- run: 20261008-213538  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-shstream0-minp0 | 1 | yes | 99696 | 0 | 99824 | 128 | 2140 | 59.8 | 94.6 | 42.0 | 6 | 689 | 16.72 |
| w2-shstream0-minp0 | 2 | yes | 99696 | 99691 | 99824 | 128 | 1778 | 72.0 | 97.6 | 48.7 | 6 | 987 | 13.89 |
| w2-shstream0-minp0 | 3 | yes | 99696 | 99691 | 99824 | 128 | 1840 | 69.6 | 98.3 | 40.9 | 6 | 1147 | 14.38 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 69.6 | 59.8 | 72.0 | 12.2 |
| decode_tps_warm | 2 | 70.8 | 69.6 | 72.0 | 2.4 |
| prefill_tps | 3 | 121.6 | 121.2 | 1193.2 | 1072.0 |
| ms_per_token | 3 | 14.375 | 13.8906 | 16.7188 | 2.8281 |
| hit_pct | 3 | 97.6 | 94.6 | 98.3 | 3.7 |
| accept_pct | 3 | 42.0118 | 40.9357 | 48.7013 | 7.7656 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-shstream0-minp0 | 1 | 57 | 3.96 | 2.25 | 37.54 | 28.24 | 20.22 | 4.16 | 0.14 | 0.50 | 0.02 | 3.49 | 0.01 | 0.17 | 4.15 |
| w2-shstream0-minp0 | 2 | 53 | 3.91 | 2.42 | 33.55 | 27.15 | 21.34 | 2.25 | 0.13 | 0.31 | 0.01 | 1.77 | 0.02 | 0.16 | 4.10 |
| w2-shstream0-minp0 | 3 | 58 | 3.95 | 2.21 | 31.72 | 26.70 | 21.66 | 1.74 | 0.12 | 0.28 | 0.01 | 1.32 | 0.01 | 0.15 | 4.10 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

