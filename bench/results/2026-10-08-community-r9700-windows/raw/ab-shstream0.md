# bench_decode - ab-shstream0

- run: 20261008-214749  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ab-shstream0 | 1 | yes | 99696 | 0 | 99824 | 128 | 2082 | 61.5 | 93.2 | 72.7 | 6 | 562 | 16.27 |
| ab-shstream0 | 2 | yes | 99696 | 99691 | 99824 | 128 | 1780 | 71.9 | 97.0 | 75.9 | 6 | 806 | 13.91 |
| ab-shstream0 | 3 | yes | 99696 | 99691 | 99824 | 128 | 1682 | 76.1 | 98.1 | 74.4 | 6 | 925 | 13.14 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 71.9 | 61.5 | 76.1 | 14.6 |
| decode_tps_warm | 2 | 74.0 | 71.9 | 76.1 | 4.2 |
| prefill_tps | 3 | 100.0 | 99.2 | 1249.2 | 1150.0 |
| ms_per_token | 3 | 13.9062 | 13.1406 | 16.2656 | 3.125 |
| hit_pct | 3 | 97.0 | 93.2 | 98.1 | 4.9 |
| accept_pct | 3 | 74.359 | 72.7273 | 75.9036 | 3.1763 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ab-shstream0 | 1 | 72 | 2.07 | 1.78 | 28.92 | 21.74 | 15.77 | 2.74 | 0.07 | 0.24 | 0.01 | 2.38 | 0.03 | 0.16 | 2.94 |
| ab-shstream0 | 2 | 68 | 2.22 | 1.88 | 26.18 | 21.25 | 16.79 | 1.55 | 0.07 | 0.18 | 0.01 | 1.28 | 0.00 | 0.15 | 3.05 |
| ab-shstream0 | 3 | 70 | 2.11 | 1.83 | 24.03 | 20.48 | 16.69 | 1.03 | 0.06 | 0.12 | 0.01 | 0.83 | 0.00 | 0.14 | 2.95 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

