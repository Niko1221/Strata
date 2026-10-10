# bench_decode - w2-shstream0-2

- run: 20261008-193515  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-shstream0-2 | 1 | yes | 99696 | 0 | 99824 | 128 | 2052 | 62.4 | 93.7 | 75.9 | 6 | 522 | 16.03 |
| w2-shstream0-2 | 2 | yes | 99696 | 99691 | 99824 | 128 | 1990 | 64.3 | 97.1 | 74.0 | 6 | 838 | 15.55 |
| w2-shstream0-2 | 3 | yes | 99696 | 99691 | 99824 | 128 | 1775 | 72.1 | 97.7 | 76.8 | 6 | 912 | 13.87 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 64.3 | 62.4 | 72.1 | 9.7 |
| decode_tps_warm | 2 | 68.2 | 64.3 | 72.1 | 7.8 |
| prefill_tps | 3 | 116.7 | 116.2 | 1107.6 | 991.4 |
| ms_per_token | 3 | 15.5469 | 13.8672 | 16.0312 | 2.1641 |
| hit_pct | 3 | 97.1 | 93.7 | 97.7 | 4.0 |
| accept_pct | 3 | 75.9036 | 74.026 | 76.8293 | 2.8033 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-shstream0-2 | 1 | 66 | 2.26 | 1.94 | 31.08 | 23.52 | 16.44 | 2.92 | 0.08 | 0.27 | 0.01 | 2.53 | 0.01 | 0.24 | 3.36 |
| w2-shstream0-2 | 2 | 72 | 2.07 | 1.78 | 27.64 | 22.31 | 16.96 | 1.50 | 0.07 | 0.16 | 0.01 | 1.24 | 0.01 | 0.25 | 3.07 |
| w2-shstream0-2 | 3 | 68 | 2.21 | 1.88 | 26.10 | 22.36 | 17.13 | 1.33 | 0.08 | 0.17 | 0.01 | 1.04 | 0.02 | 0.23 | 3.14 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

