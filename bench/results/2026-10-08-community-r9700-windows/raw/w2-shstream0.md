# bench_decode - w2-shstream0

- run: 20261008-203039  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-shstream0 | 1 | yes | 99696 | 0 | 99824 | 128 | 2293 | 55.8 | 93.6 | 71.4 | 6 | 566 | 17.91 |
| w2-shstream0 | 2 | yes | 99696 | 99691 | 99824 | 128 | 1917 | 66.8 | 97.3 | 79.2 | 6 | 793 | 14.98 |
| w2-shstream0 | 3 | yes | 99696 | 99691 | 99824 | 128 | 1846 | 69.4 | 98.1 | 72.2 | 6 | 882 | 14.42 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 66.8 | 55.8 | 69.4 | 13.6 |
| decode_tps_warm | 2 | 68.1 | 66.8 | 69.4 | 2.6 |
| prefill_tps | 3 | 121.2 | 108.5 | 1094.3 | 985.8 |
| ms_per_token | 3 | 14.9766 | 14.4219 | 17.9141 | 3.4922 |
| hit_pct | 3 | 97.3 | 93.6 | 98.1 | 4.5 |
| accept_pct | 3 | 72.1519 | 71.4286 | 79.1667 | 7.7381 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-shstream0 | 1 | 75 | 2.03 | 1.71 | 30.57 | 22.89 | 16.14 | 2.84 | 0.08 | 0.24 | 0.01 | 2.39 | 0.10 | 0.25 | 3.07 |
| w2-shstream0 | 2 | 71 | 2.01 | 1.80 | 27.00 | 22.09 | 16.95 | 1.41 | 0.07 | 0.15 | 0.01 | 1.12 | 0.05 | 0.24 | 3.03 |
| w2-shstream0 | 3 | 71 | 2.11 | 1.80 | 25.99 | 22.13 | 17.16 | 1.13 | 0.08 | 0.14 | 0.01 | 0.89 | 0.01 | 0.25 | 3.09 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

