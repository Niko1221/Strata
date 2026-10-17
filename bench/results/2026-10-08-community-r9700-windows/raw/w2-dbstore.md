# bench_decode - w2-dbstore

- run: 20261008-203602  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-dbstore | 1 | yes | 99696 | 0 | 99824 | 128 | 2873 | 44.6 | 93.6 | 71.4 | 6 | 566 | 22.45 |
| w2-dbstore | 2 | yes | 99696 | 99691 | 99824 | 128 | 2423 | 52.8 | 97.3 | 79.2 | 6 | 794 | 18.93 |
| w2-dbstore | 3 | yes | 99696 | 99691 | 99824 | 128 | 2445 | 52.3 | 98.1 | 72.2 | 6 | 882 | 19.10 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 52.3 | 44.6 | 52.8 | 8.2 |
| decode_tps_warm | 2 | 52.55 | 52.3 | 52.8 | 0.5 |
| prefill_tps | 3 | 97.6 | 85.3 | 1301.9 | 1216.6 |
| ms_per_token | 3 | 19.1016 | 18.9297 | 22.4453 | 3.5156 |
| hit_pct | 3 | 97.3 | 93.6 | 98.1 | 4.5 |
| accept_pct | 3 | 72.1519 | 71.4286 | 79.1667 | 7.7381 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-dbstore | 1 | 75 | 2.03 | 1.71 | 38.31 | 31.40 | 24.97 | 2.75 | 0.08 | 0.22 | 0.01 | 2.43 | 0.01 | 0.15 | 2.87 |
| w2-dbstore | 2 | 71 | 2.01 | 1.80 | 34.12 | 29.69 | 25.51 | 1.36 | 0.07 | 0.13 | 0.01 | 1.13 | 0.00 | 0.14 | 2.83 |
| w2-dbstore | 3 | 71 | 2.11 | 1.80 | 34.44 | 30.97 | 25.32 | 2.80 | 0.07 | 0.12 | 0.01 | 0.94 | 1.65 | 0.15 | 2.87 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

