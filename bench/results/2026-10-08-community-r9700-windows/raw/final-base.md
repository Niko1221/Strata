# bench_decode - final-base

- run: 20261008-205851  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-base | 1 | yes | 99696 | 0 | 99824 | 128 | 3415 | 37.5 | 93.2 | 72.7 | 6 | 563 | 26.68 |
| final-base | 2 | yes | 99696 | 99691 | 99824 | 128 | 2931 | 43.7 | 97.1 | 75.3 | 6 | 807 | 22.90 |
| final-base | 3 | yes | 99696 | 99691 | 99824 | 128 | 3140 | 40.8 | 98.0 | 71.4 | 6 | 943 | 24.53 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 40.8 | 37.5 | 43.7 | 6.2 |
| decode_tps_warm | 2 | 42.25 | 40.8 | 43.7 | 2.9 |
| prefill_tps | 3 | 93.6 | 80.9 | 1100.2 | 1019.3 |
| ms_per_token | 3 | 24.5312 | 22.8984 | 26.6797 | 3.7812 |
| hit_pct | 3 | 97.1 | 93.2 | 98.0 | 4.8 |
| accept_pct | 3 | 72.7273 | 71.4286 | 75.3086 | 3.8801 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-base | 1 | 72 | 2.07 | 1.78 | 47.43 | 34.18 | 26.66 | 3.15 | 0.09 | 0.25 | 0.02 | 2.74 | 0.05 | 0.27 | 9.21 |
| final-base | 2 | 67 | 2.21 | 1.91 | 43.74 | 32.54 | 27.44 | 1.82 | 0.10 | 0.18 | 0.01 | 1.45 | 0.05 | 0.28 | 9.54 |
| final-base | 3 | 76 | 2.01 | 1.68 | 41.31 | 31.56 | 27.11 | 1.24 | 0.08 | 0.13 | 0.01 | 0.96 | 0.04 | 0.29 | 8.89 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

