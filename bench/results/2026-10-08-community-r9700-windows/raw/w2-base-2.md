# bench_decode - w2-base-2

- run: 20261008-195944  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-base-2 | 1 | yes | 99696 | 0 | 99824 | 128 | 2792 | 45.9 | 93.7 | 75.9 | 6 | 522 | 21.81 |
| w2-base-2 | 2 | yes | 99696 | 99691 | 99824 | 128 | 2640 | 48.5 | 97.0 | 76.3 | 6 | 841 | 20.62 |
| w2-base-2 | 3 | yes | 99696 | 99691 | 99824 | 128 | 2650 | 48.3 | 97.9 | 72.6 | 6 | 921 | 20.70 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 48.3 | 45.9 | 48.5 | 2.6 |
| decode_tps_warm | 2 | 48.4 | 48.3 | 48.5 | 0.2 |
| prefill_tps | 3 | 92.3 | 91.6 | 1092.3 | 1000.7 |
| ms_per_token | 3 | 20.7031 | 20.625 | 21.8125 | 1.1875 |
| hit_pct | 3 | 97.0 | 93.7 | 97.9 | 4.2 |
| accept_pct | 3 | 75.9036 | 72.6027 | 76.3158 | 3.713 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-base-2 | 1 | 66 | 2.26 | 1.94 | 42.30 | 34.92 | 27.14 | 3.25 | 0.10 | 0.27 | 0.02 | 2.79 | 0.05 | 0.26 | 3.51 |
| w2-base-2 | 2 | 70 | 2.09 | 1.83 | 37.72 | 32.13 | 27.24 | 1.64 | 0.09 | 0.16 | 0.01 | 1.36 | 0.01 | 0.27 | 3.33 |
| w2-base-2 | 3 | 75 | 1.97 | 1.71 | 35.33 | 31.61 | 27.09 | 1.23 | 0.08 | 0.13 | 0.01 | 0.95 | 0.04 | 0.27 | 3.15 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

