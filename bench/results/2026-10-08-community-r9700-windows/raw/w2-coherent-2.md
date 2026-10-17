# bench_decode - w2-coherent-2

- run: 20261008-195402  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-coherent-2 | 1 | yes | 99696 | 0 | 99824 | 128 | 2822 | 45.4 | 93.7 | 75.9 | 6 | 521 | 22.05 |
| w2-coherent-2 | 2 | yes | 99696 | 99691 | 99824 | 128 | 2654 | 48.2 | 97.0 | 76.3 | 6 | 840 | 20.73 |
| w2-coherent-2 | 3 | yes | 99696 | 99691 | 99824 | 128 | 2646 | 48.4 | 97.9 | 72.6 | 6 | 921 | 20.67 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 48.2 | 45.4 | 48.4 | 3.0 |
| decode_tps_warm | 2 | 48.3 | 48.2 | 48.4 | 0.2 |
| prefill_tps | 3 | 92.8 | 91.0 | 1089.3 | 998.3 |
| ms_per_token | 3 | 20.7344 | 20.6719 | 22.0469 | 1.375 |
| hit_pct | 3 | 97.0 | 93.7 | 97.9 | 4.2 |
| accept_pct | 3 | 75.9036 | 72.6027 | 76.3158 | 3.713 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-coherent-2 | 1 | 66 | 2.26 | 1.94 | 42.75 | 34.96 | 27.21 | 3.25 | 0.11 | 0.27 | 0.02 | 2.83 | 0.01 | 0.26 | 3.31 |
| w2-coherent-2 | 2 | 70 | 2.09 | 1.83 | 37.91 | 32.34 | 27.31 | 1.71 | 0.09 | 0.17 | 0.01 | 1.38 | 0.05 | 0.25 | 3.09 |
| w2-coherent-2 | 3 | 75 | 1.97 | 1.71 | 35.27 | 31.71 | 27.16 | 1.22 | 0.08 | 0.13 | 0.01 | 0.95 | 0.04 | 0.24 | 2.93 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

