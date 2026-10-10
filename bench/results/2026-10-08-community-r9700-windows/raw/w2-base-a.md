# bench_decode - w2-base-a

- run: 20261008-202750  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-base-a | 1 | yes | 99696 | 0 | 99824 | 128 | 3061 | 41.8 | 93.4 | 74.7 | 6 | 577 | 23.91 |
| w2-base-a | 2 | yes | 99696 | 99691 | 99824 | 128 | 2509 | 51.0 | 96.9 | 76.8 | 6 | 849 | 19.60 |
| w2-base-a | 3 | yes | 99696 | 99691 | 99824 | 128 | 2220 | 57.7 | 97.9 | 74.1 | 6 | 946 | 17.34 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 51.0 | 41.8 | 57.7 | 15.9 |
| decode_tps_warm | 2 | 54.35 | 51.0 | 57.7 | 6.7 |
| prefill_tps | 3 | 92.0 | 91.9 | 1110.5 | 1018.6 |
| ms_per_token | 3 | 19.6016 | 17.3438 | 23.9141 | 6.5703 |
| hit_pct | 3 | 96.9 | 93.4 | 97.9 | 4.5 |
| accept_pct | 3 | 74.6667 | 74.1176 | 76.8293 | 2.7116 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-base-a | 1 | 73 | 2.03 | 1.75 | 41.93 | 34.27 | 26.91 | 3.15 | 0.10 | 0.24 | 0.02 | 2.76 | 0.02 | 0.18 | 2.99 |
| w2-base-a | 2 | 68 | 2.21 | 1.88 | 36.90 | 31.65 | 26.32 | 1.99 | 0.09 | 0.18 | 0.01 | 1.50 | 0.20 | 0.17 | 2.98 |
| w2-base-a | 3 | 65 | 2.31 | 1.97 | 34.16 | 30.46 | 26.12 | 1.35 | 0.08 | 0.15 | 0.01 | 1.09 | 0.01 | 0.15 | 3.02 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

