# bench_decode - w2-coherent

- run: 20261008-204103  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-coherent | 1 | yes | 99696 | 0 | 99824 | 128 | 2692 | 47.5 | 93.2 | 70.9 | 6 | 556 | 21.03 |
| w2-coherent | 2 | yes | 99696 | 99691 | 99824 | 128 | 2626 | 48.7 | 97.1 | 70.1 | 6 | 871 | 20.52 |
| w2-coherent | 3 | yes | 99696 | 99691 | 99824 | 128 | 2334 | 54.8 | 97.9 | 75.3 | 6 | 956 | 18.23 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 48.7 | 47.5 | 54.8 | 7.3 |
| decode_tps_warm | 2 | 51.75 | 48.7 | 54.8 | 6.1 |
| prefill_tps | 3 | 99.6 | 95.2 | 1285.4 | 1190.2 |
| ms_per_token | 3 | 20.5156 | 18.2344 | 21.0312 | 2.7969 |
| hit_pct | 3 | 97.1 | 93.2 | 97.9 | 4.7 |
| accept_pct | 3 | 70.9302 | 70.1299 | 75.3247 | 5.1948 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-coherent | 1 | 67 | 2.28 | 1.91 | 40.18 | 32.52 | 25.35 | 3.26 | 0.09 | 0.26 | 0.01 | 2.87 | 0.01 | 0.17 | 3.17 |
| w2-coherent | 2 | 76 | 2.01 | 1.68 | 34.55 | 29.73 | 25.33 | 1.50 | 0.07 | 0.14 | 0.01 | 1.24 | 0.04 | 0.15 | 2.77 |
| w2-coherent | 3 | 71 | 2.08 | 1.80 | 32.87 | 29.45 | 25.27 | 1.28 | 0.07 | 0.12 | 0.01 | 1.00 | 0.07 | 0.15 | 2.88 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

