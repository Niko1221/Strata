# bench_decode - long-shstream0

- run: 20261008-215357  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~49152 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm; ~200k-token prompt
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long-shstream0 | 1 | yes | 199316 | 0 | 199444 | 128 | 2340 | 54.7 | 93.5 | 74.6 | 6 | 581 | 18.28 |
| long-shstream0 | 2 | yes | 199316 | 199311 | 199444 | 128 | 2031 | 63.0 | 96.7 | 77.8 | 6 | 855 | 15.87 |
| long-shstream0 | 3 | yes | 199316 | 199311 | 199444 | 128 | 1911 | 67.0 | 98.0 | 80.0 | 6 | 903 | 14.93 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 63.0 | 54.7 | 67.0 | 12.3 |
| decode_tps_warm | 2 | 65.0 | 63.0 | 67.0 | 4.0 |
| prefill_tps | 3 | 113.2 | 103.6 | 1124.9 | 1021.3 |
| ms_per_token | 3 | 15.8672 | 14.9297 | 18.2812 | 3.3516 |
| hit_pct | 3 | 96.7 | 93.5 | 98.0 | 4.5 |
| accept_pct | 3 | 77.7778 | 74.6479 | 80.0 | 5.3521 |
| prompt_tokens | 3 | 199316.0 | 199316.0 | 199316.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 199444.0 | 199444.0 | 199444.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long-shstream0 | 1 | 76 | 1.93 | 1.68 | 30.79 | 24.26 | 17.58 | 2.66 | 0.08 | 0.24 | 0.01 | 2.32 | 0.01 | 0.24 | 2.92 |
| long-shstream0 | 2 | 72 | 2.00 | 1.78 | 28.20 | 23.63 | 18.18 | 1.62 | 0.07 | 0.18 | 0.01 | 1.33 | 0.01 | 0.26 | 2.94 |
| long-shstream0 | 3 | 72 | 1.97 | 1.78 | 26.54 | 23.29 | 18.44 | 1.07 | 0.07 | 0.13 | 0.01 | 0.83 | 0.02 | 0.23 | 2.88 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

