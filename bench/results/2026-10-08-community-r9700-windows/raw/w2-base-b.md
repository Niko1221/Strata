# bench_decode - w2-base-b

- run: 20261008-203832  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-base-b | 1 | yes | 99696 | 0 | 99824 | 128 | 2824 | 45.3 | 93.6 | 71.4 | 6 | 567 | 22.06 |
| w2-base-b | 2 | yes | 99696 | 99691 | 99824 | 128 | 2318 | 55.2 | 97.0 | 80.2 | 6 | 833 | 18.11 |
| w2-base-b | 3 | yes | 99696 | 99691 | 99824 | 128 | 2347 | 54.5 | 98.1 | 69.6 | 6 | 900 | 18.34 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 54.5 | 45.3 | 55.2 | 9.9 |
| decode_tps_warm | 2 | 54.85 | 54.5 | 55.2 | 0.7 |
| prefill_tps | 3 | 99.2 | 90.9 | 1309.8 | 1218.9 |
| ms_per_token | 3 | 18.3359 | 18.1094 | 22.0625 | 3.9531 |
| hit_pct | 3 | 97.0 | 93.6 | 98.1 | 4.5 |
| accept_pct | 3 | 71.4286 | 69.6203 | 80.2469 | 10.6267 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-base-b | 1 | 75 | 2.03 | 1.71 | 37.66 | 31.11 | 24.70 | 2.77 | 0.08 | 0.22 | 0.01 | 2.44 | 0.01 | 0.15 | 2.85 |
| w2-base-b | 2 | 66 | 2.23 | 1.94 | 35.13 | 30.27 | 25.65 | 1.67 | 0.08 | 0.17 | 0.01 | 1.40 | 0.01 | 0.15 | 3.02 |
| w2-base-b | 3 | 73 | 2.08 | 1.75 | 32.16 | 28.95 | 25.00 | 1.09 | 0.07 | 0.12 | 0.01 | 0.88 | 0.02 | 0.14 | 2.82 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

