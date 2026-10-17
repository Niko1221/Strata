# bench_decode - w2-hcplain-2

- run: 20261008-194828  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-hcplain-2 | 1 | yes | 99696 | 0 | 99824 | 128 | 2983 | 42.9 | 93.2 | 78.1 | 6 | 561 | 23.30 |
| w2-hcplain-2 | 2 | yes | 99696 | 99691 | 99824 | 128 | 2730 | 46.9 | 97.1 | 67.4 | 6 | 837 | 21.33 |
| w2-hcplain-2 | 3 | yes | 99696 | 99691 | 99824 | 128 | 2493 | 51.4 | 98.1 | 75.0 | 6 | 916 | 19.48 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 46.9 | 42.9 | 51.4 | 8.5 |
| decode_tps_warm | 2 | 49.15 | 46.9 | 51.4 | 4.5 |
| prefill_tps | 3 | 91.9 | 90.3 | 1099.7 | 1009.4 |
| ms_per_token | 3 | 21.3281 | 19.4766 | 23.3047 | 3.8281 |
| hit_pct | 3 | 97.1 | 93.2 | 98.1 | 4.9 |
| accept_pct | 3 | 75.0 | 67.4419 | 78.0822 | 10.6403 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-hcplain-2 | 1 | 71 | 2.03 | 1.80 | 42.02 | 34.23 | 27.09 | 3.05 | 0.09 | 0.24 | 0.01 | 2.66 | 0.02 | 0.27 | 3.12 |
| w2-hcplain-2 | 2 | 70 | 2.23 | 1.83 | 38.99 | 33.64 | 28.22 | 1.79 | 0.10 | 0.18 | 0.01 | 1.41 | 0.07 | 0.27 | 3.19 |
| w2-hcplain-2 | 3 | 68 | 2.18 | 1.88 | 36.66 | 32.78 | 28.14 | 1.26 | 0.09 | 0.13 | 0.01 | 0.95 | 0.07 | 0.27 | 3.19 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

