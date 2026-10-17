# bench_decode - final-shstream0

- run: 20261008-210143  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-shstream0 | 1 | yes | 99696 | 0 | 99824 | 128 | 2605 | 49.1 | 93.4 | 77.8 | 6 | 574 | 20.35 |
| final-shstream0 | 2 | yes | 99696 | 99691 | 99824 | 128 | 2308 | 55.5 | 96.9 | 77.4 | 6 | 803 | 18.03 |
| final-shstream0 | 3 | yes | 99696 | 99691 | 99824 | 128 | 2343 | 54.6 | 98.1 | 70.4 | 6 | 898 | 18.30 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 54.6 | 49.1 | 55.5 | 6.4 |
| decode_tps_warm | 2 | 55.05 | 54.6 | 55.5 | 0.9 |
| prefill_tps | 3 | 117.8 | 116.5 | 1107.5 | 991.0 |
| ms_per_token | 3 | 18.3047 | 18.0312 | 20.3516 | 2.3203 |
| hit_pct | 3 | 96.9 | 93.4 | 98.1 | 4.7 |
| accept_pct | 3 | 77.381 | 70.3704 | 77.7778 | 7.4074 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-shstream0 | 1 | 73 | 1.99 | 1.75 | 35.69 | 22.80 | 15.92 | 2.78 | 0.09 | 0.24 | 0.01 | 2.42 | 0.01 | 0.27 | 9.09 |
| final-shstream0 | 2 | 66 | 2.27 | 1.94 | 34.97 | 23.83 | 18.01 | 1.76 | 0.09 | 0.20 | 0.01 | 1.45 | 0.01 | 0.26 | 9.52 |
| final-shstream0 | 3 | 73 | 2.11 | 1.75 | 32.09 | 22.59 | 17.44 | 1.22 | 0.08 | 0.14 | 0.01 | 0.90 | 0.08 | 0.25 | 8.72 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

