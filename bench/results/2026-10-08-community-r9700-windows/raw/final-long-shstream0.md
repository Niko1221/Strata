# bench_decode - final-long-shstream0

- run: 20261008-212044  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~32768 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-long-shstream0 | 1 | yes | 132886 | 0 | 133014 | 128 | 2298 | 55.7 | 93.6 | 75.0 | 6 | 589 | 17.95 |
| final-long-shstream0 | 2 | yes | 132886 | 132881 | 133014 | 128 | 1788 | 71.6 | 97.4 | 84.8 | 6 | 720 | 13.97 |
| final-long-shstream0 | 3 | yes | 132886 | 132881 | 133014 | 128 | 1863 | 68.7 | 97.9 | 73.5 | 6 | 801 | 14.55 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 68.7 | 55.7 | 71.6 | 15.9 |
| decode_tps_warm | 2 | 70.15 | 68.7 | 71.6 | 2.9 |
| prefill_tps | 3 | 116.7 | 113.5 | 1106.2 | 992.7 |
| ms_per_token | 3 | 14.5547 | 13.9688 | 17.9531 | 3.9844 |
| hit_pct | 3 | 97.4 | 93.6 | 97.9 | 4.3 |
| accept_pct | 3 | 75.0 | 73.494 | 84.8101 | 11.3162 |
| prompt_tokens | 3 | 132886.0 | 132886.0 | 132886.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 133014.0 | 133014.0 | 133014.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-long-shstream0 | 1 | 74 | 1.97 | 1.73 | 31.05 | 23.35 | 16.53 | 2.67 | 0.08 | 0.24 | 0.01 | 2.32 | 0.01 | 0.26 | 3.02 |
| final-long-shstream0 | 2 | 64 | 2.23 | 2.00 | 27.94 | 23.42 | 17.88 | 1.52 | 0.08 | 0.19 | 0.01 | 1.23 | 0.01 | 0.25 | 3.20 |
| final-long-shstream0 | 3 | 69 | 2.20 | 1.86 | 27.00 | 23.12 | 17.83 | 1.26 | 0.08 | 0.16 | 0.01 | 0.99 | 0.01 | 0.27 | 3.12 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

