# bench_decode - ctx-short-shstream0

- run: 20261008-212901  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~4096 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctx-short-shstream0 | 1 | yes | 17154 | 0 | 17282 | 128 | 2036 | 62.9 | 91.0 | 87.8 | 2 | 688 | 15.91 |
| ctx-short-shstream0 | 2 | yes | 17154 | 17149 | 17282 | 128 | 1656 | 77.3 | 92.7 | 87.4 | 2 | 1080 | 12.94 |
| ctx-short-shstream0 | 3 | yes | 17154 | 17149 | 17282 | 128 | 1559 | 82.1 | 96.6 | 93.2 | 2 | 1368 | 12.18 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 77.3 | 62.9 | 82.1 | 19.2 |
| decode_tps_warm | 2 | 79.7 | 77.3 | 82.1 | 4.8 |
| prefill_tps | 3 | 122.2 | 112.8 | 943.9 | 831.1 |
| ms_per_token | 3 | 12.9375 | 12.1797 | 15.9062 | 3.7266 |
| hit_pct | 3 | 92.7 | 91.0 | 96.6 | 5.6 |
| accept_pct | 3 | 87.8378 | 87.3563 | 93.1818 | 5.8255 |
| prompt_tokens | 3 | 17154.0 | 17154.0 | 17154.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 17282.0 | 17282.0 | 17282.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctx-short-shstream0 | 1 | 65 | 2.14 | 1.97 | 31.33 | 22.34 | 14.29 | 3.98 | 0.09 | 0.30 | 0.02 | 3.56 | 0.01 | 0.26 | 2.97 |
| ctx-short-shstream0 | 2 | 52 | 2.67 | 2.46 | 31.84 | 24.20 | 15.88 | 4.24 | 0.12 | 0.33 | 0.02 | 3.75 | 0.01 | 0.25 | 3.19 |
| ctx-short-shstream0 | 3 | 48 | 2.83 | 2.67 | 32.47 | 24.82 | 16.93 | 3.08 | 0.14 | 0.37 | 0.02 | 2.53 | 0.01 | 0.34 | 3.39 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

