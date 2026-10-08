# bench_decode - w2-devplan-2

- run: 20261008-195653  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-devplan-2 | 1 | yes | 99696 | 0 | 99824 | 128 | 3027 | 42.3 | 93.3 | 70.9 | 6 | 589 | 23.65 |
| w2-devplan-2 | 2 | yes | 99696 | 99691 | 99824 | 128 | 2677 | 47.8 | 96.3 | 77.5 | 6 | 885 | 20.91 |
| w2-devplan-2 | 3 | yes | 99696 | 99691 | 99824 | 128 | 2501 | 51.2 | 97.6 | 76.0 | 6 | 969 | 19.54 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 47.8 | 42.3 | 51.2 | 8.9 |
| decode_tps_warm | 2 | 49.5 | 47.8 | 51.2 | 3.4 |
| prefill_tps | 3 | 93.6 | 89.1 | 1085.7 | 996.6 |
| ms_per_token | 3 | 20.9141 | 19.5391 | 23.6484 | 4.1094 |
| hit_pct | 3 | 96.3 | 93.3 | 97.6 | 4.3 |
| accept_pct | 3 | 76.0 | 70.8861 | 77.4648 | 6.5787 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-devplan-2 | 1 | 72 | 2.10 | 1.78 | 42.05 | 34.04 | 26.28 | 3.39 | 0.10 | 0.26 | 0.13 | 2.86 | 0.02 | 0.19 | 3.10 |
| w2-devplan-2 | 2 | 73 | 1.97 | 1.75 | 36.67 | 31.32 | 25.99 | 2.03 | 0.09 | 0.17 | 0.11 | 1.62 | 0.02 | 0.21 | 3.06 |
| w2-devplan-2 | 3 | 71 | 2.06 | 1.80 | 35.23 | 31.41 | 26.45 | 1.52 | 0.09 | 0.14 | 0.11 | 1.13 | 0.02 | 0.26 | 2.96 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

