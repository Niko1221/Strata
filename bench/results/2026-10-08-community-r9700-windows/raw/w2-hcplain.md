# bench_decode - w2-hcplain

- run: 20261008-203327  base_url: http://127.0.0.1:8080  model: qwen3.8-flash-next-iq3_xxs
- requested: prompt ~24576 tokens, max_tokens 128, 3 repeats, seed 1234, warm/cold: cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm
- usable 3 of 3 (rejected 0)

## Measured conditions

| label | rep | usable | prompt | reused | ctx | gen | decode_ms | decode_tps | hit_% | accept_% | ckpt | exchanged | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-hcplain | 1 | yes | 99696 | 0 | 99824 | 128 | 2894 | 44.2 | 93.8 | 72.0 | 6 | 552 | 22.61 |
| w2-hcplain | 2 | yes | 99696 | 99691 | 99824 | 128 | 2504 | 51.1 | 97.2 | 77.3 | 6 | 814 | 19.56 |
| w2-hcplain | 3 | yes | 99696 | 99691 | 99824 | 128 | 2203 | 58.1 | 97.7 | 74.2 | 6 | 907 | 17.21 |

## Median and range (usable repeats only)

| metric | n | median | min | max | range |
|---|---|---|---|---|---|
| decode_tps | 3 | 51.1 | 44.2 | 58.1 | 13.9 |
| decode_tps_warm | 2 | 54.6 | 51.1 | 58.1 | 7.0 |
| prefill_tps | 3 | 97.0 | 91.4 | 1307.8 | 1216.4 |
| ms_per_token | 3 | 19.5625 | 17.2109 | 22.6094 | 5.3984 |
| hit_pct | 3 | 97.2 | 93.8 | 97.7 | 3.9 |
| accept_pct | 3 | 74.1573 | 72.0 | 77.3333 | 5.3333 |
| prompt_tokens | 3 | 99696.0 | 99696.0 | 99696.0 | 0.0 |
| generated | 3 | 128.0 | 128.0 | 128.0 | 0.0 |
| context | 3 | 99824.0 | 99824.0 | 99824.0 | 0.0 |

## Window split (`strata decode timing`, ms/window)

| label | rep | windows | T | tok/win | ms/win | verify | wait | host | plan | actq | jobs | CPU | stage | commit | draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2-hcplain | 1 | 76 | 1.99 | 1.68 | 38.08 | 31.77 | 25.04 | 3.01 | 0.07 | 0.21 | 0.01 | 2.32 | 0.39 | 0.16 | 2.85 |
| w2-hcplain | 2 | 73 | 2.03 | 1.75 | 34.31 | 30.00 | 25.72 | 1.48 | 0.07 | 0.14 | 0.01 | 1.22 | 0.03 | 0.15 | 2.88 |
| w2-hcplain | 3 | 65 | 2.37 | 1.97 | 33.90 | 30.27 | 26.02 | 1.46 | 0.08 | 0.16 | 0.01 | 1.20 | 0.01 | 0.15 | 3.09 |

## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)

_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_

