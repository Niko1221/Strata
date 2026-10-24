# flash-next-strata-tune — 20261016T181038Z (INTERRUPTED after round 1 of 3)

model `flash-next-strata` via `http://localhost:4000`; prompt 4096; max_tokens 1024; server sampling defaults; cache_prompt false; ignore_eos true. plan step C 2026-10-16: per-request strata_tune sweep on the live v0.1.44.1 held config (engine auto pcie_frac 0.36, spec_min_p 0.5 = control); no restart. **Round 2's first request got HTTP 502: at 18:13:17 UTC a client (<LAN_IP>, curl) requested alias gemma4-moe and the proxy swapped stack 50 out for gemma4-moe-q8 (stack 28). Round 1 = one request per arm; the engine log shows every request completed normally (no error).**

| arm | tune | decode tok/s | accept | drafts | accepted | prefill tok/s | wall s | ext load |
|---|---|---|---|---|---|---|---|---|
| control | `null` | **41.1** | 0.712 | 784 | 558 | 359.5 | 36.374 | 0.0->0.0 |
| minp0.7 | `{"spec_min_p": 0.7}` | **42.4** | 0.829 | 667 | 553 | 362.7 | 35.549 | 0.0->0.0 |
| minp0.3 | `{"spec_min_p": 0.3}` | **34.9** | 0.427 | 1221 | 521 | 362.0 | 40.772 | 0.0->0.0 |
| pcie0.34+minp0.7 | `{"pcie_frac": 0.34, "spec_min_p": 0.7}` | **35.5** | 0.717 | 544 | 390 | 363.2 | 40.236 | 0.0->0.0 |
