# flash-next-strata-tune — 20261016T183446Z

model `flash-next-strata` via `http://localhost:4000`; prompt 4096; max_tokens 1024; server sampling defaults; cache_prompt false; ignore_eos true; 3 rounds, arms rotated per round. plan step C 2026-10-16 (resume after the 18:13 client swap; stack 50 back 18:34 UTC via the proxy): per-request strata_tune sweep on the live v0.1.44.1 held config (engine auto pcie_frac 0.36, spec_min_p 0.5 = control); no restart; 3 rotated rounds

| arm | tune | decode tok/s per round | median | accept per round | median | drafts per round | prefill tok/s | ext load |
|---|---|---|---|---|---|---|---|---|
| control | `null` | 36.2 / 36.5 / 36.9 | **36.5** | 0.612 / 0.586 / 0.607 | 0.607 | 744 / 806 / 769 | 360.3 / 351.6 / 360.3 | 0.0->0.0 / 1.0->0.0 / 0.0->1.0 |
| minp0.7 | `{"spec_min_p": 0.7}` | 40.2 / 42.8 / 34.4 | **40.2** | 0.787 / 0.857 / 0.727 | 0.787 | 634 / 629 / 483 | 356.4 / 351.0 / 356.3 | 0.0->0.0 / 1.0->1.0 / 1.0->1.0 |
| minp0.3 | `{"spec_min_p": 0.3}` | 34.2 / 34.8 / 34.5 | **34.5** | 0.415 / 0.434 / 0.436 | 0.434 | 1213 / 1176 / 1141 | 357.6 / 350.0 / 354.9 | 0.0->0.0 / 1.0->1.0 / 1.0->0.0 |
| pcie0.34+minp0.7 | `{"pcie_frac": 0.34, "spec_min_p": 0.7}` | 35.7 / 36.5 / 37.0 | **36.5** | 0.731 / 0.749 / 0.747 | 0.747 | 531 / 549 / 577 | 356.6 / 357.0 / 360.1 | 0.0->0.0 / 0.0->1.0 / 0.0->1.0 |
