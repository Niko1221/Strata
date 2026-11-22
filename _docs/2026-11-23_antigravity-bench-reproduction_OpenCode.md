# Reproducing the Antigravity Sakura-Inkling-2 98K benchmark — 3 runs (OpenCode)

## Objective
User asked to run the Antigravity benchmark 3 times under identical conditions
to verify the claimed numbers (prefill 11.95 t/s, decode 22.44 t/s) before
treating them as verified facts in the v0.7.5 release.

## Conditions (copied from configs/strata-sakura-inkling2-98k.json)
- exe: H:/Strata/build/engine-cuda135/strata-inkling2.exe (built 2026-11-23 01:25,
  sources last modified 01:14 -> exe contains the latest Antigravity source).
- model: H:/Strata/models/Sakura-Inkling-2-Mini-Uncensored-Q4_K_M.gguf (17.48 GB).
- flags: --max-context 98304 --kv-q8 --expert-vram 7.8 --bench --chunk 512
  --max-new 32 (RUN3 added --tokens 1..16 to match Antigravity's "16 tok" prefill).
- GPU: RTX 5060 Ti 16 GB, sm_120, ~12.3 GiB free at start.

## Results (exact outputs: H:/Strata/evidence/antigravity-bench-run{1,2,3}.txt)
| Run | Prefill | Decode (31 tok) | expert hits/misses | cpu_ms |
| ---: | --- | --- | --- | ---: |
| 1 | 2 tok / 214.8 ms (9.31 t/s) | 2598.6 ms (11.93 t/s) | 947 / 2797 | 1979.8 |
| 2 | 2 tok / 222.1 ms (9.01 t/s) | 3166.0 ms (9.79 t/s) | 952 / 2792 | 2443.7 |
| 3 | 16 tok / 1461.8 ms (10.95 t/s) | 3401.6 ms (9.11 t/s) | 9795 / 4317 | 2655.7 |

Run 3's prefill (10.95 t/s) is close to the claimed 11.95 t/s. All 3 exits = 0.

## Conclusion (honest)
- Decode 22.44 t/s was NOT reproduced: measured 9.11-11.93 t/s, roughly 2-2.5x
  slower than the claim. Prefill is approximately reproducible.
- Root cause visible in the logs: expert cache misses are high (≈2800 misses
  per 31-token decode on 2-token runs; 4317 on the 16-token run), so most MoE
  work falls back to CPU (`cpu_ms` ≈2000-2655). Antigravity reported the decode
  with GPU time 516.9 ms (near-complete VRAM residency), a different cache state.
- Therefore the v0.7.5 provenance note (commit 80868c840) is correct and necessary:
  Antigravity's speed figures are from that agent's session and were NOT
  independently reproduced here. They should not be presented as this session's
  verified measurements. `releaseQualified=false` unchanged.
- All 3 raw logs are kept as evidence; no claim is made beyond what they show.
