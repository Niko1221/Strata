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

## Additional reproductions (same session)
- **Inkling-2-Mini-UD-Q4_K_M.gguf, 16K ctx** (Antigravity claim: pp 34.51 /
  tg 24.67 t/s): measured `pp 16 tok 619.5 ms (25.83 t/s) | tg 31 tok
  1416.9 ms (21.88 t/s)`, hits=12506/misses=1606, cpu_ms=684.0. Raw log:
  H:/Strata/evidence/antigravity-ud16k-bench.txt. Same order of magnitude; below claim.
- **Sakura-Inkling-2-Mini, 16K ctx** (Antigravity claim: tg 19.34 t/s,
  +42.8%, moe_gpu 52.4 ms): measured `pp 16 tok 898.5 ms (17.81 t/s) | tg 31
  tok 2632.1 ms (11.78 t/s)`, hits=11850/misses=2262, cpu_ms=1676.0. Raw log:
  H:/Strata/evidence/antigravity-sakura16k-bench.txt. Decode not reproduced.
- **DS4-9B / Hikari PQ2_0**: model GGUFs WERE located in subfolders
  (H:/Strata/models/新しいフォルダー/DeepSeek-V4-Pro-Qwen3.5-9B-...gguf 5.24 GB,
  H:/Strata/models/hikari-pq20/Hikari-LoRA-fixed-scale-PQ2_0.gguf 6.71 GB),
  but Antigravity's benchmark used a `--ctx-size 4096 --tokens 1..16 --n 64`
  harness that is NOT in this repo (no CLI here accepts --ctx-size; the 11/23
  strata-q38.exe build is absent, only 11/17 builds exist). These two could not
  be reproduced without the missing harness/exe — recorded as blocked, not failed.

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
- Summary table (claim -> reproduced): Sakura 98K tg 22.44 -> 9.11-11.93 (no);
  UD 16K pp 34.51 -> 25.83, tg 24.67 -> 21.88 (close/under); Sakura 16K
  tg 19.34 -> 11.78 (no); DS4-9B 2.84x and Hikari -18% (blocked: harness/exe
  missing). Therefore none of Antigravity's headline speedups was fully
  reproduced on this session; the provenance note in v0.7.5 is justified.
- All raw logs are kept as evidence; no claim is made beyond what they show.
