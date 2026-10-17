# #1107 candidate A/B: fork-0.1.42-1107 vs stock v0.1.42 — RTX 5090, short prompts (gprobe)

Reported by gravitomagnetic (same machine/reporter as the 2026-10-08 interleaved folder).
Dates: 2026-10-11. Requested by @brenoperucchi on PR #1592 (candidate-build re-run offer taken up
for PR #1107: grouped expert gather on short/staged prefill walks).

## Builds

- A: stock v0.1.42 (61b3fb5d), kit built locally with the usual three CUDA pins (13.4 kit, arch 120).
- B: brenoperucchi/Strata tag `fork-0.1.42-1107` (39c01f07) = v0.1.42 + one commit touching only
  `src/prefill/prefill.cpp` (+36/−18). Verified tag-to-tag diff: `serve/`, `tools/vision/`,
  `CMakeLists.txt`, `setup.py` all empty. Same toolchain, same build command.
- Both kits report version string `0.1.42` — legs were gated by **sha256 of the live binary**
  (A `28f13ce8…`, B `52ee465a…`), not the version line. The wrapper stayed at v0.1.42 on both sides.
- Vision binary identical on both sides (copy of the verified v0142 kit).

## Machine and configuration (identical for all legs)

- RTX 5090 32 GB, 122 GB system RAM, Ryzen 9 7950X, NVMe; Linux, driver 595.99.02.
- IQ3_S GSQ-RCO pack, native experts, expert-profile; `--expert-cache 6500` floor (auto-filled to
  8338–8339 slots, 15.81–15.82 GiB — see state-hash note), `--prefill auto`, `--spec 4`,
  `--spec-min-p 0.5`, MTP runtime head, `--max-context 1048576`, YaRN scale 4, `--kv int8`,
  `--kv-resident 32768`, vision on, `--vram-reserve-mib 995`, server `parallel: 2` (no `--batch-mtp`).
- `STRATA_NO_LARGEPAGES=1` (upstream #771 guard) on every leg.
- Env per brenoperucchi's request, identical on both builds: `STRATA_STATE_HASH=1`,
  `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_ROUTE_TAIL_SKIP=0`.

## Protocol

A B A B A B (3 matched pairs), full swap between every leg, sha256 kit gate + real-completion
engine-up gate before each leg, 30 s settle. Workload: `strata-bench` core-v0.1.1,
`--sizes gprobe` only — 150/300/450-token excerpts of the pinned xlong.txt with a fixed system
prompt, 30 calls per size (first 2 warmup), temp 0, seed 42, 1 token out, `enable_thinking: false`.
Per-leg result JSONs in `data/` (six files, one per leg — every leg has its own label, no
timestamp collisions). `statehash_<leg>.txt`: the leg's 91 `STATE_HASH` lines sliced from the
serve log between its boot marker and leg end. `boot_lines_serve_log_extract.txt`: engine-start,
PCIe-probe, CPU-pool, and expert-cache-slot lines per leg.

## Result 1 — short-prompt prefill: the fork wins, consistently, ~12 ms

| size (tokens read) | stock 0.1.42 median [range] | fork-0.1.42-1107 median [range] | Δ |
|---|---|---|---|
| gprobe150 (227) | 771.5 [770.3–771.5] | 760.0 [759.1–761.2] | −11.5 ms (−1.5%) |
| gprobe300 (384) | 855.5 [855.4–856.8] | 843.3 [842.5–843.5] | −12.2 ms (−1.4%) |
| gprobe450 (521) | 931.6 [930.5–932.0] | 918.2 [917.6–919.3] | −13.3 ms (−1.4%) |

All 9 matched pairs negative; within-condition median spread ≤2 ms per size. The effect is real
and reproducible, but ~12 ms here, not the ~70 ms measured on the Swift IQ3_XXS box: our stock
floor is 2.3× higher (771 vs 331 ms at gprobe150), so the launch/wait/release grouping the patch
buys is a smaller share of our total. Our IQ3_S expert blobs are ~2× IQ3_XXS and every gprobe row
read fully cold (`reused 0`). Worth merging on the originating box's evidence; it does not close
this config's short-prompt floor.

## Result 2 — state-hash parity: build-neutral, but boot-sensitive on an auto-sizing cache

91 fingerprints per leg. The six legs split into two clusters — **not by build**:

- Pair 1 (A vs B): 91/91 lines bit-identical. Pair 3: bit-identical.
- Pair 2: 80 lines differ from L=227 onward — and A-p1 vs A-p3 (same build) differ identically.
- The clusters correlate exactly with the expert-cache slot count at boot: legs 1–3 came up at
  **8338** slots, legs 4–6 at **8339** (the auto-fill to VRAM-minus-reserve fluctuates one slot
  between boots). Same-build legs across that boundary differ as much as cross-build legs;
  same-slot-count legs are identical regardless of build.
- Requests and outputs are byte-identical across all six legs (`request_sha256` / `output_sha256`
  match leg to leg), so this is a residency-dependent accumulation path — one expert flipping
  between resident and staged changes gather grouping and FP ordering — not a semantic change.

Conclusion for #1107: **the patch does not change the state hash under matched boot conditions.**
Caveat for anyone using `STRATA_STATE_HASH` as a CI gate on an expert-cache auto-sizing box:
fingerprints are not stable across boots unless the slot count is pinned (or the comparison is
restricted to same-boot legs).

## Files

- `data/A-0142-p{1,2,3}.result.json`, `data/B-1107-p{1,2,3}.result.json` — strata-bench v1 results, one per leg.
- `data/statehash_*.txt` — per-leg STATE_HASH slices.
- `data/boot_lines_serve_log_extract.txt` — per-leg boot lines (slot counts, PCIe probe, CPU pool).
