# RTX 5090 desktop, 1M-agent regime: 0.1.42 vs 0.1.43 interleaved A/B

Reported by gravitomagnetic (same machine/reporter as `2026-10-08-community-rtx5090-1m-agent-interleaved`
and the 2026-10-02 5090 report; #466/#1134 lineage). Dates: 2026-10-11.
Both kits built locally from tags at the CUDA 13.4 kit, arch 120.

## Machine and configuration (identical for all legs)

- RTX 5090 32 GB (desktop, PCIe Gen5 x16), Ryzen 9 7950X (16c/32t, AVX-512), 122 GB system RAM, NVMe; Linux, driver 595.99.02.
- IQ3_S GSQ-RCO pack, native experts, expert-profile; `--expert-cache 6500` floor (auto-fills to 8,338–8,339 slots, 15.81–15.82 GiB, ~33% of experts resident), `--prefill auto`, `--spec 4`, `--spec-min-p 0.5`, MTP runtime head, `--max-context 1048576`, YaRN scale 4, `--kv int8`, `--kv-resident 32768`, vision on, `--vram-reserve-mib 995`, server `parallel: 2` (`--batch 2`, no `--batch-mtp`), 47 GiB pinned arena, `STRATA_NO_LARGEPAGES=1`.
- Steady-state decode expert hit rate 91–94% (4–5.5% of routed experts streamed over PCIe) — no PCIe pressure in this regime.

## Protocol

A B A B A B (A = v0.1.42 61b3fb5d, B = v0.1.43 7695859a), full swap between every leg with the
`serve/` wrapper checked out at the matching tag, version-verify gate on the live binary before
each leg (all six passed), 30 s settle. Two probe sets per leg:

1. `strata-bench` (our harness, pinned prompts): xhigh cold prefill 200K, instruct decode 256
   (temp 0.7 / presence 1.5), needle recall 200K, greedy decode 256 (doubles as the repetition gate).
2. `strata-bench core-v0.1.1 --sizes gprobe`: 150/300/450-token short prompts, 30 calls/size
   (2 warmup), temp 0, seed 42, 1 token out — the band the staged walk serves.

Decode rates are the engine's own per-request `generated in N ms (tok/s)` log lines for the
256-token probes; wall clocks round to the same tick at 1.4–2.4 s. Gates: needle 6/6 correct,
greedy count-to-300 zero adjacent-duplicate pairs 6/6 — answers clean on both sides, consistent
with the release's byte-identical claim for CUDA.

## Results

| probe | 0.1.42 | 0.1.43 | delta |
|---|---|---|---|
| instruct decode 256 | 120.3–122.6, med 121.3 tok/s | 122.5–123.6, med 122.8 tok/s | **+1.2%** |
| greedy decode 256 | 209.6–216.3, med 213.9 tok/s | 218.8–222.8, med 219.0 tok/s | **+2.4%** |
| cold prefill 200K (xhigh) | 58.0 s | 58.1 s | +0.2% (flat) |
| needle recall 200K | 57.5 s | 57.6 s | +0.2% (flat) |
| gprobe150 (227 tok read) | 772.9 [771.2–773.5] ms | 775.6 [772.3–777.0] ms | +0.36% |
| gprobe300 (384) | 854.8 [854.1–855.4] ms | 855.8 [855.8–857.3] ms | +0.12% |
| gprobe450 (521) | 930.4 [930.3–931.1] ms | 933.0 [930.8–933.3] ms | +0.28% |

- **Decode:** every 0.1.43 leg beat every 0.1.42 leg in both rows — clean separation, not a
  median artifact. `STRATA_ADAPT_OVERLAP` on by default is right for this box.
- **gprobe (#1712 pinned step-record/token-id uploads):** small consistent cost here, all 9 pairs
  positive (+1.2 to +3.5 ms on ~800–930 ms reads). The staged walk on this config is
  expert-bytes-bound; the saved pageable-copy stall is smaller than the added memcpy + event-sync
  per chunk. Not an objection to the default (the gfx1100 stale-read fix is worth more than 3 ms),
  but sm_120 + native IQ3_S + high hit rate is a measured "pageable was fine" data point.
- **Regime contrast with #2010** (5090 laptop, x8 PCIe, 28.7 GB/s probe, k8v4, 10,320 slots:
  overlap costs 21%): this box streams only 4–5.5% of routed experts in steady state. If the
  discriminator is streamed-bytes-per-window vs window compute time, a residency/PCIe-fraction-
  aware default could cover both regimes.

## Verdict

0.1.43 is the daily keeper on this machine: decode +1.2–2.4% (the workload that dominates an
agent session with warm caches), prefill flat, short prompts a rounding-error cost, all gates
clean.

## Files

- `data/version_ab_0142_*.json`, `data/version_ab_0143_*.json` — classic probe legs (six).
- `data/014[23]-p[123].result.json` — gprobe legs (six, strata-bench v1 schema).
