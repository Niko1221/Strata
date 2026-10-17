# Community benchmark B1: `--layer-split auto` on an unequal 16+12 GB pair (IQ3_S)

Measured on 2026-10-11 by @mmbeanz. Same box, same pinned profile,
same suite as run A (`../A/`) — only the split policy changed: explicit `40` -> `auto`,
and `--pcie-frac` removed so the engine probes and assigns per-stage shares.

**Headline.** On this unequal pair (RTX 5060 Ti 16 GB on PCIe 5.0 **x8** + RTX 3060 12 GB
on PCIe 3.0 **x4**), `auto` chooses **K=3** and loses badly on the prompt path
(**-25% at 32K**) and on short-context decode (**-19%**), while long-context decode ties.
Separately: **expert-cache hit rate is flat in held count** (85.9% at 8,842 experts vs
86.5% at 5,935, a 33% smaller cache) — which contradicts *both* of the planner's coverage
curves on this model/box.

## Hardware and software

Identical to run A — see `../A/README.md` for the hand-written table (7900X 12c/24t,
64 GB DDR5-6400, 990 EVO Plus 4 TB DRAM-less NVMe, driver 615.71.09, CUDA 13.4,
systemd-nspawn on bare metal, measured H2D 28.2-28.7 GB/s and 3.15-3.2 GB/s, `LnkSta`
under load: 5.0 x8 / 3.0 x4). Strata 0.1.43 from upstream `7695859a` + a `setup.py`-only
local commit.

## Model and configuration

- Model: Qwen3.8-Flash-Next IQ3_S, ISTA-DASLab GSQ-RCO pack (sha256 in `../sha256sums.txt`),
  512K context, yarn factor 2.0.
- Pinned boot profile `data/expert-profile.bin`, no `--expert-profile-save`; engine logs
  `profile ...: 24576 ranked pairs`.
- Full config in `config.json`. Effective launch line:

```text
strata --serve --pack /home/llm/Strata-data/packs/iq3_s \
  --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --expert-profile /home/llm/Strata/data/expert-profile.bin \
  --expert-cache auto --prefill auto --spec 4 --mtp rt \
  --max-context 524288 --rope-scaling yarn --rope-scale 2.0 \
  --kv int8 --kv-resident 20480 --vision --remote-expert-opt --trim-stage-weights \
  --spec-min-p 0.70 --layer-split auto
```

### What `auto` decided on this pair (from the engine log, dry start of the same config)

```
layer split: card order 1,0 (the faster card last, #1352)   -> CUDA0 = RTX 3060, CUDA1 = RTX 5060 Ti
PCIe probe: CUDA0 3.2 GB/s -> miss cost x6.26 ; CUDA1 28.7 GB/s -> miss cost x1.00
CUDA0 28 SMs at 1.78 GHz -> 1.46 ms/layer ; CUDA1 36 SMs at 2.59 GHz -> 0.78 ms/layer
K=2 would give a stage fewer than 3 layers ... chose K=3 instead (48.5 ms per decode window, not 48.2)
K=3 - predicted 48.5 ms; the caches hold 6135 of 24576 profiled pairs (~95.2% of the routed mass), best of 89 placements
NEAR TIE - K=6 is predicted 49.4 ms, within 1.9% of K=3; the cards differ, so the prediction may not rank them
           (it prices decode, not the prompt). To try the other: --layer-split 6
prompt chunk auto: 3328 tokens, a 199-slot ring      (run A with K=40: 8192)
```

Engine facts at boot (`GET /metrics`): `expert_slots 5935` (11,394 MiB), `expert_slots_primary
1536` (2,762 MiB — the 3060 holds *all* experts of its 3 layers and idles `vram_free_mib
3548`), `pcie_frac 0.09` (auto per-stage: 0.09 on the 3060, 0.55 on the 5060 Ti).

## Method

Same as run A: `strata_bench.py --url <server> --config config.json --long --runs 3`
against an operator-started server, cold boot, 0 requests before the suite;
`GET /metrics?requests=all` snapshots before/after (`B1-before.json`,
`B1-after-full.json`). Client on a separate LAN box (RTT 0.25 ms); the `machine` block in
`results.json` blends client and server fields (see `../A/README.md`, *Tool findings*) —
hardware above is hand-written. Suite time ~205 s.

## Results

| Configuration | Prompt tok/s (median, range) | Decode tok/s (median, range) | TTFT s (median, range) |
| --- | --- | --- | --- |
| short chat (9 runs) | n/a | 66.6 (62.5-75.9) | 0.48 (0.43-0.53) |
| 4K prompt, fresh | 835.6 (827.1-838.6) | 62.0* | 4.84 (4.82-4.89) |
| 16K prompt, fresh | 1227.4 (1222.1-1242.2) | 57.9* | 13.11 (12.96-13.17) |
| 32K prompt, fresh | 1290.9 (1288.1-1292.6) | 54.9* | 24.86 (24.83-24.92) |
| agent follow-up turns (15) | see `results.json` | 73.3 (50.9-85.5) | 3.92 (3.34-4.93) |

\* from the engine's per-request `decode_tok_s` in `/metrics?requests=all` (24-token cap).

Draft acceptance 1,046 / 1,485 = **70.4%**. Agent conversation-cache hit 68.8% (same as A).

### A (K=40) vs B1 (auto K=3) — same box, same day, same profile

| metric | A (held 8,842) | B1 (held 5,935) | delta |
| --- | ---: | ---: | ---: |
| prompt 32K tok/s | 1,720.8 | 1,290.9 | **-25.0%** |
| prompt 16K tok/s | 1,412.7 | 1,227.4 | **-13.1%** |
| prompt 4K tok/s | 868.1 | 835.6 | -3.7% |
| TTFT 32K | 18.66 s | 24.86 s | **+33%** |
| decode 32K tok/s | 53.2 | 54.9 | +3.2% (tie) |
| decode 16K tok/s | 58.3 | 57.9 | -0.7% (tie) |
| decode 4K tok/s | 70.7 | 62.0 | **-12.3%** |
| decode 609 tok/s | 71.6 | 57.7 | **-19.4%** |
| expert hit_rate (median, all 40) | 0.859 | 0.865 | +0.006 |
| `pcie_share` (median / max) | 0 / 0 | 0.08 / 0.247 | streaming on, no gain |

The engine's own warning was correct: its placement prediction **"prices decode, not the
prompt"** — and on this pair the prompt path is where `auto` bleeds. K=3 leaves the 3060
with 1,536 slots, which trips the #1616 chunk clamp (prompt chunk 3,328 instead of 8,192),
and the 3060's 3,548 MiB of free VRAM is stranded because it already holds every expert of
its 3 layers.

### Expert-cache coverage: the flat-hit-rate result

| held experts | f | old `(r+1)^-1.2` predicts | new `H(f)=1-(1-f)^3` predicts | **measured hit_rate** |
| ---: | ---: | ---: | ---: | ---: |
| 8,842 (run A, K=40) | 0.360 | 97.0% | 73.8% | **85.9%** |
| 5,935 (run B1, K=3) | 0.241 | 95.6% | 56.4% | **86.5%** |

Cutting the cache by **33%** changed measured hit rate by **+0.6 points** — i.e. on this
box, model and traffic, hit rate is essentially **independent of held count** across
f = 0.24-0.36. The old curve has the right *shape* (near-flat) and sits ~10 points too
high; the new `H(f)` fit has the wrong shape here (it predicts a 17-point drop that did not
happen) and the wrong level. Fitting `b` to each point separately gives b = 4.4 (A) and
b = 7.3 (B1) — i.e. no single `b` works, because the measured curve is flat where both
functional forms are steep.

Caveat, stated plainly: `hit_rate` is per-request over the whole request (prefill + decode)
with the adaptive tier churning and the prompt path borrowing cache slots, so it is not a
clean measurement of "coverage of the static held set". But it is the quantity the operator
sees, and it is what the planner's curve is implicitly compared against.

## Correctness and limitations

- Speed only; no needle/quality check in this run (needle suite: `../needles/`).
- One suite per config (3 runs each), single day; A and B1 measured the same evening.
- `auto` also changed `pcie_frac` (0.00 -> per-stage 0.09/0.55), so B1 vs A is a *policy*
  comparison, not a single-variable one. The isolation run is **B1b** (`auto` + explicit
  `--pcie-frac 0.00`); the engine-endorsed alternative is **B1c** (`--layer-split 6`).
- No concurrency, no vision latency, no mmap/low-RAM tier, Windows/HIP untested.
