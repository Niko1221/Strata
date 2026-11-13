# Decode levers on a shared 2× RTX 4090 layer-split IQ3_XXS box: `--mtp-q4 all` + four kernel switches, `--lookup-chain 3`, and a Q4-cost `--batch-groups 1` retest

Measured 2026-11-09/10 by [ken-jo](https://github.com/ken-jo) on a production shared box (two team coding
agents load it through a gateway for most of the day, so every arm below carries a concurrency regime tag).

**Short answer for this box:** `--mtp-q4 all` plus four exact-answer kernel switches moved clean-window
solo decode from a 130–140 tok/s band to 147.0 / 157.5 / 160.9 (+11–20 %) with draft acceptance untouched
(aggregate 68.4 % → 68.6 %); `--lookup-chain 3` was kept on a marginal pass (solo 150.3, acceptance −0.9 pt
within the box's own drift); `--batch-mtp --batch-groups 1` flipped from a clear loser at bf16 draft cost
(44–79 tok/s concurrent) to a strong *solo* winner at Q4 draft cost (173.3, +15 % vs pipelined) but stayed
below the bar for team-concurrent traffic and was reverted. Gating behavior worth recording: on pipelined
groups, slot drafts — MTP **and** lookup-chain alike — switch off whenever 2+ requests decode concurrently,
so drafting-only levers cannot lift the concurrent floor on this layout.

## Machine and configuration

- 2× NVIDIA GeForce RTX 4090 24,564 MiB (layer split across both), Intel i9-14900K, 123 GB RAM, Linux;
  cards capped at 450 W (measured ~440 W draw in bursts — a raise to 550 W changed nothing and was undone).
- Strata v0.2.12 (engine 0.1.58 + control 0.1.58, prebuilt CUDA 13 packages, release tag `123bad4`),
  upgraded 2026-11-09 from v0.2.12's predecessor on the same box.
- Model: SC117 Qwen3.8-Flash-Next-abliterated, GSQ-RCO IQ3_XXS pack; `--max-context 300000` with YaRN
  factor 1.1444 over the native 262,144 window; `--kv int8 --kv-resident 32768`; `--vision`;
  `--expert-cache auto` with a learned profile saved on exit and reloaded at start; expert cache
  ~10,245 CUDA0 + 7,752 CUDA1 slots; conversation cache 24,576 MiB / 10 slots; `--pool-workers 15`.
- Drafting: MTP rt pack, `--spec 4 --spec-min-p 0.5`, `--batch-mtp` **off**; the control appends
  `--layer-split auto --batch 2` to the engine command line itself — they are *not* in the run config's
  `args`, which surprised us when auditing what is live (worth a doc line somewhere).
- Serving shape: 2 batch slots in `auto` pipelined groups = 1 slot × 2 groups.

## Measurement method (client-side, regime-tagged)

Non-streaming chat completions through the local shim; decode read from the engine's own
`response.timings` (`predicted_per_second`, `prompt_ms/prompt_n`, `draft_n`, `draft_n_accepted`) —
client-side, hence unaffected by the 0.2.12 admission-time counting (#5318/#5307/#4006). Fresh nonce
prompts (list 240 invented words beginning with `z<hex>`, temperature 0, 256-token cap) plus a
~131k–196k-token random-hex-slug read arm. Solo arms fire **only** when `/metrics` shows `live.running`
0; every log line carries `waited_s`/`pre_running`/`mid_running`/`post_running` tags because the box is
shared — the tags are part of the data, and mean comparisons across regime tags are labeled as such.
This is a quick-look report, not the standard ladder.

## Files in this folder

| File | What it is |
| --- | --- |
| `bench_harness.mjs` | node harness that emits the JSON-lines runs below (reads a gateway key at runtime; no key embedded) |
| `runs-baseline-concurrent.log` | concurrent pairs on the untouched config, under live team load |
| `runs-leverA.log`, `runs-leverA-round2.log` | arms after `--mtp-q4 all` + the four kernel switches |
| `runs-leverB-chain.log` | arms after adding `--lookup-chain 3` (live config today) |
| `runs-groups1-retest.log`, `run_benchC.sh` | the `--batch-mtp --batch-groups 1` retest arms (reverted config) |
| `config-orig.json`, `config-leverA.json`, `config-leverB.json`, `config-leverBC.json` | run-config copies per arm (paths are this box's; no secrets) |

## 1. `--mtp-q4 all` + four exact-answer kernel switches (kept)

Added to `args`: `--mtp-q4 all`. Added to `env`: `STRATA_QSA_ROWS=1`, `STRATA_SIDE_MIXER=1`,
`STRATA_B6_MMVQ_ROWS=1` (exact-answer decode kernels, on by default only on sm_86/sm_12.0 — off by
default on this 8.9 box) and `STRATA_SELECT_TWO_PASS=1`.

Engine startup confirms the mechanism: `--mtp-q4: 10 projections as Q4_0 (37.3 MiB)` and `draft head over
106,299 tokens as Q4_0 (146.0 MiB read per step, was 178.4)`. Expert slots went 17,995 → 17,976 — the Q4
copies allocate, so net VRAM ≈ 0 and **the win is draft-step head bandwidth (−18 %), not freed cache**;
any expectation of "mtp-q4 frees VRAM → more slots → more speed" mispredicts on this layout.

| Arm (engine timings, clean = fired at `live.running` 0) | before | after |
| --- | --- | --- |
| Solo clean decode, tok/s | 130–140 (team-side log band, pre-change — labeled, not raw here) | **147.0 / 157.5 / 160.9** |
| Draft acceptance (aggregate, engine totals) | 68.4 % (16:12–18:27 window) | 68.6 % after restart; bench-window per-response 135–143 of 184–206 |
| Two-slot concurrent floor mean, tok/s | 77.0 / 77.7 | 83.9 / 69.9 (mean unchanged, now asymmetric) |
| Sub-solo momentary windows mid-3-way | ~77, drafts off | **131.2, drafting** |
| Fresh ~204K-token read, tok/s | 6,829 | 6,449 / 6,469 |

Two honest caveats. (a) The concurrent floor *mean* did not move — the change redistributed it: the
first-admitted slot now captures momentary sub-solo windows and drafts inside them (131.2 measured),
where those same windows ran draft-off at ~77 before. (b) The fresh-read drop is confounded: this restart
re-ran the boot PCIe probe, which re-picked `pcie_frac` 0.10 → 0.18 and shifts long-read rates by itself;
it does not implicate `SELECT_TWO_PASS`. Recommendation for the measurement guide: record `pcie_frac`
from `/metrics` in every long-read A/B — never compare fresh reads across boots without it.

## 2. `--lookup-chain 3` (kept on this box, marginal)

Deployed on top of arm 1. Reports `--spec 4 --lookup-chain 3` → engine reports spec 8 (4 MTP + 3 chain +
bonus) where the box previously ran spec 6 (4 MTP + 2 auto-ngram). `runs-leverB-chain.log`: the one
qualifying clean solo sample that night read 150.3 tok/s (draft_n 200 / accepted 143); concurrent pairs
in a *harder* regime (`mid_running` 5, vs 3–4 for the leverA baselines) read 74.7–80.3; team acceptance
66 min after boot was 69.1 % against this box's 70.0 % full-day reference. Kept: it is within every gate
we set for ourselves and the mechanism (chained drafts after MTP) only ever fires in solo/sub-solo windows
on this layout. The log lines make that gating explicit — **every concurrent sample logged `draft_n` 0,
chain included** (see #3794 for the missing pipelined-path drafts).

## 3. `--batch-mtp --batch-groups 1` retest at Q4 draft cost (reverted, kept on the shortlist)

Same box, leverA base, 2026-11-10 23:10–23:51. Earlier the same day the identical flip was measured at
bf16 draft cost: 44–79 tok/s per concurrent request, worse than pipelined, with worse time-to-first-token.
After the Q4 draft head (146.0 MiB/step): **solo clean 173.3 tok/s (+15 % vs the pipelined 150.3)**,
concurrent pairs 83.9 / 81.0 / 87.4 / 81.4 (mean 83.4, `mid_running` 4–5), acceptance 17,778/25,225 =
70.5 %. Verdict on this box: serial windows become a genuine solo winner once the draft step is cheap —
worth retesting on any low-traffic layer-split box — but the serial-ization cost kept the team-concurrent
mean at 83.4, under the 85 tok/s bar we hold for always-loaded traffic, and the pipelined layout (whose
own startup warning quotes pipelined as 25–30 % faster than serial MTP windows) stayed live. The
pipelined-vs-serial ranking in that guidance is draft-cost-dependent and deserves that caveat.

## What is live on this box now (2026-11-13)

`config-leverB.json` content (leverA args **plus** `--lookup-chain 3`), NRestarts 0 since the 2026-11-10
23:51 restart, zero startup warnings, acceptance 70.0 % over the full day on leverA and 69.1 % on leverB's
first hour. Single-step rollback backups (restore + restart) named on the box:
`~/strata/backups/strata-iq3_xxs.json.pre-leverA-20261109` (original), `.pre-leverB-20261110`.
Related upstream threads: pipelined-path drafts #3794; batch-path per-request counters read 0 — #5890
(this box's concurrent `draft_n` 0 lines are the *decode-gate* case, the *reporting* case is #5890).
