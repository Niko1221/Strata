# Blackwell RTX PRO 5000, IQ3_XXS, 4 batch slots: `--mtp-q4 all` + `--lookup-chain 3` moved idle-box solo from a ≤150 ceiling to 230–238 tok/s

Measured 2026-11-11/13 by [ken-jo](https://github.com/ken-jo) on a production shared box — the second
engine host behind the same LiteLLM gateway as the [2×4090 layer-split report](../2026-11-10-community-2x4090-layer-split-iq3xxs-mtpq4/README.md),
carrying the original (non-abliterated) model alias.

**Short answer for this box:** with drafting on the batch path the engine had a no-draft single-stream
IQ3_XXS ceiling of ~150 tok/s on this Blackwell card (response-level draft counters all read 0 — see
#5890); adding `--mtp-q4 all` + `--lookup-chain 3` took idle-box solo to 229.7 / 237.6 tok/s (+58 %),
two-at-once per-request decode to 116–131, and a true four-at-once quad to 66–77 per request — while
aggregate acceptance held (69.6 % → 68.5 % over two days of live traffic, within the box's drift).
The flag combination also *started reporting* per-request draft counters on single requests, which
looks like a decode-path change; the split of the win between the two flags is **not isolated** (the
combination was tested as one arm).

## Machine and configuration

- NVIDIA RTX PRO 5000 Blackwell, 48,935 MiB (47,341 in use while serving), AMD Threadripper PRO 9975WX
  32-core, 250 GB RAM, Linux; single card.
- Strata v0.2.12 (engine 0.1.58 + control 0.1.58, prebuilt CUDA 13, release tag `123bad4`), upgraded
  2026-11-09 from v0.2.11.
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS (original weights); native 262,144-token window, no rope
  flags; `--kv int8 --kv-resident 32768`; `--vision`; `--expert-cache auto` (34.5 GB primary, learned
  profile save/reload); conversation cache 65,536 MiB / 48 slots.
- Serving shape: 4 batch slots (control appends `--batch 4`), `--batch-groups 1`, `--batch-mtp` on;
  MTP rt pack, `--spec 4 --spec-min-p 0.5` before the change (engine reported spec 6 with auto-ngram).
- The Blackwell-specific exclusions from our side stay in force (the layer-split group-scratch switches
  abort prefill on this card; the sm_86 exact-kernel switches are not applied here).

## Measurement method

Same harness style as the paired 2×4090 report: non-streaming requests through the gateway shim, decode
from the engine's own `response.timings`, fresh nonce prompts (`q<run><hex>` invented words, temp 0,
256 cap), regime tags per arm (`pre_running`/`post_running`), solo arms fire only at `live.running` 0.
`runs-after-pairs.log` holds two-at-a-time pairs; `runs-after-quad.json` holds three four-at-once quads.
Quick-look format, not the standard ladder. One reporting caveat baked into the data: on this batch-path
box, `draft_n`/`draft_n_accepted` read **0** in every before-arm response even though the engine log
shows real batch drafting (`strata batch: MTP proposals accepted 203 of 270 (75.2 %)`, totals
1,655,458/2,377,984 = 69.6 % since boot) — decode tok/s is the only trustworthy column there (#5890).

## Files in this folder

| File | What it is |
| --- | --- |
| `bench_harness.mjs` | node harness (gateway key read from a file at runtime; none embedded) |
| `runs-before.log` | arms on the untouched config (solo, pairs, fresh read), live team traffic present |
| `runs-after.log` | same arms after `--mtp-q4 all` + `--lookup-chain 3` |
| `runs-after-pairs.log` | four two-at-a-time pair runs after the change |
| `runs-after-quad.json` | three four-at-once quads after the change (the four-slot floor) |
| `config-before.json`, `config-after.json` | run-config copies for the two arms (this box's paths; no secrets) |

## Results

| Arm (engine timings, clean = fired at `live.running` 0) | before | after |
| --- | --- | --- |
| Solo short-prompt decode, tok/s | **112.6–150.0** (two idle-box samples; the engine-log-confirmed ceiling of the no-draft batch path here) | **229.7 / 237.6** |
| Solo per-response draft counters | 0 / 0 (batch-path reporting — #5890; batch drafting itself ran at 63–75 % on team traffic) | draft_n 194–208, accepted 152–155 (78–80 %) |
| Two-at-once per-request, tok/s | 87.9–95.5 (under `mid_running` 3 team load) | 116.4–130.7 (pairs, idle box; first run transient 81.9/72.7) |
| Four-at-once per-request, tok/s | (not measured in the before arm) | **66.1–77.0** with one 101.9 draft-burst slot |
| Fresh read, tok/s | 4,056 at ~211K tokens, congested regime | 5,618 at ~174K tokens, clean regime |
| Acceptance (engine totals) | 1,655,458 / 2,377,984 = 69.6 % | 68.5 % over 4,640 requests, two days of live traffic |

Mechanism line, identical to the 4090 box: `--mtp-q4: 10 projections as Q4_0 (37.3 MiB)`, `draft head
over 106,299 tokens as Q4_0: 146.0 MiB read per step, was 178.4`.

Two findings worth maintainer attention. **(a)** The jump from ~150 to ~230 on *single* requests came
with per-response draft counters flipping from 0 to real values — the flag combination appears to move
single requests onto a drafting path on a `--batch-mtp` box. Whether single requests should draft on a
4-slot batch box, and whether that path switch is an intended side effect of `--lookup-chain`/`--mtp-q4`,
is a design question; our data only says it changed and measured +58 %. **(b)** Isolation between the
two flags was not attempted (tested as one arm; rollback is a single file restore,
`~/strata/backups/strata-iq3_xxs.json.pre-leverA-aisrv-20261111` on the box). The four-at-once floor
(66–77) is the number most comparable to the batch-MTP era's community reports on Blackwell-class cards.

## What is live on this box now (2026-11-13)

`config-after.json` content, NRestarts 0 since the 2026-11-11 11:07 restart, zero startup warnings;
team warm turns decode 168–250 tok/s in `/metrics` per-request fields, unchanged restart behavior
(boot ready 25 s). Conversation-cache pressure remains the box's real limit (48/48 slots parked,
~40 evictions/h) — unrelated to these two flags and untouched.
