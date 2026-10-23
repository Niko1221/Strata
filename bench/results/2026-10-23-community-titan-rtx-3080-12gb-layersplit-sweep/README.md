# Community benchmark: TITAN RTX 24 GB + RTX 3080 12 GB (layer split), Xeon E5-2683 v4, 125 GiB RAM — `--layer-split` sweep K=20→32 + 10-min endurance

Measured 2026-10-23 (UTC) on a headless HP Z840 by the operator's own agent (anonymous handle). Subject:
Strata app **v0.2.3** with **engine 0.1.48**, both built from source on the test box the same day, running
Unsloth **UD-IQ4_XS** Qwen3.8-Flash-Next at **262,144-token** context on an unequal two-card pair — the
*helper* (the cheap card holding the early layers) is a 12 GB sm_86 and the *main* card is a 24 GB sm_75
TITAN. I swept `--layer-split` K from the engine's automatic choice (**K=20**) up to **K=32** in steps of two
with 240-second streaming arms, then ran the engine's 10-minute endurance protocol at the ends of the range.
**Decode rose 99.8 → 114.2 tok/s (+14.5%)** from K=20 to K=32 on my client-counted streaming protocol, the
expert-cache hit fell 97.8 → 95.2%, TITAN average draw over the endurance window fell 176 → 165 W, TTFT
rose 2.4 → 2.8 s, and every one of the ~510 streamed requests across all runs errored zero. The gain
narrowed arm over arm (+2.6%, +4.8%, +1.9%, +1.8%, +3.0%) while the hit rate fell ~0.7 points per arm, and
the 3080 sat at 231 W average against its 280 W cap by K=32 — so I stopped there and did not probe K≥34;
the case for pushing further is weak (see "where the wall is" below) and this sweep is decode-centric
(`docs/MULTI_GPU.md` §"Unequal pairs" warns the *prompt-read* side can move more than decode in either
direction on a pair like this; the in-progress split search is the right long-term tool). Every number here
is a custom-protocol streaming number — **not comparable** to `strata bench` standard-suite medians — and
the absolute decode magnitudes carry a disclosed ~10-15% calibration gap against the engine's own reported
lines on long prompts. All raw per-request JSONL, configs, sanitized scripts and hashes ship with this
folder.

## Hardware and software

* Two GPUs:

  | card | nvidia-smi idx | PCIe slot | power.limit (cap set by operator) | default limit | role in this run |
  | --- | --- | --- | --- | --- | --- |
  | TITAN RTX 24 GB (sm_75) | 0 | 03:00.0, Gen3 x16 @ 8 GT/s (lspci `LnkSta`, full width and speed) | **210 W** | 250 W | main: layers 32-47 + head + MTP draft + 17.75 GiB expert cache (7,837 slots) at K=32 |
  | GeForce RTX 3080 12 GB (sm_86) | 1 | 04:00.0, Gen3 x16 @ 8 GT/s | **280 W** | 350 W | helper: layers 0-31 + 4.25 GiB expert cache (1,887 slots) at K=32 |

  The caps are a deliberate hard mandate (kept for acoustics and PSU margin on a Z840 that lives next to a
  desk); the sweep is therefore a *capped-power* sweep, not a stock-power one. The engine's own H2D probe
  read **12.1-12.4 GB/s** on both cards (its `pcie_frac` dropped to ~0.33 by itself). CUDA0/CUDA1 are
  inverted from nvidia-smi order via `CUDA_VISIBLE_DEVICES=1,0` — CUDA0 is the 3080.

* **HP Z840** (the server-class host `docs/MULTI_GPU.md` §"Servers, towers and workstations with many cores"
  recommends): Xeon E5-2683 v4, 16C/32T at 2.1 GHz base, **125.7 GiB RAM**, **no AVX-512** — the engine ran
  AVX-2 expert kernels with its automatic 15 pool workers (one per physical core, host thread pinned).
  **Caveat for reproducibility:** this BIOS ships turbo disabled and the OS released it through the MSR
  interface; a stock-clock Z840 will not reproduce these timings without the same release.
* **Storage:** the GGUFs, pack and profile live on a 953.9 GB **NVMe** (the expert arena, 55.43 GiB, mapped
  through the file cache and loaded at **8.61 GiB/s**; PLE direct-read probe 101,131 rows/s "fine"; a 7.3 TB
  SAS HDD sits idle; swap off). PLE and the arena read from the NVMe on *both* layer-split stages
  (`STRATA_STAGE_TRIM=1` makes each card load only its own span's dense weights), so a spinning disk here
  would put a stall inside every split window.
* Ubuntu 24.04.5, kernel 6.8.0-147-generic, driver **580.178.04**, nvcc CUDA **13.0.88**.
* **App v0.2.3** — source checkout, commit `5c8dd322528cc84f68f461ad54a506b9da8fa8d4`, commit date
  2026-10-23T07:57:07+02:00 — rebuilt locally with `./update.sh`. **Engine 0.1.48** local build
  (`engine/strata` mtime 2026-10-23 13:04 UTC; `BUILD.json` archs `[75]`, `build.ninja` additionally
  compiles `sm_86` cubins). **All measurements in this report — the 14:25 UTC baseline endurance and every
  arm from 15:02 to 16:06 UTC — ran this single same-day binary pair**, so no version confounds any
  cross-K delta. The machine previously ran app 0.2.1.1 / engine 0.1.46.1 (the 2026-10-20 standard-v1 log
  elsewhere on this box); those numbers are not folded in anywhere here.
* **Background load, disclosed plainly:** the box is not dedicated — the agent that operated this sweep
  shares the same server through the gateway the whole time, including a handful of ~130K-token-context
  gateway turns inside the endurance window (see the traffic excerpt). Every arm saw the same shared load;
  it inflates all TTFT numbers roughly equally, and none of the runs are "clean box" numbers.

## Model and config

* Model: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-IQ4_XS**, 3 shards of 10,946,624 + 49,835,229,856 +
  43,836,407,744 B = **93.68 GB** (the HF revision was not recorded at download; SHA-256s of all files are
  in `sha256.txt` — match those instead of a revision id). MMProj
  `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, 907,543,008 B; `--vision` on (1,024 vision tokens, sidecar on the
  3080) but **no image crosses any request in this report**.
* Native pack **built locally** with `tools/iq_pack.py` from the shards (`conversions.json` and
  `native_experts.txt` ship in `configs-and-startup/`; largest expert blob 3.48 MB, 512 experts/layer,
  `dense.bin` 1,485,688,320 B), so the expert path does not read GGUF blocks at all.
* Expert profile: **self-learned** `profile-learned.bin`, 24,576 ranked pairs, built for 24,576 slots,
  re-saved at every exit (`--expert-profile-save`). Same file across all arms and the endurance run — the
  PROFILE cache is warm *at boot* for every run (prefill log: "pre-filled 1887 of 1887 slots ... in 0.4 s").
  Honest caveat: the profile drifts across days, and today's profile was shaped by days of this agent's
  traffic, which is the same workload family as the bench prompts.
* MTP draft `--mtp ~/Strata-data/mtp/rt` (draft layer 837 MiB VRAM on the **main** card at this split; the
  startup note flags that the draft chain shares the last card with its window, the `STRATA_MTP_STAGE0=1`
  trade from `docs/MULTI_GPU.md`, kept as-is).
* Serve flags (verbatim line in `configs-and-startup/startup-k32.log.txt`):
  `--max-context 262144 --kv int8 --kv-resident 32768 --vram-reserve-mib 1000 --spec 4 --spec-min-p 0.7
  --pipeline-windows 2 --expert-cache auto --expert-profile-save ...`. Env: `STRATA_HC_Q8=1` (0.63 GiB Q8_0
  HC projections), `STRATA_STAGE_TRIM=1`, `STRATA_SPLIT_MTP_BATCH=1`; `STRATA_ROUTE_TAIL_SKIP` default ON
  (the startup spells out that answers differ slightly). KV streaming 32,768 of 262,144 cells per QSA layer
  in VRAM, 2.06 GiB pinned. At K=32 with everything loaded the free VRAM was **810 MiB (3080) / 699 MiB
  (TITAN)** and 9,724 experts were resident (39.6% of the 24,576 profiled pairs).
* Per-arm configs are byte-identical except the two `layer-split`/`layer_split` fields; `configs-and-startup/`
  ships K=20 and K=32, and arm K's file is `config-k20.json` with both fields set to K.

## Method

* **Custom streaming protocol** — not `strata bench run`, no standard suite, no fixed corpus; per
  `docs/COMMUNITY_BENCHMARKS.md` a custom protocol must say so loudly, and does not get to claim a custom
  median "beats" a standard one. This is an **A/B sweep** between same-binary arms, and the only claims that
  travel are *relative*, same-day, same-method ones.
* Workload: 3 Korean task prompts cycled 2:2:1 — arithmetic+parity (~60 tokens), 3-line summary of a
  ~1.5K-char pasted pseudo-document (~900 tokens), numbered extraction over a ~15.6K-char pasted pseudo-log
  (~8,700 tokens); `max_tokens` 200/350/250, temperature 0.3, thinking ON with reasoning deltas counted.
  Verbatim text (including a duplicated-word typo present in every measured request) in `prompts.md`.
* Timing, client-side from the SSE stream: **TTFT** = first non-empty delta (may be a reasoning token;
  keep-alives and empty deltas ignored); **decode tok/s** = counted tokens / (last-chunk − TTFT); counted
  tokens = non-empty content+reasoning chunks. Sequential within a phase; the "endurance" protocol adds
  600 s sequential → 90 s × 4 concurrent → 1 recovery request.
* **Engine-side cross-check of my counter (why absolute numbers carry slop):** the engine's own
  `prompt … N generated in M ms (X tok/s)` lines for the same window: the first endurance short request
  reads **120.3 vs my 119.0** (+1%), but a 8,632-token/250-generated request reads **80.4 vs my long-arm
  median 105.3** (−19%). Line-to-request attribution beyond the boot window was not preserved (my own
  gateway traffic interleaved), so this is an anchor, not a curve. Treat every decode figure here as
  ±10-15% in absolute terms on long prompts; cross-arm deltas share one method, one binary, one afternoon.
* **Arms:** `systemd` restart between arms (each arm starts with an empty conversation cache; the profile
  file survives), then a 240-second sequential mixed-size stream, arms K=20,24,26,28,30,32 back-to-back
  15:02-15:37 UTC with the *same* client script; K=20's "arm" reused the live K=20 boot. The 10-minute
  endurance baseline ran the same protocol at K=20 at 14:25 UTC, before the sweep, on the same binary.
* Cache state: every request carries a fresh RNG-seeded tag, so no prompt shares a prefix with its
  neighbour; within a run, nothing is reused. The shared agent gateway *does* reuse its own conversation
  prefix (visible in the traffic excerpt) — that is the deployment being used normally while measured.
* Telemetry: `nvidia-smi` sampled every 4-5 s per arm (power max/avg, temp max, util avg for the
  endurance), engine log lines for `expert cache hit rate …` and `layer split:` per arm; `lspci -vv` link
  states; free-VRAM lines from each boot.

## Results

### Table 1 — the K sweep (240 s arms, same binary, 15:02-15:37 UTC, zero errors everywhere)

| `--layer-split` K | 3080 layers | n req | decode med (range) tok/s | TTFT med s | expert hit med % | TITAN W avg (max) | TITAN °C max | 3080 W avg (max) | med tok/s by size (short/med/long) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 (engine auto) | 0-19 | 30 | **101.1** (86.0-115.2) | 2.41 | 97.8 | 176.2 (220.1) | 79 | 195.8 (270.4) | 104.2 / 100.5 / 94.0 |
| 24 | 0-23 | 31 | 103.7 (89.5-114.6) | 2.44 | 97.5 | 174.6 (216.2) | 80 | 213.5 (264.9) | 107.1 / 102.6 / 94.2 |
| 26 | 0-25 | 35 | 108.6 (95.3-118.6) | 2.41 | 97.0 | 189.3 (230.0) | 78 | 216.3 (269.8) | 110.5 / 109.4 / 101.2 |
| 28 | 0-27 | 49 | 110.7 (94.2-129.0) | 2.52 | 96.2 | 158.4 (235.4) | 76 | 208.9 (269.5) | 112.1 / 113.6 / 105.4 |
| 30 | 0-29 | 46 | 112.7 (92.3-131.0) | 2.62 | 95.5 | 152.4 (222.7) | 76 | 218.9 (275.5) | 116.7 / 111.0 / 102.9 |
| **32 (live)** | 0-31 | 48 | **115.7** (97.4-132.7) | 2.75 | 94.7 | 156.7 (218.0) | 76 | 229.0 (278.4) | 123.0 / 111.9 / 105.3 |

Arm-over-arm decode deltas vs the previous arm: +2.6%, +4.8%, +1.9%, +1.8%, +3.0%. The K=30→32 step's
hit median tripped my pre-registered 95% floor (94.7), which is why the escalation script stopped and the
K=32 verdict below rests on the endurance run instead of a 240-second arm.

### Table 2 — 10-minute endurance, the two ends of the range (same protocol, same binary)

| metric | K=20 (14:25 UTC) | K=32 (15:42 UTC) | delta |
| --- | --- | --- | --- |
| sequential requests / errors | 77 / 0 | 106 / 0 | — |
| decode median tok/s | 99.8 | **114.2** | **+14.5%** |
| last-20-request median (drift) | 98.8 (−1.0%) | 116.8 (**+2.3%** vs window median) | no decay; slight warm-up |
| TTFT median s | 2.37 | 2.75 | +0.38 |
| 4-way burst: requests / aggregate tok/s | 44 / 53.4 | 43 / 51.8 | −3% (queueing is serial either way) |
| post-burst recovery request tok/s (TTFT s) | 97.9 (1.1) | 110.6 (1.34) | +13% |
| expert hit median % (samples) | 97.8 | 95.2 (150) | −2.6 pts |
| TITAN draw avg/max W, °C max, util avg | 176.2/220.1, 79, 96%* | **164.7**/234.1, **76**, 49.7% | −11 W avg, −3 °C max |
| 3080 draw avg/max W, °C max, util avg | 195.8/270.4, 73, 81%* | **231.0**/277.7, 73, 81.1% | +35 W avg (cap 280) |

\*baseline util figures come from the earlier 5 s sampler and its split state; treat them as indicative —
the honest way to read K=20→32 util is "the TITAN does less real work over the window".

Verdict taken before publishing: **keep K=32 live.** Decision rule (pre-registered): sequential median
≥ 108 (that is ≥ −1% of the K=30 arm's 112.7 with margin), hit median ≥ 93 over the *full* window, zero
errors, recovery ≥ 90 — the run passed on all four (114.2 / 95.2 / 0 / 110.6), so the arm-level 94.7 that
stopped the escalation was a short-window artifact of the same window's own 95.2 median.

### What this buys

+14.5% decode sustained, −11 W and −3 °C off the expensive card, at a cost of +0.38 s median TTFT, −2.6
points of expert hit, and +35 W onto the cheap card (which now averages 231 W against a 280 W cap). On this
pair the 3080's *headroom is the free lunch*: every two layers handed from a 210 W-capped TITAN to a
3080 that was averaging under 200 W bought roughly +2-5% decode. The wall came into view at K=32 from three
directions at once: the hit rate's steady slide (−0.7 pt/arm, and the 3080's 1,887-slot cache at K=32 is a
quarter of the size of the K=20 one, so the slide steepens), the 3080 at ~82% of its cap averaged, and the
gains flattening. K≥34 stays unprobed on purpose: on this workload the next step looked like +0-3% against
a cache that keeps shrinking, and it would leave the 3080 ~10-20 W from a hard cap.

### Real-traffic context (not a benchmark — engine-log lines, K=32 window)

The shared agent traffic inside the endurance window, as the engine logged it (`engine-log-traffic-excerpt.txt`):
a fresh 138,986-token gateway prompt read 79.4 s at **1,749.6 tok/s** and decoded 544 tokens at 80.3; a
140,475-token turn reused 139,529 and read only 946 (3.6 s) before generating 1,373 at 96.7 tok/s; one
cancelled 138,508-token request kept 133,284 tokens of window live; sampled bench-shape lines decode at
80-102 tok/s. Long-context serving on this split stays usable — no "prompt too long", no arena eviction —
but note the direction: these are agent turns with ~99% prefix reuse, and my sweep measures the *cold* side.

## Correctness and limits

* **No quality suite ran today.** No needle, no `strata bench run`, no answer-correctness scoring. The only
  quality evidence is that this deployment answers this agent's real work (code edits, file writes,
  Korean prose) without complaint at K=32 — operator self-report, **not** a benchmark, and it does not
  establish K=32 is as correct as K=20. If you care about answers, not speed, re-run with a quality suite
  across the K's you compare; that is exactly what `docs/COMMUNITY_BENCHMARKS.md` demands and today's run
  does not meet it.
* **The sweep is decode-centric.** Prompt-read throughput through the split was not measured per arm — the
  closest are TTFT medians (2.41→2.75 s at 900-token prompts, 7.8 s at 8.7K, baseline-window long TTFT
  7.8 s) and the one logged 8.6K-prompt line. On this workload the sweep direction was uniform, but
  `docs/MULTI_GPU.md`'s measured 4060 Ti + 3060 pair shows the prompt side can move ±30% while decode
  moves ±2-14%; a heavy-document workload on this box could prefer a *different* K than mine. The in-tree
  split search is the tool built for this; ping it.
* **Profile bias, stated plainly:** the profile cache is shaped by this agent's own traffic, which is the
  same task family as the bench (Korean prose, arithmetic, log extraction, ~1-9K prompts). A workload far
  from that distribution will see a lower hit rate at every K, and the K-slope of that penalty is
  unmeasured here. The profile also drifts day to day — cross-day comparisons of these numbers are void.
* **Calibration:** chunk-counted tok/s; engine-line anchors show +1% (short) to −19% (long) gaps against
  my counter; all absolutes inherit that, deltas do not (same ruler end to end).
* **Single-shot arms** (one 240 s run per K, no repeats), ~30-49 requests each, background agent traffic
  unequal by luck between arms, ~1-2% of its own; temperature 20-40 °C GPU, cap-bound so mostly settled;
  turbo state must be reproduced (BIOS ships it off); caps are operator mandates, not defaults; the
  endurance util columns mix samplers (flagged in the table note); the medium/long prompts carry their
  verbatim typo; `sm_86` cards run a build whose declared arch is `[75]` (extra cubins compile fine — worth
  knowing if you drive the same build system).
* **Reproduce:** scripts (sanitized, password and paths placeholdered) + configs + prompts + consolidated
  JSON + per-request JSONLs all ship here; expect the same shape, not the same digits.

## Files in this folder

* `README.md` — this report.
* `results-consolidated.json` — every run, arm and config in one machine-readable file (machine, build,
  protocol, tables above, anchors).
* `per-request/` — raw client JSONL: `baseline-endurance-k20.jsonl` (122 rows incl. burst+tail),
  `arm20/24/26/28/30/32.jsonl`, `endurance-k32.jsonl`.
* `sweep-results.json`, `sweep-escalation-results.json`, `endurance-k32-results.json` — the scripts' own
  verdict files (arms, hit medians, per-GPU power/temp roll-ups, decision lines verbatim).
* `configs-and-startup/` — `config-k20.json`, `config-k32.json`, `startup-k32.log.txt` (the full boot
  block of the K=32 run: probes, sizing, cache lines, serve flags verbatim), `iq_pack-conversions.json`,
  `iq_pack-native_experts.txt`.
* `scripts/` — sanitized harness: `bench-10min.py`, `sweep-3arm.py`, `sweep-escalation.py`,
  `sweep-endurance-k32.py` (sudo password, hostname and home paths placeholdered; nothing else removed).
* `prompts.md` — the three task prompts verbatim, with the typo disclosure and translations.
* `engine-log-traffic-excerpt.txt` — long-context gateway lines + sampled engine `generated in` lines from
  the K=32 window (the Method cross-check).
* `sha256.txt` — SHA-256 of the model shards, mmproj, MTP draft files, profile and dense pack.

## Suggested docs updates

* `docs/MULTI_GPU.md`, §"Unequal pairs" (the `layer_split 0` / TITAN-only failure line): add a *measured*
  same-pair-shape datapoint — a 12 GB sm_86 helper + 24 GB sm_75 main on a 15-worker PCIe 3.0 host, sweep
  up from the auto K, bought +14.5% sustained decode to the auto-K+12 with hit −2.6 pts and the helper card
  climbing to ~82% of its power cap; and note the sweep direction was decode-only measured, so treat it as
  "one workload moved the optimum to auto-K+12 on this shape", not a new default.
* §"Servers, towers and workstations with many cores": this is the predicted case arriving — the Z840's
  15 AVX-2 workers plus PLE direct reads (101k rows/s) on NVMe made raising K *faster* instead of
  register-hungry; a line there saying "with a server CPU and NVMe, sweeping the split upward can be a real
  win, measured +14.5% on a 210W-capped TITAN + 3080 pair" would have saved an hour of wondering.
* §"How to read a report that disagrees with your hardware": the ~0-1% to ~19% engine-line vs client-chunk
  calibration band is a concrete instance of "measure with the ruler you ship".

## Follow-up, same day: standard-v1 cross-check and the same-box single-card baseline

After the sweep finished, the K=32 live config was re-measured with the official protocol — `strata bench run --url http://127.0.0.1:8000 --suite standard-v1` (frozen corpus `e75eff8b9feea04f...`) — with the concurrent bulk model **download paused** for a clean measurement (see the limitation below). The runner's own report, results.json and per-request outputs are committed as [`2026-10-23-community-titan-3080-standard-v1/`](../2026-10-23-community-titan-3080-standard-v1/README.md).

| Case (median) | TITAN RTX 24 GB alone, 2026-10-20 (engine 0.1.46.1, 250 W cap) | TITAN RTX + RTX 3080 layer split K=32, 2026-10-23 (engine 0.1.48, 210+280 W) | Delta |
| --- | --- | --- | --- |
| short decode tok/s | 62.6 | **94.1** | +50% |
| 4K decode tok/s | 60.0 | **88.2** | +47% |
| 16K decode tok/s | 54.6 | **82.3** | +51% |
| 4K prompt tok/s | 900.6 | 846.0 | -6% |
| 16K prompt tok/s | 1,270.3 | **1,507.3** | +19% |
| TTFT s (short/4K/16K) | 1.5 / 4.6 / 12.9 | 1.3 / 4.9 / 10.9 | improved |

The single-card column is the same machine's own official-suite run six days earlier (new engine defaults, TITAN only, before the 3080 was fitted), so the before/after isolates "add the 3080 and split K=32" from machine differences — but it does **not** isolate it from the engine bump (0.1.46.1 -> 0.1.48) nor from the TITAN power-cap change (250 -> 210 W): the +47-51% decode gain and the -6% 4K-prompt cell carry both confounds and should be read as "this box, before vs after", not as the layer split alone.

Sanity: 11/12 (I2 JSON-value fail, engine 0.1.48). Concurrency (capacity 1): 74.4 / 71.1 / 69.7 tok/s aggregate across three four-request rounds.

Measurement hazard found the hard way: a 1.7 MB/s bulk download writing its 97 GB file through the page cache evicted the expert pool from RAM and crushed decode run-over-run 87.9 -> 36.5 -> 9.2 -> 1.9 tok/s while prompt rates stayed flat within 1%. On a box that streams experts from the page cache, **any bulk file write must be paused during measurement** — and not just the process: also the flock watchdog that restarts it (killing it without muting its cron is not enough).
