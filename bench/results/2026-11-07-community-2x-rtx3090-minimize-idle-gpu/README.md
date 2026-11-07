# Filling the idle half of a layer split — community report, 2× RTX 3090 (l28, Linux)

`--pipeline-windows 2` on NVIDIA does fill the second card, and on this box that is worth
**+5…+11 % decode** depending on the task. The 2026-10-20 idle-GPU report (2× RTX A4000) measured the
same switch as noise; the difference is not the driver, it is what the two boxes put into a window.

> **Same day, second pass.** After this report was written the box was re-tuned — power limit back to the
> 350 W default on both cards, memory clock +1000 MHz on both, cooling improved, node rebooted. A renewed
> sweep then ran **autonomously over ~2 h** on `qwen3.8-flash-next-unsloth-ud-iq4_xs` at this deployment's
> full **786 432-token (≈770 k) context**: 746 requests, 16 sweeps, 26 server restarts, and a 1 Hz load
> capture of GPU / PCIe / RAID / CPU running 123 minutes without a gap. It confirmed the mechanism (§ 1.1,
> § 3), moved two numbers that turned out to be artefacts of the 300 W limit (§ 1.1, § 4), and re-tested the
> parameter ranges listed in § 6. Everything from that pass is marked **`[350 W]`**.

**Machine.** l28: **2× RTX 3090 24 GB** (`nvidia-smi` names both the same, 82 SMs @ 1.70 GHz each, one
PCIe 4.0 x16 link apiece, driver 615.78.08), AMD EPYC 7402 (24c/48t), 125 GiB RAM, CachyOS, kernel
7.3.0-2-cachyos-bore. Engine **0.1.55** (`src a22f54c6e3e07cd9`, built on the box, CUDA 13.4.92,
gcc 16.2.1, `-DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES=86`), `serve/server.py`, `parallel: 1`.

**Model and config.** Qwen3.8-Flash-Next Unsloth **UD-IQ4_XS** (`--native` GGUF + pack, 55.43 GiB of
experts in RAM, MTP draft `--mtp mtp/rt`), started as `--expert-cache auto --prefill auto --spec 4
--max-context 786432 --rope-scaling yarn --rope-scale 3 --kv int8 --ple-io ram --kv-resident 32768
--vision --vram-reserve-mib 1280 --trim-stage-weights --pcie-frac 0.08 --spec-min-p 0.65
--conversation-cache-mib 32768 --conversation-cache-slots 16 --pipeline-windows 2 --layer-split auto`,
env `STRATA_PIPELINE_DEBUG=1 STRATA_PIPELINE_SWITCH=pl_switch.txt` (= `pw=2 theta=0.35`). The auto cut
lands at K=23: layers 0-22 on CUDA0, 23-47 + output head + draft chain on CUDA1.

**Method.** Both arms on **one** running server: `STRATA_PIPELINE_SWITCH=<file>` is re-read per
request, so `pw` switches between two requests with the same expert placement and KV state - no
restart per arm. Round-robin, greedy, same prompt in every arm of a round, paired per round, `pw=0`
as baseline. `STRATA_DECODE_TIMING` stays off: it costs the serial loop 7.7-8.1 % and would inflate
the difference by ~8 pp. Utilisation is `nvidia-smi -lms 100` on the serving host, sliced to the
**decode window only** with the millisecond counts the engine reports for that request, so prefill
and queue time are not averaged in.

## 1 · The idle half is real, and the switch fills it

Mean per-card utilisation during decode, 100 ms samples, 2 684 rows, 2026-11-07 morning,
**at the 300 W limit** then in place (this folder):

| arm | task | n | CUDA0 | CUDA1 | **sum of both** | pair power |
|---|---|---:|---:|---:|---:|---:|
| `pw=0` (serial) | prose | 6 | 44 % | 60 % | **104 %** | 520 W |
| `pw=0` (serial) | code | 7 | 46 % | 59 % | **105 %** | 531 W |
| `pw=2 theta=0.35` | prose | 5 | 61 % | 83 % | **144 %** | 525 W |
| `pw=2 theta=0.35` | code | 7 | 60 % | 85 % | **145 %** | 537 W |

The serial columns add up to ~100 % of one card: the cards take turns, which is what a serial layer
split is. The switch moves that to ~144 % — and, **at that 300 W cap**, for only **+6…+12 W**; the same
fill measured above the cap costs 50…64 W (§ 1.1), so read that figure as "the cap was already binding",
not as "the second card is free". Coarser samplers see the same thing: 1 s
samples on this box before the switch existed gave 40 / 53 % (2026-11-01), and on 2× RTX A4000 the
same switch took 45 / 58 % to 76 / 75 % (`2026-10-20-0.2.1-idle-gpu`). Filling the idle half is not
the hard part - whether the filled time is useful is.

## 1.1 `[350 W]` Replicated with a 1 Hz sampler — and the watt figure corrected

The second pass sampled the same box continuously at 1 Hz on the serving host (80 channels: per-card
utilisation, VRAM, power, SM and memory clock, temperature, fan, PCIe link and `rxpci`/`txpci`, plus the
RAID0 that holds the model, its four members, CPU and RAM) while 746 requests ran. Over 123 minutes and
26 server restarts: **7021 samples, median step 1.00 s, no gap over 8 s, no sample without an answer.**

| arm | sum of both cards | pair power | tokens/s per kW |
|---|---:|---:|---:|
| `pw=0` serial | **101…105 %** | 586…603 W | 230…244 |
| `pw=2 theta=0.35` | **142…151 %** | 636…650 W | 243…264 |
| `pw=2 theta=0.05` / `0.70` | 152…157 % / 129…131 % | 648…656 / 618…621 W | 251 / **221** |

Two things follow, and both correct § 1:

* **The fill is not a 6 W affair.** Above 300 W per card it costs **+50…+64 W** for the pair (586…603 →
  636…650 W). The "+6…+12 W" in § 1 was an artefact of the old limit: with the cards already pinned at
  their cap, extra work arrived as clock they were not allowed to use. What survives the power change is the
  **per-watt** statement — 243…264 against 230…244 tokens/s per kW, i.e. the filled windows are still the
  more efficient way to run the pair, just not for free.
* **The gate has an optimum, and `theta=0.70` is past it**: 137.4 tok/s against 138.4 serial at *higher*
  power, i.e. worse in rate and in watts (221 against 230 tokens/s per kW). `pw=1` is the efficiency
  outlier — filling only the prompt path buys +3.0 % code for ~4 W.

Memory clock under load sat at **10001 MHz** (max 10251 — the +1000 offset is doing what it says), SM at
1890…1935 MHz, **65…68 °C, fan below 70 %, and not one throttle bit in 746 requests**. At decode this box
was never at its power limit; see § 5.1 for the workload that is.

## 2 · What the fill is worth

| Setting (same box, same server, paired) | `pw=2` against `pw=0` | n |
|---|---|---|
| **live, code** (this folder, 4K context, 400 tokens) | **+5.7 %** (107.6 → 114.0 tok/s) | 4/5 rounds ahead |
| **live, prose** (this folder, 2K context, 300 tokens) | −0.5 % (120.0 → 120.2) | 1/3, n too small |
| 8 documents, production config, code / prose | +8.0 % / **+10.6 % [+5.8, +15.6]** | 8 pairs, prose 8/8 |
| production config without debug timing, code / prose | +5.0 % / +6.6 % | 8 + 6 rounds |
| reasoning switched on (what this box really serves) | **+23.7 % code**, +7.4 % prose, −0.3 % chat | 2 rounds |
| 120 k-token context | +1.3 % code, +8.1 % prose | 2 rounds |
| first measurement of the switch, 4K context | +4.7 % story, +2.5 % code | 6 rounds |
| null control `pw=1` (pipelines only the prompt path) | +1.5 / −0.5 / +1.0 % | 11 rounds |
| `force_miss=1` (build every window, hit none) | **−9.9 % prose, −5.4 % chat** | 3 rounds |

The live pass shared the box with a live agent session: 15 of its 40 requests were dropped because a
foreign request landed in the window, which is why those two rows are thinner than the rest.

**`[350 W]` the same rows, larger n.** The second pass re-ran the pairing 8 rounds deep and pooled two
identical prints of the same config (n = 16 per arm and task, `pw=2` against `pw=0` on the same document in
the same round): **+11.7 % code (11/16 rounds ahead) and +10.4 % prose (14/16)**, with `theta=0.20` the best
single arm at +14.8 % / +11.8 %. With reasoning switched on — the mode this box actually serves — code went
to **+32.4 % [+18.4, +46.3]**, and a 120 760-token prompt to **+8.0 % code / +10.0 % prose** (3/3 rounds).

That pass also put a number on how much a row of this table is worth. Two prints of the **same config**,
same protocol, 100 minutes apart, differ by **sd 11.5 pp**, and 47 of 96 paired documents move more than
5 pp. So: single-digit percentages here are direction, not evidence; the rows above n = 8 are the ones that
carry, which is why § 2's live rows are read together with the pooled ones and not against them. (Prefill is
the exception — no draft lottery in it: 11 k and 131 k fresh prompts reproduced to **0.4 %**.)

## 3 · Why +5…+11 % here and noise on 2× A4000

The engine says it per request: a window that arrives prebuilt costs 13.4 ms against 20.8 ms for a
fresh one, so **34 % of a window is coverable and no parameter makes that bigger** - `theta`,
`spec_min_p`, the task and the context length all leave the ceiling where it is. Over 215 paired
requests the gain is linear in what was actually covered:

    gain % = 0.88 · (share of windows that arrived prebuilt × 34 %) − 3.8 %

Break-even is 13 % of windows prebuilt; the ceiling of two windows is ≈ +26 %. What decides is the
draft in front of the window: this box runs at 68-75 % accepted drafts and 2.5-2.7 tokens per window,
so the gate lets 38-82 % of windows be built ahead and most of them land. On the A4000 the fill was
the same and the gain was not (+5.4 % code, −1.0 % prose): a window that rolls back costs 5-10 %, and
a miss-prone draft chain spends the covered time on rebuilds.

**`[350 W]` the ceiling did not move, which is the interesting part.** Measured again with
`STRATA_DECODE_TIMING=1` on the re-tuned box: a window that came in on the path cost **12.19 ms** against
**18.39 ms** for a fresh one — **33.7 % coverable**, against the 35.6 % above and the 34 % of the 215-request
series. The second gate tested, `theta=0.20`, gives 33.8 %. So 50 W more per card, +1000 MHz of memory
clock and a better-cooled box change the rate under the ceiling, not the ceiling itself: two windows on two
cards can hide about a third of a window, full stop. What did grow is the share of windows that arrive
prebuilt and land (72.2 % built ahead, 64.2 % of those landing, against ~31 % hitting in the 300 W era),
which is exactly the `s` of the formula — the reason the paired gain roughly doubled while `c` stood still.

## 4 · What it costs, and what it does not do

* **Rent:** the startup says `--pipeline-windows 2: 272 MiB of CUDA0 kept out of the expert cache`
  (second verifier + GDN snapshots) = **118 expert slots, −1.9 %**; the `force_miss` row above is what
  that rent costs when nothing hits.
* **Not bit-identical** to the serial path ("up to CPU rounding"). For token-SHA regression runs put
  `pw=0` in the switch file - no restart needed.
* **Two windows is the maximum:** `--pipeline-windows 5` is silently clamped to 2, and `--batch` /
  `parallel: N` turns the pipeline off - parallel chats and parallel windows want the same cards.
  `STRATA_MTP_STAGE0=1` (draft chain onto CUDA0) lifts the ceiling to 46 % and the absolute rate not
  at all (153.3 against 154.0 tok/s) while CUDA0 loses 474 slots.
* **Long context was the weak spot — less so at 350 W.** The 05.11 pass on this box gave +1.3 % code /
  +8.1 % prose at 120 k and the 300 W reading above concluded that windows get dearer the longer the
  context is. Re-measured `[350 W]` on real 120 760-token and 59 719-token documents, three rounds, paired
  per document: **+8.0 % code / +10.0 % prose, 3/3 rounds ahead of serial** — close to the short-context
  numbers rather than below them. What is left of the caveat: acceptance does drop with context, and with
  three rounds this row is the softest in the report.
* **`--spec` and `--spec-min-p` are coupled, and the coupling is the finding.** On the same server, per
  request: `spec_min_p` 0.65 against 0.50 costs **−9.7 %** in the serial loop and only **−3.5 %** with the
  windows on; 0.95 is a cliff on its own (114.5 tok/s code serial, −20 % against its own baseline) while
  being the best single code arm with windows. A stricter draft floor buys shorter, safer windows — worth
  having only if something builds the next one while you wait.

## 5 · Production cross-check

The switch has been on this box since 2026-11-03 and the log agrees: 2 329 requests on the current
config at **128.1 tok/s mean decode (median 131.2)**, draft acceptance 74.7 %, expert cache 92.3 %.
The era before it (12.10-03.11) ran 105.0 tok/s - config against config, not an A/B, which is why the
paired rows above carry the argument.

**`[350 W]` a second window, same config, 06.11–07.11:** 2 417 requests, 1.30 M tokens generated,
**mean 138.1 / median 132.9 / p95 172.1 tok/s** end to end (reasoning included), expert cache 91.1 % hit,
PCIe spill p95 0.300 % of routed experts, largest prompt served 671 420 tokens, 10 starts and 6 idle
unloads in 36 h. Two independent production windows on the same configuration now bracket the claim: the
switch is not a bench artefact.

## 5.1 `[350 W]` Where the power limit actually sits: not in decode

The load capture was taken to answer a different question — what does the box burn while it does this — and
it turned up one thing this report had no room for. Across everything measured at decode, **no card ever
reached its limit**: 230…330 W per card, SM 1890…1935 MHz, 65…68 °C, zero throttle events. Raising the cap
from 300 to 350 W changed nothing in the rate either (serial 144.5 → 141.5 tok/s, pooled n = 16).

The one workload that does press against it is a **large fresh prompt**: reading a 131 657-token document
drives both cards to 97.5…98 % (sum **195.5 %**) at **328…333 W each — 94 % of the 350 W cap — and 657…665 W
for the pair**, with the SM clock back from 1935 to **1582…1665 MHz**, **77 °C** at the hot card, ~15 GB/s of
PCIe per card, while the host engine runs on a **single core** and the RAID0 that stores the model reads
**nothing** (the 55.43 GiB of experts live in RAM once loaded; the disk works only during the 13…16 s model
load, at 3.5…3.9 GB/s spread evenly over its four strips). That clock back-off at 94 % of the cap is what a
power-limited workload looks like here, and it appears only on this path. Prefill throughput itself is where
the wattage shows: **4718…4738 tok/s** on that prompt, reproducing to 0.4 %.

So: on a two-card layer split the extra 50 W are not a decode story, they are a **prompt-reading** story —
and the decode gain of the windows is a scheduling gain, which is why it survives (and grows) without ever
touching the limit.

## 6 · `[350 W]` What was re-tested, and with which ranges

One 2 h autonomous pass over the same deployment: 746 requests in 16 sweeps, 26 server restarts (the pod was
allowed to go offline as often as it took), prompts from real documents — the coding agent's own TypeScript,
the Linux-history articles, chat tasks — and a decision rule written down before it ran: **a value replaces
the running one only if it wins the paired comparison (same document, same round) by more than 2 pp in both
task kinds and in at least 70 % of the pairs.** No knob below cleared it, so this is a list of ranges tried,
not a results matrix:

* **`--layer-split`** `auto` (= K 23), 22, 23, 24, 25, 26 — two passes, 4 then 8 rounds, plus fresh 16 k
  prompts for prefill. No lever. The control pair `23` against `auto` (the same configuration on this box)
  printed 0.0 %, and prefill sat at 2579…2674 tok/s across all six (±1.8 %).
* **`pw`** 0, 1, 2 and **`theta`** 0.05, 0.20, 0.35, 0.50, 0.70 — per request, on one server. `theta` is flat
  from 0.05 to 0.50 (≤1.2 pp apart); 0.70 drops under serial in rate *and* in watts. `pw=1` (prompt path
  only) is nearly free; `pw=2` is the win.
* **`--spec-min-p`** 0.35, 0.50, 0.65, 0.80, 0.95 — per request via `strata_tune`. Flat 0.50…0.95 (±4 %),
  0.35 too cheap, 0.95 a cliff in the serial loop; the interesting part is the coupling to `pw` (§ 4).
* **`--spec`** 2, 4, 6; **`STRATA_MTP_STAGE0`** off/1; **`--vram-reserve-mib`** 1024, 1280, 1536 — all behind
  a restart, all draws. `spec 2` bought the same rate with ~50 W more, `MTP_STAGE0` cost −8.5 % code, and the
  reserve moved ±76 expert slots per 256 MiB without the rate noticing.
* **Context length** 2 k, 11 k, 16 k, 60 k, 120 k, 131 k and **mode** (code / prose / chat, reasoning on and
  off) — the axis where the numbers moved most: reasoning + code is where the windows pay (§ 2), large fresh
  prompts are the only place the box meets its power limit (§ 5.1).

Two caveats worth more than any of the rows: absolute tok/s on this box resolve to about **±10 %** (§ 2), so
only the paired figures above are arguments; and `--pipeline-windows 3` is not a thing to test — the engine
clamps it to 2 (`generate.cpp:3258`), as § 4 already said.

## Reproduce and raw

```sh
# arms switch per request; kubectl drives the pod, ssh <host> runs the sampler.
# strata_pw_idle_ab.py imports strata_paths.py and strata_pipeline_ab.py (Strata-Tools/).
KUBECONFIG=~/.kube/config python3 -u strata_pw_idle_ab.py \
    --url http://g27:8081 --pod-match strata-l28 --host l28 \
    --arms 'pw=0,pw=2 theta=0.35' --runs 6 --context 2000 --tokens 300 --json raw.json
```

Raw: `pw_idle_l28.json` (every round: rate, per-card utilisation, power, window counts),
`pw_idle_l28_summary.txt` (both passes as the tool printed them), harness `strata_pw_idle_ab.py`.
The sweeps behind § 2 and § 3: `Measurements/strata/2026-11-01-l28-2x3090-split-sweep`,
`2026-11-03-l28-pipeline-windows`, `2026-11-05-l28-pipeline-acceptance` (AI-Umbrella).

**`[350 W]` second pass.** Its raw material is in the AI-Umbrella folder
`Measurements/strata/2026-11-07-l28-350w-sweep/`: `results/*.jsonl` (one record per request: rate, reused and
fresh prompt tokens, acceptance, window counts, node-stamped start/end), `results/*-resources.csv` (the same
records joined to the 1 Hz load capture), `machine-logs/startup-*.txt` (the engine's own argv per restart —
the proof of which configuration ran), `results/endtable.txt`, and `PLAN.md` (the sweep order, including the
two phases deliberately dropped). The tools, all in `Strata-Tools/`:

```sh
# 1 Hz load capture on the serving node (survives pod restarts and dropped ssh)
python3 Strata-Tools/strata_l28_monitor.py start --tag 350w --interval 1
# one server start per configuration, every request-level arm inside it
python3 -u Strata-Tools/strata_l28_sweep.py --groups 'split=auto; split=24' \
    --arms 'pw=0; pw=2 theta=0.35 smp=0.65' --kinds code,prose --runs 8 --jsonl out.jsonl
# what the box was doing while each arm ran
python3 Strata-Tools/strata_l28_resource_summary.py --csv l28-350w.csv --jsonl out.jsonl --md
```

