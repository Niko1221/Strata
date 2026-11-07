# Filling the idle half of a layer split — community report, 2× RTX 3090 (l28, Linux)

`--pipeline-windows 2` on NVIDIA does fill the second card, and on this box that is worth
**+5…+11 % decode** depending on the task. The 2026-10-20 idle-GPU report (2× RTX A4000) measured the
same switch as noise; the difference is not the driver, it is what the two boxes put into a window.

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

Mean per-card utilisation during decode, 100 ms samples, 2 684 rows, 2026-11-07 (this folder):

| arm | task | n | CUDA0 | CUDA1 | **sum of both** | pair power |
|---|---|---:|---:|---:|---:|---:|
| `pw=0` (serial) | prose | 6 | 44 % | 60 % | **104 %** | 520 W |
| `pw=0` (serial) | code | 7 | 46 % | 59 % | **105 %** | 531 W |
| `pw=2 theta=0.35` | prose | 5 | 61 % | 83 % | **144 %** | 525 W |
| `pw=2 theta=0.35` | code | 7 | 60 % | 85 % | **145 %** | 537 W |

The serial columns add up to ~100 % of one card: the cards take turns, which is what a serial layer
split is. The switch moves that to ~144 % for **+6…+12 W**. Coarser samplers see the same thing: 1 s
samples on this box before the switch existed gave 40 / 53 % (2026-11-01), and on 2× RTX A4000 the
same switch took 45 / 58 % to 76 / 75 % (`2026-10-20-0.2.1-idle-gpu`). Filling the idle half is not
the hard part - whether the filled time is useful is.

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
* **Long context is the weak spot:** windows get dearer and acceptance drops, so the gain slides
  toward the prose numbers.

## 5 · Production cross-check

The switch has been on this box since 2026-11-03 and the log agrees: 2 329 requests on the current
config at **128.1 tok/s mean decode (median 131.2)**, draft acceptance 74.7 %, expert cache 92.3 %.
The era before it (12.10-03.11) ran 105.0 tok/s - config against config, not an A/B, which is why the
paired rows above carry the argument.

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
