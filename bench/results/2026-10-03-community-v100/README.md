# 2026-10-03: Tesla V100-PCIE-32GB, dual Xeon E5-2696 v3, 128 GB RAM (Windows)

Field measurements from a daily-use install and a capability battery - not a fixed-cap
benchmark harness. Each number is a single run unless stated; the `--calibrate` sweep
is the engine's own repeated measurement. Reported per the guide in
[COMMUNITY_BENCHMARKS.md](../../../docs/COMMUNITY_BENCHMARKS.md).

The sections below describe the **v0.1.38 source build** as first submitted; an
**update for the v0.1.39 ready-made CUDA 12 engine** (verified on this sm_70 card and
re-calibrated, decode roughly 2x) is at the end of this file.

## Hardware

- Tesla V100-PCIE-32GB (sm_70, TCC mode), solo, PCIe **Gen3** x16
- dual Xeon E5-2696 v3 (36C/72T, two sockets, NUMA)
- 128 GB DDR3L-1600 ECC REG (8x16 GB, Samsung M393B2G70QH0-YK0) - the DDR3 variant of this
  X99-generation board; first posted as DDR4, corrected after checking the module part numbers
- Windows 10 Enterprise LTSC (19044)

## Software

- Strata **v0.1.38**, engine built **from source** for sm_70 (`STRATA_EXPERIMENTAL_SM60=1`),
  CUDA Toolkit 12.4, MSVC 14.44; driver 581.80
- Build note: linking needed `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared` on MSVC, see #585

## Model

- Qwen3.8-Flash-Next **IQ3_XXS** (ISTA-DASLab GSQ-RCO, pinned revision ed59f920, verified by setup)
- Context **131,072**; KV int8 streamed to RAM (`--kv-resident 32768`); MTP draft (q2_0)
- 15,553 / 24,576 experts (63%) cached in VRAM (25.2 GiB); expert cache hit 95-97% measured

## Settings

Before / after `--calibrate` (kept: `--pcie-frac 0.55 --spec-min-p 0.70 --pool-workers 18`):

| Setting | Sweep (tok/s) |
|---|---|
| pcie-frac | 0.00: 17.8 / 0.20: 16.2 / 0.34: 20.0 / 0.35: 19.2 / **0.55: 25.4** / 0.75: 21.5 |
| spec-min-p | 0.30: 19.2 / 0.50: 18.3 / **0.70: 21.1** |
| pool-workers | 35: below 28.6 / 23: 28.6 / **18: 33.5** |

Takeaway: on this dual-socket NUMA box 18 CPU workers beat 35 by 20%+; defaults tuned on a
6-core Ryzen were ~2x off. **Calibration doubled decode: 16.5 -> 33.5 tok/s.**

## Decode speed (real requests, thinking on, temperature 0.6)

| Workload | Output | tok/s (pre-calibration) |
|---|---|---|
| First request after engine start (JIT warm-up) | 175 | 7.5 |
| Short answer | 39 | 10.1 |
| Medium answer | 289 | 13.7 |
| ~400-word essay (thinking + prose) | 322 | 14.2 |
| Long-form writing | 1,492 | 15.8 |
| Long multi-method math reasoning | 1,761 | 16.8 |
| 31k-token interactive session (web UI) | 31,796 | 16.7 steady |

After calibration, a 196-token prose request measured 22.0 tok/s (same class of workload
as the 39/289-token rows above, which were 10.1-13.7 before). Speed is flat through a
31k-token session (KV streaming).

## Prompt processing (scaling with document size)

| Prompt | tok/s |
|---|---|
| ~90 tokens | 57 |
| ~280 tokens | 133 |
| 4,340 tokens (synthetic log document) | **863** (5.03 s) |

For comparison, Qwen3.8-27B (UD-Q4_K_M, llama.cpp b10545, thinking build, same box) read
the same 4,340-token document at 629.5 tok/s.

## MTP

- Acceptance 97.8% on repetitive output, ~65% on free prose (measured over the battery above)

## Notes

- Decode is memory/PCIe-bound on this card: GPU util ~19%, 42-48 W of the 250 W limit, 49 C.
- Host memory bandwidth is the main decode lever left on this box after calibration: DDR3L-1600
  has no headroom, and same-CPU-generation dual-socket DDR4-2133 boxes report roughly 1.8x our
  decode on comparable MoE traffic (community anecdote, unverified).
- Full capability comparison against Qwen3.8-27B (llama.cpp) on the same machine, and the
  thinking-budget pitfall (`reasoning_budget_tokens`), are described in issue #617.

## Update (2026-10-04): v0.1.39, the ready-made CUDA 12 engine

v0.1.39 ships an experimental ready-made engine for Pascal/Volta
(`strata-windows-x64-cuda12.zip`, sm_60/61/70 + sm_75-89 + PTX, CUDA 12.9). Installed over
this setup with `--cuda 12`: the model pack, MTP draft and settings carried over. It runs
correctly on this real sm_70 card (the release notes check the build only - no Pascal/Volta
card on the test PC). Driver 581.80.

Re-calibrated for 0.1.39 (0.1.38 values in parentheses): `--pcie-frac 0.35` (was 0.55),
`--spec-min-p 0.70` (unchanged). `--pool-workers` is no longer set: 35 / 23 / 18 workers
measured within 2% of each other (76.9 / 77.2 / 78.3 tok/s in the engine's own bench),
where on 0.1.38 eighteen workers beat 35 by 20%+. Full sweep: PCIe share 0.00: 73.4 /
0.20: 72.0 / 0.34: 75.8 / 0.35: 76.2; draft floor 0.30: 71.5 / 0.50: 73.5 / 0.70: 79.2.
The calibrate bench itself went 33.5 -> 76.9 tok/s (**2.30x**) between the two engines.

Same-prompt API measurements (timings reported by the server, single runs):

| Workload | 0.1.38 (source, calibrated) | 0.1.39 (ready-made, calibrated) |
|---|---|---|
| Short prose, ~150 tok out | 22.0 tok/s | 54.6 tok/s |
| Long-form, 2.2-3.4k tok out | 33.5 tok/s steady | 64.2 tok/s |
| 4,340-tok synthetic log doc, prompt processing | 863 tok/s | 1,127 tok/s |
| Same doc, decode, ~800 tok out | 11.9 tok/s (pre-calibration) | 72.3 tok/s |

The decode gains far exceed the RTX 5070 medians in the 0.1.39 notes (+2.5-6%): #646's
savings are host-side launches and round trips, which weigh far more on an older
dual-socket box (2x E5-2696 v3, DDR3L-1600) than on a modern desktop. Consistent with
the GPU-side evidence above - decode here is host-bound, not compute-bound. The
0.1.38-time `--pool-workers` sensitivity disappearing fits the same story: less CPU work
per token, fewer threads needed to feed the GPU.

## Update (2026-10-04): `--parallel 4` concurrent decode

Four simultaneous ~700-token requests against the calibrated XXS server (`"parallel": 4`
in the config; a single request still takes the solo path):

| Mode | Decode tok/s |
|---|---|
| Solo (one request) | 59.1 |
| Per-request, 4 concurrent | 22.2 / 23.9 / 23.4 (48.3 for the early finisher) |
| Group throughput | **84.4 = 1.43x solo** |

All four answers landed within 31.9 s of wall time (serial worst case ~50 s). Cost:
2.9 GiB of expert cache (13,715 experts resident) for the extra slots. Worth it for
multi-agent or multi-user bursts; pointless for a single user, and it was removed
again on this box afterwards.

## Update (2026-10-05): images on the ready-made CUDA 12 engine (sm_70)

The release notes verify the CUDA 12 build only ("we have no Pascal or Volta card"); these
checks cover the **vision path** on a real sm_70 card. Enabled with the config's `vision`
entry plus `--vision --vram-reserve-mib 700` in the engine args (~1.4 GiB from the expert
cache). With images on, text decode stayed in the same 58-63 tok/s range as the vision-off
runs above.

**Speed.** A control image (shapes + caption) was described correctly at 63 tok/s decode.
A 3-round real session with an image every turn: decode 78-80 tok/s, expert-cache hit
96.7-97.4%, image+text prompts read at ~760 tok/s. Images cost no decode speed here.

**Quality.** Three user photos through a strict constrained prompt (a fixed six-section
answer: an 8-object inventory, an OCR list capped at 10, exactly 4 risk lines, a 5-item
inspection record, a limits section, and a self-check line - "write 'cannot confirm' for
anything not visible, never fill in from general knowledge"):

- An illustration with no text: 8/8 objects real and positioned right; every unreadable
  field honestly "cannot confirm"; zero invented content.
- A vintage cast-metal nameplate: OCR exact to the character - "№2906", "SIEMENS-SCHUCKERT".
  It listed 2 of the 4 mounting holes: a completeness miss, not a hallucination.
- A dense motor nameplate (CG Power): 10/10 OCR lines correct to the character, including
  "IS 12615", "MACHINE NO : 0.75KNE4FLG", "CM/L-7800028417", "kW(HP): 0.75(1.00) RPM : 1410",
  "VOLT: 415±10% AMP : 1.82Y" (the ± sign and the Y winding code included), "REF : XEGM27571",
  and the one low-legibility line was marked low confidence by the model itself. Only two
  screws were visible in the photo; the model said so and marked the fastening check
  "cannot confirm" instead of guessing.

No invented identifiers, serials or risks in any of the three answers. One minor wrinkle:
the self-check line's "read count" used a slightly different counting rule between two
answers (with and without the "cannot confirm" rows).

Vision on this card is production-usable for inspection-style work: exact OCR on clean
nameplates, honest abstention on unreadable fields, and no decode-speed cost.

## Update (2026-10-05): IQ3_S vs IQ3_XXS on the same box

The best-quality quant ("matches the full model") installed next to the IQ3_XXS one
(`SETUP.bat --setup --model IQ3_S --gguf-dir ... --vision gpu`; new pack in the data dir,
MTP draft and expert profile reused, `run-iq3_s.bat` + `strata-iq3_s.json` generated, the
XXS install untouched). Calibrated on its own (settings are remembered per model):

- IQ3_S: `--pcie-frac 0.20`, `--spec-min-p 0.70` - 61.6 tok/s in the calibrate bench
  (IQ3_XXS on the same PC: `--pcie-frac 0.35`, `--spec-min-p 0.70` - 76.9 tok/s, so S runs
  at 80% of XXS here; the release notes' expectation for S is 8-10% slower).
- Worker sensitivity is back on S: 35 / 23 / 18 workers measured 61.6 / 58.7 / 55.4 tok/s
  (an 11% spread) where XXS on 0.1.39 was flat within 2%. More bits per weight = more host
  work per token, so thread count matters again on this dual-socket box.

Same-prompt API measurements, both calibrated, images on, single runs (the two quants
choose slightly different answer lengths; the trend is stable across repeats):

| Workload | IQ3_XXS | IQ3_S | S / XXS |
|---|---|---|---|
| Short prose, ~170-220 tok out | 54.6 tok/s | 49.3 tok/s | 90% |
| Long-form, 2-3.8k tok out | 64.2 tok/s | 59.4 tok/s | 92% |
| 4,340-tok doc, prompt processing | 1,127 tok/s | 1,101 tok/s | 98% |
| Same doc, decode, ~700-800 tok out | 72.3 tok/s | 53.3 tok/s | 74% |
| Image prompt (control image), decode | 61.8 tok/s | 52.5 tok/s | 85% |

Prompt processing is compute-bound and barely moves (-2%); decode is host/memory-bound and
pays 8-26%, worst on the long-KV document workload. The expert cache holds fewer, larger
experts: 12,321 (23.4 GiB incl. the vision reserve) vs XXS's 14,660 (23.8 GiB).

**Quality.** Three hard prompts (H1: smallest n with exactly 2019 trailing zeros in n!,
H3: minimum bracket flips with a proof of optimality, H5: find exactly 4 bugs in a Python
class), identical params (thinking cap 4000, seed 42): both quants answered all three
correctly with the same final values, at comparable token counts (both burn through a
5,000-token cap on the hardest one; a re-run of S with 8,000 finished correctly at 4,525).
On these prompts no quality gap is visible - on this DDR3-bandwidth box the ~20% decode
cost of S buys nothing measurable, so XXS stays the daily driver and S is the
quality-first option.

## Update (2026-10-05): UD-IQ4_XS (Unsloth ~4-bit) on the same box

The third tier installed next to the other two (`--family unsloth --model UD-IQ4_XS
--vision gpu`). Two install notes: the pack is only 1.38 GiB - on unsloth-family quants
the engine reads experts in place from the GGUF, and the pack is an index plus the dense
side - and `--resident-budget-gib 55` keeps all 59.5 GiB of experts resident in RAM on
this 128 GB box, so steady-state decode reads nothing from the model disk.

Calibration kept **only `--pool-workers 18`**: 35 / 23 / 18 workers measured 31.9 / 30.4 /
**38.0** tok/s in the calibrate bench - the strongest worker sensitivity of the three
tiers (XXS within 2%, IQ3_S 11%, UD-IQ4_XS 20%+; bigger experts mean more host bytes per
token). The other sweeps stayed flat enough that nothing else was written (PCIe share
0.00-0.75: 36.2-39.1; draft floor 0.70 best at 39.1). Hand overrides on top of the kept
settings did not help real loads either - median long-form tok/s over 3 runs each: kept
settings 42.1 vs `--spec-min-p 0.70` 38.8, `--pcie-frac 0.20` 40.8, NUMA pinned to
node 0 / node 1: 40.3 / 35.6. The calibrate bench's pick transferred to real requests
better than any hand tuning on this box.

Same-prompt API measurements, calibrated, images on:

| Workload | IQ3_XXS | IQ3_S | UD-IQ4_XS | UD / XXS |
|---|---|---|---|---|
| Short prose, ~150-220 tok out | 54.6 | 49.3 | 27.3 | 50% |
| Long-form, ~2-3.4k tok out | 64.2 | 59.4 | 42.6 | 66% |
| 4,340-tok doc, prompt processing | 1,127 | 1,101 | 541.7 | 48% |
| Same doc, decode, ~700-1,000 tok out | 72.3 | 53.3 | 35.0 | 48% |
| Image prompt, decode (hand-run) | 61.8 | 52.5 | 45.5-46.8 | 75% |

Calibration itself bought +13-31% on real loads (long-form 37.6 -> 42.6, doc decode
30.1 -> 35.0, H1 thinking decode 32.2 -> 38.7) - on this tier the workers pick matters
more for real requests than the calibrate probe suggests. Measurement note: the fixed
500-token-cap suite run on an image prompt reads 30.7 tok/s because the whole cap goes
to thinking; the hand-run numbers in the table are the comparable ones.

**Quality.** The same three hard prompts as the IQ3_S section (cap 4000, seed 42,
max 8000): **3/3 correct** - H1 n = 8090 with the minimality check, H3 answers identical
to IQ3_S (including the odd-length no-solution case), H5 all four bugs with fixes.
14,251 tokens total vs IQ3_S's 14,369 on the same suite. No quality gap between tiers
is visible on these prompts.

**Quirks.** The first answer after a cold start runs ~25 tok/s (expert cache ~65%) and
needs 2-3 rounds to reach the ~97% steady-state hit rate. And `enable_thinking: false`
combined with an image request looks like the image is dropped (the prompt collapses to
a few tokens) - keep thinking on for image requests and cap it with
`reasoning_budget_tokens` instead.

Positioning on this DDR3-bandwidth box: UD-IQ4_XS gives up 25-52% of XXS decode speed
and half the prompt throughput. XXS stays the daily driver, IQ3_S the quality-first
pick, UD-IQ4_XS the highest-bit tier that still carries vision (UD-Q4_K_XL remains
experimental without vision).

## Update (2026-10-07): v0.1.40, the Volta data point the release notes ask for

Upgraded to the 0.1.40 ready-made CUDA 12 engine (`git fetch origin && git reset --hard
origin/main` after the history cleanup, then SETUP; the config's vision block and the
`--vision --vram-reserve-mib 700` args had to be re-added by hand, since SETUP without
`--vision gpu` rewrites the config without them). Same model, same workload suite,
server timings, seed 42.

Re-calibrated (the 0.1.39 settings were `--pcie-frac 0.35 --spec-min-p 0.70`):

- PCIe share 0.00 / 0.20 / **0.33** / 0.35 / 0.55 / 0.75: 77.2 / 78.3 / **80.1** /
  79.2 / 79.0 / 78.2 tok/s - the curve stays flat, 0.33 is kept.
- Draft floor 0.30 / 0.50 / **0.70**: 76.5 / 77.7 / **84.1** tok/s - 0.70 stays the
  clear winner, as on 0.1.39.
- CPU workers 9 / 17 / 18 / 23 / 35: 83.0 / 82.4 / 77.4 / 80.4 / 81.8 tok/s - flat
  again (no setting written), like 0.1.39 and unlike 0.1.38.
- **Calibrate bench: 76.9 -> 81.8 tok/s (+6.4%).**

API workload suite, calibrated settings, images on:

| Workload | 0.1.39 | 0.1.40 |
|---|---|---|
| Short prose, ~180-220 tok out | 54.6 tok/s | 59.4 tok/s |
| Long-form, 3.3-4.1k tok out | 64.2 tok/s | 69.4 tok/s |
| 4,340-tok doc, prompt processing | 1,127 tok/s | 1,175 tok/s |
| Same doc, decode, ~670-920 tok out | 72.3 tok/s | 72.6 tok/s |
| H1 (n! trailing zeros), thinking decode | ~78 tok/s | 78.6 tok/s |

Decode +8%, prompt processing +4%, and the H1 math spot-check (smallest n with exactly
2019 trailing zeros, thinking cap 4000) still answers n = 8090 with the minimality
check at 5,394 tokens. This is the Volta (sm_70) result the 0.1.40 release notes ask
for under "Testers wanted" ("we now have a P100 for sm_60, but not Volta").

## Update (2026-10-07): decode speed vs context length, three tiers (0.1.39)

Ladder runs, one server per tier (calibrated settings, thinking on, cap 4000), walking
the prompt up 1K -> 16K -> 48K -> 96K -> 128K tokens and generating after each step;
server timings, repeat runs in parentheses where taken.

| Prompt tokens | IQ3_XXS | UD-IQ4_XS (warm) | IQ3_S |
|---|---|---|---|
| ~1K | 55.5 (58.8) | 40.9 | 45.7 (58.2) |
| ~16K | 69.5 | 39.7 | 53.0 (54.1) |
| ~48K | 56.9 (67.0) | 39.4 | 53.8 |
| ~96K | 60.6 | 38.1 | 60.4 |
| ~128K | 69.2 | 31.9 | 58.1 |

- **IQ3_XXS: no decay.** 55-70 tok/s across the whole range, and 128K is among the
  fastest runs (draft acceptance 0.82 there). Prompt processing does fall with size, as
  expected: ~1,450-1,480 tok/s at 16K -> 1,263 at 96K -> 1,130 at 128K.
- **IQ3_S: no decay either** - 45.7 at 1K rising to 60.4 at 96K, 58.1 at 128K.
- **UD-IQ4_XS: flat to 96K, then drops at 128K** - 38.1 -> 31.9 tok/s (roughly -20%
  against its 39-41 plateau). Draft acceptance falls with it, 0.73 -> 0.58 at that
  step: longer KV and a lower MTP acceptance rate compound on the widest experts of
  the three tiers. Warm prompt processing held 1,033-1,287 tok/s from 16K to 128K.
- Warm-up matters more than context on UD: the cold first pass read a 1K prompt at
  143 tok/s (318 warm) - expert pages come off the GGUF on first touch.

Deployment note from the same session: with the IQ3_S GGUF sitting on an HDD (and the
page cache under pressure from a 59.5 GiB resident set elsewhere), decode pinned at
13-19 tok/s regardless of context until the files moved back to NVMe. The numbers above
assume model files on NVMe/SSD or fully RAM-resident.

## Update (2026-10-08): v0.1.40.3 engine + the 0.1.40.2 opt-in switches

Upgraded to the 0.1.40.3 ready-made engine (setup replaced 0.1.40 by itself; the old
engine stays in `engine-cuda12\.previous`). Re-calibrated on the same XXS install: the
kept settings did not change (`--pcie-frac 0.33 --spec-min-p 0.70`) but the calibrate
bench reads **74.7 tok/s (0.1.40: 81.8, -8.7%)**, and the real-load suite moves the
same way - long-form 64.0-67.7 (0.1.40: 69.4), 4.3K-doc prompt 1,104 (1,175), doc
decode 73.8 (72.6), short prose unchanged at 59.4, H1 still correct. The 0.1.40.2
notes report +1.7-3.9% on RTX 3060/5070/P100; on this Volta/DDR3L box the same
release reads as a small regression. One calibrate run also died mid-sweep ("the
engine stopped unexpectedly (exit code None)") after the PCIe phase; a clean retry
completed and wrote the same settings.

The opt-in switches from 0.1.40.2, measured on this box:

| Switch | Measurement | Result |
|---|---|---|
| `STRATA_PREFILL_CPU_SHARE=auto` | 587- and 977-token prompts, 10 interleaved runs each, server timings | 1,605 -> 1,345 ms (**-16%**) and 2,076 -> 1,841 ms (**-11%**); H1 with it on still answers n = 8090 |
| `strata_prefix` pin | 30K-token ops-log document, 8 questions, `tools/research_run.py` | follow-up first token 2.98 -> **0.87 s (3.4x)**, whole run 57.9 -> 20.3 s. Without the field, 0.1.40.3's default checkpoint reuse already holds follow-ups at ~3 s on this box - the 48-149 s re-reads from the release notes do not happen here |
| `STRATA_SPEC_PROB=1` | long-form decode, temp 0.6, 3 runs per arm | 65.9-67.6 -> **68.8 tok/s (+2-4%)**, drafts accepted 0.77 -> 0.66 |
| `STRATA_SPEC_COUPLED=1` (+ `STRATA_SPEC_GUMBEL=1`) | same | decode flat (64.5-65.1), drafts accepted 0.77 -> 0.62-0.63 - the opposite direction from the AMD and RTX 3060 reports |
| `--prefill auto:32768` | 4.3K-doc and 30K-token cold prompt reads | 1,104 -> 1,119 and 1,430 -> 1,419 tok/s - no effect on this host-bound box (the 21-35% setup tip does not transfer) |

Takeaways for a box like this one: `STRATA_PREFILL_CPU_SHARE=auto` is the only switch
with a clear win here (agent-style short prompts), `STRATA_SPEC_PROB=1` is a small
real gain for sampled long-form decode, and the coupled/Gumbel drafts and the 32K
prefill chunks buy nothing on Volta + dual DDR3 sockets.
