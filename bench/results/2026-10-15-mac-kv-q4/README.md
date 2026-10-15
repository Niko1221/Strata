# 2026-10-15: Mac (M4 Pro) KV q4_0 measurement round — metal-kv-q4 branch, task M3-a

The q4_0 K/V format (`--kv q4_0`, PR #21 lineage, landed for Metal in M1-b/M2-b of this branch) measured on the
Mac backend: teacher-forced quality, batched-prompt-path quality, same-day paired speed, KV bytes/arena, and
needles. Every number below was measured on this machine on 2026-10-15; nothing is extrapolated from another
machine. The PC round (bench/results/2026-09-27-kv-q4, commit history) is a different backend and a
`--kv-resident` streaming configuration — its numbers are not comparable and are not cited as results here.

## Hardware / software

- Apple M4 Pro (Mac16,8), 20 cores, 48 GB unified memory; macOS 27.0.1.
- Engine `build-metal/strata`, 0.1.44-metal, built from branch `metal-kv-q4` at 19b8b58b (plus this round's
  `tools/quant_quality.py` change; no engine change in this round).
- Model: Qwen3.8-Flash-Next GSQ-RCO Q2_0 (~80 GB, `/Users/berton/Github/AI-Infra/Strata-data`), experts
  streamed from the SSD (`--mmap-experts`, `STRATA_METAL_STREAM=1`), as in `strata-q2_0.json`.
- No server was started: the engine ran as one-shot `generate` or stdin `--serve` processes; nothing bound a
  port.

## The backend fact that frames every table: on Metal, fp16 and int8 are one path

The Metal decode window has exactly two K/V implementations: q4_0, and one shared non-q4 path whose pools are
the int8 layout. `metal/src/engine/kv_format.h`: "fp16 / int8 are today's path (the window's pools are the
int8 ones either way)"; `window.cpp` logs the format as `kv_q4_ ? "q4_0" : "int8"` — the `--kv fp16` run's own
log says `KV int8`. Measured consequence: the fp16 and int8 teacher-forced TSVs of this round are
**byte-identical** (md5 `4e942a7375715ac014956fbdbf6110f6` both). So on this backend:

- "fp16 reference" below means the shared non-q4 path;
- an fp16-vs-int8 delta cannot be measured (there is nothing to measure);
- the real quality question is q4_0 against that shared path, which is what the tables give.

## Teacher-forced quality, decode window (`tools/quant_quality.py run` + `compare`)

Five segments of the committed set (code, docs, english, multilingual, agent; 999 scored positions each,
4,995 total), every token read through the decode windows (`--short-read 1064`), STRATA_LOGPOS top-64 TSVs in
`data/teacher-forced/`. Same day, same engine args, only `--kv` differs; reference = the fp16-flagged run
(the shared non-q4 path, per above).

| run | segment | pos | top-1 | mean KL | KL p99 | PPL | PPL ref |
|---|---|---:|---:|---:|---:|---:|---:|
| q4_0 | code | 999 | 96.70% | 0.01182 | | 2.115 | 2.110 |
| q4_0 | docs | 999 | 90.99% | 0.03196 | | 10.499 | 10.464 |
| q4_0 | english | 999 | 94.69% | 0.01406 | | 4.592 | 4.546 |
| q4_0 | multilingual | 999 | 96.80% | 0.00900 | | 2.806 | 2.812 |
| q4_0 | agent | 999 | 93.09% | 0.10385 | | 6.573 | 6.128 |
| q4_0 | **all** | 4,995 | **94.45%** | **0.03414** | 0.4467 | **4.517** | 4.442 |
| int8 | all | 4,995 | 100.00% | 0.00000 | 0.0000 | 4.442 | 4.442 |

- int8's row is the byte-identical TSV, by construction (one path on Metal).
- q4_0: top-1 agreement 90.99–96.80% per segment, 94.45% overall; mean KL 0.034; perplexity +1.7% overall
  (per segment −0.2% to +7.3%, worst on the agent transcript).

## Batched prompt path quality (`prompt` + `pcompare`)

The M2-b format-aware prompt fast path (prompts on the matrix units) answering the same set: 20 prompts
(5 segments × first 256/512/768/1000 tokens), full-vocabulary float32 logits of the first window
(STRATA_DUMP_FIRST_LOGITS), fp16-flagged run as reference.

| run | prompts | top-1 | KL mean | KL median | KL max |
|---|---:|---:|---:|---:|---:|
| q4_0 | 20 | **100.00%** | 0.03565 | 0.00434 | 0.19546 |

The prompt fast path under q4_0 never flips the argmax on these 20 prompts; its full-vocab KL is the same
magnitude as the decode window's (0.036 vs 0.034 mean). The `first.*` float files (~20 MB per run) are not
committed; both runs regenerate them with the commands in *Reproducing* (`data/prompt-path/` keeps the
indexes, stderr summaries and the pcompare table). `data/prompt-path/per-prompt.tsv` is the offline check:
the 20 per-prompt top-1/KL rows from a regeneration on 2026-10-15 (the original run's raw logits lived in
/tmp and are gone); the regenerated aggregates are identical to the table above — worst row agent@768
tokens, KL 0.1955.

## Speed (same-day pairs, `speed.sh`, logs in `logs/speed-*.log`)

Engine args exactly the `strata-q2_0.json` template (`--expert-cache 64 --prefill auto --spec 2
--suffix-draft 0 --max-context 32768 --mmap-experts`, `STRATA_METAL_STREAM=1`), greedy, one run per cell,
only `--kv` differs. "none" = no `--kv` flag (the fp16 default).

| cell | q4_0 | int8 | none | q4_0 vs int8 |
|---|---:|---:|---:|---:|
| decode tok/s (4-id prompt, --max-new 64) | 20.84 | 21.09 | 20.82 | −1.2% |
| prefill tok/s (2200-id prompt, 2199 tokens prefilled, --max-new 16) | 213.15 | 215.54 | | −1.1% |
| time to first token, 2200-id prompt | 10.71 s | 10.60 s | | +1.0% |
| decode tok/s after the 2200-id prompt | 19.84 | 19.87 | | −0.2% |

The noise scale: `none` and `int8` run the same code on this backend (see above), so their −1.3% decode
delta is pure run-to-run noise. The q4_0 deltas (−1.2% decode, −1.1% prefill) sit at that noise edge: q4_0
is, at worst, a ~1% cost on this machine — the dequant arithmetic roughly pays for itself against the halved
KV traffic, and the KV saving shows up as resident experts instead (next section). Single runs per cell, no
repeats; the 2200-id prompt (2199 tokens prefilled; the last id seeds generation) is
`data/long-prompt-2200.ids` (ids 100..2299, the M1 baseline file; the engine's own line reads "prefill 2199
tokens in 1 chunks").

## KV bytes and arena (`kvmem.sh`, logs in `logs/kvmem-*.log`)

Short runs (--max-new 8) at both contexts, pairs {q4_0, int8}. Arithmetic: the engine allocates
`max-context + 128` cells (the log's "32896/65664 cells"), 12 QSA layers, 1,056 B/cell/layer int8 vs
576 q4_0 (`metal/src/engine/kv_format.h`). The KV allocation sits inside the log's "N GB allocated before
the arena" line; the q4-vs-int8 difference there is the KV saving.

| ctx | cells | int8 KV (calc) | q4_0 KV (calc) | delta (calc) | delta (log, before-arena) | arena slots int8 → q4_0 |
|---|---:|---:|---:|---:|---:|---|
| 32768 | 32,896 | 416,858,112 B (0.417 GB) | 227,377,152 B (0.227 GB) | 0.189 GB | 3.69 − 3.50 = **0.19 GB** | 15,840 → 16,081 |
| 65536 | 65,664 | 832,094,208 B (0.832 GB) | 453,869,568 B (0.454 GB) | 0.378 GB | 4.16 − 3.78 = **0.38 GB** | 15,163 → 15,657 |

The arithmetic and the log agree to 0.01 GB at both sizes. The planner spends the saving on resident
experts: the planned resident footprint is 31.2–31.3 GB in all four runs (cap 43.8 GB), and under q4_0 the
expert arena holds 241 more expert slots at 32K, 494 more at 64K. RAM lines, copied as logged:
32768: q4_0 "RAM 51.5 GB (30.6 GB available, 7.7 GB kept for macOS)" vs int8 30.3; 65536: q4_0 30.0 vs
int8 29.3 — q4_0 leaves ~0.3–0.7 GB more available (the Metal logs carry no watchdog line; the RAM planning
lines are what was copied).

## Needles (`needle.py`, results in `needle-results.tsv`, logs in `logs/needle-*.log`)

A codeword (`MARBLE-7742-KUMQUAT`) in one sentence, hidden in natural text (the committed quant-quality
set's five segments, cycled, as token ids — the prompt is byte-reproducible from committed files), depths
10/50/90% of the filler span, lengths 8K/16K/32K (id budgets 7936/16128/32512 = nominal length − 256 so
prompt + 64 new tokens fit --max-context 32768). Greedy, --max-new 64, one `generate` load per case. Judge:
the decoded reply, whitespace-stripped and uppercased, contains the full codeword; partial retrievals are
visible in the recorded reply text.

| length \ depth | 10% | 50% | 90% | int8 @ 50% |
|---|---|---|---|---|
| 8K (7,936 ids) | found | found | found | found |
| 16K (16,128 ids) | found | found | found | found |
| 32K (32,512 ids) | found | found | **missed** | found |

**q4_0: 8/9 retrieved; int8 controls: 3/3.** The one miss is the deepest needle at the longest length
(32K, 90% depth): the model recognized the sentence as an embedded "access code" but treated it as a prompt
injection and refused to repeat it (full reply in `needle-results.tsv`) — a behavior miss, not a garbled
retrieval; every retrieved answer was the exact codeword. No partial retrievals in this pass. (A first pass
at full-budget prompt lengths — before the 256-id margin fix — had one partial `MARBLE-7742` and one
injection-suspicion miss at 16K; that pass is superseded and its per-case outcomes are in the task report.)

Two disclosures on the needle form. The question ends with "The access code is" — a completion cue: the
model completes that phrase rather than answers a neutral question; all arms share it, so the q4-vs-int8
comparison is internally fair, but the 8/9 is under an easier phrasing than a neutral QA form. And the one
miss has no int8 control at 90% depth (the controls ran at 50%), so it cannot be excluded the shared path
would refuse there too; its reply was also cut at the 64-token budget mid-sentence — the injection-suspicion
reasoning is explicit, a final refusal sentence is not.

The q4_0 50%-depth engine logs were lost to a log-name collision with the int8 controls (same tag) and are
not in `logs/`; those three cases' outcomes and replies are recorded in `needle-results.tsv`. The script
now names logs `needle-<tag>-<kv>.log`.

## What was not tested

- The M5 k4 numeric arm (fc-7 right-input layouts): this M4 Pro is !k4 hardware; the k4 specializations
  compile but are never probed here.
- MTP + q4 end to end: every run used the template's `--spec 2 --suffix-draft 0`; the MTP drafter over q4
  pools was not exercised.
- PC consistency: no PC run this round; the 2026-09-27 PC table is a different backend and config.
- `--kv k8v4`: not in the matrix (and refused by the Metal window: kv_format_check).
- Speed repeats: one run per cell (the same-code none-vs-int8 pair puts noise near 1%).
- Needle breadth: one codeword, one filler corpus, one run per case; no distractor needles; int8 controls
  only at 50% depth.

## Reproducing

Worktree `/Users/berton/Github/AI-Infra/Strata-metal-kv-q4` at 19b8b58b + this round's quant_quality.py; the
venv `.venv` (python with numpy/regex). Engine loads: 26 accounted in the task report (plus 3 refused
arg-guard startups and 6 superseded first-pass needle cases, also accounted there).

```sh
# teacher-forced quality (3 loads) + the math
.venv/bin/python tools/quant_quality.py run  --config strata-q2_0.json --out Q-fp16.tsv --kv fp16
.venv/bin/python tools/quant_quality.py run  --config strata-q2_0.json --out Q-int8.tsv --kv int8
.venv/bin/python tools/quant_quality.py run  --config strata-q2_0.json --out Q-q4.tsv   --kv q4_0
.venv/bin/python tools/quant_quality.py compare Q-fp16.tsv Q-int8.tsv Q-q4.tsv --segments

# prompt-path quality (2 loads) + the math
.venv/bin/python tools/quant_quality.py prompt  --config strata-q2_0.json --out P-fp16 --kv fp16
.venv/bin/python tools/quant_quality.py prompt  --config strata-q2_0.json --out P-q4   --kv q4_0
.venv/bin/python tools/quant_quality.py pcompare P-fp16 P-q4

# speed (5 loads) and KV bytes/arena (4 loads)
./speed.sh
./kvmem.sh

# needles (12 loads)
.venv/bin/python needle.py
```

`tools/quant_quality.py` grew this round's only repository change: an opt-in `--kv` on `run`/`prompt` that
replaces FIXED's fp16 pin when passed. Without it the constructed engine args are byte-identical to before
(diff of `engine_args()` output over the three gpu_experts modes, before vs after the change: identical).

### Files

- `speed.sh`, `kvmem.sh`, `needle.py` — the runners (engine args = the `strata-q2_0.json` template).
- `data/teacher-forced/` — the three LOGPOS TSVs + seg indexes + compare table/JSON + run logs.
- `data/prompt-path/` — prompt-run indexes, stderr summaries, pcompare table, `per-prompt.tsv` (float
  logits not committed).
- `data/speed-runs.txt`, `data/kvmem-runs.txt` — the tok/s and init lines the scripts print.
- `data/long-prompt-2200.ids` — the 2200-id prefill prompt (ids 100..2299; 2199 tokens prefilled, the last
  id seeds generation).
- `logs/` — full engine logs for the speed, kvmem and needle runs.
- `needle-run.txt`, `needle-results.tsv` — the needle matrix outcome.
