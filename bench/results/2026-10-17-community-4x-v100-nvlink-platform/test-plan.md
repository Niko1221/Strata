# test-plan.md — the stand for testing a change on this machine

Purpose: **one fixed set of checks** that can be run against any new branch of Strata on this machine and
will produce comparable numbers. The stand does not re-measure what has already been taken (§7); it answers
one question: *did the result change against this machine's baseline line, and if so, because of what*.

All scripts are in `scripts/`, results go to `raw/ab/<UTC stamp>-<build label>/`. Nothing is overwritten. The
working configuration (`<HOME>/strata185/Strata/strata-185-iq3_s.json`, `--layer-split auto`) is not changed:
variants are written to separate files `strata-185-iq3_s.ab-<cell>.json` and removed at the end of a run.

## 1. What the stand consists of

| Component | File | What it does | Status |
|---|---|---|---|
| the 2x2 matrix orchestrator | `scripts/ab-run.sh` | manifest -> cells (start a variant -> acceptance -> benchmark -> counters -> stop) -> restore the working start -> summary | ready |
| build identity | `scripts/ab-manifest.py` | `manifest.json`: sha256 of the binary, the config (without secrets), every script and probe, tool versions, `build_info` from `/props`, an nvidia-smi snapshot | ready |
| a cell's config | `scripts/ab-mkconfig.py` | an isolated config variant: replacing the `--layer-split` value, pointing `exe` at another binary | ready |
| returning the service | `scripts/ab-ready.py` | waiting for start-up + checking **three signs**: processes alive, VRAM at or above a threshold on every card, a real `chat/completions` returning `choices` | ready |
| bit acceptance | `scripts/ab-text-check.py` | transport acceptance **before** the benchmark: warm-up + repeats on stable prompts, comparing `text_sha` and the common-prefix length with the previous cell's reference | ready |
| the benchmark | `scripts/baseline-profile.py` (unchanged) | prompt/decode/TTFT/VRAM/PCIe with a fixed output length, >= 3 repeats, medians | ready |
| the engine's counters | `scripts/ab-extract-log.py` | from a slice of the working log: `MB of expert reads`, `the OS read … (major faults)`, `decode expert cache hit rate`, `expert cache … slots`, `the prompt path borrows …`, `layer split auto:` / the chosen split, CUDA error lines | ready |
| summary and comparison | `scripts/ab-report.py` | `summary.csv` / `summary.json` / `summary.md`: the delta against the base cell, the delta against the machine's baseline line, the noise threshold from the repeated base, verdicts against the significance thresholds | ready |
| low-level checks | `build/kernel-load-audit`, `build/sm70-tests`, `build/p2p-probe-ext.check`, `build/peer-read-probe` | loading every kernel of a new build, 13 numeric SM70 tests, P2P correctness and bandwidth, the bandwidth of direct kernel access to a peer's VRAM | ready |
| direct kernel access to a peer | `probes/peer-read-probe.cu`, `scripts/run-peer-read.sh` | the bandwidth of a kernel peer read/write against `cudaMemcpyPeerAsync`, opposing load, a cold start | ready, taken 2026-10-17T05:13Z |

A worked example: `raw/peer-read/2026-10-17T051351Z/` (the full 1…128 MiB ladder, 0 correctness errors).

## 2. The fixed test configuration (do not change without a reason)

| Parameter | Value | Why |
|---|---|---|
| matrix cells | `A0:auto:base -> A1:auto:patched -> B0:split12:base -> B1:split12:patched -> A0r:auto:base` | the 2x2 matrix in `patch-notes.md` §9.2; the repeat of the base at the end sets the run's noise threshold and catches the drift of `auto` |
| the pinned split | `12,25,37` | the only layout already measured on this machine (baseline-profile.md §8); **a diagnostic row of the matrix, not the working configuration** |
| prompt lengths | `4k,32k,131072` | short (the noise threshold), medium (the linear regime), long (the dominant expert-streaming regime). Correction of 2026-10-17T06:40Z: it used to be `125k`, which `baseline-profile.py` turns into a target of 128 000 -> an actual `prompt_n = 122 087`, so the ladder did not match `baseline-profile.csv` (125 011). The validation run of 2026-10-17 still used `125k` - see `ab-stand.md` §6.4. A further correction at 06:54Z changed the default in `scripts/ab-run.sh` to `4k,32k,131072` (the script's sha after the change `91b23d66…`) |
| repeats | 3, medians | the within-row spread is 0.19…0.29 % (up to 1.31 % at 4k) |
| fixing the output length | `logit_bias = -100` for the EOS ids 248044/248046, checking `gen_n == max_tokens` and `finish_reason = "length"` | without this the decode column is invalid |
| warm-up | the first request after a start / config change is not a measurement | the first request after a change of state returns different text |
| bit acceptance | targets of 1024 and 8192 tokens, `max_tokens = 128`, a "stabilize-then-compare" protocol: requests are repeated until two consecutive `text_sha` match (at most 6), the first ones are warm-up | corrected on 2026-10-17T05:49Z after an actual run: one warm-up is not enough - the engine's state warms over 2…3 requests. Analysis in `ab-stand.md` §5, §7.1 |
| significance thresholds | prompt tok/s: |delta| > 1 % at lengths >= 32k and |delta| > 2 % at 4k; decode: |delta| > 10 % and only as a "must not get worse" control | the measured noise of a run: 0.06…0.42 % session-to-session for prompt (with a within-row spread up to 1.19 % at 4k) and up to 7.8 % for decode - `ab-stand.md` §4 |
| the "the model is loaded" threshold in `ab-ready.py` | a sum over the four cards >= 57 000 MiB AND every card >= 13 000 MiB | corrected on 2026-10-17T06:16Z: the previous threshold of 14 000 MiB per card failed for the pinned `12,25,37`, where CUDA0 legitimately holds 13 744 MiB. Under auto: 61 712 MiB total; under 12,25,37: 59 862; a wrapper without the engine: 28 |
| GPUs for external probes | 0,1,2 only | an external CUDA context cannot attach to GPU3 while the service runs |
| the result folder | `raw/ab/<UTC stamp>-<label>/` + `manifest.json` | a separate folder per build; there is no git on the machine, so the primary identity is the binary's sha256 + `build_info` |

## 3. How to run it

```bash
cd <HOME>/strata185/'Strata+4xV100+NVlink'
mkdir -p raw/ab
# the base binary + the patched binary (the same build twice = validating the method)
setsid nohup scripts/ab-run.sh --patched-binary <HOME>/strata185/Strata-patched/engine-cuda12/strata \
    --label v0.1.45-p2b > raw/ab/run-$(date -u +%H%M).log 2>&1 &
tail -f raw/ab/run-*.log
```

Options: `--skip-bench` (bit acceptance and counters only - a smoke run), `--skip-text` (speeds only),
`--restore-only` (restore the working start, measure nothing), `--arms "A0:auto:base A0r:auto:base"`
(measuring noise without a change).

**The stop window:** running the matrix stops the service for about 6 minutes per cell (about 40 minutes in
total, of which 66 s is start-up). Before stopping, the script updates `<HOME>/strata185/run/server-iq3_s.pid`
to the actual wrapper pid (`ss -ltnp | grep ':8080 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2`),
otherwise the stop script would take down only the engine.

The low-level checks run separately and need no window (except pairs involving GPU3):

```bash
scripts/run-peer-read.sh --max-size 128M --note "…"      # 1…128 MiB, GPU0-2
build/sm70-tests                                          # 13 tests, dev0/dev1/dev2
build/kernel-load-audit                                   # loading every kernel of a new build
build/p2p-probe-ext.check --out /tmp/x --max-size 8M      # P2P correctness (the contention_pair output is fixed)
```

## 4. The order inside a cell (mandatory)

1. `manifest.json` (one per run) - sha256 of binary/config/scripts, `build_info`, versions, nvidia-smi.
2. A VRAM snapshot before the stop -> update the pid file -> stop through the normal script.
3. Start the variant **the same way** as the main one (an absolute python path).
4. `ab-ready.py` - three signs. Not passed -> the cell is `FAILED`, nothing further is measured.
5. **Bit acceptance before the benchmark** (`ab-text-check.py`, compared with the previous cell's reference).
   Within a cell a stable pair of `text_sha` is required (not reached in 6 requests -> no benchmark).
   A mismatch with the reference: with a change applied -> FAIL and no benchmark; in method-validation mode
   (no change) -> `PASS_WITH_WARNING`, the benchmark runs and the difference is recorded as the object of
   measurement (`ab-run.sh` passes `--allow-ref-mismatch` when `--patched-binary` is not given).
6. The benchmark (`baseline-profile.py`): warm-up + 3 repeats x 3 lengths, with a fixed output length.
7. Extracting the engine's counters for the cell's time window (the mandatory lines are in §1) + a slice of the raw log.
8. Stop, move to the next cell. At the end, restore the working config and verify it by the three signs.

## 5. What exactly separates "less PCIe" from "more streaming"

Every run records these in pairs:

* `for <N> MB of expert reads` and `the OS read <M> MB from storage (<F> major faults)` - if the first grew
  and the second did not, the change did not "save PCIe", it "added streaming";
* `expert cache <S> slots` and `the prompt path borrows <K> … slots (<G> GiB)` - the VRAM price of the
  change; by the relation in `patch-notes.md` §3.3 every 1 MiB taken from the cache costs (T/8192) MiB of
  extra stream;
* `decode expert cache hit rate` - a fall in the hit rate explains a change in the decode column better than
  "the change made it slower";
* `layer split auto:` and `layer split: layers …` - which layout the search chose (after P5, with both scores);
* per-card `rxpci` from `dmon` - the sign of a working peer path: **`rxpci` falling while prompt tok/s rises**,
  with idle traffic remaining exactly 0.

## 6. Comparing two builds

`ab-report.py` builds `summary.csv|json|md`: the delta against the run's base cell, the delta against the
machine's baseline line (`baseline-profile.csv`), the noise threshold (how far the repeated base drifted), and
verdicts against the thresholds. To compare two runs by hand: match only cells of the same name (`A0` with
`A0`, `B1` with `B1`) and only when bit acceptance passed; if it did not, compare the common-prefix length
and analyse the computation rather than the speeds.

## 7. What the stand deliberately does not include (the data already exists)

The full benchmark sweep; the P2P capability matrix and the card<->card / host-path bandwidths; the per-phase
PCIe profile; the A/B of `STRATA_PREFILL_HELP`; the P2P size ladder and the six-pair competition; the kernel
inventory and the SM70 gate table; the 0.1.44.1 baseline profile and the idle-machine control point; the
attribution experiment; the SM70 and multi-card correctness checks; the recall needles.

## 8. Known limits of the stand

1. Binding to a tree commit SHA is unavailable: `git` is not installed on the machine. Build identity is
   recorded as the binary's sha256 and `build_info` from `/props` - enough to tell builds apart, not enough to
   map to a source tree.
2. Pairs involving GPU3 are reachable in the low-level probes only in a stop window.
3. There are no NVLink byte counters on this machine (DCGM 4.x without the profiling module, NVML Sampling &
   Reporting absent) - routing is proven by "counting PCIe in reverse" and by the engine's counters.
4. The stand measures **speed columns and counters**, not causes: a conclusion about the mechanism comes from
   the whole set (§5), never from one number.
5. The load profile is single-threaded (`concurrency/serving = 1`): the server also serves other clients, so
   foreign traffic can appear between cells; it lands in the log slice and is visible in
   `engine-request-counters.csv` (rows with other `prompt_n` values).
6. **The decode column is not a measure on this machine** (measured): session-to-session spread up to 7.8 %
   with an unchanged config, explained by draft acceptance moving 3-7 percentage points between sessions
   (`ab-stand.md` §2, §4.3).
7. **The text of greedy generation is reproducible across restarts only with an unchanged placement
   configuration.** Changing `--layer-split` changes the output on a long prompt (a common prefix of 32
   characters out of 647) while each configuration is bit-reproducible internally - finding F-8,
   `ab-stand.md` §5.2. For placement changes bit acceptance does not apply by construction.

## 9. Validation of the stand (run 2026-10-17T05:43…06:38Z, no change applied)

The 2x2 matrix on the base binary was run: 5/5 cells, the run's noise threshold measured (prompt tok/s
0.06…0.42 %), the known `auto` / `12,25,37` difference reproduced (+4.48 % at 31 307, +6.66 % at 122 087,
TTFT -4.2 % / -6.2 %) in two independent pairs of cells. Analysis: **`ab-stand.md`**; data:
`raw/ab/2026-10-17T061723Z-validation3/` (`summary.csv|json|md`, `MERGE-NOTE.txt`), and the interrupted
attempt `raw/ab/2026-10-17T054547Z-validation/`. Two defects of the stand were found in that run and fixed:
premature bit-stability requirement, and a VRAM threshold in `ab-ready.py` tuned to the auto layout.
