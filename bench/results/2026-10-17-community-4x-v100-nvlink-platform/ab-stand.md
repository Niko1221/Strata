# Validating the A/B stand: the 2x2 matrix on the current binary

```
captured_utc:        2026-10-17T05:43:44Z … 06:38:13Z (a service-stop window, agreed with the owner of the machine)
raw:                 raw/ab/2026-10-17T061723Z-validation3/  (the full protocol; the A0/ and A1/ directories were
                   copied from raw/ab/2026-10-17T054922Z-validation2/ - see MERGE-NOTE.txt in that directory)
                   the interrupted attempt: raw/ab/2026-10-17T054547Z-validation/ (A0 did not pass acceptance)
                   a kernel-access probe taken in the same window: raw/peer-read/2026-10-17T054509Z/
binary:              <HOME>/strata185/Strata/engine-cuda12/strata  sha256 bdfec9f553d07c2b…  build_info "Strata 0.1.44.1"
configs of the auto cells: strata-185-iq3_s.ab-{A0,A1,A0r}.json  sha256 c2bf41eff223c53f…  (--layer-split auto)
configs of the pinned cells: strata-185-iq3_s.ab-{B0,B1}.json     sha256 51fdf0ef8aa86b1c…  (--layer-split 12,25,37)
tooling:             scripts/ab-run.sh add3cdea… (edited 05:49Z: --allow-ref-mismatch in validation mode,
                   props.json per cell), ab-manifest.py 13b53d7b…, ab-mkconfig.py 29d2dee9…,
                   ab-ready.py d6015bc6… (edited 06:16Z, the VRAM threshold - see §7.2; the earlier variant
                   `3440eb23…` was used in the first attempt), ab-text-check.py 51e49bc3…
                   (edited 05:49Z, the "stabilize-then-compare" protocol - see §7.1), ab-extract-log.py e59db664…,
                   ab-report.py 10f5bfd1…, the benchmark scripts/baseline-profile.py 01e2f4c3… (unchanged).
                   Every sha is recorded in the manifest.json of both attempts (the artifacts_sha256 key).
method:              test-plan.md §2-§5; lengths 4k/32k/125k (actual prompt_n 3971 / 31307 / 122087),
                   3 repeats, medians, a fixed output length (EOS banned, finish=length in all 12x5 requests),
                   warm-up, bit acceptance before the benchmark, the engine's counters for every cell
status:              measured (the method reproduces a known difference and measures its own noise threshold)
no change applied:   all five cells are the same binary; the purpose of the run was to validate the method
```

---

## 1. Verdict

**The stand works.** Three independent signs:

1. **A known difference was reproduced at the same size.** `auto` -> pinned `12,25,37`:
   **+4.48 % prompt tok/s at 31 307** (measured earlier, baseline-profile.md §8.2: +4.48 %) and
   **+6.66 % at 122 087** (there it was +7.33 % at 125 011 - the same effect corrected for a shorter
   prompt, §6.4). Both independent pairs of cells (`A0` with `B0`, `A1` with `B1`) produced the difference,
   and both repeats within a configuration did.
2. **Its own noise threshold was measured directly** (the repeated base `A0r` against `A0`, same config,
   another session): **0.42 % at 4k, 0.14 % at 32k, 0.06 % at 122k** prompt tok/s - so the 1 % significance
   threshold has margin at lengths >= 32k and is marginal at 4k (§4).
3. **Transport acceptance works and separates "route" from "computation":** within each cell the text is
   bit-stable; two cells with one config give an identical `text_sha` (A0 = A1, A0r = A0, B0 = B1); cells
   with different configs diverge - and that divergence was caught (§5, a separate finding).

5/5 cells, 0 FAILED, bit acceptance passed 5/5, `finish_reason = "length"` and `gen_n == max_tokens` in every
benchmark run (24 requests per cell), 0 CUDA error lines in the log growth of any cell.

## 2. The matrix of results

prompt tok/s (median of 3 repeats), delta against the base cell A0:

| cell | config | 3 971 | 31 307 | 122 087 | TTFT at 31 307 | TTFT at 122 087 |
|---|---|---|---|---|---|---|
| **A0** (base) | auto | 872.6 | 1897.8 | 2428.0 | 16.53 s | 50.37 s |
| **A1** (same config, 2nd start) | auto | 864.4 (-0.94 %) | 1895.5 (-0.12 %) | 2426.1 (-0.08 %) | 16.56 | 50.43 |
| **A0r** (repeat of the base, end of the run) | auto | 868.9 (-0.42 %) | 1895.2 (-0.14 %) | 2426.5 (-0.06 %) | 16.56 | 50.41 |
| **B0** | 12,25,37 | 877.7 (+0.58 %) | **1982.9 (+4.48 %)** | **2589.8 (+6.66 %)** | **15.84 (-4.2 %)** | **47.23 (-6.2 %)** |
| **B1** (same config, 2nd start) | 12,25,37 | 884.9 (+1.41 %) | **1985.1 (+4.60 %)** | **2589.6 (+6.66 %)** | 15.82 (-4.3 %) | 47.24 (-6.2 %) |

decode tok/s (medians): A0 102.3 / 98.4 / 91.7 (4k/32k/122k), A1 101.0 / 102.5 / 97.7, A0r 96.1 / 102.6 /
92.3, B0 101.0 / 99.9 / 98.4, B1 103.0 / 99.8 / 96.1.
**The decode column does not separate the configurations** (the spread between sessions of one configuration
reaches -6.1…+7.8 %, larger than any difference between them). The reason is visible in the counters: draft
acceptance between `auto` sessions varies 71.8…78.4 % with an unchanged config, so on this machine the decode
column is not a discriminator even at a 5 % threshold (§4.3).

## 3. The engine's counters: what the pinned layout's gain was bought with

| Measure (a 122 087-token request) | auto (A0 / A1 / A0r) | 12,25,37 (B0 / B1) | delta |
|---|---|---|---|
| `for … MB of expert reads` per request | 253 528 / 253 618 / 253 624 MB | 240 442 / 240 440 MB | **-13.1 GB (-5.2 %)** |
| `the OS read … (major faults)` | 2.8…13.9 MB, 0 faults | 3.8…6.6 MB, 0 faults | the stream comes from the page cache, not from disk |
| PCIe `rxpci` over 3 requests at 122k (sum over four cards) | 776 960 / 776 949 / 790 666 MB | 744 681 / 755 592 MB | **-32 GB (-4.1 %)** |
| of which GPU3 | 236 050 / 224 377 / 247 495 MB | 197 593 / 209 835 MB | **-12 %** |
| `decode expert cache hit rate` at 122k | 99.5…99.8 % | **100.0 %** | - |
| `expert cache … slots` (CUDA0) | 5632 | **6144** | **+512 slots = +1.05 GiB of cache** |
| profile coverage on the narrow card (CUDA3) | 5469 of 6144 profiled pairs (89.0 %) | 5443 of 5632 (96.6 %) | +7.6 pp |
| `97 % / 99 % of the experts resident` | 97 % | **99 %** | +2 pp |
| `the prompt path borrows …` (summed over cards) | 5178 slots | 5259 slots | **+81 (not reduced)** |
| VRAM peak per card | 15244 / 15588 / 15180 / 16054 MiB | **13850** / 15306 / 14954 / 16060 | CUDA0 -1394 MiB |
| the split | 0-10 / 11-23 / 24-35 / 36-47 | 0-11 / 12-24 / 25-36 / 37-47 | - |

**What this means (confirmed by observation; the mechanism is from the code):** the pinned layout's gain was
bought **not** by reducing slot borrowing (there is none: 5178 -> 5259) but by (a) +512 cache slots on CUDA0 -
a direct consequence of `stage_trim` applying to CUDA0 only with an explicit `--layer-split`
(`generate.cpp:3082`, `const bool stage_trim = trim_asked && multi_gpu && !split_auto`), and (b) higher
profile coverage on the narrow CUDA3 (89.0 % -> 96.6 %). Both are exactly the P5.5 and P5.4/P5.6 items in
`patch-notes.md`; both are now confirmed not only by arithmetic but by measurement in two independent pairs of cells.

Agreement with an independent estimate: -13.1 GB per request from the engine's counter against -10.7 GB per
request from `rxpci` (dmon, 1 s sampling = a lower bound) - an 18 % difference, within the method; the
direction and order of magnitude agree in both configurations.

## 4. The noise threshold of a run (the main product of validating the method)

### 4.1 The repeated base A0r against the base A0 (same config, another session, 40 minutes and three service restarts later)

| length | A0 | A0r | delta | within-row spread (3 repeats) |
|---|---|---|---|---|
| 3 971 | 872.6 | 868.9 | **-0.42 %** | 1.19 % (A0), 0.96 % (A0r) |
| 31 307 | 1897.8 | 1895.2 | **-0.14 %** | 0.47 % / 0.42 % |
| 122 087 | 2428.0 | 2426.5 | **-0.06 %** | 0.19 % / 0.19 % |
| gen-only | 200.3 | 200.8 | +0.25 % | 1.35 % / 2.19 % |

### 4.2 A second cell of the same configuration (A1 against A0)

-0.94 % at 4k, -0.12 % at 32k, -0.08 % at 122k. B1 against B0: +0.82 % / +0.11 % / 0.00 %.

### 4.3 Conclusions about thresholds (they update test-plan.md §2 and patch-notes.md §8.8)

1. **prompt tok/s:** session-to-session noise **<= 0.42 % at 4k and <= 0.14 % / <= 0.06 % at 32k / 122k**.
   The |delta| > 1 % threshold is right at lengths >= 32k and **does not have enough margin at 4k** (there the
   within-row spread reaches 1.19 % and the session-to-session 0.94 %): this method cannot support a claim
   that "the change improved 4k by 1 %". Recommendation: require |delta| > 2 % at 4k, or draw no conclusion at 4k.
2. **decode:** session-to-session spread **up to 7.8 %** with an unchanged config, so a 5 % threshold **does not
   work** as a criterion. The decode column should be used only as a "must not get worse" control with a 10 %
   threshold and with draft acceptance and hit rate recorded (they explain the swing).
3. **`auto` does not drift in its split choice within one profile:** A0, A1 and A0r all chose the same
   `K=11,24,36` and the same 5632 slots; the layout was stable in this run. Whether `auto` drifts is a question
   of a changing access profile (experiment E1), not of search randomness.

## 5. Bit acceptance: it works, and a new limit was found

### 5.1 Results

| cell | config | stable pair | `text_sha` at 1024 | `text_sha` at 8192 | verdict |
|---|---|---|---|---|---|
| A0 | auto | requests 2-3 | `1080a176…` | `50246f69…` | PASS |
| A1 | auto | 2-3 | `1080a176…` (the same) | `50246f69…` (the same) | PASS, ref_match = true |
| A0r | auto | 2-3 | `1080a176…` (the same) | `50246f69…` (the same) | PASS, ref against B1 = false (expected: B1 is another config) |
| B0 | 12,25,37 | 1-2 | `1080a176…` | **`caf2b35b…` (different)** | PASS |
| B1 | 12,25,37 | 1-2 | `1080a176…` (the same) | `caf2b35b…` (the same) | PASS, ref_match = true |

**Confirmed:** the text of greedy generation is reproducible **across service restarts** with the same config
(A0 = A1 = A0r, bit-for-bit, at 1259 and 10 012 prompt tokens). This matters: the criterion "a transport has no
right to change bits" is applicable when the reference was taken in the same placement configuration.

### 5.2 Finding F-8: changing `--layer-split` changes the output on a long prompt

The same fixed prompt (`prompt_sha 736928ee…`, `prompt_n = 10 012`), `temperature = 0`, EOS banned, 128 tokens:

- `auto` (A0/A1/A0r): 647 characters, sha `50246f69…`
- `12,25,37` (B0/B1): 530 characters, sha `caf2b35b…`
- **a common prefix of 32 characters out of 647 (4.95 %)**; at a 1259-token prompt the texts are fully identical.

Status: **confirmed as an observation** (two independent sessions per configuration, each bit-reproducible
internally). **The mechanism is not established (hypothesis).** What measurement rules out: `prompt chunk auto:
8192 tokens, a 96-slot ring`, the context, KV int8, spec, MTP and control vectors are all the same;
`stage_trim` (`generate.cpp:3069-3082`) changes only which dense weights are loaded into VRAM, not the
computation. The hypothesis the counters point to: at 10 012 tokens expert streaming is armed
(`stream_all_min() = 1024`), and the share of experts arriving through a ring slot rather than a cache slot
differs (97 % against 99 % residency, hit rate 99.5…99.8 % against 100.0 %) - different service paths for the
same blobs. The check for someone who wants it: compare logits at the first diverging token and establish
whether experts served from a ring slot give bit-identical results to the same experts from a cache slot.

**The consequence for accepting a change (more important than the mechanism itself):** the "common prefix at
least 95 %" criterion for placement changes is **not met by the real difference between the two layouts**
(4.95 %). Therefore:
* for P1-P4 (transport) the bit criterion remains valid - with an unchanged placement configuration;
* for a placement change bit identity is unreachable, and **any change that alters the auto decisions must pass
  an output-quality check, not only a speed one**: the "prefix at least 95 %" comparison should be replaced by
  recording the text itself and evaluating it (or comparing logits), otherwise a speed-up could be bought with a
  different answer. This also bounds the practical side of staying on `auto`: moving to a pinned layout changes
  more than speed.

## 6. Limits and corrections of the method

### 6.1 Two defects of the stand found along the way (fixed)

1. **Acceptance demanded stability too early.** The first version of `ab-text-check.py` (1 warm-up + 2
   measurements) diverged in A0 (a common prefix of 559 out of about 630 characters) while the engine's own
   lines showed the expert-cache hit rate rising 98.8 % -> 99.5 % -> 99.8 % over those same requests. The
   engine's state warms over 2…3 requests, not one. Fixed: requests are repeated until two consecutive
   `text_sha` match (at most 6), the first are warm-up, and the stable pair is the cell's fingerprint. Checked
   against a live service: `raw/ab/selfcheck-textcheck-0548/` (PASS, stability at 2-3). In all five cells
   stability actually arrived after 1-2 warm-ups.
2. **The VRAM threshold in `ab-ready.py` was tuned to the auto layout.** A threshold of 14 000 MiB per card
   fails for the pinned `12,25,37`, where CUDA0 legitimately holds 13 744 MiB (baseline-profile.md §8.2). Cell
   B0 would have been marked FAILED with a fully working service (health 200, chat answering, hit rate 100 %).
   Fixed: "the model is loaded" means **a sum of at least 57 000 MiB and every card at least 13 000 MiB**
   (auto: 61 712 MiB total; 12,25,37: 59 862; a wrapper without the engine: 28 MiB). Checked against a live
   service with a pinned split: READY in 0 s.

### 6.2 The run is assembled from two attempts

`A0`/`A1` come from the first attempt (`raw/ab/2026-10-17T054922Z-validation2/`, 05:49-06:02Z), `B0`/`B1`/`A0r`
from the second (`raw/ab/2026-10-17T061723Z-validation3/`, 06:17-06:37Z); the `A0`/`A1` directories were copied
into the second attempt's directory, the order `A0 -> A1 -> B0 -> B1 -> A0r` is preserved, and build identity is
confirmed by two manifest files (identical sha256 for the binary and the cell configs). The reason was the
stand fixes between the attempts (§6.1). Full description: `MERGE-NOTE.txt` in the run directory. The
intermediate attempt `2026-10-17T054547Z-validation` (05:45Z) was interrupted: A0 did not pass acceptance under
the old rule.

### 6.3 How long the window took

The window took 05:43:44Z…06:38:13Z, about **55 minutes** (about 40 had been agreed). Of that, the service was
up for about 44 minutes (on the stand's configurations and under the stand's load); port 8080 was unavailable
only in the pauses between cells (about 25 s x 6). One cell: stop about 22 s + start-up 61 s + acceptance about
40-70 s + benchmark about 5.5 minutes + extracting counters. Returning the working config was verified by the
three signs (READY in 61 s, VRAM 15124/15510/15106/15972) and additionally by an integration test
`T11_long_context_determinism` (prompt_n 80 999, bit-matching sha over two repeats) -
`raw/correctness/2026-10-17T063813Z/`.

### 6.4 The "125k" length does not match the baseline profile's length

The stand fixes `4k,32k,125k`; `baseline-profile.py` turns `125k` into a target of 128 000 -> an actual
`prompt_n = 122 087`, while `baseline-profile.csv` was taken at a target of 131 072 -> `prompt_n = 125 011`.
That is why the "delta against the baseline line" column for that point in `summary.*` is None, and the
comparison +6.66 % (here) against +7.33 % (§8.2) is only correct with a correction for length (a shorter
prompt -> a smaller gain in the streaming regime). **Correction for the future:** state `131072` instead of
`125k` in the test plan so the ladder matches `baseline-profile.csv` (done in test-plan.md §2).

### 6.5 What this run does not show

* The effect of any change: there is no change, all cells are one binary. Only the stand's ability to measure a
  difference and reproduce it was tested.
* Whether `auto` drifts under a different access profile (that is experiment E1, and needs separate agreement).
* The route of traffic directly (there are no NVLink counters) - the conclusion rests on the engine's counters and `rxpci`.
* The stability of the decode column as a measure (on the contrary: it was shown not to be one).

## 7. Added in the same window: the direct kernel-access probe (pairs with GPU3 and the 256 MiB point)

Taken with the service stopped (`raw/peer-read/2026-10-17T054509Z/`, 576 rows, **0 correctness errors**, all 12
ordered pairs, 1…256 MiB, `--blocks 5`, private nvcc 12.8 + g++-14, `-arch=sm_70`). This closes the last gaps
of the P2P work (earlier only 3 of 6 pairs were reachable, and the ladder stopped at 128 MiB).

| size | `peer_read` (a kernel reads a peer) | share of `memcpy_peer` | `peer_read_write` (a copy inside a kernel) | `peer_write` | `memcpy_peer` |
|---|---|---|---|---|---|
| 32 MiB | 32.13 GB/s | 0.838 | 36.41 | 36.46 | 38.37 |
| 64 MiB | 34.42 | 0.892 | 36.53 | 36.56 | 38.58 |
| 128 MiB | 35.64 | 0.922 | 36.60 | 36.62 | 38.68 |
| **256 MiB** | **36.25** | **0.936** | **36.63** | **36.64** | **38.75** |

* **Pairs with GPU3 are indistinguishable from GPU0-GPU2 pairs**: the median over 12 pairs and the median over
  the 6 pairs involving GPU3 agree within 0.06 GB/s (for example `peer_read` at 256 MiB: 36.25 against 36.25).
  The earlier restriction was about occupied VRAM, not a property of the card.
* **The 256 MiB point:** the plateau continues (+1.7 % over 128 MiB), and the kernel read's share of a copy
  grows (0.922 -> 0.936). The conclusion in favour of **P2b** (a kernel reading a peer's VRAM) is unchanged and
  strengthened: the hand-off payload (335.5 MiB) lies above the 90 % plateau threshold.
* Opposing kernel reads on one pair (all 6 pairs): **65.1…65.14 GB/s** in total (32.56 + 32.56); a kernel read
  plus an opposing copy: **51.4…51.5 GB/s**. The earlier three-pair figures (64.0 / 50.8) were reproduced.
* **A caveat on the `local_read` column:** the same kernel on its own memory gives 362 GB/s at 128 MiB and
  519 GB/s at 256 MiB - above the rated HBM2 bandwidth of V100 (300 GB/s). The column is usable only as "the
  kernel is not the limiter" and never as a memory-bandwidth measurement; it is not part of any conclusion.
