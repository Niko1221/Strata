# Inkling-2-Mini turbo (opt-in tail skip)

Two opt-in presets for Inkling-2-Mini on NVIDIA (CUDA) and AMD (HIP) cards. **Both are off by default and never the default.** They skip
some of the experts the model would read for a word, so the card reads less per word. **Answers differ slightly from exact** with either one.

Inkling-2-Mini reads 8 of its 128 experts per layer for every word. The router gives each of the 8 a weight. The experts with the smallest
weights change the answer least, so the preset leaves them out. Speed only improves where reading experts is what takes the time, that is on
cards that hold only part of the model and fetch the rest from RAM or the SSD (8 to 12 GB cards), and on unified-memory machines.

## How to turn it on

| Where | Turbo | Turbo max |
| --- | --- | --- |
| Terminal | `strata launch --turbo` | `strata launch --turbo-max` |
| Same, written by hand | `--set STRATA_INK_TAIL_SKIP=turbo` | `--set STRATA_INK_TAIL_SKIP=turbo-max` |
| Engine | `strata-inkling --turbo ...` or `STRATA_INK_TAIL_SKIP=turbo` | `strata-inkling --turbo-max ...` or `STRATA_INK_TAIL_SKIP=turbo-max` |

Take the flag away (or set `STRATA_INK_TAIL_SKIP=off`) and the answers are exact again. A value you set yourself in `STRATA_INK_TAIL_SKIP`
wins over the flag. The Terminal shows a warning when a preset is on. The app does not have a switch for it. Metal, SYCL and CPU builds do
not have it: the engine says so at the start and the answers stay exact there. On HIP it applies to the one-token and verify-window decode
(the direct expert path).

## The two presets

| | Turbo | Turbo max |
| --- | --- | --- |
| What is left out | the 7th and 8th expert when their weight is 8 percent or less, in the last pass only (visits 30 to 41), the rest are not renormalised | the 6th to 8th expert when their weight is 8 percent or less, in every visit, the rest renormalised |
| `STRATA_INK_TAIL_SKIP` expands to | `thr=0.08,rank=7,visits=30-41,cap=0,renorm=0` | `thr=0.08,rank=6,visits=all,cap=0,renorm=1` |
| Expert reads skipped (measured) | about 3 percent | about 16 percent |
| Difference from exact answers (KL, measured) | about 1e-3 (agent steps 1.0e-3, code 1.3e-3, maths 1.3e-3) | about 1.3e-2 to 2.6e-2 |
| Next word the same as exact (measured, the model's own text) | 99.0 to 99.2 percent | 96.5 to 97.3 percent |
| Speed, tokens per second, plain decode (measured) | RTX 5070 12 GB: +1 percent; RTX 5060 8 GB: +0.4 to +3 percent; RTX 4090 (everything on the card): +1 percent | RTX 5070 12 GB: +22 to +23 percent; RTX 5060 8 GB: +18 to +21 percent; RTX 3070 8 GB: +9 percent; RTX 4090: +5 percent |

Plain words: **turbo is a very small step**: the answers stay very close to exact, and the speed gain is about one percent, so you will
hardly notice it. **Turbo max** is the one that is faster on cards that hold part of the model (around a fifth faster on a 12 GB card),
and its answers drift further from exact (about one word in 30 differs from what exact would have written next, and the text can then go a
different way). In our test suite (tool calls, agent loops, coding, memory, maths, needle documents up to 64K tokens, 24 picture questions) the
number of passed items stayed within one or two items of exact for both presets, so we did not see a loss there, but the test suite is
small; the difference in the measured KL is real.

Strix Halo and other unified-memory machines were not measured: reading experts is almost all of the time there, so the speed gain
should be close to the share skipped (about 3 percent and 16 percent) (computed, not measured).

Measured with the speed tests of `phaseA-tests/rel0223/tailskip/RESULTS.md` (same binary in every arm, alternating, 3 to 5 rounds); the
numbers differ by a few percent between runs on the hybrid cards.

## Other settings (advanced)

`STRATA_INK_TAIL_SKIP` also takes a list of keys: `thr` (the weight limit, 0.08 is 8 percent), `rank` (the first rank that may be left out, 2
to 8), `visits` (`all`, or ranges such as `30-41` or `0-5+42-47`; visits 0 to 5 are the first layers, 6 to 17, 18 to 29 and 30 to 41 the three
passes through the repeated layers, 42 to 47 the last layers), `cap` (the most weight left out per layer, 0 = no limit) `renorm`
(1 = the remaining weights are renormalised) and `cold=1` (only experts that are not on the card are left out). Examples and measurements of other settings are in `phaseA-tests/rel0223/tailskip/RESULTS.md`:
the first pass (visits 18 to 29) is the most sensitive; leaving out experts in all visits costs far more than in the last pass only.

Ideas that were tested and are **not** part of the presets: a per-layer correction for the left-out experts (`sq.tail_comp`, the table
that ships as zeros) lowered the difference from exact by 0 to 6 percent for the presets' kind of setting (up to 19 percent for a setting that
leaves out only experts that are not on the card), so it does not allow skipping more; swapping a cold expert for a fixed resident one
(`sq.resident_fallback`) gave a larger difference than leaving the expert out and was not faster. `cold=1` (leave out only experts that are not
on the card, for cards that hold part of the model) removes 25 to 35 percent of the reads from RAM or SSD in the last two passes for a KL of
about 2e-3 to 4e-3 and was +4 to +6 percent faster on an 8 and a 12 GB card; it is available as an advanced setting and is not a preset. The numbers
are in `phaseA-tests/rel0223/tailskip/RESULTS.md`, section 10.
