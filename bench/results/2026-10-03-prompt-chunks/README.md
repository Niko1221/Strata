# Chunks by prompt size

**Hardware:** RTX 5070 Ti 16 GB on PCIe 3.0 x16 (the board's limit), Ryzen 9 5900XT, 64 GB DDR4-2133, Windows 11.

**Engines:** v0.1.38 (99f3dbd), and v0.1.38 with this change. Both were built from source for sm_120 and run behind
the v0.1.38 server.

**Model and settings:**
- IQ3_XXS with a 400K context (YaRN 1.5625).
- int8 KV with 32,768 cells in VRAM, and the MTP draft.
- `STRATA_PF_FUSED=1`, images off.
- `--expert-cache auto --vram-reserve-mib 1600 --pcie-frac 0.35 --spec-min-p 0.70`, giving 4,170 expert slots.

**Method:**
- Every request goes through `serve/server.py`, with a fresh server for each run.
- The runs went in this order: v0.1.38, the change with `--prefill auto`, the change with `--prefill auto:16384`. Then
  all three ran again.
- Each run sends a 2K warm-up, then prompts of this repository's code from 9K to 100K tokens. Each prompt has its own
  first line, so nothing is reused. Each asks for 16 tokens out, greedy, with thinking off.
- The speed is the engine's own figure for the prompt, from its log line. It is the mean of the two rounds, which
  differed by less than 0.5%.

Every request is in [`ab.json`](ab.json).

## Prompt reading (tokens/s)

| Prompt (tokens) | v0.1.38 | this change, `--prefill auto` | this change, `--prefill auto:16384` |
| --- | ---: | ---: | ---: |
| 9,010 | 1,511 | 1,517 (+0.3%) | **2,410 (+59%)** |
| 16,402 | 2,289 | 2,303 (+0.6%) | 2,449 (+7%) |
| 20,036 | 2,101 | 2,148 (+2.2%) | **2,898 (+38%)** |
| 26,700 | 2,143 | 2,172 (+1.4%) | 2,783 (+30%) |
| 32,716 | 2,570 | 2,583 (+0.5%) | 3,120 (+21%) |
| 40,013 | 2,546 | 2,547 (0.0%) | 3,088 (+21%) |
| 65,426 | 2,597 | 2,602 (+0.2%) | **3,392 (+31%)** |
| 100,117 | 2,428 | 2,451 (+1.0%) | **3,275 (+35%)** |

- **`--prefill auto:16384`:**
  - v0.1.38 tries 16,384, which does not fit in this cache, and falls back to 8,192.
  - With the change, auto finds 13,312.
  - The 9K prompt then reads in one chunk instead of two.
  - From 20K on, the change is 21-38% faster.
- **`--prefill auto` (what setup writes):**
  - The limit is 8,192, as before.
  - Equal chunks change only the prompts whose last chunk would be 1,024 tokens or more and smaller than the others:
    - 20,036 tokens read as 3 × 6,912: +2.2%.
    - 26,700 tokens read as 4 × 6,912: +1.4%.
    - 100,117 tokens read as 13 × 7,936: +1.0%.
  - The other sizes are within 0.6%.
- **A short last chunk stays:**
  - A chunk under 1,024 tokens moves only the experts its own tokens route to; a larger one streams every expert the
    GPU does not hold. So 16,402 tokens still read as 2 × 8,192 + 18.
  - A first version of this change split them into 3 equal chunks. It read 22% slower: 1,791 tok/s.
  - 26,700 tokens with 13,312-token chunks read as 2 × 13,312 + 76 for the same reason.

## Answers

- **Each run is repeatable:** each build gave the same 16 answer tokens in both rounds.
- **One exception, in v0.1.38 itself:**
  - Its two runs answered the 100K prompt differently.
  - Those runs started with 4,171 and 4,163 expert slots, because a few MiB more VRAM were in use the second time.
  - The answer tokens can change with which experts the cache holds.
- **Read alone,** right after the warm-up, the 65K prompt gives the same 16 tokens on both builds.
- **Needles:** 9 of 9 found with the change and `--prefill auto:16384`. The test is `tools/needle_bench.py` at 32K,
  128K and 262K, depths 10, 50 and 90. Every result is in [`needles.json`](needles.json).
