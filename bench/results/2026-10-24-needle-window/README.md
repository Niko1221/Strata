# 2026-10-24: needle recall, int8 against the production window

Same-day pair. Fresh server per arm. Retrieval only: both arms found every needle, so the test shows no regression and no separation. It does not replace the workpackage's three-arm gate (plain, 256/64, 512/128, 6/6 each, 18/18).

## Machine and build

- GPU: AMD Radeon RX 9060 XT (gfx1200). Windows. HIP runtime `amdhip64_7.dll` (ROCm 7.1 on the library path).
- Engine: `build-hip-port\strata.exe`, session line `engine 0.1.47`. Started 16:08 (int8) and 16:22 (window). The logs are stored as `.txt` because `*.log` is gitignored.
- Weights: Swift 1.5 IQ3_XXS. `--spec 4`, `--spec-min-p 0.5`, `--mtp`, `--prefill auto`, `--max-context 133120`.
- Both arms set `STRATA_PREFILL_CPU_SHARE=0`.

## Arms

| Arm | KV | Window |
|---|---|---|
| int8 | `--kv int8` | off (no window line in the log) |
| window | `--kv q4lm8v` | on |

The window arm's config sets `STRATA_KV_WINDOW=1`, `STRATA_KV_WINDOW_S=256`, `STRATA_KV_WINDOW_R=64`. The server copies a config's `env` into the engine process (`serve/server.py`, `child_env`). The engine log does not print S and R. What it does print, and the int8 log does not, is:

```
strata: KV window ring sized for the prompt chunk (6144 cells): STRATA_KV_WINDOW_BATCH 512 -> 6144
```

That line is the window turning on and the automatic chunk being capped at 6,144. S=256 and R=64 are the config values, not a printed confirmation. The code defaults, if those two variables were missing, would be 128 and 8 (`src/core/layer.cpp` on `KV_QUANT_research`).

## Protocol

`tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`. One code word per (length, depth). Greedy, thinking off, max 40 tokens. The haystack is this repository's text files in a fixed order; the words come from `random.Random(7)`. The two arms therefore saw the same six prompts. A hit is the planted word occurring in the answer. Here each answer is the word itself.

Prompt tokens, both arms: 31,643 / 31,643 / 31,644 at 32K, and 126,017 / 126,015 / 126,015 at 128K.

## Result

6/6 found on int8, 6/6 found on the window. Same words, same answers.

| Length | Depth | Word | int8 | window |
|---|---:|---|---|---|
| 32k | 10% | meadow-copper-504 | found | found |
| 32k | 50% | falcon-quartz-940 | found | found |
| 32k | 90% | willow-cobalt-696 | found | found |
| 128k | 10% | falcon-saffron-138 | found | found |
| 128k | 50% | quartz-tundra-528 | found | found |
| 128k | 90% | quartz-glacier-192 | found | found |

## What these files are not

- Not a throughput result. The JSON `seconds` are one-shot request wall times. The third 128K probe is about 133 s on both arms; the first two are about 232–235 s on both arms. Same shape on both arms, so it is not a window effect, and it is not used as a speed claim.
- Not the Table 4 expert-cache A/B. The logs print slot counts after the Windows reserve shrink (int8 2,923, window 3,484). Different session, different build label from the 0.1.44.1 runtime table. Not used.
- Not a quality (KL) measurement, and not a second task family. One needle type, one probe per cell, no repeats.
