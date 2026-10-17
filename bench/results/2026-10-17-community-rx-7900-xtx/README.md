# Community benchmark on AMD Radeon RX 7900 XTX 24 GB on Windows 11: Flash-Next IQ2_XS and Coder IQ1_M at 262,144 tokens

Measured on 2026-10-17 by @IREvil. Two Flash-Next quantizations run at the model's native
262,144-token context on a **64 GB** machine — the common size, not a large-RAM workstation — on Windows with the
ready-made AMD engine (no source build). There are three parts: the standard `tools/strata_bench.py --long` suite
per model, a separate long-context sweep out to 255K prompt tokens, and a HumanEval correctness check.

The main finding beyond the speed numbers: on this machine the **Windows commit limit, not RAM or VRAM, was what
capped the GPU expert cache**, and the engine said so only in its log. Raising the page file from 48 GB to 64 GB
removed a documented silent-crash warning and let the cache more than double. Details under
["The commit limit capped the expert cache"](#the-commit-limit-capped-the-expert-cache).

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GiB (24560 MiB; Windows budgets 23748 MiB of it to the process).
  An integrated Radeon (gfx1036) is present; setup listed it as unsupported and used the 7900 XTX.
- CPU: AMD Ryzen 9 9950X3D, 16 cores / 32 threads, AVX-512.
- Installed RAM: 61.4 GiB. Page file: **64 GB, fixed, on a second NVMe** (it was 48 GB when the runs started — see
  the commit-limit section).
- Storage: model files and the install on a 1908 GB NVMe SSD.
- OS: Windows 11 Pro (10.0.26300). GPU driver: 32.0.32015.2008.
- ROCm: none installed system-wide. The engine carries its own — ROCm 10.2.0a20260930 from AMD's TheRock builds,
  in `engine\rocm\bin` — and setup verified the ready-made zip's SHA-256 against GitHub's.
- Strata: 0.1.44.1, commit `296b59d2`; engine 0.1.44.1 (the official prebuilt Windows HIP zip).
- Background load: no game and no browser during the published runs. A Minecraft client was running during some
  of the exploratory measurements and is called out where it matters. No power limits set; Balanced power plan.

## Model and configuration

| Folder | Model | Quant | Experts cached | Engine |
| --- | --- | --- | ---: | --- |
| [`iq2_xs/`](iq2_xs/README.md) | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | IQ2_XS | 9903 slots, 13.26 GiB | 0.1.44.1 |
| [`coder-iq1_m/`](coder-iq1_m/README.md) | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` | IQ1_M | 6673 slots, 12.69 GiB | 0.1.44.1 |

Both use the full-model MTP draft layer packed by setup (`tools/mtp_fetch.py`, `tools/mtp_pack.py`). The Coder has
half the experts of the original, so the same VRAM holds a larger *fraction* of its expert set (6673 of ~12288 vs
9903 of ~24576), which is why it prefills faster despite the lower quantization label.

Settings are the install's own, unchanged, with `--max-context 262144` (the model's trained length; no rope
scaling), `--kv int8`, `--kv-resident 32768` (KV streaming on: 3.09 GiB of pinned RAM), `--expert-cache auto`,
`--prefill auto`, `--spec 4 --spec-min-p 0.5`, MTP on. Vision is off — the AMD backend has no image encoder yet
(`docs/AMD_HIP.md`). Calibration is not offered on AMD; the experimental speed projection is off.

```text
strata.exe --serve --pack <data>\packs\iq2_xs
  --native <data>\models\IQ2_XS\Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
  --ple-gguf <data>\models\IQ2_XS\Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
  --expert-profile <repo>\data\expert-profile.bin --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --mtp <data>\mtp\rt
  --max-context 262144 --kv int8 --kv-resident 32768
```

The exact launch line of each model is in its folder's `README.md` and `results.json`.

## Results

Stock suite (`python tools/strata_bench.py --config strata-<model>.json --long`), medians over 3 runs per test
(9 requests for chat), 262,144-token context, thinking off, temperature 0.

| Model | Chat decode tok/s | 4K prompt tok/s | 16K prompt tok/s | 32K prompt tok/s | Agent decode tok/s | Chat first token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Flash-Next IQ2_XS](iq2_xs/README.md) | **80.2** | 1,198 | 1,367 | 1,279 | **90.2** | 0.34 s |
| [Coder IQ1_M](coder-iq1_m/README.md) | 65.6 | **1,568** | **1,631** | **1,567** | 83.4 | 0.41 s |

Ranges and per-request data are in each folder. The two quants trade places: the Coder ingests prompts
**24-33 % faster**, IQ2_XS decodes **22 % faster**. Both hold a 68.6-68.7 % conversation-cache hit over the
agent session (5 turns, ~14.5K prompt tokens at the last turn), i.e. re-reading ~2.9K tokens per turn.

### Long-context behaviour to 255K

A separate sweep ([`long-context/`](long-context/README.md)) with the same request shape out to the full window.
These numbers come from a different session and a different prompt construction than the table above; compare them
with each other, not across tables.

| Model | 4K prefill | 32K | 128K | 255K | Decode 4K → 255K |
| --- | ---: | ---: | ---: | ---: | --- |
| Coder IQ1_M | 1,378 | 1,631 | 1,526 | **1,346** tok/s | 68.1 → 63.7 tok/s |
| IQ2_XS | 1,068 | 1,288 | 1,289 | 1,158 tok/s | 79.5 → 68.3 tok/s |

Prompt throughput is close to flat to 128K and only ~10 % down at 255K; decode loses 7-14 % across the whole
range. A 255K-token prompt is ingested in **3.2 min (Coder)** / **3.7 min (IQ2_XS)**, first token at 189 s / 220 s.
That is the practical argument for KV streaming at this memory size: the K/V sits in 3.09 GiB of RAM and the VRAM it
frees goes to experts.

## The commit limit capped the expert cache

This is worth knowing for any 64 GB Windows machine with this card. Windows counts both RAM and page file as
*commit*, and a WDDM VRAM allocation such as the expert cache must be backed by commit. With the 48 GB page file
this machine started with (commit limit **108.3 GiB**), the engine wanted 11083 expert slots and `cudaMalloc`
failed four times in a row:

```text
ExpertCache: cudaMalloc(15.59 GiB) failed: out of memory (Windows has 5.3 GiB of commit left: RAM + page file)
             ... 8733 ... 6543 ... 4902 ... 3667 slots
WARNING the Windows commit limit is 108.3 GiB (RAM + page file) and 0.5 GiB of it is left ... it can end at the
        limit with no message (exit code 0xC0000409, #1607) while RAM is still free.
```

so it settled at **3667 slots (4.93 GiB)** — under a third of the 24560 MiB card — and printed a warning that the
engine could die silently while RAM was still free. After the page file was raised to **64 GB (commit limit
123.9 GiB)** the warnings disappeared and, with the desktop idle, the cache reached **9903 slots (13.26 GiB)**.

Two practical notes for Windows AMD users:

- The symptom is not "out of memory" to the user; the engine runs, just slowly, with a third of the card's
  potential cache. It shows up only as `cudaMalloc ... commit left` lines in `strata-<model>.log`.
- Setup's own page-file tips and #1607 cover this; on a 64 GB machine a fixed page file of 64 GB rather than 48 GB
  was the difference between 6.57 GiB and 13.26 GiB of expert cache here.
- Anything else holding VRAM competes directly: with an 8.5 GB Minecraft client running, the same 64 GB page file
  still only reached **6530 slots (8.75 GiB)** and 4K prefill fell from 1,198 to 759 tok/s. Close games first.

## Correctness: HumanEval

[`humaneval/`](humaneval/README.md) — the first 82 HumanEval problems, greedy, thinking off, 512-token cap, each
candidate executed against the canonical tests in a subprocess with a 10 s timeout.

| Model | pass@1 | Wall time |
| --- | ---: | ---: |
| Coder IQ1_M | **82/82 (100 %)** | 288 s |
| Flash-Next IQ2_XS | 81/82 (98.8 %) | 284 s |

With 82 problems the 95 % interval on a single score is roughly ±4 points, so the two are statistically
indistinguishable; the point is that a ~2-bit quantization of this model is a competent coder on this hardware.
The single miss was HumanEval/81 (an assertion failure).

## Method

- `python tools/strata_bench.py --config strata-<model>.json --long --out <dir>`, one model at a time. Each run
  started its own server from the install configuration; model loading is not timed. One warm-up chat and a 1K
  prompt are discarded first, and every prompt carries a unique tag so the cache cannot help fresh-prompt tests.
- The long-context sweep used a lightly adapted copy of `bench/bench_vs_llama.py` ([`long-context/bench3.py`](long-context/bench3.py));
  the request body and the server-side `timings` fields are unchanged from it. Fresh prompts of exactly 4K / 32K /
  128K / 255K tokens built from 8 MB of the repository's own docs and sources, 256-token cap, one run per cell.
- Speeds are the engine's own timings (`prompt_per_second`, `predicted_per_second`); first-token times are measured
  by the client over HTTP.
- The machine was kept awake for the whole series; the published runs are the ones taken with no game or browser
  running.

## Limitations

- One benchmark session per model. Single decode runs vary widely (see the ranges), so differences of a few percent
  between the two quants are not meaningful; the prefill gap between them is.
- The long-context sweep is one run per cell and a different session from the stock suite. Treat it as ±5 %.
- HumanEval is one small correctness check, not a measure of agentic or long-context quality. No needle/recall test
  and no tool-calling evaluation were run here.
- Windows AMD limits what could be tested: images are not available (no image encoder in the Windows AMD engine),
  one card per model, no calibration, no `rocm-smi` (so no VRAM/power telemetry was captured).
- No power or clock figures; the Balanced power plan was left as-is.

## Appendix: a dense 27B on the same card (not Strata)

For readers wondering about a dense model of similar size, `Qwen3.8-27B` (`qarch=qwen35`, native 262,144 tokens)
was run through **llama.cpp b11207 (HIP)**, IQ4_XS, `-ngl 99 -c 262144 -ctk q8_0 -ctv q8_0`: prefill 330 / 279 /
180 tok/s and decode 30.9 / 20.5 / 10.0 tok/s at 4K / 32K / 128K, and it **did not finish a 255K-token prompt
within a 3600 s timeout**. At 24 GB the dense model must hold 14.5 GB of weights plus ~8.7 GB of KV at this
context, which is most of the cause; `llama-bench` at its default context reports 984 tok/s prefill and 42.6 tok/s
decode for the same file. It is included only as context — it is not a Strata measurement and is not part of the
tables above.
