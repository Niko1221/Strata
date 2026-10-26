# Two prompt switches on RTX 40 cards: STRATA_PROMPT_ATTN_IMMA and STRATA_HC_UPMIX (2x RTX 4060 Ti, sm_89)

Measured on 2026-10-26 on my own machine, with Strata 0.2.6 built from source (tag `v0.2.6`, `4b9e05a2`).

I run Strata for a small office: the assistant reads long documents, so for us the time to read the prompt matters
more than the writing speed. Both switches are in DETAILS.md as CUDA opt-ins measured on sm_86 (RTX 3060 and A4000).
I wanted to know what they do on an Ada card, where FP16 with FP32 accumulation runs at half rate on GeForce. On these
cards cuBLAS gives me about 45 TFLOPS in that mode, against 74 with FP16 accumulation and 151 in INT8. IMMA moves the
prompt attention to INT8 tensor cores, and UPMIX saves a full FP32 round trip through VRAM in the hyper-connection read.

**Short version:** with both switches on, prompts read **4.8% faster at 20K and 6.1% at 69K** in my A/B, and Niko's
`bench.sh` shows **+5.3% at 16K and +4.5% at 32K**. Nothing changes at 4K or in decode. UPMIX was bit-exact on every
check it ran. IMMA passes Strata's own parity test, but its answers are not bit-identical (expected, it changes the rounding).

## Machine and settings

- 2x RTX 4060 Ti 16 GB (AD106, 165 W limit, PCIe 4.0 x8 each, no P2P), Threadripper PRO 3975WX (32 cores, AVX2),
  128 GB DDR4, Pop!_OS 24.04, NVIDIA driver 610.57.04.
- Engine built here for sm_89 with CUDA 12.6, Release, default CMake options, ggml from the pinned llama.cpp `3cf03257`.
- Unsloth UD-IQ4_XS, 262,144-token context, layer split `auto` (CUDA0 layers 0-22, CUDA1 23-47), experts partly in
  VRAM, `--kv int8 --kv-resident 32768 --spec 4 --mtp`, vision on the CPU. The full config is in
  [strata-config.json](strata-config.json) (API key removed).
- Nothing else ran on the GPUs. One request at a time.

## How I measured

**A/B on four configurations**, each a fresh server start with the same config. Only environment variables change:

| | variables |
| --- | --- |
| A | none (baseline) |
| B | `STRATA_PROMPT_ATTN_IMMA=1` |
| C | `STRATA_HC_UPMIX=1` |
| D | both |

All four also had `STRATA_PREFILL_TIMING=1`, and C and D had `STRATA_HC_UPMIX_CHECK=4`. I ran A B C D, then D C B A,
to keep heat and drift from looking like an effect. In each start: one 2K warm-up, then twice a fresh ~20K and a fresh
~69K prompt (Strata's own docs, a random number in front so no cache helps, 1 output token), two 400-token decodes at
temperature 0, and one fixed 20K prompt with a short answer (thinking off) to compare the text. Power and clocks were
sampled with `nvidia-smi` every 250 ms. Speeds are the engine's own `timings`.

**Niko's `bench.sh --long`**, run once on the baseline and once with both switches, each on a freshly started engine.

## Results

Prompt read, tok/s, median and [min-max] of 4 requests:

| | ~20K | ~69K |
| --- | ---: | ---: |
| A baseline | 1,616 [1,578-1,626] | 2,106 [2,080-2,128] |
| B IMMA | 1,653 [1,636-1,656] (+2.3%) | 2,178 [2,150-2,186] (+3.4%) |
| C UPMIX | 1,665 [1,661-1,669] (+3.0%) | 2,192 [2,190-2,193] (+4.1%) |
| D both | 1,694 [1,691-1,741] (+4.8%) | 2,235 [2,231-2,239] (+6.1%) |

I only count an effect where its range does not touch the baseline's. All six do.

The engine's prompt timer shows where the time went (ms per 1,000 prompt tokens, both GPUs together):

| | hc read | qsa attn | everything |
| --- | ---: | ---: | ---: |
| A baseline | 40.0 | 26.9 | 404.5 |
| B IMMA | 39.9 | **18.9** | 392.0 |
| C UPMIX | **26.0** | 26.8 | 389.0 |
| D both | **26.1** | **18.9** | 380.7 |

So each switch does exactly its own job: qsa attn -30% with IMMA, hc read -35% with UPMIX, and they add up.

`bench.sh --long`, medians of 3 runs:

| | baseline | both switches |
| --- | ---: | ---: |
| 4K prompt tok/s | 872.7 | 879.2 (within noise) |
| 16K prompt tok/s | 1,670.0 | 1,758.6 (+5.3%) |
| 32K prompt tok/s | 1,956.0 | 2,043.3 (+4.5%) |
| 32K first token | 16.45 s | 15.74 s |
| chat decode tok/s | 55.4 [48.6-75.6] | 52.0 [47.6-75.7] (within noise) |
| agent cache hit | 69.1% | 69.1% |

Full reports: [bench-sh-baseline](bench-sh-baseline/README.md), [bench-sh-imma-upmix](bench-sh-imma-upmix/README.md).

Decode in the A/B: 60.9 / 62.0 / 60.2 / 61.7 tok/s (A/B/C/D), MTP drafts accepted 70-72% in all four, so no change, as
expected for switches that only touch the prompt path. Power during the 69K prompts went from 228 to 232 W (both cards),
but the prompt finishes sooner, so energy per prompt went down 4% (7,516 to 7,215 J). Clocks stayed at about 2,760 MHz,
both cards at their 165 W limit, peak 73 °C.

## Do the answers change?

- **UPMIX:** `STRATA_HC_UPMIX_CHECK=4` compares the fused read with the default pair on the first four reads of every
  start. All 16 checks: relative RMS 0, max difference 0. That covers layers 0-1 of the warm-up only, not every layer.
- **IMMA:** `qsa_prompt_attn_parity` from the same source (built by hand; `STRATA_BUILD_TESTS` is off in my build)
  passes on both cards, 0 failures. IMMA sits closer to FP64 than the current kernel (2.9e-6 against 3.8e-6 at 2,100
  context) and is 1.41x faster per chunk. Output in [parity/](parity/).
- **Text at temperature 0:** not useful here. Two baseline runs already gave different answers to the same 20K
  prompt, so I cannot tell IMMA's rounding from the run-to-run difference. All eight answers say the same thing.
  They are in each `answer.jsonl`.

## What I did not do

- **No KL.** `tools/quant_quality.py` reads the text through the decode windows, and these two switches only touch the
  prompt path, so it would not exercise them. I have no other way to get prompt-path logits from the server.
- One machine, one model, one quant. No other GPU, no HIP or SYCL (no code changed, so nothing to build there).
- I measured the same A/B on 0.1.45 too, with the same picture (+6.0% at 69K). Those data are not in this folder.

## Files

- `ab/<config>-round<n>/`: `prefill.jsonl`, `decode.jsonl`, `answer.jsonl` (every request, engine timings), `power.csv`,
  `engine-excerpt.txt` (the timer lines, the UPMIX checks and the engine's start and warning lines; paths replaced).
- `bench-sh-*/`: `bench.sh` output as it wrote it, plus the console.
- `parity/`: `qsa_prompt_attn_parity 32768 2048 5` on each GPU.

I did this with an AI coding assistant: it wrote the scripts and helped me read the engine code. I ran everything on my
machine and checked the numbers.
