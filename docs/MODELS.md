# Which model? Sizes, versions and what fits

Strata runs one model, [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), in several **sizes**
(the same model, compressed more or less) and several **versions** (the original, a coding version, a fine-tune).
The installer recommends one for your PC; this page explains the choice. Back to the [README](../README.md).

> **On this page:** [Pick by RAM](#pick-by-ram) · [Speed](#how-fast-is-each-size) · [The sizes](#the-sizes) ·
> [Will it fit?](#will-it-fit) · [The versions](#the-versions) · [Adding another model](#adding-or-switching-models)

## Pick by RAM

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB; best at code, weaker at everything else ([why](#coder)). With a 24 GB card, Q2_0 and IQ2_XS run too ([low-RAM mode](#a-big-graphics-card-and-little-ram)) and are the better pick for general use |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits (IQ3_S with little else open) |
| **96 GB or more** | **IQ3_S**, or [Unsloth's UD-IQ4_XS or UD-Q4_K_XL](#unsloths-files-ud-iq4_xs-to-q8_0) (~4-bit) | room for the largest sizes |
| **128-192 GB** | [Unsloth's UD-Q5_K_XL, UD-Q6_K_XL or Q8_0](#unsloths-files-ud-iq4_xs-to-q8_0) | every expert of the 5-8-bit files in RAM |

Not sure? Take **IQ2_XS**. The **Coder** is the one that fits a 32 GB PC, but it keeps only 256 of the 512 experts,
chosen on code data, so it is weaker outside code and in languages other than English. For general use, or whenever
your RAM allows, take a full model (Q2_0, IQ2_XS or IQ3_S).

## How fast is each size

Measured on an RTX 5070 (12 GB), a Ryzen 5 7600 and 64 GB of RAM:

| Size | Writes answers (short chat) | Writes answers (128K context) | Reads your prompt |
| --- | ---: | ---: | ---: |
| **Q2_0** | 94 tokens/s | 76 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 63 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 49 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 46 tokens/s | 1,620 tokens/s |
| **Coder** (IQ1_M) | 55 tokens/s | 43 tokens/s | 2,180 tokens/s |

Measured on an AMD RX 9070 XT (16 GB), a Ryzen 9 3900X and 47 GB of RAM (Linux, setup's own install; Q2_0 is
above setup's RAM estimate for 47 GB and was installed with `--model Q2_0 --yes`):

| Size | Writes answers (short chat) | Writes answers (128K context) | Reads your prompt |
| --- | ---: | ---: | ---: |
| **Q2_0** | 60 tokens/s | 48 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 36 tokens/s | 1,110 tokens/s |
| **Coder** (IQ1_M) | 44 tokens/s | 33 tokens/s | 1,420 tokens/s |

- **Writes answers** = how fast the reply appears (tokens per second; a token is about ¾ of a word).
- **Reads your prompt** = how fast it takes in what you send (long documents, code, chat history), measured on a
  32K-token prompt; a 4K prompt reads at 910-1,580 tokens/s. A 32K prompt takes about 15 seconds with Q2_0.

A card with more VRAM is faster, because more of the model fits on the GPU: an RTX 3090 (24 GB) should do roughly
100-140 tokens per second. All measurements, long-context numbers and estimates for other cards are in the
[details](DETAILS.md#speed-measured). The AMD measurements per card (RX 9070 XT, Radeon AI PRO R9700, RX 7800 XT,
RX 9060 XT, RX 6900 XT) are in [AMD_HIP.md](AMD_HIP.md#rdna4-gfx1201).

Every PC is different: `START-HERE.bat --calibrate` measures a few engine settings on yours and keeps the fastest
(about 15-30 minutes, longer on a slow card; on the PC above it made the Coder 7% faster; NVIDIA cards for now). Measured Strata on your own
PC? See [Community benchmark results](COMMUNITY_BENCHMARKS.md) for a report template and how to share your results
in a pull request.

## The sizes

| Model | RAM+VRAM Requirements | Speed | Quality |
| --- | ---: | --- | --- |
| **Q2_0** | 37.6 GB | fastest | good |
| **IQ2_XS** | 39.2 GB | fast | better (**recommended**) |
| **IQ3_XXS** | 47.0 GB | slower | great |
| **IQ3_S** | 54.8 GB | slowest | best: matches the full model on the published tests |

The download is 66-76 GB for the three smaller sizes ([details](DETAILS.md#which-model)); the first start also
fetches the MTP draft layer (~6 GB, +1 GB with images).

## Will it fit?

Shard 1 is the part of the model that gets loaded when it starts: its experts go into your **RAM**, the rest onto
your graphics card (the second shard, a 29 GB lookup table, stays on the SSD). So it fits when your **RAM is at least
the experts + about 10 GB** for Windows and your other programs - the experts are most of shard 1: 34 GB for Q2_0,
35.5 for IQ2_XS, 43 for IQ3_XXS, 50 for IQ3_S, 23 for the Coder (the dense weights in shard 1 go to the graphics
card). With 64 GB of RAM every size fits (IQ3_S with little else open); with 48 GB, Q2_0 and IQ2_XS. A bigger graphics card makes it faster, but it doesn't lower the RAM needed
- except in the low-RAM mode below.

### A big graphics card and little RAM

On a PC whose RAM cannot hold the experts beside the system, setup maps them from the model's files instead and keeps
in RAM only what the graphics card does not hold (the [low-RAM mode](DETAILS.md#speed-measured), chosen by setup). For
example, a 32 GB PC with a 24 GB GPU runs Q2_0, IQ2_XS and the Coder this way, and a 32 GB PC with a 12-16 GB GPU the
Coder. With a small card most experts then come from the SSD and it is much slower (setup says so).
`START-HERE.bat --setup --low-ram on|off` overrides the choice.

## The versions

### Qwen3.8-Flash-Next

The original, in all four sizes. With images, and with the
[experimental speed projection](DETAILS.md#experimental-speed-projection-experimental-off-by-default) as an option.

### Coder

**[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** - ISTA-DASLab's coding version:
it keeps 256 of each layer's 512 experts, chosen on code (with agentic and image) data, and drops the rest (91% of
the full model's SWE-bench Verified score, 99% of LiveCodeBench, by its authors). One size (IQ1_M: its experts stored
like IQ3_S): shard 1 is **29.6 GB**, so it fits a PC with **32 GB of RAM**, runs 262K context on 64 GB, and reads long
prompts the fastest of all. With half the experts gone it is weaker outside code and in languages other than
English, Chinese and other CJK text included (#438: Chinese answers came out wrong or looping where English was
fine). For general use, or whenever your RAM allows, take a size that keeps every expert: Q2_0, IQ2_XS or IQ3_S.
More: [details](DETAILS.md#or-the-coder-half-the-experts-for-code).

```
START-HERE.bat --setup --family coder
```

### Swift 1.5

**[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)** - a fine-tune by UkisAI that
thinks much shorter before answering, so you get the answer sooner, with about the same quality. Same speed per
token, and about the same RAM as the same size of the original. Its own license applies (see its page).
More: [details](DETAILS.md#or-swift-15-a-fine-tune-that-thinks-shorter).

```
START-HERE.bat --setup --family swift --model IQ2_XS
```

<a id="unsloth-ud-iq4_xs"></a><a id="unsloth-ud-q4_k_xl-experimental"></a>

### Unsloth's files: UD-IQ4_XS to Q8_0

The fourth version in setup's menu (`--family unsloth`) is **Unsloth's quantizations** of the same model, from ~4 to 8
bits. They keep the original's experts at more bits than the 2-3-bit sizes above, so they need more RAM. Setup keeps
every expert in RAM where they and 16 GB more fit (the fastest way); with less RAM it keeps as many as fit and reads
the rest from the SSD while it answers (much slower), and for the bigger files it asks first. Images are on by default
on an NVIDIA card of 16 GB or more (and on AMD with the CPU encoder where the experts fit in RAM).

| Size | Download | Experts | RAM for all of them in RAM | Same top token as Q8_0 | AMD R9700 (32 GB): story / code / 32K prompt |
| --- | ---: | ---: | ---: | ---: | --- |
| **UD-IQ4_XS** (setup's first) | 94 GB | 55 GiB | ~72 GB | 88.9% | - |
| **UD-Q4_K_XL** | 111 GB | 72 GiB | ~88 GB | 93.1% | 63 / 87 tok/s / 1,110-1,320 tok/s |
| **UD-Q5_K_XL** | 158 GB | 92 GiB | ~108 GB | 94.6% | 57-60 / 82 / 810-1,260 |
| **UD-Q6_K_XL** | 169 GB | 102 GiB | ~118 GB | 94.6% | 54 / 76 / 1,020 |
| **Q8_0** | 188 GB | 120 GiB | ~136 GB | 100% (the reference) | 53-57 / 66-74 / 1,230-1,300 |

For comparison, IQ3_S on the same card: 75 / 95 tok/s, 87.7% the same top token as Q8_0. The quality numbers, the
speed bound and NVIDIA/AMD/Intel details: [QUANTS.md](QUANTS.md); the files: [UD-Q4_K_XL](UNSLOTH_Q4.md),
[UD-Q6_K_XL](UNSLOTH_Q6.md).

```
START-HERE.bat --setup --family unsloth --model UD-Q5_K_XL
```

### OrcaRouter Uncensored IQ3_XXS

For **OrcaRouter's Flash-Next Uncensored IQ3_XXS**, see the [manual compatibility setup](ORCA.md). It needs an
explicit packing conversion and is not an installer menu option.

## Adding or switching models

You can add another model any time with `SETUP.bat` (the same as `START-HERE.bat --setup`; on Linux
`./setup.sh --setup`). Files another model shares are not downloaded again (the Coder uses the original's shard 2 and
vision encoder). With more than one model installed, `START-HERE.bat` asks which one to start; `run-<model>.bat`
(Linux: `run-<model>.sh`) starts one directly.
