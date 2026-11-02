# The all-cards split engine (experimental, opt-in)

`strata-split` gives every card a share of **every** layer instead of some of the layers, and the cards exchange the layer sums through
page-locked host memory. It is built from [sFNX](https://github.com/johnfink8/sfnx) by John Fink (MIT; the licence and credit travel with the
code in `third_party/sfnx`). It is **off by default**: it only runs when you turn it on, only for the file and PC described below, and Strata
starts the normal layer split by itself when anything is off.

## What it is for

Unsloth's UD-Q4_K_XL file of Qwen3.8-Flash-Next on two or more big identical NVIDIA cards on Linux, when the whole model fits in the cards'
memory. With the normal layer split one card works while the other waits; here both read the weights at the same time, so answers are
written faster. It reads long prompts 1.4-2.3x more slowly than the normal layer split (the expert kernels for prompts are written for an older GPU instruction; a port is the known fix), so it pays where most of the time is spent writing.

Measured on two RTX A6000 (48 GB), UD-Q4_K_XL, 128K context, Strata f-0291-int3 against the same engine with the split on (3 interleaved
pairs, medians of 3 runs; `phaseA-tests/sfnx/RESULTS.md` sections 8 and 9):

| | normal layer split | all-cards split |
| --- | --- | --- |
| code edit | 195 tok/s | 377-394 |
| new code | 139-144 | 189-232 |
| prose | 93 | 139-146 |
| agent loop (sampled, up to 123K-token prompts) | 112-119 | 165 |
| 8K / 32K / 114K prompt | 4.6 s / 14.8 s / 37 s | 6.4 s / 24.5 s / 86 s |
| one, 2, 4, 6 at once (combined tok/s) | 69, 104, 144, 168 | 92, 141, 170, 181 |
| a 28K prompt joins 4 decoders: its first token / the decoders' longest stall | 8.0 s / 8.0 s | 20.7 s / 5.8 s |
| 32K prompts, 2 or 4 at once (tok/s over the run) | 11.7 / 12.9 | 5.3 / 5.3 |
| the 61-task quality suite (temperature 0) | 57 of 61 | 56 of 61 (the verdicts of four tasks differ in both directions; both engines repeat themselves exactly) |

## Turn it on

`"split_engine": true` in the run config, or `strata launch --split-engine on` (when the build has the option). Nothing else to set: the
engine's own settings are written into the config by Strata.

## When it runs (strata_control/split_engine.py)

All of these, else the normal layer split with a plain line saying which:

- Linux, NVIDIA cards, 2 to 4 of them, the same model and memory size;
- a card count and architecture that have been run: **today two cards of compute capability 8.6 (RTX A6000)**; other counts and
  architectures are built but stay off, and the line says what has not been tested;
- the GGUF's routed experts are Q4_K / Q5_K / Q5_1 / Q8_0 (checked from the file, not its name), its token embeddings and shared expert Q8_0,
  Qwen3.8-Flash-Next's 48 layers and 512 experts;
- **all routed experts fit in the cards' memory** (each card needs half the experts plus about 9 GiB: 2 x 48 GB cards, not 2 x 24 GB);
- enough free RAM for what stays in RAM: the 27.5 GiB embedding table (locked), about 3 GiB of buffers, the conversation cache (4 GiB) and
  8 GiB spare, about 43 GiB. Measured: 31 GB at start, 35 GB after long prompts;
- a context of at most 131072.

If the engine refuses or fails to start, the server says so in one line and starts the normal layer split (`serve/server.py`,
`start_engine_or_fallback`). A crash after it was running restarts the same engine (the fallback is for starting only).

## No card writes into another card's memory

sFNX as published needs the cards to write into each other (CUDA peer access): GeForce drivers refuse that, and its start checks need the
IOMMU in identity mode, which most Linux PCs do not have by default. Here the buffers the cards exchange sums through live in page-locked
host memory that both cards read and write with ordinary loads and stores. No peer API is called anywhere: `tests/split/test_split_engine.py`
fails if one appears in `src/split`, and `packaging/build_split_cuda.sh` fails if the built engine can call one. The IOMMU and BAR1 start
checks are removed because the path they protect no longer exists.

## Not proven

No GeForce pair has been tested (there is none in the lab). Three and four cards, other architectures, Windows and macOS, a 64 GB RAM PC
with two big cards, and contexts over 128K have not been run. The copy-engine ring now used for prompts differs in its last bits from sFNX's GPU-thread ring (`FNX_PF_RING_SM=1`, kept): 39 of 61 suite answers are identical between the two. Answers differ from the normal layer split's in the same way (29 of 61 identical), because the arithmetic differs.

## Build

`-DSTRATA_BUILD_SPLIT=ON -DSTRATA_GGML_DIR=<llama.cpp checkout> [-DSTRATA_SPLIT_CARDS="2;3"] [-DSTRATA_SPLIT_ARCH=86]` builds
`strata-split-N` per card count; `packaging/build_split_cuda.sh` does that with the release engine's toolchain rules and checks the result.
