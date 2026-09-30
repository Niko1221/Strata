# This fork, upstream Strata and llama.cpp on one PC

Qwen3.8-Flash-Next in two quantizations, IQ3_S and UD-Q4_K_XL, through three engines on an RTX 3090 + RTX 5070 Ti PC.
The llama.cpp and upstream columns were measured once (2026-09-30). This fork's columns are kept current: measured at
0bbd8f5 (2026-09-30).

**In short**, with both GPUs:

- **IQ3_S:** this fork reads prompts 4.2-5.0x as fast as llama.cpp and 2.3-2.4x as fast as upstream Strata, and
  writes answers 4.2-7.9x as fast as llama.cpp and 1.5-2.0x as fast as upstream (the most on long contexts and on
  answers that quote the prompt). On the 3090 alone, against upstream there: prompts 1.6x, answers 1.4-1.7x.
- **UD-Q4_K_XL:** prompts 5.0-6.2x and answers 4.8-8.8x as fast as llama.cpp; upstream cannot run the file.
- A follow-up turn after a 16K-token prompt starts in 0.2-0.3 s (llama.cpp 0.6-0.7 s; upstream 0.35 s on the 3090,
  1.1 s on both GPUs).
- The next-token distributions match llama.cpp's as closely as llama.cpp's own CPU and CUDA backends match each other
  (IQ3_S; UD-Q4_K_XL somewhat further, from llama.cpp's 8-bit rounding of the file's Q8_0 hyper-connection weights).

## The PC and the engines

- **PC:** Ryzen 7 9700X (8 cores), 160 GB DDR5-5600 (2 x 32 + 2 x 48 GB), Windows 11. An RTX 3090 (24 GB, PCIe 4.0
  x4, nothing else on it) and an RTX 5070 Ti (16 GB, PCIe 5.0 x16, driving the display).
- **Models:** ISTA-DASLab's GSQ-RCO **IQ3_S** (shard 1 54.8 GB: experts 46.8 GiB, i-quants with Q2_0 and IQ4_NL down
  projections) and unsloth's **UD-Q4_K_XL** (111 GB: experts 71.7 GiB, Q4_K / Q5_K with Q5_1 / Q8_0 downs). Each has
  a 28.8 GB IQ4_NL n-gram table (the per-layer embeddings) besides.
- **llama.cpp** build 11116 (709fe755d, MSVC), its ggml-cuda.dll patched to pin the CPU experts' buffers (prompts
  1.8x faster than unpinned). Both GPUs with the layers split (the 5070 Ti first), the experts that do not fit in VRAM
  in RAM: UD-Q4_K_XL by `--fit`, IQ3_S by fit's placement with its Q2_0 down projections kept on the GPUs (llama.cpp
  has no x86 SIMD kernel for Q2_0: 30.5 instead of 27.2 t/s). 8-bit KV cache, 8 threads, 1024-token micro-batches,
  the n-gram table in RAM. No speculative decoding: llama.cpp has no graph for this model's MTP layer.
- **Upstream Strata** a790805 (engine 0.1.27) with the config its setup writes for IQ3_S on this PC: MTP drafts, 8-bit
  KV kept in RAM past 32K cells, the expert cache sized by the free VRAM. On the 3090, and on both GPUs as its setup
  recommends here (a layer split with the newer 5070 Ti first: layers 0-25 there, 26-47 on the 3090). It cannot run
  UD-Q4_K_XL: it has no K-quant expert kernels.
- **This fork** with its configs ([examples](../examples/README.md)): the 3090 runs the model and the 5070 Ti is a
  second expert tier (12.2-12.4 GiB of experts; its share of each layer's experts, the next layer's likely experts
  copied ahead, the prompt path's experts the 3090 lacks, and with IQ3_S part of the output head). MTP drafts and
  prompt lookup, 8-bit KV in VRAM, 16K-token prompt chunks. And on the 3090 alone.

## Speed through the API

The same requests to each server's OpenAI API: greedy, one at a time, the prompt caches off except for the follow-up
turn. Prompts: 16K-128K tokens of llama.cpp's sources with a question (400 tokens of answer), a 28-token coding request
(up to 300), a coding task with thinking on (3,000), and a Python module to return with a class renamed (the edit, up
to 3,000 tokens that mostly quote the prompt). The follow-up turn sends the 16K prompt, its answer and a new question:
its prompt time is the processing of what the engine did not keep. Each engine writes its own text and the texts part
at the first near-tie, so the generation speeds compare different texts of one kind, one run each. The memory rows
come from a second start of each server (the model file then in the OS cache): the GPUs' memory taken, and the RAM
taken as the drop in the system's available memory.

### IQ3_S

<!-- matrix:api-iq3_s -->
| | llama.cpp, both GPUs | upstream Strata, 3090 | upstream Strata, both GPUs | this fork, both GPUs | this fork, 3090 |
|---|---|---|---|---|---|
| Prompt, 16K tokens (t/s) | 581 | 1029 | 1060 | 2466 | 1605 |
| Prompt, 32K tokens (t/s) | 562 | 1120 | 1119 | 2666 | 1768 |
| Prompt, 64K tokens (t/s) | 551 | 1139 | 1148 | 2688 | 1791 |
| Prompt, 128K tokens (t/s) | 527 | 1111 | 1143 | 2650 | 1787 |
| Generation: code, 28-token prompt (t/s) | 31.0 | 74.1 | 84.8 | 167.7 | 113.9 |
| Generation: thinking, 3,000 tokens | 31.3 | 77.1 | 89.0 | 130.2 | 111.2 |
| Generation: edit, the prompt's file returned (2.4K tokens) | 31.1 | 99.4 | 117.2 | 234.8 | 171.3 |
| Generation after the 16K prompt | 29.8 | 75.7 | 85.4 | 151.2 | 108.1 |
| Generation after the 32K prompt | 26.9 | 80.6 | 94.2 | 150.2 | 122.1 |
| Generation after the 64K prompt | 22.0 | 73.3 | 85.3 | 139.6 | 113.9 |
| Generation after the 128K prompt | 16.4 | 67.6 | 76.8 | 129.2 | 110.2 |
| Follow-up turn after the 16K prompt: new tokens, prompt time | 34 in 0.62 s | 34 in 0.35 s | 440 in 1.12 s | 34 in 0.21 s | 34 in 0.34 s |
| Start to the API, the model file cached (s) | 21 | 16 | 21 | 21 | 20 |
| VRAM taken: 3090 / 5070 Ti (GiB) | 22.9 / 13.0 | 22.9 / 0.0 | 21.9 / 14.6 | 22.5 / 13.1 | 22.6 / 0.0 |
| RAM taken (GB) | 57 | 56 | 60 | 53 | 52 |
<!-- /matrix:api-iq3_s -->

### UD-Q4_K_XL

Upstream Strata cannot run this file: it has no kernels for K-quant experts.

<!-- matrix:api-ud-q4_k_xl -->
| | llama.cpp, both GPUs | this fork, both GPUs | this fork, 3090 |
|---|---|---|---|
| Prompt, 16K tokens (t/s) | 467 | 2341 | 1099 |
| Prompt, 32K tokens (t/s) | 453 | 2645 | 1184 |
| Prompt, 64K tokens (t/s) | 444 | 2662 | 1209 |
| Prompt, 128K tokens (t/s) | 426 | 2627 | 1221 |
| Generation: code, 28-token prompt (t/s) | 23.5 | 121.3 | 76.8 |
| Generation: thinking, 3,000 tokens | 23.7 | 114.3 | 74.3 |
| Generation: edit, the prompt's file returned (2.4K tokens) | 23.7 | 160.0 | 113.4 |
| Generation after the 16K prompt | 22.6 | 126.7 | 67.6 |
| Generation after the 32K prompt | 21.7 | 138.4 | 75.7 |
| Generation after the 64K prompt | 18.6 | 119.7 | 66.4 |
| Generation after the 128K prompt | 14.5 | 127.5 | 69.2 |
| Follow-up turn after the 16K prompt: new tokens, prompt time | 34 in 0.69 s | 34 in 0.26 s | 34 in 0.48 s |
| Start to the API, the model file cached (s) | 68 | 41 | 36 |
| VRAM taken: 3090 / 5070 Ti (GiB) | 22.3 / 12.9 | 22.7 / 13.1 | 22.7 / 0.0 |
| RAM taken (GB) | 86 | 81 | 80 |
<!-- /matrix:api-ud-q4_k_xl -->

## Generation on fixed text (this fork)

Each run emits the model's reference text (`--spec-follow`), so runs and builds compare on the same tokens and
experts: two runs, averaged, tokens a verify pass in parentheses. The code text's reference repeats its answer after
the end of the turn, which prompt lookup drafts and a server never generates.

<!-- matrix:fixed -->
| Text | IQ3_S t/s (tokens a pass) | UD-Q4_K_XL t/s (tokens a pass) |
|---|---|---|
| code (400 tokens) | 181.5 (4.02) | 114.6 (3.69) |
| thinking (1,200) | 137.8 (2.07) | 105.4 (1.71) |
| after the 16K prompt (400) | 145.2 (2.71) | 124.7 (2.65) |
| edit (2,397) | 243.7 (7.51) | 160.8 (7.51) |
<!-- /matrix:fixed -->

## Agreement with llama.cpp

Argmax agreement and mean KL divergence (nats) of the next-token distributions along llama.cpp's greedy text, which
this fork follows (`--spec-follow`, `--window-logits`), with `--adapt-every 0`. The two baselines differ from each
other only in rounding: llama.cpp's CPU backend against its CUDA backend, and this fork in one-token windows against
its four-token windows. Any change of rounding moves 2-4% of the thinking text's argmaxes, all near-ties.

<!-- matrix:parity -->
| | Code, ~260 tokens | Thinking, 1,200 | After the 16K prompt, 300 |
|---|---|---|---|
| IQ3_S: this fork vs llama.cpp | 100.0%, 1.3e-04 | 98.0%, 2.1e-03 | 99.3%, 1.0e-03 |
| llama.cpp's CPU vs CUDA backend | 100.0%, 2.2e-04 | 96.9%, 3.2e-03 | 99.0%, 1.4e-03 |
| this fork, one-token vs four-token windows | 100.0%, 1.8e-04 | 98.3%, 1.9e-03 | 99.7%, 3.8e-04 |
| UD-Q4_K_XL: this fork vs llama.cpp | 99.3%, 5.3e-04 | 95.8%, 6.7e-03 | 97.7%, 3.5e-03 |
| llama.cpp's CPU vs CUDA backend | 99.6%, 1.3e-04 | 96.8%, 3.3e-03 | 100.0%, 1.6e-03 |
| this fork, one-token vs four-token windows | 99.6%, 1.0e-04 | 97.2%, 2.3e-03 | 98.7%, 5.4e-04 |
<!-- /matrix:parity -->

UD-Q4_K_XL's CPU baseline after the 16K prompt covers 181 positions: llama-server's greedy text itself differed between
two runs there, and the CPU backend was forced along the other one, which parts after 180 tokens.

## The two files

llama.cpp's perplexity over 28 chunks of 4,096 tokens of llama.cpp's own docs, C++ and Python sources (lower is
better): UD-Q4_K_XL 1.8849, IQ3_S 1.8946 (+0.5%), and ISTA-DASLab's IQ3_XXS, which IQ3_S replaced in this fork's
configs, 1.9215 (+1.9%). IQ3_S's experts take 46.8 GiB, UD-Q4_K_XL's 71.7: on this PC IQ3_S generates 16-58%
faster on fixed text, and reads prompts about as fast.

## Features

| | llama.cpp | Upstream Strata | This fork |
|---|---|---|---|
| Model files | any GGUF of the model | ISTA-DASLab's GSQ-RCO sizes (Q2_0 to IQ3_S), their Coder, Swift 1.5: i-quant and Q2_0 experts | those, and every quant of unsloth's (K-quant, Q5_1 and Q8_0 experts) |
| Speculative decoding | none: no graph for this model's MTP layer | the MTP layer's drafts, up to 3 a pass; prompt lookup (up to 5 tokens where it pays) | the MTP layer's drafts; prompt lookup (up to 7 tokens after a 16+-token match) |
| Several GPUs | a layer or row split | a layer split, each card holding its layers' experts | the second card as an expert tier: its share of each layer, the next layer's likely experts copied ahead, the prompt's experts the first card lacks, part of the output head |
| Experts in VRAM | a fixed placement | ranked by use, updated every 4 verify passes | ranked by use on both cards, updated every pass |
| KV cache | 8-bit, in VRAM | 8-bit, in RAM past 32K cells | 8-bit, in VRAM |
| Prompt cache | the live slot, states saved to RAM | checkpoints at turns and every 16K tokens | the live sequence, checkpoints at turns and chunks, replaced conversations kept in RAM with their K/V |
| Sampling, APIs | full; OpenAI and Anthropic | per request; OpenAI and Anthropic, a web app, MCP tools | upstream's |
