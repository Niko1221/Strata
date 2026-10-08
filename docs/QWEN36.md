# Qwen3.6-35B-A3B

A second, smaller model next to Qwen3.8-Flash-Next: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
(35B parameters, 3B active per token), in Unsloth's GGUFs with the MTP draft layer
([unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), Apache 2.0). It is the
same family as Flash-Next - DeltaNet layers with a full-attention layer every fourth, a mixture of experts with a
shared expert - but 40 layers of 256 experts instead of 48 of 512, so its experts take 12-14 GB of RAM instead of
23-50 GB. It is the model for PCs with 16-32 GB of RAM and 8-12 GB graphics cards. Back to the [models](MODELS.md).

> **On this page:** [Setup](#setup) · [What fits](#what-fits) · [Speed](#speed-measured) ·
> [Small cards](#small-cards) · [Against llama.cpp](#against-llamacpp) · [Ornith-1.5](#ornith-15-35b-a3b) ·
> [What is different](#what-is-different-from-flash-next) · [Limits](#limits)

## Setup

```
./setup.sh --setup --family qwen36                      (Linux)
START-HERE.bat --setup --family qwen36                  (Windows)
```

Setup offers two sizes and picks the larger one that fits:

| Size | Download | Experts in RAM | RAM it asks for |
| --- | ---: | ---: | ---: |
| **UD-IQ4_XS** (recommended, ~4-bit) | 18.2 GB | 14.2 GiB (measured) | 26 GB |
| **UD-IQ3_S** (smaller, faster) | 15.3 GB | 12.0 GiB (measured) | 24 GB; a 16 GB PC with a 12 GB card in the [low-RAM mode](MODELS.md#a-big-graphics-card-and-little-ram) (with an 8 GB card it does not fit) |

The file goes where every model goes (`<data>/models/qwen36-UD-IQ4_XS/`), its pack is made the usual way
(`tools/iq_pack.py --compat-bf16`, two seconds), and the MTP draft layer is read from the same file: there is no
separate draft download. The engine finds out which model it runs from the file (`general.architecture`):
nothing in the command line says "Qwen3.6".

No ready-made engine runs it yet: the published Windows engines (0.1.40 to 0.1.41) were built before this model was
added. Setup looks for it in the engine and, when the ready-made one lacks it, compiles the engine on the PC instead
(10-20 minutes, once; setup installs the compiler and the CUDA toolkit when they are missing, asking first). On Linux
setup compiles the engine anyway. The compiled engine runs every model, Flash-Next too.

## What fits

Every expert lives in RAM; the graphics card holds the dense weights, the context and as many experts as fit in the
rest of its memory (the expert cache). Measured at 32K of context, 8-bit K/V:

| Graphics card | UD-IQ4_XS experts on the GPU | UD-IQ3_S experts on the GPU |
| --- | ---: | ---: |
| 12 GB (RTX 4070 Ti) | 4,407 of 10,240 (43%) | 5,587 (55%) |
| 8 GB (emulated, see below) | 1,372 (13%) | 2,003 (20%) |

### Small cards

How small a card it starts on, measured the same way as the 8 GB rows (the 4070 Ti with the rest of its VRAM held by
another process; "free" is what `nvidia-smi` showed free before the start), UD-IQ3_S with its draft layer:

| Settings | Starts from | Writes (short chat) | Reads (8K prompt) |
| --- | ---: | ---: | ---: |
| setup's: 32K context, 8-bit K/V | 4.17 GB free | 73 tokens/s | 591 tokens/s |
| 8K context, `--draft-vocab en` | 3.9 GB free | 74 tokens/s | 962 tokens/s |

UD-IQ4_XS needs more: 4.68 GB free with setup's settings (64 tokens/s) and 4.3 GB with the second ones (66 tokens/s),
so setup recommends UD-IQ3_S on cards under 5.5 GB. At 3.65 GB free with the second settings the short prompt ran and the 8K one stopped at the start (free VRAM differs
by a few tens of MiB from one start to the next). With less, the engine stops at the start and says how many MiB are
short and what makes room. Setup recommends the 8K context on cards under 5.5 GB (4 GB cards). These are emulated
sizes, not runs on 4 and 6 GB cards.

## Speed (measured)

RTX 4070 Ti (12 GB), Ryzen 9 5950X, 62 GB of RAM, Linux, CUDA 13.0; 32K context, 8-bit K/V, the MTP draft layer on
(`--spec 4`). "8 GB" is the same card with 4 GiB of VRAM held by another process for the whole run, so the engine
sizes itself as on an 8 GB card with the same desktop load; it is not a measurement on an 8 GB card. Short chat = a
21-token prompt, 128 tokens written; long = an 8,014-token prompt, 64 tokens written.

| Size, card | Short chat, writes | Long prompt, reads | Long prompt, writes |
| --- | ---: | ---: | ---: |
| UD-IQ4_XS, 12 GB | 105 tokens/s | 4,460 tokens/s | 86 tokens/s |
| UD-IQ4_XS, 8 GB | 75 tokens/s | 2,669 tokens/s | 68 tokens/s |
| UD-IQ3_S, 12 GB | 127 tokens/s | 4,284 tokens/s | 106 tokens/s |
| UD-IQ3_S, 8 GB | 89 tokens/s | 3,076 tokens/s | 70 tokens/s |

These tables were measured with a first expert profile that ranked nothing (every expert in turn). The shipped one,
`data/expert-profile-qwen36.bin`, ranks the experts by how often 24 varied prompts (code, agent tool calls, chat,
French and Chinese) routed to them; the profile decides which experts the GPU holds when the engine starts. Against the
first one, on prompts not among the 24 (three runs each, UD-IQ4_XS): 12 GB, short chat 99 -> 106 tokens/s and long
prompt writes 84 -> 93; with an 8 GB card's VRAM no difference (73 and 60-62 both). Ornith uses the same profile (on
Ornith IQ4_XS: long prompt writes 74 -> 87, short chat unchanged; a profile from Ornith's own routing was not
clearly better - faster on the short chat, slower on the long prompt). It was made with the
server's `--expert-profile-save` and a 64-slot expert cache, so the ranking is the routing count and not what one
cache happened to hold.

A 256K context (setup's 256K choice) runs on the 12 GB card too: UD-IQ3_S wrote 78 tokens/s and read the 8K prompt
at 3,802 tokens/s with it (the engine keeps 393 MiB beside the expert cache for the attention over the whole context).

The draft layer's drafts are accepted 0.84-0.96 of the time on these prompts (2.4-3.1 tokens per verify window).
Through the server, chat answers with thinking ran at 104-118 tokens/s (UD-IQ4_XS, 12 GB).

## Against llama.cpp

The same PC, the same UD-IQ4_XS file, llama.cpp b11438 (`llama-server --fit on`, which puts what fits on the card
and the rest of the experts on the CPU), the same prompts:

| | Strata 12 GB | llama.cpp 12 GB | Strata 8 GB | llama.cpp 8 GB |
| --- | ---: | ---: | ---: | ---: |
| Short chat, writes | 105 | 51 | 75 | 39 |
| Long prompt, reads | 4,460 | 708 | 2,669 | 490 |
| Long prompt, writes | 86 | 49 | 68 | 36 |

(tokens/s). The answers are the same: greedy decoding gives llama.cpp's tokens exactly on UD-IQ4_XS (three short
prompts and the 8K prompt, 32 tokens each, with the expert cache at 3,000 slots). The experts in the GPU's cache
round slightly differently from the CPU's, so which experts the cache holds can move a near-tie: with the cache sized
by `auto` (4,420 slots) the 8K prompt took the second choice at its 17th token, where llama.cpp's top two are 28% and
26%. On UD-IQ3_S the first tokens agree and a later token can differ where llama.cpp's own top two are within a few
percent (37% against 35% at the first difference seen) - a rounding-level tie, both continuations correct.

## Ornith-1.5-35B-A3B

[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (ornith-ai, MIT) is a fine-tune of the same
model for coding agents: the same architecture (`qwen35moe`), tokenizer and sizes, so it runs on the same engine path
with Qwen3.6's expert profile and draft vocabulary. There are no Unsloth files for it; setup uses bartowski's
single-file GGUFs ([bartowski/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF)),
which keep the MTP draft layer (stored Q4_0):

```
./setup.sh --setup --family ornith                      (Linux)
START-HERE.bat --setup --family ornith                  (Windows)
```

| Size | Download | Experts in RAM | RAM it asks for |
| --- | ---: | ---: | ---: |
| **IQ4_XS** (recommended, ~4-bit) | 19.3 GB | 15.9 GiB (measured) | 28 GB |
| **IQ3_XXS** (smaller, faster) | 15.3 GB | 12.4 GiB (measured) | 24 GB; a 16 GB PC with a 12 GB card in the low-RAM mode |

Speed, measured as [above](#speed-measured) (the same PC, prompts and settings):

| Size, card | Short chat, writes | Long prompt, reads | Long prompt, writes |
| --- | ---: | ---: | ---: |
| IQ4_XS, 12 GB | 102 tokens/s | 4,423 tokens/s | 73 tokens/s |
| IQ4_XS, 8 GB | 75 tokens/s | 3,724 tokens/s | 52 tokens/s |
| IQ3_XXS, 12 GB | 110 tokens/s | 4,549 tokens/s | 103 tokens/s |
| IQ3_XXS, 8 GB | 78 tokens/s | 3,739 tokens/s | 75 tokens/s |

Its draft layer helps less than Qwen3.6's: with every draft proposed, 61% of them were accepted on a chat prompt and
38% on a code prompt, against 80% and 98% for Qwen3.6 UD-IQ4_XS. A Q8_0 copy of the same layer (mudler's
APEX-MTP-Compact file) did no better, 44% and 29%, and the Q4_0 kernels give llama.cpp's results (`iq_parity`), so the
difference is the draft layer itself, which the fine-tune did not retrain. The answers are the main model's either way: greedy decoding gives llama.cpp b11438's tokens except where
3-bit rounding moves a close choice (IQ3_XXS: one difference seen, at a token llama.cpp gave 0.73 and Strata 0.63).

## What is different from Flash-Next

The engine keeps one code path for both and switches on the model's geometry:

- **One residual stream** instead of Flash-Next's four hyper-connection streams: RMSNorm before the mixer and before
  the MoE, the output added back (llama.cpp's `qwen35moe` graph).
- **Dense attention**: no sparse-attention indexer; every cached position is attended (the prompt path's tensor-core
  kernel has a dense mode for it, 8 query heads per K/V head).
- **No PLE** block, no n-gram table on the SSD.
- **The DeltaNet output gate** is `silu(z)` instead of `sigmoid(z)`.
- **The MoE** routes 8 of 256 experts (Flash-Next: 10 of 512); 2048-wide, 512-wide experts.

Flash-Next's own arithmetic is unchanged: the same tokens and the same speed as the engine before this model was
added (Coder, three short prompts and an 8K prompt, back to back on the same PC).

## Limits

- **NVIDIA only for now.** The prompt path's dense attention is a tensor-core kernel (RTX 20 and newer). On AMD and
  older NVIDIA cards the engine reads the prompt through the decode windows instead (on the 4070 Ti that path read
  ~80 tokens/s, against 4,000+ batched) - and it has not been run on an AMD card at all; setup says so and asks.
- **One GPU.** The layer split across several cards is not done for this model.
- **No images** yet (its vision encoder is not wired in).
- The low-RAM mode (`--resident-experts`) gives the same tokens as the normal mode on this PC (UD-IQ3_S: 9.0 GiB of
  experts held in RAM); it has not been measured on a 16 GB PC.
