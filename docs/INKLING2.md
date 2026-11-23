# Inkling-2-Mini (`strata-ink2`)

`strata-ink2` is Strata's engine program for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(Thinking Machines Lab, Apache 2.0): a 128-expert, top-8 mixture of experts whose layers 6-17 run three times, so a
token goes through 48 layer applications. The engine reads the published checkpoint as it is: NVFP4 experts in layers
0-19, MXFP8 experts in layers 20-23, BF16 for everything else, and the ViT that reads pictures. Nothing is requantized.

To install and start it, see [MODELS.md](MODELS.md#inkling-2-mini): `./setup.sh --family inkling2` (the official
checkpoint, one of the two sources of setup's Inkling family; the `.strata` SQ builds run `strata-inkling` instead)
picks the flags below for your card. This page is for reading what setup chose, or for running the engine by hand.

NVIDIA (RTX 20 series or newer) or AMD RDNA (wave32, gfx10 to gfx12) through ROCm, one GPU, Linux. The AMD port is
jingx8885's, measured on a gfx1151 (Radeon 8060S); the speeds below are NVIDIA ones.

## What setup chooses

Measured on an RTX 3090 with its VRAM limited to the card size (the expert pools only see that much), 2x Xeon Gold
6252, 257-token chat prompt, 512 greedy tokens; prompt speed on a 16K-token prompt.

| Card | Engine flags (besides `--model`, `--chunk 1024 --tc 1`) | Decode | Prompt |
| --- | --- | --- | --- |
| 24 GB | `--max-context 65536 --kv f16 --spec 1` (every expert on the card) | 88-90 tok/s (108 on short prompts with `--spec`) | 2,380 tok/s |
| 16 GB | `--max-context 131072 --kv f16`, offload as below | not measured | not measured |
| 24 GB, longer contexts | `--max-context 131072 --kv f16 --kv-grow 1 --vram-reserve-mib R --cpu-threads N --spec 1` | 84-86 tok/s (103 on short prompts) | not measured |
| 12 GB, with a desktop | `--max-context 65536 --kv f16 --kv-grow 1 --vram-reserve-mib R --cpu-threads N` | 73-75 tok/s | 717 tok/s |
| 8 GB, nothing else on it | `--max-context 32768 --kv f16 --kv-grow 1 --vram-reserve-mib R --cpu-threads N` | 66 tok/s | 568 tok/s |
| 8 GB, with a desktop | the same with `--kv i8r` | 60-61 tok/s | 560 tok/s |

Every config also gets `--conversation-cache-mib 4096` (2048, or off, when RAM is short). In offload the engine pins
about 17 GiB of RAM for the experts that are not on the card; setup asks for 32 GB of RAM.

With the offload, setup adds `--spec 1` only where the card still holds 70% of the experts beside the drafter at the
full context. The drafter takes about 2 GiB from the expert pools (its weights, its K/V, and the prompt buffers, which
stay mapped with it), and a check computes eight tokens' experts to keep three to five, so the experts that are not on
the card cost more per token than in plain decoding. Decode in tok/s, plain and with `--spec 1` (same box, 512 greedy
tokens, mean of two runs; 16 GB: one run):

| Card and context | Experts on the card with the drafter | Chat prompt (257 tokens) | Short prompt (24 tokens) |
| --- | --- | --- | --- |
| 8 GB, 32K | 19% | 66.9, with `--spec 1` 51.8 | 64.6, with `--spec 1` 59.9 |
| 12 GB, 64K | 45% | 80.6, with `--spec 1` 72.3 | 81.5, with `--spec 1` 87.9 |
| 16 GB, 128K | 70% | 84.1, with `--spec 1` 84.4 | 85.7, with `--spec 1` 100.2 |
| 24 GB, 128K | 98% | 85.9, with `--spec 1` 83.9 | 86.7, with `--spec 1` 103.1 |
| 24 GB, 256K, `--kv i8r`, no `--kv-grow` | 92% | 82.9, with `--spec 1` 84.7 | 83.3, with `--spec 1` 100.0 |

These are speeds at the start of a conversation: with `--kv-grow` the K/V of a long conversation takes expert slots,
which is why the 16 GB card at 128K stays without the drafter.

## Flags

**Model and context**

- `--model DIR` the checkpoint folder (or a GGUF file, below).
- `--max-context N` (alias `--ctx`) the longest conversation, prompt plus answer.
- `--chunk N` prompt tokens read per step (`--serve` needs one; setup uses 1024). With `--kv-grow` a chunk shrinks by
  itself when the card is short of room.
- `--tc 0|1` tensor cores for the prompt (1, the default in setup). `--tc 0` is the exact fp32 path, slower, used to
  check the other one.
- `--serve` the line protocol the Strata server speaks.

**Memory**

- `--vram-reserve-mib N` leaves N MiB of VRAM free and turns on the expert offload: the experts that do not fit stay
  in pinned RAM, and the card keeps the most used ones (decayed use counts, promotions in the background).
- `--cpu-threads N` computes the experts that are not on the card on the CPU instead of copying them over PCIe: one
  pool of N threads per NUMA node (at most two), each expert's memory bound to its node. Needs `--vram-reserve-mib`,
  AVX2 + FMA + F16C, and Linux. `STRATA_NUMA_POOL=0` turns the NUMA placement off.
- `--kv-grow 1` the K/V cache starts small and grows into the expert pools as the conversation grows, so a long
  context costs expert slots only when it is used. Each pool keeps at least 64 slots: a prompt that would need more
  ends with a "context is full" error, and the engine keeps serving. `--kv-grow-step N` sets the growth step.
- `--kv FORMAT` the K/V format. Mean KL against fp32 K/V (expert routing fixed, 23K-token mixed text):

  | `--kv` | Bytes per token and layer pass | KL mean | Top-1 |
  | --- | --- | --- | --- |
  | `f16` (default) | 4,096 | 0 | 100% |
  | `i8r` (int8, rotated) | 2,112 | 3.5e-5 | 99.80% |
  | `k8v8` | 2,112 | 5.3e-5 | 99.73% |
  | `i8` | 2,112 | 1.3e-4 | 99.61% |
  | `k8v4` | 1,632 | 1.5e-3 (1.6K-token text) | 98.75% |

  `f32` exists for checks.

**Speed**

- `--spec 1` speculative decoding with the model's own drafter (`mtp.safetensors`): a draft token is kept only when it
  is the model's own choice (greedy) or passes the usual acceptance test (sampled). It falls back to plain decoding
  while that is faster. The check reads the drafts through other kernels than plain decoding, so a near-tie can go the
  other way: a 512-token greedy answer matched plain decoding for 273 tokens, then parted where the top two logits were
  0.2 apart. With `--cpu-threads` the CPU also computes a check's experts that are not on the card (same answer as
  with every expert on the card over those 512 tokens); it pays only where most experts are on the card (the table
  under [What setup chooses](#what-setup-chooses)).
- `--verify-tc 1` (off by default) runs the dense products of a `--spec` check and of the drafter (up to 8 vectors,
  matrices of more than 256 rows) on tensor cores: fp16 weights and inputs, fp32 accumulation, the numbers of the
  prompt path's products. Without it, those products follow plain decoding's order. CUDA cards of compute capability
  8.0 or later. RTX 3090, every expert on the card, `--kv i8r`, a check at every step, three runs each: 36.0 ms per
  check without, 32.6 with. With the expert routing fixed, the 8-token prompt lane is at mean KL 3.0e-7 against the
  fp32 token path, against 8.7e-8 without the flag, top-1 100%.

**Conversations and pictures**

- `--conversation-cache-mib N` (0, off, by default) keeps conversations in pinned RAM so that a request never reads
  again a prompt the engine already read: a checkpoint at the end of each prompt lets the next turn continue from
  there even when the client sends the history back without the model's reasoning, conversations set aside (an
  agent and its subagents taking turns) are taken up again, and conversations sharing a long beginning (a system
  prompt and tools) read it once. `--conversation-cache-min-tokens N` (512): shorter conversations are not kept.
- Pictures need no flag. The ViT is read into RAM at the first picture and streamed to the card for each one, so
  text-only use costs no VRAM; with offload it borrows a few hundred MiB of expert slots while it runs. A picture
  over `--image-max-tokens N` (4096; one token is a 40x40 pixel tile) is scaled down.

**Checks and debugging**: `--tokens IDS` / `--tokens-file F` with `--n N` run one prompt without the server,
`--dump-logits F`, `--record-routing F` / `--force-routing F`, `--image F`, `--dump-image-emb F`, `--vis-pixels F`.
`terminal/tools/inkling2/README.md` describes the comparison against the transformers implementation.

## GGUF files

`--model FILE.gguf` reads llama.cpp's merged `inkling2_loop` layout with IQ3_S, IQ4_NL and Q8_0 experts (work by
merbanan). GGUF models run without `--cpu-threads` and `--spec` for now, and setup does not install them.
Layers 20-23 may use a different expert type than layers 0-19, as in ggml-org's `Inkling-2-Mini-IQ4_NL.gguf` (IQ4_NL,
Q8_0 in layers 20-23): on the 1,600-token text its mean KL against the fp32 reference is 5.9e-3 (llama.cpp on the same
file: 5.4e-3). Any other mix, or an expert type outside these three (NVFP4, for example), stops the load with an error
that names the layers and types.

## Quality

Against the transformers implementation in fp32 on a 1,600-token text: the token-by-token path with fp32 K/V is at
mean KL 1.4e-12 (top-1 100%), and the tensor-core prompt path at 2.1e-7 with the expert routing fixed. Pictures: mean
KL 1e-7 or less in fp32, with the same greedy answers.

## Limits

- One GPU. `--cpu-threads` on Linux only, at most two NUMA nodes.
- `--spec` with the expert offload and no `--cpu-threads` copies a check's missing experts over PCIe: 16 tok/s on an
  8 GB card. Setup does not write that pair.
- Not offered in `strata launch` or the desktop app yet.
