# Nemotron-H MoE (Nemotron 3.5 Super VL 120B-A12B)

Strata runs NVIDIA's Nemotron 3.5 Super VL from unsloth's GGUFs (`unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF`,
architecture `nemotron_h_moe`) on an engine path of its own, `src/nemotron_h`. `strata` switches to it from the
file's architecture, the same way it switches to Qwen3.6's path; `strata-nh` runs it on its own. Flash-Next's and
Qwen3.6's paths are unchanged.

## What the model is

88 flat sublayers, each `x += op(rms(x) * attn_norm)`:

- 40 Mamba-2 mixers: `in_proj` (z | xBC | dt), a causal conv of 4 taps with a bias and SiLU, the selective scan
  (128 heads of 64, state 128, 8 groups, a scalar decay per head), `y + D x`, the SiLU(z) gate, an RMS norm per group
  of 1024 channels, `out_proj`;
- 8 attention mixers: 32 query heads over 2 KV heads, head 128, and no positional encoding at all;
- 40 MoE blocks: a sigmoid router with a selection bias (512 experts, top 22, the weights normalized and scaled by
  5), the routed experts in a 1024-wide latent space (`down(relu(up x)^2)`, no gate), and a shared expert
  `down(relu(up h)^2)` on the full 4096 width;
- one MTP block (`blk.88`): `eh_proj` over `[enorm(embedding) | hnorm(hidden)]`, an attention and a MoE sublayer, a
  LayerNorm head norm, the main output head.

Images go through the RADIO vision tower and projector in the repository's `mmproj-BF16.gguf`.

## How Strata runs it

The routed experts are 78 GB of the 87 GB `UD-Q4_K_XL` file, so this path is always in the hybrid mode:

- every dense weight (Mamba, attention, latent and shared-expert projections, router, embedding, head; about 8 GiB)
  is in VRAM;
- each MoE layer gets the same number of VRAM slots for experts from the VRAM that is left, and the slots follow the
  routing (the most used experts move in between requests and every 32 decoded tokens);
- a batch of up to `--hybrid-rows` rows (decode, MTP verify windows, short prompts) runs its resident experts on the
  GPU and the others on the CPU (ggml-cpu's dot products), side by side;
- a bigger prompt chunk (`--chunk`, 2048 by default) streams every non-resident expert through a ring of VRAM buffers
  (`--stream-ring`) and runs each expert's rows as two FP16 GEMMs.

What this path took over from Flash-Next's:

- **the expert arena**: every routed expert is copied once into pinned host memory at start (73.4 GiB, ~70-90 s), on
  2 MiB pages (transparent huge pages, then registered with CUDA; `STRATA_NH_ARENA_4K=1`: cudaHostAlloc's 4 KiB
  pages; `--no-arena`: the mapped file). The CPU experts read it and every upload is a DMA at full PCIe speed. It
  needs that much free RAM (checked; otherwise the mapped file is read);
- **the adaptive VRAM tier**: the slots follow the routing counts (every 32 decoded tokens, up to 512 swaps after a
  prompt);
- **the expert profile**: `--expert-profile F` starts every layer's slots with its most-routed experts,
  `--expert-profile-save F` writes the counts every 8 requests and at exit (they add up across restarts when both
  name the same file). Format `NHP1`: per MoE layer (the MTP block last) a float count per expert;
- **the PCIe share of the misses** (`--pcie-frac F`, up to `--pcie-max N` experts a layer): that part of a layer's
  missed experts uploaded and run on the GPU while the CPU computes the rest;
- **the prefetch** (`--prefetch N`): the next MoE layer's router applied to the hidden one sublayer early, its N most
  likely non-resident experts uploaded while that sublayer runs (92-96% of the uploaded ones were then routed, at
  N = 2-4);
- **turn checkpoints**: the Mamba state at the end of each request's prompt (but its last token), up to 4, in pinned
  RAM (173 MB each). The state cannot step back, the KV cache can, so a request that returns to one of them along
  the live sequence (a regenerated or edited last answer) reads only the rest of its prompt.

The PCIe share and the prefetch are off by default: on the PC below the CPU experts read RAM at its ceiling already
(~40 GB/s of DDR4-2133), and an upload reads the same RAM, so they measured neutral or slower there. A PC with a
faster PCIe link than RAM per core, or a slower CPU, is where they pay - the same reasoning as Flash-Next's
`--pcie-frac`.
Also:

- the KV cache is FP16, 9 attention layers (8 + the MTP block): 9 KiB per token of context;
- `--mtp N` drafts N tokens with the MTP block for greedy requests; the main model verifies them and the Mamba state
  is then committed for the accepted rows only, so the text is the plain greedy text.

The formulas are llama.cpp's for the same tensors (`src/models/nemotron-h.cpp`, `nemotron-h-moe.cpp`,
`mamba-base.cpp`, ggml's `ssm_conv` / `ssm_scan`); `tools/nemotron_h/parity.py` compares the two.

## Files

| File | What |
|---|---|
| `src/nemotron_h/nh_engine.cu` | the engine: loading, the sublayers, the hybrid experts, the stream, MTP, `--serve` |
| `src/nemotron_h/nh_kernels.cuh` | its kernels: Mamba-2 conv / scan / gated norm, NoPE attention, the router, combines |
| `src/nemotron_h/nh_cpu.cpp` | the CPU experts (ggml-cpu `vec_dot`, relu^2) |
| `src/kernels/cuda/iq_kernels.cu` | `native_expert_grouped_relu2`: the GPU experts without a gate (new, opt-in) |
| `tools/nemotron_h/nh_llama_dump.cpp` | llama.cpp's side of the parity check (`-cmoe`, sublayer dumps) |
| `tools/nemotron_h/parity.py` | KL / top-1 against llama.cpp, per-sublayer errors |
| `tools/strata_tokenizer.py` | the `tekken` pre-tokenizer (with `ignore_merges`) next to the Qwen one |
| `serve/server.py` | `<img> <image> x n </img>` for this model's pictures; stop ids that are single tokens only |

## Running it

```bash
# the tokenizer (once): vocab, merges, the chat template from the GGUF
STRATA_GGUF_PY=<llama.cpp>/gguf-py .venv/bin/python tools/strata_tokenizer.py --gguf <first shard> --out <pack dir>
# a quick greedy run
strata-nh --model <first shard> --tokens 10,1785,... --n 32 [--mtp 2]
# the server (strata-<name>.json: "exe" strata, "args" ["--native", "<first shard>", "--max-context", "65536",
# "--mtp", "2"], "tokenizer" <pack dir>/tokenizer, "arch" "nemotron_h_moe", "vision" {"exe": strata-vision, ...})
python serve/server.py --engine strata --config strata-<name>.json --port 8092
```

The image encoder needs llama.cpp's mtmd with `PROJECTOR_TYPE_NEMOTRON_OMNI` (upstream since PR #30520): build
`tools/vision` with `-DLLAMA_DIR=<that checkout>`.

## Measured

On one PC: RTX 3090 (24 GB, PCIe), i7-7800X (6 cores, AVX-512), 95 GiB RAM; `UD-Q4_K_XL` (87 GB), llama.cpp 0.7.0
(`620d2225c`) as the reference, 2026-10-15.

- **Parity** (`tools/nemotron_h/parity.py`, 5 chat prompts, thinking off, 32 teacher-forced steps each, 87 slots per
  layer): the first answer token matched on all 5; over the answers (122 steps up to `<|im_end|>`) mean KL 0.00106,
  median 0.000015, top-1 0.992 against llama.cpp CUDA (-cmoe).
- **A 981-token prompt** (the streamed prompt path): KL 0.00328 against llama.cpp CUDA, while llama.cpp's own CPU
  backend is at 0.00277 against the same; the per-sublayer error over the prompt's positions grows with depth exactly
  as llama.cpp CPU's does (layer 40: 0.176 vs 0.174, layer 80: 0.311 vs 0.302) - the text repeats one paragraph 12
  times, where both engines' rounding drifts the same way.
- **MTP**: with 2 and 3 drafts the greedy text is the same as without (96 tokens); 85% / 82% of the drafts accepted,
  but no faster on this PC (10.9 / 10.8 vs 11.2 tok/s): a verify window routes ~3x the distinct experts, and most of
  them are CPU misses. The config leaves `--mtp 0`.
- **Speed** (`strata-nh --bench-pp 2048 --bench-tg 32`, two runs): prompt 124-129 tok/s, decode 11.7-12.6 tok/s.
  Through the server: a 2795-token prompt in 27.2 s (103 tok/s), 256 tokens decoded at 10.7 tok/s.
- **The server**: text, thinking (reasoning separated), a tool call (`get_weather` with `{"city":"Kyiv"}`), a picture
  (a red square, a blue circle and the text "HELLO STRATA", described correctly; 300 image tokens, 59 ms in the
  encoder on the GPU), streaming, and the second turn of a conversation reusing the first (45 cached tokens).
- `--pin-experts`: page-locking the mapped GGUF failed on this PC (cudaHostRegister on the read-only mapping), the
  stream used its pinned buffers; same speed.
- The CPU experts against a dequantized f64 reference (`nh_cpu_test`, every MoE layer, 2 experts x 3 tokens): worst
  relative error 1.2% (the activations rounded to Q8_K / Q8_0, as llama.cpp's CPU backend does).

**With the Flash-Next mechanisms** (same PC, 2026-10-15 evening; 128-160 greedy tokens of one coding prompt unless
said, `--expert-slots 78` for the A/B rows so every row has the same VRAM tier):

| change | decode tok/s | note |
|---|---|---|
| expert arena, 4 KiB pages | 9.16 | the CPU experts at ~30 GB/s |
| expert arena, 2 MiB pages | **11.75** | +28%: ~0.094 ms an expert, ~40 GB/s = this RAM's measured read ceiling |
| + prefetch 2 / 4 / 8 | 11.62 / - / - | 4 KiB-page runs: +2.5% / +7% / +1%; 92-96% of the uploaded experts routed at 2-4 |
| + PCIe share 0.15 and prefetch 4 | 11.31 | the uploads read the same, already saturated RAM |
| + expert profile (10 mixed prompts), held-out prompt | 13.48 (11.83 without) | VRAM share of the routed experts 34% (16% without) |
| adaptive swaps 8 / 32 / 128 every 32, or 64 every 16 | 12.98 / 12.91 / 12.91 / 12.83 | 160 tokens; the rate does not matter, the profile does |

- `strata-nh --bench-pp 2048 --bench-tg 32` with the arena: prompt 337-350 tok/s (124-129 before), decode
  12.2-13.1 tok/s.
- Through the server (vision on the GPU, 63 slots a layer): a 2795-token prompt in 11.3 s (246 tok/s; 103 before), 256
  tokens decoded at 11.2 tok/s.
- A regenerated answer (the same prompt again) restored its turn checkpoint: the prompt in 88 ms instead of 2011 ms,
  the same text.
- Parity with the arena and the profile (2 chat prompts + the 981-token one): answers-only KL 0.00124, top-1 0.984;
  with the PCIe share 0.6 and prefetch 16 (82% of the routed entries on the GPU): KL 0.00279 over all steps, top-1
  0.979 - the GPU and CPU experts round their activations differently (q8_1 / Q8_K), the text stays the model's.
- The arena takes 70-130 s at start (from the page cache, or the disk when it was evicted) and leaves ~15 GiB of this
  PC's 95 GiB free.

Not done yet: MMQ for the prompt's experts (it takes a per-expert FP16 GEMM today), video input, HIP and SYCL builds,
a setup entry.
