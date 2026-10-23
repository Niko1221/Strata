# DeepSeek V4-Flash (GGUF architecture `deepseek4`)

This is a separate model family, not a Qwen3.8 variant: the engine path is `src/dsv4` (`strata::dsv4::run`)
and the build switch is `STRATA_HAVE_DSV4`. The Qwen4EXP gate in `gguf_reader.hpp` is untouched — a
`deepseek4` file is only accepted by this family, and a `qwen4exp` file only by the other one.

## What the model does differently

- 43 layers. Layers 0 and 1 route by **token-id hash** (`tid2eid`), not by a router matrix; layer 2 is
  hash + the sparse-attention indexer.
- Attention is **MLA**: 64 heads of 512-wide latent per position, a sliding window of 128, and a
  compressor that turns every 4th and every 128th position into a compressed slot. Even layers (`r=4`)
  also run the **DSA indexer** (a 256-wide scoring branch that picks the top 512 compressed slots);
  odd layers (`r=128`) only use the window plus their single compressed slot.
- 256 routed experts per layer, top-6, plus one shared expert; `moe_intermediate_size` 2048 on a 4096
  hidden size, so an expert is 3 × 4096 × 2048.
- The checkpoint is **NVFP4** (group 16) for the routed experts; attention, shared expert, embeddings
  and the head ship as Q8_0/BF16.

## What this port supports

- Loading a **canonical pack** (`pack/experts.bin` + `pack/dense.bin` + `pack/native_experts.txt`)
  produced from a converted GGUF, and running prefill and decode on one or two GPUs (layer split with
  the existing peer path).
- Decode with the sparse attention stack: window 128 | compressed slots (indexer top-512), 8-group
  output projection, inverse rotary on the attention output, YaRN rope for the compressor.
- `--dsv4-decode N` (decode N tokens), `--dsv4-forward` (one forward, dump with `--dsv4-dump-dir`),
  `--dsv4-verify-rows R` (verify R draft rows in one forward — the batching the engine's own speculator
  would use), and `--dsv4-mrow-bench R` for throughput.
- A CPU reference forward, `src/dsv4/dsv4ref.cpp` (binary `strata-dsv4ref`), which is what the dumps are
  compared against; `tools/dsv4/oracle.py` is an independent PyTorch reference that reads the original
  safetensors.

## What it does not support

- **No MTP/DSpark draft head.** The checkpoint has one (`mtp.0.*`) and it is not wired. Measured on a
  2×L20 with this expert pool, the block verify costs `c(5) ≈ 2.76` forwards per round while the accept
  rate gives ~2.4 accepted tokens, and the draft weights would take resident expert memory away; the
  model works without it.
- **Hadamard-folded packs (format v5) are refused at start-up.** The CPU native path rotates activations
  (`hadamard.hpp`); the CUDA GEMV here does not, so a folded pack would load and then compute
  plausible-looking garbage. A pure-v3 pack is what the conversion below produces.
- **Context is capped at 65536** (the compressor's slot table is sized from `ctx / 128`); `--ctx ` larger
  is rejected at start-up rather than overrunning.
- **Prefill is chunked at the 128-token window.** A prompt longer than 128 tokens must be fed in chunks
  by the caller; the engine rejects a longer single prefill instead of reading past the window.
- Only NVFP4 experts are wired. IQ2/IQ4 packs of this family are not converted or tested.

## Convert

The pinned llama.cpp in `third_party/llama.cpp` already knows the architecture (`deepseek4`) and writes
`compress_ratios` correctly, so the conversion is a thin layer over `tools/gguf-py`:

```bash
python tools/nvfp4_convert_dsv4.py --src <hf checkpoint dir> --out dsv4-flash-nvfp4.gguf
python tools/iq_pack.py --gguf dsv4-flash-nvfp4.gguf --out pack
```

`nvfp4_convert_dsv4.py` patches the ModelOpt NVFP4 tensor naming (`*_nvfp4.weight`,
`*_nvfp4.weight_scale`, `*_nvfp4.weight_scale_2`) and merges the per-expert tensors into
`ffn_{gate,up,down}_exps`. `--use-temp-file` is not optional on a 128 GB machine: `GGUFWriter` otherwise
holds the whole 164 GB in RAM before writing.

## Run

```bash
strata --model dsv4-flash-nvfp4.gguf --native dsv4-flash-nvfp4.gguf \
       --pack pack --ctx 32768 --dsv4-decode 64 --dsv4-prompt-ids prompt.txt
```

`strata` reads the GGUF header and hands the run to `dsv4::run` when the architecture is `deepseek4`;
`strata-dsv4` is the same entry point on its own command line. A pack from a `deepseek4` GGUF carries
`native_experts.txt` with the 43-layer table, so `--pack` alone is enough.

### Prefill attention batching

A chunked prefill pass runs 128 rows through one attention block, and the default path used to launch one
MQA-512 grid per row (4 % of the SMs) with a host-built topk table.  `STRATA_DSV4_ATTN_BATCH` (now **on by
default**) batches the row dimension instead: one launch for the whole chunk, tables built on the device,
the same per-row fp32 accumulation order.  It is bit-identical on the 176-file prefill gate and the 4k
chunked gate, and on 2xL20 a 4k prompt takes 98.8 s with it against 170.5 s without (same binary,
same pass).  `STRATA_DSV4_ATTN_BATCH=0`
restores the per-row path for A/B runs.

## Validation

The three test binaries in `src/dsv4` (`strata-dsv4-attn-test`, `strata-dsv4-comp-test`,
`strata-dsv4-probe`) gate the engine against the reference forward:

- golden prompts: prefill argmax and three decode steps match the reference exactly;
- multi-row verify R=2..6 vs single-row decode: bit-identical dumps;
- long context (start_pos 130, window wrapped): bit-identical;
- `compute-sanitizer --tool memcheck`: 0 errors.

Expected numbers on 2×L20 (96 GiB VRAM, 125 GiB RAM, expert pool 145.13 GiB, 57.5 % resident):
~30 t/s for one decode stream (33 ms/pass) and ~47 t/s with six rows in one forward (127 ms/pass).
