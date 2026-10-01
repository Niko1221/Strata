# Unsloth UD-Q4_K_XL (experimental, manual import)

**Experimental, not in setup yet.** Engine 0.1.31 or newer. Validated on one PC: Windows 10, an RTX 5070 (12 GB),
64 GB of RAM and an AVX-512 CPU, on 2026-10-01. Everything below is a manual workflow; `START-HERE.bat` / `setup.sh`
do not offer this model.

The target is Unsloth's 4-bit quantization of the same model the other packs use:
[unsloth/Qwen3.8-Flash-Next-GGUF, `UD-Q4_K_XL`, revision `38bb39e`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL).
"UD-Q4_K_XL" is a mix of formats, not Q4_K everywhere: the routed experts are Q4_K (Q5_K in layer 2) for gate/up and
Q5_1 (Q8_0 in layers 2, 4, 30, 46, 47) for down; the token embedding, the output head, the attention and shared-expert
projections and the PLE key are Q8_0; the 28.8 GB PLE table is IQ4_NL. The routed experts are 71.7 GiB, about twice
the 3-bit models'.

## The files

Four shards, 111,334,654,784 bytes (111.3 GB) together. Shard 1 holds only the metadata; layer 11's down projection is
in shard 2 and its gate/up in shard 3, which this engine handles (`native_experts.txt` v4).

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | 10,946,624 | `4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` | 49,859,583,136 | `3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` | 49,376,141,504 | `56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` | 12,087,983,520 | `753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a` |

Keep the four files together in one folder under these names (the engine finds shards 2-4 from shard 1's name; a
missing shard is an error that names it). Check them before packing (`sha256sum -c`, or `Get-FileHash` on Windows).
Hugging Face snapshot symlinks work if you pass the snapshot's file name, not the hash-named blob it points to.

## What it needs

| | |
| --- | --- |
| Disk | 111.3 GB for the four files, 1.4 GB for the pack, ~6 GB for the MTP draft layer if you have none yet. **No `experts.bin`**: the engine reads the experts from the GGUF files in place. An NVMe SSD matters: the experts that fit neither VRAM nor the RAM budget are read from it for every token. |
| RAM | 64 GB measured. The RAM budget (below) holds the most-used experts; the rest come from the SSD through the OS file cache. |
| GPU | 12 GB measured (RTX 5070): after the weights, the draft layer and the buffers, the expert cache held 1,280 of the 24,576 experts (3.7 GiB) at 4K context. |

## The pack

Build Strata (or install 0.1.31+), then, from the repository root, with gguf-py from the pinned llama.cpp
(setup installs it; `STRATA_GGUF_PY` can point to its `gguf-py` folder):

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --out packs/ud-q4_k_xl --compat-bf16
```

About 30 seconds. It writes `index.txt`, `dense.bin` (1.4 GB), `native_experts.txt`, the tokenizer, `conversions.json`
and `compat-bf16.json`. **`--compat-bf16` is required**: this file stores 195 small projections (the hyper-connection
up/down matrices, `output_hc_*` and the PLE value) as Q8_0, while the engine reads them as BF16. Without the flag the
packer refuses and writes nothing; with it, they are dequantized and rounded to BF16 (nearest-even). Do not add
`--experts-bin`: it would write a 77 GB copy of the experts that this mode does not need.

## The numbers: what is native and what is converted

The routed experts, the token embedding, the output head, the attention and shared-expert projections, the PLE key and
the PLE table are used as the GGUF stores them (the engine has GPU kernels and CPU (ggml) paths for Q4_K, Q5_K, Q5_1 and
Q8_0). The small tensors the engine reads as floats are converted, and `conversions.json` lists every one with its
source/destination type, method, whether it is exact, the largest absolute error and the source bytes' SHA-256:
264 F32 tensors (routers, injections, SSM gates) whose values are already BF16's are stored as BF16 exactly; the 195
Q8_0 projections above are rounded to BF16 (largest absolute error 0.0144, in `hc_ffn_up`); the PLE convolution (F32)
is narrowed to F16 when the engine loads it (largest error 3e-8). That is the compatibility Unsloth's file needs; it is
not the original BF16 checkpoint. The output has not yet been compared with llama.cpp on the same GGUF (argmax
agreement); treat quality as unmeasured.

## The MTP draft layer

The draft layer is the base model's, the same one the other packs use. If setup installed Strata, you have it
(`mtp/rt` in `Strata-data`). Otherwise build it as [docs/ORCA.md](ORCA.md#preparation) shows (`tools/mtp_fetch.py`,
`tools/mtp_pack.py`, `tools/mtp_rt.py`, then copy `data/draft_vocab.bin` to `mtp/rt/`).

## The server

Save as `strata-ud-q4_k_xl.json` at the repository root (or next to setup's other `strata-*.json`), with `/path/to/`
pointing at **shard 1**. `--ple-gguf` is not needed: the engine finds the shard that holds the PLE table
(`per_layer_token_embd.weight`, shard 2) by name. `--resident-budget-gib` turns on the mapped mode with a RAM budget
(it implies `--mmap-experts`).

```json
{
  "exe": "engine/strata.exe",
  "args": [
    "--pack", "packs/ud-q4_k_xl",
    "--native", "/path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
    "--resident-budget-gib", "40",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "8192"
  ],
  "cwd": ".",
  "tokenizer": "packs/ud-q4_k_xl/tokenizer",
  "model_name": "qwen3.8-flash-next-ud-q4_k_xl",
  "log": "strata-ud-q4_k_xl.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

```sh
.venv/bin/python -m serve.server --engine strata --config strata-ud-q4_k_xl.json --port 8080
```

(`engine/strata.exe` is setup's engine on Windows; a self-built one is `build/strata`.) The expert profile is the base
model's (`data/expert-profile.bin`); it ranks the same 48 x 512 experts and decides which ones the GPU cache and the RAM
budget take first.

**Choosing the RAM budget N:** your RAM minus 20-24 GB (the OS, the engine itself, the 126 MB router copy, and room for
the OS file cache that serves the rest). On 64 GB, 40. The engine clamps a budget larger than the RAM it finds free
(minus 4 GB) and says so. Everything above N comes from the SSD for every token, so N is the setting that matters
most; a bigger budget was faster in every measurement (24 / 32 / 40 GiB). When the driver page-locks the whole budget
(24 GiB did on this PC, 32 and 40 did not), the GPU also computes a share of the misses over PCIe, as in the resident
low-RAM mode; `STRATA_RESIDENT_PIN=0` keeps the budget locked only (the CPU then computes every miss).

**Context:** measured at 4K. The KV cache takes VRAM from the expert cache, so a longer context makes decoding slower;
8K is a reasonable start on 12 GB. `--kv int8` halves the KV cache's VRAM.

## What to expect (RTX 5070 12 GB, 64 GB RAM, Windows, one short greedy prompt)

| RAM budget | Greedy output | Notes |
| --- | --- | --- |
| 24 GiB | 5.1-5.2 tok/s (2 runs, before the routing prefetch below) | 2.2 GB read from the SSD per verify round (~3.5 tokens) |
| 40 GiB | **7-8.5 tok/s** (6 runs: 6.7-8.6, mean 7.9) | 1.1 GB read from the SSD per verify round, ~0.33 GB per token |

- The first answer comes about 60 s after the engine starts (mostly loading: the RAM budget is copied from the GGUF
  files at start). A 30-token prompt then took 5-6 s. Longer prompts have not been measured and will be slow: the
  prompt path dequantizes these formats to FP16 (fast prompt kernels for Q4_K/Q5_K/Q5_1 are future work).
- The speed varies from run to run (6.7-8.6 tok/s at 40 GiB for the same prompt), with what the OS file cache holds.
- For comparison, IQ3_S (all its experts in RAM) writes ~53 tok/s on the same PC ([the speed tables](DETAILS.md#speed-measured)).
- Where the time goes (`--stats` on the command line): at 40 GiB about 550-750 ms of each verify round (3.5 tokens) is
  reading experts from the SSD, ~75 ms the CPU's expert kernels, ~13 ms the GPU.

## Opt-ins and switches (environment variables)

| | |
| --- | --- |
| `STRATA_LOOKAHEAD=0` | Turns off the routing-aware prefetch (on by default in this mode): while the CPU works on a layer, a thread applies the next layer's router to this layer's input and asks the OS to read the predicted experts' pages. Pages only, the answers are the same. About half of the SSD reads were predicted; mean +14% (6.9 -> 7.9 tok/s). `STRATA_LOOKAHEAD_K` sets the experts per token (default 10). |
| `STRATA_KQ256=1` | Multi-token AVX2 kernels for the Q4_K / Q5_1 / Q8_0 experts. Bit-exact with ggml's, but measured no faster, so off. |
| `STRATA_PARTIAL_PIN=1` | Registers the hottest part of the RAM budget (up to `STRATA_PARTIAL_PIN_GIB`, default 24) with the GPU driver, so the GPU computes a share of the misses over PCIe (`--pcie-frac`). Measured no faster on this PC, and it changes the numerics of those experts (GPU kernels instead of the CPU's), so off. |
| `STRATA_FETCH_THREADS=N` | Threads that read the experts from the GGUF (default 8; 16 was no faster). |

## Scope and validation

- Only UD-Q4_K_XL at revision `38bb39e` is targeted. Other Unsloth quantizations use formats this engine may not
  have kernels for; the engine checks every layer's formats at start and refuses an unsupported one by name.
- Tests: the packer's synthetic 4-shard and conversion tests (`.venv/bin/python -m unittest discover -s tools -p
  test_iq_pack.py`); CTests `gguf_split_test`, `expert_layout_test`, `native_expert_parity_*` (the three real expert
  format pairs against ggml-cpu, the Q5_1 min term, Q8_0 rows); the in-place mode against `experts.bin` on the Coder
  (identical tokens and logits).
- Real runs: greedy answers to a coding prompt (correct) at every budget and setting above, identical across them. Not
  yet: llama.cpp argmax agreement, long prompts, sampled decoding, long conversations. Please report what you see.

## Credits

The split-artifact loader, the expert formats' GPU kernels and the packing of Unsloth's files follow
[eddoursul/Strata](https://github.com/eddoursul/Strata) (PR #245), which ran these files first. The Q8_0 kernels, the
per-role shard column and the PLE convolution's F32 -> F16 fix come from @gopinath87607's PR #255; per-role shard names
were also proposed by @jagsan-cyber in PR #247. The quantization is [Unsloth](https://huggingface.co/unsloth)'s; the
model is Qwen's ([license](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).
