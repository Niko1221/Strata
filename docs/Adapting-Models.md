# Adapting another model to Strata

Strata's app, installer and `strata launch` ship a fixed list of models, each with a pinned file, size and
SHA-256. This page is for a GGUF that is **not** on that list: a fine-tune, another quantization, or a file that
mixes two quantization levels ("hybrid"). It describes the route that works today, the five steps, and where it
stops.

Models that already went this way have their own pages: [OrcaRouter IQ3_XXS](ORCA.md),
[OrcaRouter Q4_K_S](ORCA_Q4_K_S.md), [Gyro](GYRO.md), [Unsloth UD-Q4_K_XL](UNSLOTH_Q4.md),
[UD-Q6_K_XL](UNSLOTH_Q6.md), [NVFP4](NVFP4.md), [Intel Arc](INTEL.md), [Radeon 780M](RADEON_780M.md).
Read this page first, then the one closest to your file.

> **In short:** all shards in one folder → `tools/iq_pack.py` → MTP draft layer (optional) → a server JSON with
> `--native`, `--ple-gguf` and `--mtp` → start it and measure it yourself.

## What runs, and what does not

The engine has a path for the **Qwen3.8-Flash-Next** geometry (the engine calls it `qwen4exp`) and a separate
one for **Qwen3.6-35B-A3B** ([MODELS.md](MODELS.md#qwen36-35b-a3b)). This page's packing route is the `qwen4exp`
one; a Qwen3.6 fine-tune is not covered here. A GGUF of any other architecture is refused when the engine loads it,
whatever the packer did with it.

The **Mac (Metal) engine** and the **Intel Arc engine** are built from the same sources but not every path is
validated on them; the Arc engine has no picture reader yet. Check the per-model page or
[TEST_REQUESTS.md](TEST_REQUESTS.md) before you count on a card the maintainers do not have.

The model lists in the app, in setup and in `strata launch` stay as they are. `strata models import <folder> --plan`
inspects a folder and prints what it sees, but it is a preview: it does not adopt, copy or download anything
([TERMINAL.md](TERMINAL.md)). Adapting a model is a local, explicit step.

## The checklist

A model needs all of this. If one line fails, the engine refuses the load or answers wrongly, and the message
usually names the missing piece.

| What | Why | Where it is checked |
|---|---|---|
| **Architecture** | only `qwen4exp` (Qwen3.8-Flash-Next) and the Qwen3.6 path | engine start; `strata models import --plan` |
| **Every tensor type known** | the packer writes them, the kernels run them | `tools/iq_pack.py` output; [QUANTS.md](QUANTS.md) |
| **All shards, one folder, bounded names** | the packer finds the others by shard 1's name | `...-0000N-of-0000M.gguf` |
| **The PLE table's shard** | the 28.8 GB lookup table stays on disk and is read from its GGUF | model card; engine message |
| **A picture reader (`mmproj`)** | only for pictures | the model's own mmproj file |
| **A tokenizer** | comes out of the GGUF into the pack | `tokenizer/` in the pack |
| **An MTP draft layer** | optional; without it the server drafts by lookup only | `tools/mtp_*` |

## Not all tensors cost the same

"Q2" or "3-bit" is an average over very different tensors, and the average hides where the loss sits.

- The **routed experts** are the cheap place to save. A token uses 10 of 512 per layer, so one expert's
  rounding error reaches few tokens and averages out over the mixture.
- The **tensors read on every token** carry the loss: attention, the shared expert, the embeddings, the output
  head, the PLE key, the hyper-connection projections. Rounding there applies to every token.

Two measurements show the split:

- In Unsloth's UD-IQ4_XS, the smallest of their files (93.7 GB), everything except the routed experts is Q8_0;
  only the experts are IQ3_S/IQ4_NL ([QUANTS.md](QUANTS.md)).
- In shefowl's hybrid (IQ3_XXS trunk, Q2_0 routed experts) the routed experts alone cost +2.6% perplexity against
  IQ3_XXS. The full Q2_0 tier, which also cheapens the dense part (shared experts Q2_0, attention Q3_K, `ple_key`
  Q2_0), costs +4.3%. Those are the uploader's numbers, on his hardware, not this repository's.

So a file's name is not its quality: read what each tensor stores with `tools/quant_formats.py` before you pack it.

## Step 1 — Get the files and check them

```sh
hf download <owner>/<repo> --local-dir /path/to/models/<quant>
```

- Every shard must land in **one** folder, and keep its `-0000N-of-0000M.gguf` name. The packer reads the other
  shards beside the one you pass; a missing shard is an error, not a smaller model.
- Compare size and SHA-256 with the repository's file list before you pack. Nothing checks provenance for you.
- **Do not rename** the files to a name setup knows. Setup compares size and SHA-256 of its own pinned files and
  will stop with a mismatch; a file that pretends to be a shipped model only produces a broken install.
- Xet, the Hub's transfer protocol, writes its progress in the background: the target folder can stay still for a
  while and then jump. On the PC this page was written on (Windows, RX 9060 XT 16 GB, 2026-10-25) Xet moved
  4-7 MB/s while a plain HTTP range request managed 0.3 MB/s, so keep Xet on. The live number is in
  `%USERPROFILE%\.cache\huggingface\xet\logs` (newest file, field `observed bytes sent so far`).

`tools/quant_formats.py` reads the GGUF headers only and prints what each tensor stores, before you download
everything or pack anything:

```sh
.venv/bin/python tools/quant_formats.py /path/to/models/<quant>/*.gguf
```

## Step 2 — Build the pack

The pack holds what the engine cannot read from the GGUF directly; the experts stay in the GGUF unless you ask for
`experts.bin`.

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/models/<quant>/<model>-00001-of-00002.gguf \
  --out packs/<name>
```

It writes `dense.bin`, `index.txt`, `native_experts.txt`, `conversions.json` and `tokenizer/`.
`native_experts.txt` is written last, so a pack without it is not finished.

| Flag | When |
|---|---|
| `--compat-bf16` | ordinary quants whose small projections are not float: the packer dequantizes them to BF16, with rounding. Experts, embeddings and the PLE table stay unchanged |
| `--experts-bin` | also write `experts.bin`. Needed for `--mmap-experts` and `--resident-experts`; costs about as much disk as the experts ([AMD_HIP.md](AMD_HIP.md)) |
| `--base <pack>` | reuse a Q2_0 pack's `dense.bin` when the float tensors are byte-identical |
| `--skip-experts` | rewrite the index only |

Measured on the OrcaRouter IQ3_XXS file (two shards, 85.20 GB): 460 tensors converted, 1.39 GiB of converted BF16
weights, 1.43 GiB total dense weights ([ORCA.md](ORCA.md)). The OrcaRouter Q4_K_S pack comes out at about the same
1.43 GiB ([ORCA_Q4_K_S.md](ORCA_Q4_K_S.md)).

A pack with nothing to convert is quick: on a 67.2 GB hybrid (two shards, IQ3_XXS trunk and Q2_0 experts) the
command took **5 s** and wrote 1.55 GB, on a Windows PC with an NVMe SSD (2026-10-25). The packer reads each shard's
tensor directory and the 1.43 GiB of float tensors, not the 34 GB of expert payloads; `--compat-bf16` reads the
tensors it converts.

`iq_pack.py` needs NumPy and the `gguf-py` module from the pinned llama.cpp dependency (`STRATA_GGUF_PY` can point
to that directory). In this checkout, `.venv` has both.

## Step 3 — The MTP draft layer (optional)

One runtime per expert quantization, not per model. Without `--mtp` the server still answers; it just drafts by
lookup only, which is slower.

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

`--experts` names the draft layer's expert quantization: `q2_0`, `q4_0` or `q8_0`. The draft head comes from the
original model and the target verifies every proposal, so a fine-tune's acceptance rate can differ from the base
model's. Measure it; do not assume it.

## Step 4 — Write the config and start the server

Save it beside the checkout as `strata-<name>.json` and replace the paths. Two of them matter most: `--native`
points at **shard 1** and `--ple-gguf` at **the shard that holds the PLE table**, which is not always the same
shard (GSQ-RCO: shard 2; Orca and Swift 1.5: shard 1).

```json
{
  "exe": "/path/to/engine/strata",
  "args": [
    "--pack", "packs/<name>",
    "--native", "/path/to/models/<quant>/<model>-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/models/<quant>/<model>-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt",
    "--max-context", "32768", "--kv", "int8", "--vram-reserve-mib", "1024",
    "--vision"
  ],
  "cwd": "/path/to/strata",
  "tokenizer": "packs/<name>/tokenizer",
  "model_name": "<name>",
  "log": "strata-<name>.log",
  "host": "127.0.0.1",
  "port": 7602,
  "vision": {
    "exe": "/path/to/engine/strata-vision",
    "mmproj": "/path/to/models/<quant>/mmproj-....gguf",
    "model": "/path/to/models/<quant>/<model>-00001-of-00002.gguf",
    "gpu": false,
    "max_tokens": 300,
    "threads": 8
  }
}
```

```sh
.venv/bin/python -m serve.server --engine strata --config strata-<name>.json --port 7602 --open
```

- `--vram-reserve-mib 1024` leaves room for the desktop. If windows stutter or disappear, raise it to 2048 or 3072.
- `--max-context 32768` is a quiet start. A larger window costs VRAM for the KV cache; `--kv int8` halves it.
- `"gpu": false` reads pictures on the CPU. That is the usual choice under 24 GB of graphics memory on AMD
  ([DETAILS.md](DETAILS.md)).
- On AMD set `"backend": "hip"` and `"lib_dirs": ["/opt/rocm/lib"]` (Windows: `C:\Program Files\AMD\ROCm\7.1\bin`).
  On NVIDIA `"cuda"`. The engine package's own start scripts are the reference for your card.
- With little RAM, build the pack with `--experts-bin` and add `--mmap-experts`; the OS then keeps what fits in RAM
  ([AMD_HIP.md](AMD_HIP.md), [DETAILS.md](DETAILS.md)).

## Step 5 — Measure it

A model the maintainers never ran has no speed and no quality here. The uploader's numbers are the uploader's:
different card, different build, different context.

Write down, at least: engine version, backend and card, RAM, quantization, context length, prompt-processing rate
on a fixed prompt, decode rate on a fixed prompt, and what the answer looked like. Then compare with a same-day run
of a shipped model on the same PC. [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) and
[COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) describe the form; a single short measurement is a data point,
not a throughput claim.

Quality is the harder half. A fine-tune can be better at its specialty and worse elsewhere; a hybrid mixes two
quantization levels, so its experts are noisier than its trunk. The only honest statement without a measurement is
"not measured on this PC".

## Limits

- **No provenance check.** You compare size and SHA-256; the packer trusts the file.
- **The shipped lists do not change.** The app, setup and `strata launch` keep their pinned models, hashes and
  memory tiers. A local pack is a local pack.
- **One model per pack.** Tokenizer, `dense.bin` and `native_experts.txt` belong together; do not mix packs.
- **The PLE table stays on disk.** It is the 28.8 GB `per_layer_token_embd` tensor, read from its GGUF; keep that
  shard on an SSD.
- **Quantization support is a list, not a promise.** A type outside [QUANTS.md](QUANTS.md) and
  [NVFP4.md](NVFP4.md) is refused, not approximated.
- **Licenses travel with the weights.** Qwen3.8-Flash-Next is under the Qwen Community License; a fine-tune has
  its own terms. Read them before you pass a pack on.

## If something goes wrong

| Message or symptom | Cause and fix |
|---|---|
| `unknown architecture` at start | not a `qwen4exp` file; this route does not apply |
| `iq_pack.py`: missing shard | all shards must sit in one folder, names `-0000N-of-0000M.gguf` |
| `iq_pack.py`: unsupported type | the quant is outside the known list; try `--compat-bf16`, or take another quant |
| `PLE` message at start | `--ple-gguf` points at the wrong shard; swap the shards |
| Answers are garbage, no error | usually a wrong `--ple-gguf` or a mixed pack/tokenizer |
| `--mmap-experts` refuses | the pack has no `experts.bin`; rebuild with `--experts-bin` |
| Desktop stutters, windows vanish | raise `--vram-reserve-mib` |
| No draft speed-up | `--mtp` missing or built for another expert quant |
| Engine loads, then exits on a big context | the KV cache does not fit; lower `--max-context` or keep `--kv int8` |

## Sources

- Per-model routes: [ORCA.md](ORCA.md), [ORCA_Q4_K_S.md](ORCA_Q4_K_S.md), [GYRO.md](GYRO.md),
  [UNSLOTH_Q4.md](UNSLOTH_Q4.md), [UNSLOTH_Q6.md](UNSLOTH_Q6.md), [NVFP4.md](NVFP4.md), [INTEL.md](INTEL.md),
  [RADEON_780M.md](RADEON_780M.md)
- Formats and kernels: [QUANTS.md](QUANTS.md); engine internals and settings: [DETAILS.md](DETAILS.md)
- AMD specifics: [AMD_HIP.md](AMD_HIP.md); several cards: [MULTI_GPU.md](MULTI_GPU.md)
- Measuring: [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md), [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)
- The numbers on this page are from the pages linked beside them, or measured on the Windows PC named in
  Step 1. Nothing here was measured on a card the maintainers do not have.
