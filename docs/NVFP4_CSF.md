# FP4-Local / CSF text conversion (experimental)

`tools/nvfp4_csf_convert.py` converts the text weights of
[`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
for Strata's existing NVFP4 expert and BF16 PLE readers. Run it explicitly;
the normal installer, runtime and `nvfp4_convert.py` path are unchanged.
Full-model serving, quality and throughput have not been validated. Vision
and MTP weights are excluded; this tool does not produce their sidecars.

## What it writes

- **Main GGUF:** CSF streams become the original E4M3 expert scales. Routed
  expert codes and global scales are retained, and the pinned llama.cpp
  converter repacks them as NVFP4. MXFP8 and non-expert NVFP4 weights are
  decoded first, then follow `nvfp4_convert.py`'s BF16/F16/Q8_0 policy. This
  second conversion rounds dense weights; it is not a lossless conversion
  of the complete checkpoint.
- **PLE GGUF:** NVFP4 n-gram shards are read in row chunks, decoded and
  rounded to BF16. The result is one `per_layer_token_embd.weight` tensor,
  in numerical shard order, with 160 values per row. For FP4-Local this is
  102,400,491,520 payload bytes (about 95.4 GiB), plus a small GGUF header.
- Each completed output gets a `.manifest.json` describing the precision
  changes and exclusions. No source checkpoint file is written.

The CSF codec is `byte-window4-fixed-stream-u24-exceptions/1`: a base byte
per row, 4-bit offsets and a list of 24-bit positions with replacement
bytes. The format reference and an independent loader implementation are
in [vLLM PR #60646](https://github.com/vllm-project/vllm/pull/60646).
This conversion expands the scales; it does not keep CSF's runtime memory
savings or require new GPU kernels.

## Run

Use setup's pinned `third_party/llama.cpp` source, and a Python environment
with numpy, torch, transformers, safetensors, sentencepiece and protobuf.
The main GGUF groups experts by layer before repacking and uses llama.cpp's
temporary-file writer; place temporary files
on a drive with enough room for the expert data. The final GGUF, temporary
GGUF storage, PLE table and subsequent expert pack each need disk space.

Inspect metadata and output sizes first. This reads shard headers, not
tensor payloads, and writes no output:

```sh
python tools/nvfp4_csf_convert.py --model /path/to/checkpoint --dry-run
```

Convert the main GGUF and stream the PLE table:

```sh
python tools/nvfp4_csf_convert.py --model /path/to/checkpoint \
  --outfile /path/to/output/main.gguf --ple-outfile /path/to/output/ple-bf16.gguf
python tools/iq_pack.py --gguf /path/to/output/main.gguf --out /path/to/output/pack
```

Either output flag can be used alone. `--chunk-rows` sets the PLE chunk
size (default 16384); it does not change the output bytes. Existing output
files, manifests and partial files are refused. A failed write can leave
a `.part` file for inspection. Review it before removing it or retrying.

The engine would use the pack and main GGUF as in `NVFP4.md`, with
`--ple-gguf /path/to/output/ple-bf16.gguf`. This is not an end-to-end tested
launch recipe. Use a RAM budget on hosts that cannot hold the expert arena;
the checkpoint being small enough on disk does not establish that it fits
in RAM or VRAM.

## Validation

```sh
python tools/test_nvfp4_csf.py
python tools/test_nvfp4_csf_convert.py
```

CPU tests cover the byte codec, stream validation, BF16 rounding, MXFP8
decoding, numerical PLE shard/row order across chunk sizes, metadata-only
planning and the actual pinned Python converter writing a small GGUF. A
multi-shard fixture checks that interleaved expert layers flush within one
layer's buffer budget rather than accumulating the entire expert arena.
The opt-in adapters are restored on context exit.

The initial sample experiment used
`FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300` and the uncompressed
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4@6909a5bed089a48fa07e956d3915af2537de9368`:

- Nine expert projections (gate/up/down at layer/expert 0/0, 17/311 and
  47/511) had identical decoded scale, code and global-scale bytes. Their
  packed bytes matched the pinned converter and dequantized values matched
  independently, without rounding.
- Strata's existing `nvfp4_expert_gpu_parity` passed all three expert blobs
  on Ubuntu with an RTX 5090 32 GB and Core Ultra 9 285K. Against its double
  reference, output relative L2 errors were 1.096%, 1.156% and 1.159%
  (the benchmark's threshold is 3%; its activation quantization remains).
- One MXFP8 shared-expert projection matched a PyTorch reconstruction.
  Q8_0 re-quantization added 0.709% relative weight L2 error.
- Sixteen NVFP4 PLE rows rounded to the same BF16 bytes as PyTorch; rounding
  added 0.141% relative weight L2 error.

These measurements validate sampled tensor paths. They do not measure
full-model answer quality, model throughput, memory use, vision or MTP.

The new command was also run end to end on a bounded fixture containing
those nine real projections, one real MXFP8 dense projection, sampled
embeddings and n-gram rows, the original hash constants and tokenizer.
The experts were remapped into one layer for this file-conversion check;
the fixture is not a runnable language model. All nine packed projection
hashes matched the independent sample experiment, and the separate PLE
output was readable as BF16 with the expected shape and row order.
