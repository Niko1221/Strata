# GLM-5.3-Flash on Strata: deployment guide (2× V100)

This guide deploys the GLM-5.3-Flash engine path (`strata-glm`) and the
multimodal extension (`strata-vision` + GENI) on a dual-GPU workstation, and
reproduces every number in it.  A Chinese version sits next to this file
(`GLM53-FLASH-DEPLOY-ZH.md`).

Everything measured below was measured on this exact machine:

- 2× Tesla V100-SXM2-32GB (sm70), one PCIe host
- 64 GB host RAM, Ubuntu 24.04, CUDA 12.x toolkit (`/usr/local/cuda`)
- model: `GLM-5.3-Flash-UNCENSORED` (dealignai's FP8 export, re-quantized by us
  to a UD-Q2_K_XL-grade mixed-precision GGUF; see §4)

## 1. What the feature adds

- **`strata-glm`**, a second engine binary behind `STRATA_BUILD_GLM=ON` (opt-in,
  default off — the default path stays byte-identical).  It runs the glm5-next
  architecture: 45 blocks, 288-expert MoE with a shared expert, GDN (gated
  delta net) attention, a DSA indexer on 12 layers, and the NextN/MTP draft
  block (blk.45).  It serves on stdin/stdout (`GEN`, `GENI`, `LOGP`, `STOP`,
  `QUIT`) and reuses the engine's expert-cache scheme: hot experts in VRAM
  (two pools, one per card), the working set on a pinned RAM tier, the rest
  read from the model file.
- **A quantization pipeline** for the 355B-class MoE: mainline-llama.cpp
  conversion → Q8_0 intermediate → imatrix → per-tensor type map → final
  mixed-precision GGUF → `tools/iq_pack.py --compat-bf16` pack (index +
  tokenizer + dense side-file the engine reads).
- **A terminal integration**: the family shows up as `glm53` in
  `strata launch --plan/--family glm53`, plus `tools/glm/start.sh` for a
  one-command OpenAI-compatible LAN server.
- **Multimodal**: `tools/vision/strata_vision.cpp` (the image encoder process)
  and the engine's GENI wire command.  Image rows replace the model's
  `<|image_pad|>` run; GLM image tokens take no rope positions of their own.
- **chat and test tools**: `tools/glm/chat.py` (engine + tokenizer + GLM chat
  template in one command), tokenizer tests, pack tests.

## 2. Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DSTRATA_BUILD_GLM=ON
cmake --build build --target strata-glm -j8
```

Notes:

- `GGML_CUDA` stays OFF: the GLM engine carries its own CUDA kernels
  (`src/glm_port`, attention/expert/mtp kernels), it does not go through
  ggml-cuda.
- `CMAKE_CUDA_ARCHITECTURES=70` matches the V100.  For other cards set yours.
- Only CUDA was built and tested.  HIP/SYCL are untouched by this feature and
  were not built.

## 3. The model file and the pack

The engine reads a *pack* directory (an `index.txt` plus tokenizer files plus a
dense side-file) and the experts from the GGUF next to it.  Two ways to get
them:

**a) Use the produced artifacts** (what this guide's numbers use):
`GLM-5.3-Flash-UNCENSORED-UD-Q2_K_XL.gguf` (108.8 GB, 1412 tensors) and its
`glm-pack/` (1.6 GB).  Put them in the same directory.

**b) Quantize it yourself** from a safetensors source (the UNCENSORED FP8
release; the author's NVFP4 export is broken, see the runbook).  The pipeline,
short form:

```sh
# 1. HF->GGUF conversion (mainline llama.cpp converter), FP8 tensors decoded to F16/BF16
python convert_hf_to_gguf.py <fp8-src-dir> --outfile GGUF_A_q8_0.gguf  # q8_0 intermediate
# 2. imatrix over ~34 calibration chunks (calibration text in tools/glm/build_calib_corpus.py)
./llama-quantize --imatrix imatrix.dat GGUF_A_q8_0.gguf mid.f16.gguf ...      # (see runbook)
# 3. per-tensor type map (the reference UD-Q2_K_XL's map, tools/extract_ref_typemap.py)
./llama-quantize --tensor-type-file tensor_types_iq.txt mid.gguf final.gguf
# 4. the pack the engine reads
python tools/iq_pack.py --gguf final.gguf --out glm-pack --compat-bf16
```

Download sources: prefer ModelScope, then hf-mirror.  Disk: keep ≥ 250 GB free
(source + intermediate + final).

`--hc-npz`: the UNCENSORED FP8 export carries the hc_* recursion tensors
crushed by a broken safetensors header.  `tools/glm/nvfp4_dequant_gguf.py
--hc-npz <official-values.npz>` grafts the official values during dequant (270
tensors).  Without this step those layers decode from zero-filled scales.

## 4. Run the engine

```sh
./build/strata-glm \
  --glm-pack /models/GLM-5.3-Flash-UNCENSORED-UD-Q2/glm-pack \
  --serve --max-context 8192 --kv int8 --parallel 1
```

- `READY 8192 stop pause` on stdout when warm (~4-5 min: dense weights, the
  MTP block, the expert pools, the RAM tier).
- Layer split is automatic: layers [0,24) on CUDA0, [24,45) on CUDA1.
- Measured steady state (8192 ctx, kv int8): VRAM ~31.8 GB/card (dense ~3.5 G
  + expert pool ~27.5 G + KV + drafter), host RSS ~40 GB (the pinned RAM tier),
  disk reads 0.00/token once warm.
- **Host memory rule**: the engine plus the OS must stay under 90% of RAM
  (here: 57.7 GB of 64 GB).  At 8192 ctx the engine sits at ~40 GB anon — fine.
  Do not wrap the engine in a `MemoryMax` cgroup limit: the 108 GB GGUF's page
  cache is charged to the unit and systemd-oomd kills the engine on the
  resulting pressure (measured twice).  Watch `MemAvailable` instead.
- Requests, one per line on stdin (what `tools/glm/chat.py` speaks):
  - `GEN <max_new> [sampling keys] <id,id,...>`
  - `GENI <max_new> <embeddings-file> <id,id,...>` — images (§6)
  - `LOGP`, `STOP`, `QUIT`.

## 5. Serve to apps

```sh
tools/glm/start.sh            # one command: engine + OpenAI-compatible server on :8000
strata connect <app> --check  # send one prompt and one tool call the way that app does
```

`start.sh` runs the engine, starts `serve/server.py` in front of it
(OpenAI/Anthropic-compatible, /v1/chat/completions), and rewrites GLM tool
calls.  Measured: two concurrent conversations at 512K context each (1M tokens
verified) on this machine.

A hand-written serve config (`--config`, a custom model) needs two keys:
`"arch": "glm5-next"` (without it the family reads as qwen and picture requests
take the wrong vision branch) and `"reasoning_effort": "high"` (without it
requests default to low; the official base does not think at low, but this model
still does, with no markers, so its thinking lands in the answer's text with
nothing to split on).  Bigger contexts decode slower through expert-pool
coverage (the KV takes the VRAM the pool wants; pool-external experts arrive
over PCIe): 512K x 2 covers 81% of the routed experts -> 22.6 tok/s; 1M x 2
covers 64% -> 10.5 tok/s (same prompt, 150 tokens).  524288 x 2 is the
recommended point on 2x32 GB.

## 6. Multimodal (images)

The text model has no vision weights; vision comes from a separate **mmproj**
file (vision encoder + projector, glm5v projector, 448 px / patch 14, 348
tensors).  We used unsloth's base-model `mmproj-F16.gguf` (1.13 GB); the
UNCENSORED source ships no vision weights, so the projector is the base
model's.

Build the encoder (links llama.cpp's mtmd; use a checkout that knows glm5v):

```sh
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_DIR=/path/to/llama.cpp      # must contain src/models/glm5-next.cpp
cmake --build build-vision --target strata-vision -j8
```

Encode an image (resident process, one `ENC` line per picture):

```sh
./build-vision/bin/strata-vision --mmproj <mmproj-F16.gguf> --model <the .gguf>
# stdin:  ENC /path/to/picture.png /tmp/pic.bin
# stdout: OK 256 256 1 2259        (n_tokens, nx, ny, ms) → SVE1 file
```

Ask the engine (the prompt carries `xCA` once; the driver expands it to the
256-token run; the engine places the SVE1 rows there):

```python
# multimodal/e2e_test.py, short form
ids = tok.encode(rendered_template_with_image, parse_special=True)
i = ids.index(154854)                      #  tokens_to_decoded: 154854
ids = ids[:i] + [154854]*256 + ids[i+1:]   # one pad per image row
stdin.write(f"GENI 300 /tmp/pic.bin {','.join(map(str, ids))}\n")
```

Measured end to end on this machine: a synthetic picture (red canvas, blue
circle upper-middle, yellow square bottom-right) is described correctly through
the engine's GENI path; prefill 280 tok in 1.4 s, decode 29.9 tok/s,
speculative acceptance 85.9%.  The same pair passes llama.cpp's
`llama-mtmd-cli` too:

```sh
llama-mtmd-cli -m final.gguf --mmproj mmproj-F16.gguf --image pic.png \
  -p "describe" -ngl 20 -t 6 -c 2048 -ub 256 -b 256
```

(the smaller `-c/-ub/-b` matter: the defaults want a 4.7 GB compute buffer that
does not fit next to the expert pools.)

## 7. Measured accuracy and performance

All on this machine (2× V100-SXM2-32GB, 64 GB host):

| item | value | how |
|---|---|---|
| decode speed | 26.9–35.9 tok/s | steady chat, kv int8, spec on |
| speculative acceptance | 81.9% (740/903) | 10 varied prompts + counts |
| prefill | ~96 tok/s | 128-token chunks |
| VRAM hit rate | 83–86% | expert pool hit counter |
| MMLU (228-question zero-shot, first-letter judging) | 79.8% | see note 1 |
| HarmBench-50 refusals | 1/50 (2%) | regex judging, no classifier |
| weight fidelity vs the unsloth base UD-Q2_K_XL | attn cos 0.9993, experts cos 0.900 | 21-tensor probe vs the FP8 source |
| PPL ratio (ours / unsloth base quant), wikitext-2, 20 chunks | 1.115 | llama-perplexity -ngl 20 |
| top-1 agreement | 77.7% | same run |

Note 1: the answer letter leads the reply ("B Explanation: …") in 221/228
replies — judge the first standalone A–D, never the last (stray option letters
in explanations cost 54 questions in an earlier pass).  The reference UD-Q2 is
the *base* model's quant, so the PPL/top-1 numbers bundle the finetune drift
with the pipeline difference; they are a same-grade sanity check, not a pure
quantization-loss measurement.

## 8. Limits

- CUDA only (sm70 tested).  HIP/SYCL not built.
- The mmproj is the base model's projector; the UNCENSORED source carries no
  vision weights, so image quality claims stop at "correctly describes simple
  scenes".
- `serve/server.py`'s picture path and the BGENI parallel path were not
  exercised end to end (the terminal chat path and e2e_test.py were).
- The quantization pipeline (§3b) is documented, not packaged: the imatrix and
  type-map steps were run by hand per the runbook.
