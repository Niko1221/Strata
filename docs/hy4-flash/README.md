# Run hy4-flash on a single NVIDIA GPU

[简体中文](README.zh-CN.md) · [Apple Silicon (Metal)](README_MAC-Metal.md)

## Models

Keep both GGUF shards in the same directory. Pass the directory or the first shard as `MODEL`.

| Model | GGUF files | Description |
|---|---|---|
| `Hy4-flash-Q4_K_M_MTP_Q8_0` | `Hy4-flash-Q4_K_M_MTP_Q8_0-00001-of-00002.gguf`, `Hy4-flash-Q4_K_M_MTP_Q8_0-00002-of-00002.gguf` | Full expert set with built-in MTP. |
| `Hy4-flash-Q4_K_M_MTP_Q8_0_Prune25` | `Hy4-flash-Q4_K_M_MTP_Q8_0_Prune25-00001-of-00002.gguf`, `Hy4-flash-Q4_K_M_MTP_Q8_0_Prune25-00002-of-00002.gguf` | 25% of the experts pruned, reducing host RAM requirements. |

## Hardware and memory

This guide uses Linux x86_64, CUDA Toolkit 12.8, CMake 3.24+, Ninja, a C++20 compiler and Python 3.10+.

**We recommend a GPU with at least 16 GiB of VRAM.** With the full model, a 64K context limit and Q8_0 KV, the launcher budgets about 12 GiB for resident weights, the KV cache and workspace before allocating the GPU expert pool. On a 16 GiB card, start with `CTX=32768` and try [DFly](#speculative-decoding) for generation (about 5% faster for prose and 27% faster for code than the default with drafting off in the test below).

Use `--plan` on the target card to check the GPU expert pool, especially after changing the context limit or enabling a drafter. The Engram table in the second shard is read on demand. Accessed pages can remain in the OS page cache.

By default, RAM holds a complete copy of the trunk experts (about 83.4 GiB for the full model), and VRAM caches active experts. Allow at least 128 GiB of available host RAM for this configuration. If RAM is tight, use `--resident-experts` to split the experts between VRAM and RAM. For the full model, subtract the GPU expert pool size shown by `--plan` from 83.4 GiB to estimate the expert weights in RAM. Allow additional RAM for swap slots, transfer buffers and other process memory.

## Build

Run these commands from the checkout root. The RTX 4090D uses `CMAKE_CUDA_ARCHITECTURES=89`. For another GPU, use its compute capability or `native`.

```bash
cmake -S engine -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -G Ninja
cmake --build build --target strata --parallel 16

python3 -m venv .venv
.venv/bin/python -m pip install -r terminal/requirements.txt -r terminal/requirements-monitor.txt
```

- The CUDA build uses the GGML sources shipped under `engine/third_party`. Keep `libstrata_kq.so` beside `strata` for K-quant prefill. Offline Python installation requires local wheels.
- `requirements-monitor.txt` installs Textual for the terminal screens. A headless HTTP service can use `requirements.txt` alone and launch with `--no-tui`.
- The CUDA build includes the GGUF vision projector reader. Pillow is in `requirements.txt`. Keep `libstrata_kq.so` beside `build/strata` for quantized prefill.

## Launch and query

The launcher selects the engine from the main model file. A hy4-flash GGUF uses `strata`. A native SQ `.strata` model uses `strata-hy4`. The steps below use GGUF.

With the build above, the launcher finds `build/strata` automatically. Set `MODEL` once in your shell to the model directory or first GGUF shard.

```bash
export MODEL=/path/to/Hy4-flash-Q4_K_M_MTP_Q8_0
./tools/hy4_flash/launch-hy4-flash.sh --plan
./tools/hy4_flash/launch-hy4-flash.sh
```

The first launch prepares `pack/` in the model directory. Set `PACK` to a writable directory if necessary. The launcher plans the expert pool, MTP depth, prefill chunk and CPU expert settings using the model and available hardware. The plan is an estimate; actual memory usage depends on the request.

When the service is ready, query its OpenAI-compatible endpoint from another terminal:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash","messages":[{"role":"user","content":"What is two plus two?"}],"temperature":0,"reasoning_effort":"none","max_tokens":20}'
```

Add `--no-tui` for headless operation. Stop the default service with `.venv/bin/python terminal/strata.py launch --stop --preset hy4-flash`. If you set `PRESET` or `PORT`, use that preset when stopping or that port when calling the API.

### Speculative decoding

The launcher selects built-in MTP automatically when the model and available VRAM support it. To enable DFly, download the published [hy4-flash DFly Q8_0 drafter](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/blob/main/hy4-flash_dfly_Q8_0.gguf) to a separate directory. Keep `MODEL` set to the main model, as in the launch example above.

```bash
./tools/hy4_flash/launch-hy4-flash.sh --dfly-model /path/to/drafts/hy4-flash_dfly_Q8_0.gguf
```

With a 16 GiB VRAM budget and a 33,024-token context limit, the Q8_0 DFly drafter needed about 1 GiB of the planned budget. The GPU pool for trunk experts was 4.90 GiB with DFly and 4.22 GiB when MTP was enabled with its draft experts in host RAM. With the full model, we measured each workload once using 4,096 input tokens and 512 generated tokens. DFly increased decode speed from 36.8 to 38.7 tok/s for prose and from 37.5 to 47.6 tok/s for code relative to the default with drafting off.

Passing `--dfly-model` selects DFly. The default is three draft tokens per verification window. Add `--draft-tokens 4` to try four or `--plan` to preview the settings.

## Launch options

| Option | Default | Meaning |
|---|---|---|
| `MODEL` | Required | Main model directory or first GGUF shard. |
| `STRATA_BIN` | Auto-detected | Optional path to the GGUF engine if you built it elsewhere or want to select a specific build. |
| `PACK` | `<model directory>/pack` | Prepared pack directory, created on first launch. |
| `GPU` / `PORT` / `PRESET` | `0` / `8080` / `hy4-flash` | GPU index, HTTP port and saved preset. |
| `CTX` | `196608` | Maximum context in tokens. |
| `EXPERT_VRAM` | Automatic | GPU trunk expert budget in GiB. See `--plan` for the selected context and drafter. |
| `MTP=on/off` or `--mtp on/off` | Automatic | Enable or disable built-in MTP. |
| `DRAFTING` or `--drafting` | `auto` | `auto`, `mtp`, `dfly` or `off`. |
| `DFLY_MODEL` or `--dfly-model PATH` | Unset | External Q8_0 DFly GGUF. |
| `DRAFT_TOKENS=N` or `--draft-tokens N` | MTP automatic, DFly `3` | MTP accepts 1–7, DFly 1–5. When MTP is enabled, the full model typically drafts three and Prune25 drafts one. |
| `KV` | `int8` | Latent KV precision, `int8` (Q8_0) or `fp16`. |
| `CPU_MISS_MAX` | `16` | Maximum cache misses computed on the CPU per layer. Zero disables CPU expert execution. |
| `--resident-experts` | Off | Split expert storage between VRAM and RAM. |
| `--vision on/off` | `off` | Accept `image_url` in chat requests. The main model decodes image requests. Text-only requests can use the selected drafter. |
| `STRATA_HY4_FLASH_VISION_SHARD` | Auto-detected with vision on | Select the mmproj GGUF if it is outside the model directory or there are multiple candidates. With a single `mmproj*.gguf` beside the main model, use `--vision on` alone. |
| `--set KEY=VAL` | Unset | Override a generated environment setting or advanced engine argument. |

### Engine settings

Use `--set KEY=VAL` to override the launcher's settings.

| Setting | Default | Meaning |
|---|---|---|
| `STRATA_HY4_FLASH_CPU_WORKERS` | `32` | CPU expert worker threads. |
| `STRATA_HY4_FLASH_PREFILL_CHUNK` | Hardware-dependent | Tokens per prefill chunk, for example `512`, `1024`, `2048`, `4096` or `auto`. |
| `STRATA_HY4_FLASH_PREFILL_BORROW` | Selected by the launcher | `1` allows prefill to borrow VRAM from the trunk expert pool. Use `0` to disable it. Split residency (`--resident-experts`) skips borrowing. |
| `STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR` | `1` | Process long prompts layer by layer. Can be disabled under a tight VRAM budget. |
| `STRATA_HY4_FLASH_PREFILL_GROUP_ROWS` | `1024` | Prefill row capacity per expert. For an `exceeds row capacity` error, try `2048`. |
| `STRATA_HY4_FLASH_MTP_HOST_EXPERTS` | Automatic | Place the MTP experts in host RAM under tight VRAM budgets, saving about 3.6 GiB on the full model. |
| `STRATA_HY4_FLASH_ENGRAM_IO` | `mmap` | Read Engram rows through the page cache or request `direct` for direct I/O. |

For supported single-GPU configurations with a full host copy, the launcher considers available VRAM, chunk size and the expert pool budget. With borrowing enabled, it can allocate the allowance for larger prefill chunks to the decode expert pool. When prefill needs workspace, the engine temporarily reduces the number of GPU expert slots. On return to decode it restores pool capacity and refills slots. KV cache, dense weights and the drafter retain their own budgets.

Append `--plan --output json` to your launch command and inspect `plan.hy4_flash_prefill.borrow` and `chunk`. Add `--set STRATA_HY4_FLASH_PREFILL_BORROW=0` to compare the budget with borrowing disabled. In the engine log, `HY4_FLASH_PREFILL_LOAN phase=prefill` and `phase=decode` record the actual loan and return. The `scratch` field is in bytes.
