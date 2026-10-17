# Strata — configuration reference

## NAME

**strata configuration** — every flag, config-file key and environment variable you can set, and what each one changes.

This is a reference. For decisions ("which model, which context, which settings for my PC") use
[AI_SETUP.md](AI_SETUP.md) and [INSTALL.md](INSTALL.md#tuning-for-your-pc).

## SYNOPSIS

```
python setup.py [options]                                    # install, set up, start (START-HERE.bat / setup.sh run this)
python -m serve.server --engine strata --config strata-<model>.json   # the server, by hand
python chat.py [--port 8080] [--think low]                   # terminal chat with a running server
python tools/strata_mcp.py [--root DIR]                      # the MCP server (docs/MCP_SERVER.md)
strata generate --pack DIR --tokens "1,2,3" [engine options] # the C++ engine, by hand
```

## DESCRIPTION

Strata has five configuration layers. Each one is read by a different program.

| Layer | Where you set it | Read by |
|---|---|---|
| Setup flags | `python setup.py --model IQ3_S --context 32768` | `setup.py` |
| Run config | `data/config/strata-<model>.json` | `serve/server.py` and the engine |
| Server flags | `python -m serve.server --port 8080` | `serve/server.py` |
| Engine flags | the config's `"args"` list, or `strata generate …` | the C++ engine binary |
| Environment | `STRATA_…` in the shell, or the config's `"env"` block | setup, the server or the engine |

**Precedence:** a command-line flag wins over the config file, which wins over the built-in default. Some
server flags read an environment variable as their starting value (`--api-key` and `$STRATA_API_KEY` are the
same setting). `setup.py` writes the run config, and the config's `"args"` list is what the engine actually
receives — so anything setup has no flag for goes there.

### The words used below

- **expert** — the model is a *mixture of experts*: 24,576 small networks, of which a few are used per token.
  Most sit in system RAM or on the SSD; the busiest ones are kept on the GPU.
- **KV cache** — what the model keeps of the text it has already read. It grows with context length and lives
  in GPU memory unless it is streamed from RAM.
- **PLE table / prompt table** — a large lookup table on the SSD the model reads while it reads your text.
- **MTP draft layer** — a small helper model that guesses the next few tokens; the big model checks them in one
  pass. "Speculative decoding" is that guess-and-check loop.
- **QSA** — the model's sparse attention: it reads a subset of the KV cache instead of all of it.
- **pack** — the folder of model files the engine reads.
- **verify window** — one guess-and-check step: several tokens checked at once.
- **prompt path / prefill** — reading your text, which is done in chunks, as opposed to producing tokens.

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for the longer version.

---

## setup.py — install, set up, start

`START-HERE.bat` (Windows) and `./setup.sh` (Linux) install Python if needed and run `setup.py`. The first run
asks four questions (model, size, context, images); every later run goes straight to starting the model. Setup
recommends but never forces: a bigger choice than it recommends is kept, with what it risks. With `--yes`, an
explicit flag is the consent to a risk setup would otherwise stop at — `--yes` alone is not.

What each choice means and what fits: [MODELS.md](MODELS.md), [INSTALL.md](INSTALL.md#setups-questions).

| Flag | Values / default | What it does |
|---|---|---|
| `--family` | `qwen` (default), `swift`, `coder`, `unsloth` | Which model family: the original Qwen3.8-Flash-Next, the Swift 1.5 fine-tune, the Coder, or Unsloth's quants ([MODELS.md](MODELS.md)) |
| `--model` | `Q2_0`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ1_M`, `UD-Q4_K_XL`, `UD-IQ4_XS` | Model size (quantization). Pick by RAM: [MODELS.md](MODELS.md#pick-by-ram) |
| `--context` | e.g. `32768` | Maximum context length in tokens |
| `--rope-scaling` | `none`, `linear`, `yarn` | How the context is extended past the trained 262144 tokens. Omitted for a scaled context: setup picks `yarn`. An explicit `none` is refused for such a context |
| `--rope-scale` | final context ÷ 262144, at least 1 | The extension factor (1.5 for 384K, 2 for 512K, 1 inside the trained range) |
| `--kv` | `int8` (default), `q4_0`, `k8v4` | KV cache precision above 8K context. `q4_0` is half the memory and a little less precise; `k8v4` is a hybrid (8-bit K + 4-bit V) |
| `--vision` | `yes`, `no`, `none`, `gpu`, `cpu` | Let the model read images; `yes`/`gpu` put the image encoder on the GPU ([DETAILS.md](DETAILS.md#images-vision)) |
| `--vision-tokens` | number | How many tokens one image costs |
| `--experimental-speed-projection` | `on`, `off` (default), `GGUF` | Experimental faster projection path |
| `--port` | `8080` | The port the server listens on |
| `--gpu` | e.g. `0` | Which GPU to run on, numbered as `nvidia-smi` numbers them |
| `--gpus` | e.g. `0,2`, or `all` | Several GPUs sharing the model ([MULTI_GPU.md](MULTI_GPU.md)) |
| `--layer-split` | e.g. `24`, or `auto` | With `--gpus`: where each later card's layers start |
| `--no-remote-expert-opt` | — | Turn off the extra optimizations used when experts sit on a second card |
| `--host` | `127.0.0.1` (default), `0.0.0.0` | `0.0.0.0` lets other devices on your network reach it — set `--api-key` too |
| `--api-key` | key, or `key1,key2` | Require a key on `/v1/*` |
| `--no-browser` / `--browser` | — | Open (or don't open) the local page when the model is ready |
| `--data-dir` | directory | Where Strata puts its data (default `data/`) |
| `--models-dir` | directory | Where model files are stored |
| `--gguf-dir` | directory | Use GGUF files you already have instead of downloading |
| `--yes` | — | Take the recommended answers, ask nothing |
| `--setup` | — | Open the setup questions again (install another model, change settings) instead of starting |
| `--no-start` | — | Install and prepare, but don't start the server |
| `--update` | — | Update the engine and the model files ([INSTALL.md](INSTALL.md#updating)) |
| `--rollback-engine` | — | Go back to the previous engine build |
| `--build` | — | Compile the engine from source instead of using the ready-made build |
| `--source` | `auto`, `modelscope`, `huggingface` | Which site to download the model from |
| `--inspect SOURCE [VARIANT …]` | — | Print what setup would download and install, and change nothing |
| `--cuda` | `12`, `13`, `auto` | Which CUDA toolkit to build with |
| `--prebuilt` | URL | Fetch the ready-made engine from this URL instead of the release |
| `--check` | — | Only check this PC (GPU, driver, RAM, disk); install nothing |
| `--calibrate` | — | Measure this PC and write the speculative-decoding threshold that fits it (`--spec-min-p`) |
| `--draft-vocab` | `cjk`, `en`, `cyrillic`, `fr` | Restrict the draft layer's vocabulary to one language's tokens |
| `--low-ram` | `auto`, `on`, `off`, `resident`, `mmap` | How experts are held when RAM is short ([TROUBLESHOOTING.md](TROUBLESHOOTING.md)) |
| `--resident-budget-gib` | GiB | For `UD-Q4_K_XL` / `UD-IQ4_XS`: how many GiB of experts stay in RAM |
| `--vram-reserve-mib` | MiB | Keep this much GPU memory free for other programs ([DETAILS.md](DETAILS.md#sharing-the-gpu-with-other-programs-optional)) |
| `--parallel` | 1..8 | Decode several requests at once ([BATCHING.md](BATCHING.md)) |
| `--replicas` | N | Run N copies of the engine |
| `--kv-streaming` | `auto`, `on`, `off` | Keep part of the KV cache in RAM to free GPU memory for experts |
| `--backend` | `cuda`, `hip`, `sycl` | Which GPU backend. Chosen by itself on a PC with no usable NVIDIA card. `sycl` delegates to `sycl/setup_intel.py` ([INTEL_ARC.md](INTEL_ARC.md)) |

---

## serve/server.py — the server

Usually started for you by `run-<model>.bat` / `run-<model>.sh`. Endpoints, request and response fields and the
API itself: [DETAILS.md](DETAILS.md#using-it).

| Flag | Values / default | What it does |
|---|---|---|
| `--engine` | `mock` (default), `strata` | `mock` is a scripted stand-in engine, for testing clients without a GPU |
| `--config` | path | The run config JSON (below) |
| `--host` | `127.0.0.1` (default), `0.0.0.0` | `127.0.0.1` = this PC only. Also `"host"` in the config |
| `--script` | text, repeatable | The mock engine's answer(s) |
| `--port` | `8095` by hand; `8080` from the run scripts | Listen port |
| `--gpu` | `0` or `0,2` | GPU(s) to run on. Also `"gpu"` in the config |
| `--tokenizer` | directory | The model's tokenizer folder; falls back to a byte tokenizer if absent |
| `--open` | — | Open the local page in the browser once the model is ready |
| `--fit-max-tokens` | — | Clamp `max_tokens` to what the context has left instead of refusing with 400 |
| `--api-key` | key, or `key1,key2` | Require `Authorization: Bearer …` or `x-api-key` on `/v1/*`. Also `$STRATA_API_KEY` |
| `--mcp-config` | path | A JSON file of MCP servers in Claude Desktop's format; their tools are offered to the web app ([DETAILS.md](DETAILS.md#tools-from-mcp-servers)) |
| `--lazy` | — | Start the engine only when the first request arrives |
| `--api-monitor` | — | Keep the last 100 requests' prompts and answers in memory for `/api-monitor` (readable by anyone who can reach the server if there is no API key) |
| `--idle-unload` | seconds, 0 = off | Unload the model from GPU memory after this much idle time |
| `--min-free-vram-mib` | MiB, 0 = off | Refuse a request unless this much GPU memory is free |
| `--before-load` | command | Run this command before the engine loads |
| `--slot-save-path` | directory | Where conversation checkpoints are written |
| `--replicas` | N | N engine processes, each taking requests of its own |

---

## chat.py — terminal chat

| Flag | Default | What it does |
|---|---|---|
| `--port` | `8080` | The server's port |
| `--host` | `127.0.0.1` | The server's address |
| `--think` | the model's own level | How long the model thinks before answering: `none`, `low`, `medium`, `high` |
| `--no-think` | — | Same as `--think none` |
| `--max-tokens` | unlimited | Cap the answer length |

Inside the chat: `/image <path>`, `/think <level>`, `/reset`, `/quit`.

## tools/strata_mcp.py — the MCP server

| Flag | Default | What it does |
|---|---|---|
| `--root` | the folder this file is in | Which Strata checkout to install and run |

Setup, tools and safety: [MCP_SERVER.md](MCP_SERVER.md).

---

## The run config — `data/config/strata-<model>.json`

`setup.py` writes this file and the run scripts start the server with it. You can hand-edit it for settings
setup has no flag for. A bad value usually refuses to start the server rather than being quietly ignored.

### What to run

| Key | What it holds |
|---|---|
| `exe` | The engine binary |
| `args` | The engine's command-line flags ([Engine flags](#engine-flags--strata-generate)) |
| `cwd` | The engine's working directory |
| `lib_dirs` | Extra folders to search for the engine's libraries |
| `log` | Where the engine's output is written |
| `engine_silence_s` | End a request when the engine has said nothing for this many seconds (default 300; 0 = wait forever) |
| `env` | Extra environment variables for the engine — the usual way to set an engine switch |
| `backend` | `cuda`, `hip` or `sycl` |
| `tokenizer` | Tokenizer directory |
| `model_name` | The name clients see on `/v1/models` |
| `aliases` | Other names the model answers to |
| `vision` | Image settings: the `mmproj` file and the encoder's `model` file. Their presence is what enables image input |
| `threads` | Passed to the engine as `--threads` |
| `max_tokens`, `min_tokens` | Passed to the engine as `--max-tokens` / `--min-tokens` |

### Where it listens

| Key | What it does |
|---|---|
| `host` | `127.0.0.1` (this PC only) or `0.0.0.0` (also your network) |
| `api_key` | Required key on `/v1/*`; several keys separated by commas |
| `allowed_hosts` | Extra `Host:` names the server answers to. Needed when you reach it by hostname or through a proxy ([LLAMA_SWAP.md](LLAMA_SWAP.md#the-trap-host--is-not-allowed)) |
| `trusted_origins` | Browser origins allowed to send requests |
| `cors_origins` | CORS origins for `/v1/*` |
| `api_monitor` | Turn on `/api-monitor` |
| `open_browser` | Open the local page when the model is ready |

### How much of the model sits where

| Key | What it does |
|---|---|
| `gpu` | GPU(s), as `nvidia-smi` numbers them |
| `gpu_order` | `"as_given"` keeps the config's GPU order instead of sorting cards fast-first |
| `hip_ordinal` | On Windows, the HIP ordinal setup found for your card; it wins for a one-card config |
| `layer_split` | With several GPUs: where each later card's layers start, or `auto` |
| `split_skip_if_fits` | With `layer_split: auto`, run on one card when the model fits on it |
| `vram_elastic` | Expert cache in segments, so `POST /v1/vram` can hand GPU memory back to other programs ([VRAM_ELASTIC.md](VRAM_ELASTIC.md)) |
| `vram_segment_mib` | Segment size for the above (engine default 512) |
| `expert_profile_save` | Save what the adaptive GPU tier learned, so the next start begins from it |
| `expert_profile_save_every` | Minutes between saves (default 10; 0 = at shutdown only) |
| `parallel` | Decode 1..8 requests at once ([BATCHING.md](BATCHING.md)) |
| `replicas` | Number of engine processes |
| `replica_cpus` | Pin each replica to its own set of CPU cores (`false` turns this off) |
| `replica_load_tokens` | How many tokens a new replica must handle before it counts as warm |
| `lazy_load` | Start the engine on the first request only |
| `idle_unload_s` | Unload after this many idle seconds |
| `min_free_vram_mib` | Refuse a request unless this much GPU memory is free |
| `before_load` | Command to run before the engine loads |
| `prevent_sleep` | Keep the PC awake while the model runs (Windows, opt-in) |

### Answers and reasoning

| Key | What it does |
|---|---|
| `sampling` | Defaults for `temperature`, `top_p`, `top_k`, `min_p` (and llama.cpp's spellings of the same fields). A request's own values win; a bad value refuses to start the server |
| `reasoning_effort` | The thinking level of requests that name none (`none`, `low`, `medium`, `high`). A `--reasoning-effort` flag put in `"args"` is moved to this key at startup with a note — the engine has no such option |
| `reasoning_budget_tokens` | Token budget for thinking |
| `anthropic_thinking` | `"model"` (default) or `"on_request"` (only when the Anthropic request asks for thinking) |
| `effort_position` | Where the effort instruction is placed in the prompt (`"start"` by default) |
| `fit_max_tokens` | Clamp `max_tokens` to the remaining context instead of refusing |
| `repeat_stop_tokens` | Stop a repeated token run (0 turns the guard off) |
| `tool_call_recovery` | Try to repair a malformed tool call (opt-in) |
| `reasoning_loop_recovery` | `false` (default), `"stop"` or `"recover"` when the model loops inside its thinking |
| `reasoning_close_retry` | Retry when the thinking block fails to close (opt-in) |
| `codex_thread_titles`, `codex_compaction_cache` | Codex CLI compatibility extras (opt-in; [DETAILS.md](DETAILS.md#the-responses-api-and-codex-cli)) |
| `mcp_servers` | MCP servers whose tools the web app may use |
| `slot_save_path` | Where conversation checkpoints are written |

---

## Environment variables

### Install and setup (read by `setup.py`, `setup.sh`, the Dockerfile)

| Variable | What it does |
|---|---|
| `STRATA_UV` | Path to the `uv` command setup uses to make `.venv` |
| `STRATA_CUDA` | Force the CUDA version to build with |
| `STRATA_NVCC` | Path to `nvcc` |
| `STRATA_VCVARS` | Path to the Visual Studio `vcvars` script (Windows) |
| `STRATA_ISA_FLOOR` | The lowest CPU instruction set to build for (`avx`, `sse`; [INSTALL.md](INSTALL.md#older-cpus-experimental)) |
| `STRATA_ROCM_VERSION` | Which ROCm to install or use |
| `STRATA_ROCM_INDEX` | Which ROCm package index to use |
| `STRATA_EXPERIMENTAL_GFX1103` | Allow a GPU setup normally refuses (gfx1103) |
| `STRATA_EXPERIMENTAL_SM60` | Allow a GPU setup normally refuses (sm_60) |
| `STRATA_PREBUILT_URL` | Fetch the ready-made engine from this URL |
| `STRATA_SOURCE` | Force the download source (`huggingface`, `modelscope`) |
| `STRATA_SKIP_SHA256` | Skip download checksum checks |
| `STRATA_GGUF_PY` | Set by setup while it prepares a model: the folder holding llama.cpp's `gguf-py` package |
| `STRATA_EXECV` | Replace the shell with the server process instead of forking (the Docker image sets this) |
| `STRATA_DATA` | Docker: where the image stores its data (default `/data`) |
| `STRATA_CPUINFO` | Docker: which file to read CPU information from (default `/proc/cpuinfo`) |
| `HF_ENDPOINT`, `MODELSCOPE_ENDPOINT` | Alternate download sites |
| `ROCM_PATH`, `LD_LIBRARY_PATH`, `PATH`, `CC`, `CXX`, `CUDAHOSTCXX` | Ordinary build variables setup honours |

### Server (read by `serve/*.py`)

| Variable | What it does |
|---|---|
| `STRATA_API_KEY` | Same as `--api-key` |
| `STRATA_ALLOWED_HOSTS` | Extra `Host:` names, comma-separated |
| `STRATA_MAX_BODY_MIB` | Largest request body accepted |
| `STRATA_HTTP_BACKLOG` | The listening socket's backlog |
| `STRATA_REQUEST_LINES` | Log one line per request |
| `STRATA_DEBUG` | Extra server-side debug output |
| `STRATA_FAIL_ONCE` | Fail one request (a test of the retry path) |
| `STRATA_ENGINE_READY_S` | How long to wait for the engine to load |
| `STRATA_ENGINE_STALL_S` | How long a request may make no progress before the server calls it stalled |
| `STRATA_VISION_READY_S`, `STRATA_VISION_ENCODE_S` | The same two, for the image encoder |
| `STRATA_KEEP_EMPTY_TURNS` | Keep turns a client sends with no text ([TROUBLESHOOTING.md](TROUBLESHOOTING.md)) |
| `STRATA_PARALLEL_SOLO` | Allow `parallel` on a single-slot engine |
| `STRATA_REPLICA_START` | Delay between starting replicas |
| `STRATA_BATCH_MTP` | Same as the engine's `--batch-mtp` ([BATCHING.md](BATCHING.md)) |
| `STRATA_BATCH_TRACE` | Log how batch slots are used |
| `STRATA_VERIFY_TRACE` | Log the engine's verification steps at the server level |
| `STRATA_TOKENIZER` | Tokenizer directory |
| `STRATA_SYCL_BIN` | Which SYCL engine binary to run |
| `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, `HIP_VISIBLE_DEVICES` | Which GPUs the process sees; the engine normally sets these itself from `--gpu` |

### Docker (`docker-entrypoint.sh`)

These are the container's front door; each one becomes a `setup.py` flag.

| Variable | Default | Maps to |
|---|---|---|
| `FAMILY` | `qwen` | `--family` |
| `MODEL` | `IQ2_XS` | `--model` |
| `CONTEXT` | `32768` | `--context` |
| `VISION` | `no` | `--vision` (`no`, `yes`, `cpu`) |
| `HOST` | `0.0.0.0` | `--host` — set `API_KEY` before publishing a container |
| `PORT` | `8080` | `--port` |
| `API_KEY` | empty | `--api-key` |
| `KV` | setup's default | `--kv` |
| `GPU`, `GPUS` | — | `--gpu`, `--gpus` |
| `LAYER_SPLIT` | `auto` | `--layer-split` |
| `LOW_RAM` | `auto` | `--low-ram` |
| `GGUF_DIR` | — | `--gguf-dir` |
| `RESIDENT_BUDGET_GIB` | setup's pick | `--resident-budget-gib` |
| `KV_STREAMING` | `auto` | `--kv-streaming` |
| `CONFIG` | — | Start with this config file instead of the model's own |
| `REINSTALL` | `0` | `1` re-runs setup even when a config already exists |
| `CUDA_ARCHITECTURES`, `BUILD_VISION` | build-time | `docker build --build-arg` ([INSTALL.md](INSTALL.md#docker-linux)) |

### Engine switches

Everything below is read by the engine. Set it in the shell, or — the usual way — in the run config's `"env"`
block. Unset means the default path, which is the one the maintainers measured; the rows below name what each
one changes and where it is measured. Some of them change the answer bits, and the linked doc says which.

**Memory, and where the experts live**

| Variable | What it does | Measured in |
|---|---|---|
| `STRATA_RESIDENT_HEADROOM_GIB` | RAM the engine leaves free when it decides how many experts to hold (default 4 GiB) | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| `STRATA_RESIDENT_PIN=0` | Keep the expert copy in ordinary RAM instead of page-locked RAM | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| `STRATA_PIN_GUARD`, `STRATA_PIN_RESERVE_GIB` | Page-lock in steps, following how much RAM is free | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| `STRATA_ARENA_MMAP=1` | Map a native pack's expert file instead of copying it (Linux, small RAM) | [DETAILS.md](DETAILS.md) |
| `STRATA_ARENA_PIN_GIB=N` | Pin at most N GiB of that arena (0 = all) | [MULTI_GPU.md](MULTI_GPU.md) |
| `STRATA_NO_ARENA_THP=1` | Don't ask Linux for huge pages for the arena | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| `STRATA_FILE_RELEASE=1` | Let Windows release mapped expert pages | [DETAILS.md](DETAILS.md) |
| `STRATA_UNBUFFERED_LOAD=0` | Load the model through the OS file cache instead of around it | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| `STRATA_MEMORY_LIMIT_MIB=N` | Tell the engine the RAM limit its container cannot see (the `--memory-limit-mib` flag wins) | [DETAILS.md](DETAILS.md) |
| `STRATA_MIRROR_MIB=N` | Read every expert without a GPU slot into a pinned host mirror (0 = off) | [INTEL.md](INTEL.md) |
| `STRATA_UMA_HEADROOM_GIB` | On a shared-memory chip, how much RAM to keep back (default 6 GiB) | [STRIX_HALO.md](STRIX_HALO.md) |
| `STRATA_PARTIAL_PIN=1`, `STRATA_PARTIAL_PIN_GIB` | Register the hottest part of the RAM budget with the GPU driver | [UNSLOTH_Q4.md](UNSLOTH_Q4.md) |
| `STRATA_WDDM_BUDGET=0` | Turn off the Windows display-driver VRAM allowance | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_OWNED_PRICE=exact` | Count the prompt path's buffers when sizing the expert cache | [DETAILS.md](DETAILS.md) |
| `STRATA_EMB_REUSE_ACCOUNT=0` | Stop counting buffers the prompt path reuses, so auto chunks get bigger | [DETAILS.md](DETAILS.md) |

**Reading the model from disk**

| Variable | What it does | Measured in |
|---|---|---|
| `STRATA_IO_PREFETCH=1` | Linux: read experts with explicit prefetch instead of page faults | [DETAILS.md](DETAILS.md) |
| `STRATA_IO_PREFETCH_DEPTH`, `STRATA_IO_PF_THREADS`, `STRATA_IO_PF_AHEAD`, `STRATA_IO_PF_STAGE` | How far ahead, on how many threads, and what to stage | [DETAILS.md](DETAILS.md) |
| `STRATA_IO_STATS=1` | Log page-cache hits and misses and the read-ahead used | [DETAILS.md](DETAILS.md) |
| `STRATA_IO_THREADS=N` | The prompt-table reader's thread count (Intel default 64) | [INTEL.md](INTEL.md) |
| `STRATA_LOOKAHEAD=0`, `STRATA_LOOKAHEAD_K`, `STRATA_LOOKAHEAD_STATS` | Predict the next layer's experts and ask the OS for their pages | [DETAILS.md](DETAILS.md), [UNSLOTH_Q4.md](UNSLOTH_Q4.md) |
| `STRATA_FETCH_THREADS=N` | Threads that read experts from the GGUF (default 8) | [DETAILS.md](DETAILS.md) |
| `STRATA_STAGER_THREADS`, `STRATA_STAGER_RING`, `STRATA_STAGER_SSD`, `STRATA_STAGER_SLEEP` | The threads and queue depth that copy experts in place from an SSD | [DETAILS.md](DETAILS.md), [UNSLOTH_Q4.md](UNSLOTH_Q4.md) |
| `STRATA_READ_AHEAD=0`, `STRATA_FILL_AHEAD=N` | Turn read-ahead off, or set how far ahead it goes (default 256) | [DETAILS.md](DETAILS.md) |
| `STRATA_FILL_SERIAL=1` | The older one-at-a-time fill | [INTEL.md](INTEL.md) |
| `STRATA_RING_BYTES=N` | Size of the expert ring | [DETAILS.md](DETAILS.md) |
| `STRATA_PREFILL_RING=N` | The prompt path's ring slots | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_PLE_TRACE=1` | Trace the prompt-table reader | [INTEL.md](INTEL.md) |

**The GPU expert cache**

| Variable | What it does | Measured in |
|---|---|---|
| `STRATA_FS_SLOTS=N` | Foresight swap space: N GPU slots per layer, refilled on a copy stream | [DETAILS.md](DETAILS.md) |
| `STRATA_FS_AHEAD=0` | Stop prefetching the experts the next layer's router predicts | [DETAILS.md](DETAILS.md) |
| `STRATA_ADAPT_LAG=2` | Apply the adaptive tier's swaps two windows late (default 1) | [DETAILS.md](DETAILS.md) |
| `STRATA_ADAPT_JOB_CPU` | Pin the adaptive tier's job thread to one CPU | [DETAILS.md](DETAILS.md) |
| `STRATA_EXCHANGE_ROTATE=1` | Hand buffer ownership over in an adaptive swap instead of copying | [EXCHANGE_ROTATION.md](EXCHANGE_ROTATION.md) |
| `STRATA_DISJOINT_ADAPT=1` | Avoid duplicate expert slots across two cards | [DISJOINT_EXPERT_CACHE.md](DISJOINT_EXPERT_CACHE.md) |
| `STRATA_STAGE_PIN=1` | Page-locked staging buffers | [DETAILS.md](DETAILS.md) |
| `STRATA_STAGE_TRIM=1` | Each card loads only its own layers' dense weights | [MULTI_GPU.md](MULTI_GPU.md) |
| `STRATA_ROUTE_RESIDENT`, `STRATA_ROUTE_RESIDENT_MTP`, `STRATA_ROUTE_PRIOR` | Routing variants that keep more experts resident (`ROUTE_PRIOR` costs answer quality) | [DETAILS.md](DETAILS.md) |
| `STRATA_ROUTE_TAIL_SKIP=0` | Stop skipping the tail of missed experts in a verify window | [DETAILS.md](DETAILS.md) |

**KV cache and sparse attention**

| Variable | What it does | Measured in |
|---|---|---|
| `STRATA_KV_GROW=1` | Same as `--kv-grow`: take KV memory as the context grows | [DETAILS.md](DETAILS.md) |
| `STRATA_KV_ROT` | The rotation used by the 4-bit KV formats | [DETAILS.md](DETAILS.md) |
| `STRATA_KV_PREFETCH=1` | Overlap streamed KV uploads with prompt computation | [KV_PREFETCH.md](KV_PREFETCH.md) |
| `STRATA_KV_HOST_DMA=1` | Don't use a GPU kernel to write the RAM copy of the KV cache | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| `STRATA_ARGMAX_MULTI=0`, `STRATA_QSA_CLUSTER=0` | Turn off the two sparse-attention decode kernels | [DETAILS.md](DETAILS.md) |
| `STRATA_TOPK_STREAM=0` | The older prompt top-k selection kernel | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| `STRATA_SNAPSHOT_FULL_CAPTURE=1` | Turn off the shared capture snapshot (more VRAM, same output) | [DETAILS.md](DETAILS.md) |

**Prompt processing (prefill)**

| Variable | What it does | Measured in |
|---|---|---|
| `STRATA_PREFILL_CPU_SHARE`, `STRATA_PREFILL_CPU_SHARE_MAX` | Let the CPU compute part of a short prompt's experts (on by default on one NVIDIA GPU, under 1024 tokens) | [DETAILS.md](DETAILS.md#short-prompts-the-cpu-shares-the-experts-on-by-default-on-one-nvidia-gpu-strata_prefill_cpu_share) |
| `STRATA_PREFILL_EQUAL=1` | Read a prompt in equal chunks instead of full chunks plus a short one | [DETAILS.md](DETAILS.md) |
| `STRATA_PREFILL_FIRST=N` | A short first chunk so the GPU starts while the rest is still being read (Intel default 256) | [INTEL.md](INTEL.md) |
| `STRATA_PREFILL_SYNC=1` | Synchronize after every prompt phase (a debug switch) | [INTEL.md](INTEL.md) |
| `STRATA_PREFILL_STREAM_ALL=1`, `STRATA_PREFILL_STREAM_MIN=N` | Which prompt lengths take the streamed expert path | [INTEL.md](INTEL.md), [STRIX_HALO.md](STRIX_HALO.md) |
| `STRATA_PREFILL_MMQ=ON` | Build-time: the GGML MMQ kernels for prompt projections | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_PF_FUSED=0`, `STRATA_PF_FUSED_KQ=1`, `STRATA_PF_FUSED_NATIVE=0` | Which fused prompt expert kernels to use | [AMD_HIP.md](AMD_HIP.md), [DETAILS.md](DETAILS.md) |
| `STRATA_PF_GEMM=1`, `STRATA_PF_SWITCH_MIN_T`, `STRATA_PF_PAD`, `STRATA_CVEC_FUSE` | Matrix-core prompt GEMMs, the length they take over at, padded row strides, a fused steering write | [STRIX_HALO.md](STRIX_HALO.md) |
| `STRATA_PF_STEP_SYNC=1` | Synchronize each prompt step — used to find a step that hangs | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| `STRATA_DENSE_MMQ=1` | HIP: dense prompt projections on MMQ | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_HIP_PROMPT_F16=1` | HIP: 16-bit prompt GEMMs in and out | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_PROMPT_ATTN_OLD=1`, `STRATA_PROMPT_ATTN_XMX=1`, `STRATA_SELECT_WMMA=1`, `STRATA_SELECT_SIMT=1`, `STRATA_HIP_WMMA=1` | Alternate prompt-attention and block-scorer kernels, per card | [NVIDIA_V100.md](NVIDIA_V100.md), [INTEL.md](INTEL.md), [STRIX_HALO.md](STRIX_HALO.md), [OLDER_GPUS.md](OLDER_GPUS.md) |
| `STRATA_GDN_CHUNKED=1` | The prompt's gated-delta-net recurrence in chunks of 32 | [DETAILS.md](DETAILS.md) |
| `STRATA_GDN_HEAD`, `STRATA_GDN_PP`, `STRATA_GDN_CONVL2`, `STRATA_GDN_NOY`, `STRATA_GDN_SPLIT`, `STRATA_GDN_KEYHEAD` | Variants of that recurrence, measured per card | [STRIX_HALO.md](STRIX_HALO.md), [INTEL.md](INTEL.md) |
| `STRATA_QFUSE`, `STRATA_PLE_BATCH` | Fused activation images; a window's PLE projections at once | [STRIX_HALO.md](STRIX_HALO.md) |
| `STRATA_CACHE_MESSAGE_BOUNDARY=1` | Reuse chat history when the last user message is edited | [MESSAGE_BOUNDARY_CACHE.md](MESSAGE_BOUNDARY_CACHE.md) |
| `--prompt-cache-tail` (an engine flag, not a variable) | An extra checkpoint near the prompt tail | [PROMPT_CACHE_TAIL.md](PROMPT_CACHE_TAIL.md) |

**Speculative decoding (the draft layer)**

| Variable | What it does | Measured in |
|---|---|---|
| `STRATA_SPEC_PROB=1`, `STRATA_SPEC_DEPTH`, `STRATA_SPEC_MIN_TEMP` | Sampled drafting: how the draft layer guesses when temperature is above 0 | [DETAILS.md](DETAILS.md) |
| `STRATA_SPEC_COUPLED=1`, `STRATA_SPEC_GUMBEL=1` | Coupled drafts with Gumbel-max picks | [DETAILS.md](DETAILS.md) |
| `STRATA_BATCH_MTP=1` | Same as `--batch-mtp` | [BATCHING.md](BATCHING.md) |
| `STRATA_BATCH_DECODE_SHARE` | How long a batch slot keeps decoding after a chunk (default 0.5) | [BATCHING.md](BATCHING.md) |
| `STRATA_IQ_MT_MIN=1` | Multi-token CPU expert kernels for every group, for reproducible greedy output | [DETAILS.md](DETAILS.md) |
| `STRATA_LOGPOS`, `STRATA_LOGPOS_TOPK` | Log the top tokens per position, to compare answers with another program | [UNSLOTH_Q4.md](UNSLOTH_Q4.md) |

**Several GPUs**

| Variable | What it does | Measured in |
|---|---|---|
| `STRATA_SPLIT_OWN=1` | Give every card its own buffers | [MULTI_GPU.md](MULTI_GPU.md) |
| `STRATA_SPLIT_RING` | The prompt path's loan and streamed ring under a split | [MULTI_GPU.md](MULTI_GPU.md) |
| `STRATA_SPLIT_MTP_BATCH=1` | The draft layer's prompt KV in batches on a split | [MULTI_GPU.md](MULTI_GPU.md) |
| `STRATA_MTP_BATCH_RING=0` | Send a draft KV ring back to the older pass | [MULTI_GPU.md](MULTI_GPU.md) |
| `STRATA_PIPELINE_ADAPT_ASYNC=0`, `STRATA_PIPELINE_DEBUG=1` | A blocking adaptive tier under pipelined windows; pipeline debug output | [MULTI_GPU.md](MULTI_GPU.md) |
| `STRATA_PREFILL_HELP=1`, `STRATA_PREFILL_HELP_FRAC=f` | The idle card helps a one-chunk prompt | [MULTI_GPU.md](MULTI_GPU.md) |
| `STRATA_PCIE_FRAC_DEFAULT=old` | Keep the older rule for the PCIe share of missed experts | [DETAILS.md](DETAILS.md) |
| `STRATA_DMA_BATCH=1` | Batch the DMA submissions used with `--pcie-mode dma` | [BATCHED_DMA.md](BATCHED_DMA.md) |

**CPU, threads and cores**

| Variable | What it does | Measured in |
|---|---|---|
| `STRATA_HOST_CORE=last` | Put the host thread on the last physical core (Windows sends GPU interrupts to the first) | [DETAILS.md](DETAILS.md) |
| `STRATA_AUX_CPUS=auto`, `STRATA_AUX_STAGER=0` | Where the helper threads run | [DETAILS.md](DETAILS.md) |
| `STRATA_FORCE_ISA=avx\|sse` | Use an older CPU instruction set | [INSTALL.md](INSTALL.md#older-cpus-experimental) |
| `STRATA_NO_AVXVNNI=1` | Turn off the AVX-VNNI forms of some row kernels | [DETAILS.md](DETAILS.md) |
| `STRATA_IQ3S_MT1=1`, `STRATA_KQ256=1`, `STRATA_Q2_BITPLANE=1` | Alternate CPU expert row kernels (some change the last bits) | [DETAILS.md](DETAILS.md), [UNSLOTH_Q4.md](UNSLOTH_Q4.md) |

**Per-card kernel and precision switches**

Each family is measured in one document; that document says which of them change the answer bits.

| Variable | What it does | Measured in |
|---|---|---|
| `STRATA_BF16_TC=1` | Run BF16 projections on the FP16 tensor cores (RTX 20) | [OLDER_GPUS.md](OLDER_GPUS.md) |
| `STRATA_ATTN_PRE75=0`, `STRATA_PROMPT_ATTN_OLD=1` | The attention kernels other cards run | [NVIDIA_V100.md](NVIDIA_V100.md) |
| `STRATA_SM70_TABLE=1` | The Volta decode-kernel table | [NVIDIA_V100.md](NVIDIA_V100.md) |
| `STRATA_MMVQ_IL=0`, `STRATA_MMVQ_IL_ROWS="…"` | The interleaved small-batch matvec, and a rows table for your card | [MMVQ_IL_TABLE.md](MMVQ_IL_TABLE.md) |
| `STRATA_MMVQ_A2=0`, `STRATA_MMVQ_SG`, `STRATA_MMVQ_WAVE=0`, `STRATA_MMVQ_WIDE_32=0`, `STRATA_MMVQ_WIDE_K=0` | Alternate matvec kernels and their shapes | [INTEL.md](INTEL.md), [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_EXP_MODE=N` | Which routed-expert kernel layout | [NVIDIA_V100.md](NVIDIA_V100.md), [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_GR_FAST=0`, `STRATA_GR_SPLIT=1` | The hyper-connection read and its split-along-K variant | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_GR_V3=1`, `STRATA_GR_DOWN_DIRECT=0`, `STRATA_GR_DOWN_MAX4=1`, `STRATA_GR_DOWN_SLICED=0` | Alternate hyper-connection kernels (Intel) | [INTEL.md](INTEL.md) |
| `STRATA_HC_Q8=1`, `STRATA_HC_UPMIX=1`, `STRATA_HCD_EXACT`, `STRATA_HC_CHECK=1` | The hyper-connection read from the file's own 8-bit projections, its fused epilogue, and exact-order variants | [STRIX_HALO.md](STRIX_HALO.md), [INTEL.md](INTEL.md) |
| `STRATA_Q8_PACKED`, `STRATA_Q6_PACKED`, `STRATA_MMVF_ROWS`, `STRATA_ATTN_LANECELL` | Packed 8-bit / 6-bit weight layouts, 4-row BF16 matvec, one-cell-per-thread attention scores | [STRIX_HALO.md](STRIX_HALO.md) |
| `STRATA_EXPERT_V2`, `STRATA_TSUM`, `STRATA_LFUSE` | The grouped decode experts, single-butterfly warp sums, fewer launches | [STRIX_HALO.md](STRIX_HALO.md) |
| `STRATA_SH_STREAM=1` | The shared expert on a second stream (pays on gfx1151) | [STRIX_HALO.md](STRIX_HALO.md) |
| `STRATA_HIPBLASLT_TUNING=<file>` | Which hipBLASLt solution table to use. A table is valid only for the GPU and hipBLASLt version named on its first line (`STRATA_HIPBLASLT_TUNING_V1 <arch> <version>`); the engine refuses any other table | [AMD_HIP.md](AMD_HIP.md), [STRIX_HALO.md](STRIX_HALO.md) |
| `STRATA_HIPBLASLT_VERBOSE=1` | Count hipBLASLt fallbacks at start | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_HIP_ADAPT_KERNEL_COPY=1` | Copy the adaptive tier's swaps with a kernel instead of the copy engine | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_HIP_ROUTER_OLD=1` | The portable routing kernel | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_ROUTE_PER_TOKEN=1` | Route a whole verify window in one projection (HIP) | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1`, `STRATA_GFX906_ATTN_REDUCE12=1` | Community-measured attention variants for gfx906 | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_GFX1151_DEFAULTS=0` | Turn off the whole Strix Halo default table | [STRIX_HALO.md](STRIX_HALO.md) |
| `STRATA_PA_FAST=1` | Prompt attention with single FP16 q and p | [STRIX_HALO.md](STRIX_HALO.md) |
| `STRATA_WARM_GRAPHS=0` | Don't capture the window graphs at load (the first request then pays for them) | [INTEL.md](INTEL.md) |
| `STRATA_ARENA_ALIAS_CHECK=0` | Skip the arena aliasing check (Intel) | [INTEL.md](INTEL.md) |

**Timing limits and diagnostics**

| Variable | What it does |
|---|---|
| `STRATA_WATCHDOG_S` | How long a request may make no progress before the engine calls it stalled |
| `STRATA_DECODE_TIMING=1` | Print where a decode window's time goes |
| `STRATA_VERIFY_PROFILE=1` | Print the GPU side of the same measurement |
| `STRATA_IO_STATS=1` | File-read statistics (see above) |
| `STRATA_DBG_NAN=1` | Count non-finite outputs, for chasing a numerical problem |
| `STRATA_VERIFY_DEBUG=1` | Print the residual after every layer (find which layer goes wrong) |
| `STRATA_VERIFY_FIND=1`, `STRATA_VERIFY_DUMP=<dir>` | Read every expert slot back and compare it with the file |
| `STRATA_VERIFY_ALL_SLOTS`, `STRATA_VERIFY_NO_HOST`, `STRATA_VERIFY_DEVICE_PLAN`, `STRATA_VERIFY_EAGER` | Intel/SYCL checks and device selection |
| `STRATA_DUMP_SEL=<file>` | Dump what sparse attention selected, for measuring locality |
| `STRATA_SESSION_BUFFERED=1` | Write session files with buffered I/O |
| `STRATA_PLE_GGUF=<path>` | Build-time: where the parity tests find the prompt-table shard |

---

## Engine flags — `strata generate`

The C++ engine binary. Normally the server starts it for you with the run config's `"args"`; you edit these
only when you want something setup has no flag for. `strata --help` prints the same list.

The SYCL (Intel) port is a separate program with its own option list. Everything below works on the CUDA and
HIP builds; `--stream-experts` and `--prefill-borrow` exist only in the SYCL one, and the SYCL build leaves out
eight flags the CUDA/HIP engine has (`--aux-cpus`, `--pool-tasks`, `--adapt-min-gain`, `--gpu`,
`--memory-limit-mib`, `--conversation-cache-min-tokens`, `--session-save-reclaim`, `--version`). Its own list:
[INTEL.md](INTEL.md).

`--pack DIR` and `--tokens LIST` (or `--tokens-file PATH`) are required for a one-shot run. `--serve` is what
the server uses: the engine stays up and takes requests on stdin.

**Model, input and files**

| Flag | Default | What it does |
|---|---|---|
| `--pack DIR` | `pack/full` | The pack directory of model files |
| `--tokens LIST` | — | The prompt as comma-separated token ids |
| `--tokens-file PATH` | — | The same, read from a file (commas or whitespace) |
| `--ple-gguf PATH` | — | The prompt-table shard. **Without it the PLE layer is silently skipped and every number downstream changes** — pass it for any real run. With `--native`, the engine finds the shard that holds it if you don't say which |
| `--no-ple` | — | Skip the PLE layer (a diagnostic ablation) |
| `--ple-io direct\|mmap\|ram` | `direct` | How the prompt table is read: `direct` = unbuffered SSD reads that never enter RAM or the file cache; `mmap` = the comparison arm; `ram` = the whole table locked in RAM at start, so no SSD read sits on the token path |
| `--ple-row-cache N` | 1048576 | Bounded cache of fetched table rows (0 = off) |
| `--ple-inflight N` | 256 | Outstanding SSD reads |
| `--ple-delay-us U` | — | Fault injection: each row read completes no earlier than U microseconds |
| `--ple-sync-submit` | — | Submit table reads on the token thread instead of an I/O thread |
| `--no-ple-prefetch` | — | Read the table's rows one at a time instead of in one call |
| `--native SHARD1` | — | The model's first shard: turns on every full-context native path at once (the engine's fast arithmetic for each part of the model). Individual `--native-*` flags stay for comparing one at a time |
| `--native-dense-gguf PATH` | — | Native GDN / sparse-attention / shared projections from this model shard; repeat per shard |
| `--native-head-gguf PATH` | — | The native output head from model shard 1; needs `--stream-token` |
| `--embd-gguf PATH` | — | Take the token embedding from this file (mapped host memory, no GPU memory) instead of `--native`'s |
| `--native-bf16`, `--native-bf16-extra`, `--native-gdn`, `--native-qsa`, `--native-qsa-indexer`, `--native-rope`, `--native-router`, `--native-ple-key`, `--native-ple-postops`, `--native-moe-combine`, `--native-head`, `--gr-native-mmvf` | off | One native path each, for comparing a single part |
| `--gr-fp32-activations` | off | CUDA-oracle hyper-connection activation precision |
| `--native-flash-attn-short` | off | A plain vector attention for diagnostics; needs `--max-context <= 256` |
| `--cpu-oracle-q8-0` | off | CPU expert quantization and dot product in the reference form |
| `--no-capture` | — | Run the layers directly instead of replaying captured graphs |
| `--no-token-graph` | — | Two graphs per layer, launched by the host, instead of one per token |
| `--version`, `--help` | — | Print the engine version / this list |

**Context and RoPE**

| Flag | Default | What it does |
|---|---|---|
| `--max-context N` | 4096 | KV and state capacity |
| `--max-new N` | 16 | Tokens to generate |
| `--rope-scaling none\|linear\|yarn` | the model file's own keys | Extend the context past the trained length. Fixed for the whole run: the K in the cache is already rotated |
| `--rope-scale F` | the model file's factor, else 1 | The extension factor |
| `--rope-freq-base N`, `--rope-freq-scale F` | 0 = the model's | The raw llama.cpp knobs: frequency base and angle shrink |
| `--yarn-orig-ctx N` | 0 = 262144 | The trained context the YaRN correction targets |
| `--yarn-ext-factor F`, `--yarn-attn-factor F`, `--yarn-beta-fast F`, `--yarn-beta-slow F` | -1, 1, 32, 1 | YaRN's own knobs (-1 = auto) |

**Sampling and stopping**

| Flag | Default | What it does |
|---|---|---|
| `--greedy` | yes | Pick the highest-scoring token |
| `--seed S` | — | Turn on sampling with this seed |
| `--top-k N`, `--top-p F`, `--temperature F` | — | The sampling settings |
| `--stop-eos` | off | Stop when the model produces an end-of-turn token |
| `--eos-ids LIST` | `248044,248046` | Which token ids count as end-of-turn; giving it also turns on `--stop-eos` |
| `--coupled-draft` / `--no-coupled-draft` | off; on when `STRATA_SPEC_PROB=1` | The draft layer proposes a token from the same random draw the model used, instead of its own best guess |

**KV cache and GPU memory**

| Flag | Default | What it does |
|---|---|---|
| `--kv fp16\|int8\|q4_0\|k8v4` | `int8` for the setup path | KV storage. `int8` = 8-bit codes plus a scale per 64 values, half the memory; `q4_0` = 4-bit after a rotation, half of int8's memory and slightly lower precision; `k8v4` = 8-bit K (exact scores) + rotated 4-bit V |
| `--kv-resident N` | 0 | KV streaming: keep N cells per sparse-attention layer in GPU memory (minimum 20480) and the whole KV in pinned RAM; the freed GPU memory goes to expert slots. A context of N cells or fewer is not streamed |
| `--kv-grow` / `--no-kv-grow` | off | Take KV memory only for the cells requests actually reach, and hold the rest in the expert cache, giving slots back as the context grows. One GPU, with `--expert-profile` |
| `--vram-reserve-mib N` | 700 | Keep this much GPU memory free. Giving it explicitly also stops the engine from shrinking the reserve on its own |
| `--vram-reserve-later-mib N` | -1 = the same as `--vram-reserve-mib` | The reserve applied to later allocations rather than the first one |
| `--vram-elastic` | off | `--serve`, NVIDIA: keep the expert cache in segments so `POST /v1/vram` can hand GPU memory back to other programs and take it back ([VRAM_ELASTIC.md](VRAM_ELASTIC.md)) |
| `--vram-segment-mib N` | 512 | The segment size for the above |
| `--memory-limit-mib N` | 0 = none | `--serve`: the total RAM this engine's container or cgroup may use, when the container cannot see its real limit (the `STRATA_MEMORY_LIMIT_MIB` variable does the same; the flag wins) |

**Experts: the GPU cache, RAM and the file**

| Flag | Default | What it does |
|---|---|---|
| `--expert-cache N` | 0 | Keep N expert blobs resident in GPU memory and compute their rows there |
| `--expert-cache-per-layer` | — | Give each layer its own slots instead of letting the first position take them all. This is what makes a small cache pay: 2.97% hits one way, 21.4% at 8 slots per layer, 70.4% at 64 |
| `--expert-cache-device1 N\|auto`, `--expert-cache-device2`, `--expert-cache-device3` | — | Fill the second, third and fourth cards with N experts, or as many as fit |
| `--expert-cache-cpu-order` | off | GPU expert reduction that matches the CPU's summation order |
| `--expert-cache-remote-placement stripe\|layer` | `stripe` | Spread expert ranks, or whole layers, across the extra cards |
| `--remote-expert-opt` | off | Extra optimizations for experts that live on another card (`--serve`) |
| `--expert-profile P` | — | Start the GPU tier from a prepared `profile.bin` (see `tools/make_profile.py`) instead of admitting experts on first use |
| `--expert-profile-save P` | — | `--serve`: save what the adaptive tier learned, on shutdown and every `--expert-profile-save-every` minutes (default 10; 0 = shutdown only) |
| `--adapt-every N` | 4 | How often the adaptive tier re-decides which experts to keep |
| `--adapt-swaps N` | 96 | How many experts it swaps per step |
| `--adapt-min-gain F` | 1.5 | Swap an expert in only when its usage count beats the one it replaces by more than F (higher = fewer, surer swaps) |
| `--adapt-decay F` | 0.7 | Usage counts are multiplied by this after each adaptation, so old habits fade |
| `--adapt-async 1` | 0 | `--serve` with the resident RAM mode: swaps advance between verify windows instead of a window waiting for a whole round. Not bit-identical run to run |
| `--resident-budget-gib N` | setup's pick | How many GiB of experts to keep in RAM |
| `--resident-experts` | — | The low-RAM mode setup uses: `--mmap-experts --resident-cpu-experts`, page-locked when it fits, plain mmap when it doesn't. Same answers as `--mmap-experts` for the same placement |
| `--resident-cpu-experts` | — | With mmap and a static profile, keep the experts the GPU cache doesn't hold in ordinary RAM, so none is read from the file twice |
| `--mmap-experts` | — | Opt out of the resident expert arena and map the file instead. The mapped rate depends on the OS file cache holding the whole model |
| `--stream-experts` | off | SYCL port only: no host arena at all — every expert is read from the GGUF on demand and the model's first shard sits in GPU memory ([INTEL.md](INTEL.md)) |
| `--shared-expert-arena FILE` | — | Linux: back the resident arena with one shared file. Put it on `/dev/shm`, not on ordinary SSD |
| `--no-hit-poke` | — | Don't nudge the driver right after a launch, so the GPU work no longer overlaps with the CPU pool (the comparison arm) |

**CPU expert pool and cores**

| Flag | Default | What it does |
|---|---|---|
| `--no-pool` | — | Skip the CPU expert pool entirely (the GPU-only floor) |
| `--pool-workers N` | 0 = every physical core but the host's | Pool worker count |
| `--pool-tasks N` | 0 = 3 per thread | Batched CPU expert tasks per phase (0..4096) |
| `--pool-affinity all\|auto\|p-cores` | `all` | Which cores workers get (hybrid CPUs: performance cores first) |
| `--host-core first\|last` | `first` | Which core the host thread spins on. Windows sends a GPU's interrupts to one logical processor, usually the first |
| `--aux-cpus off\|auto\|24,26` | `off` | Where the threads that are neither pool workers nor the host thread run |
| `--no-host-worker` | — | Leave the host thread out of the pool (the comparison arm) |
| `--no-split-rows` | — | One whole expert per pool thread instead of splitting a large expert across threads |

**Prompt path**

| Flag | Default | What it does |
|---|---|---|
| `--prefill CHUNK` | — | Read the prompt in chunks of CHUNK tokens (needs `--native`). `auto` = the largest chunk up to 8192 whose buffers the expert cache can lend; `auto:16384` / `auto:32768` allow bigger ones |
| `--prefill-until N` | 0 | Batch the prompt only through token N, then fall back to the per-token loop (used to compare the two paths) |
| `--no-prefill-borrow` | — | Don't borrow expert slots from the other card during the prompt path. `--prefill-borrow` (SYCL port only) decides the other way explicitly; without either, the choice follows `--expert-profile` |
| `--trim-stage-weights` | — | With an explicit `--layer-split`: each GPU loads only its own layers' dense weights |
| `--shared-late` | — | Run the shared expert after the CPU pool instead of overlapping with it |
| `--keep-canonical` | — | Also keep reference copies of tensors served natively (more GPU memory) |
| `--no-fused-gr` | — | The six-kernel hyper-connection read and a separate write, instead of the fused one |
| `--no-fast-attn`, `--no-fast-select`, `--no-fused-gdn`, `--no-publish-kernel` | off | Turn off one fast kernel each and use the plain reference version, to compare the two |

**Speculative decoding (the draft layer)**

| Flag | Default | What it does |
|---|---|---|
| `--mtp PATH` | — | The draft layer's directory |
| `--spec N` | setup writes 4 | How many tokens the drafter proposes per check |
| `--spec-min-p P` | setup writes 0.5 | How sure the draft layer must be to extend a window by another guess (`--calibrate` measures it for your PC) |
| `--mtp-max-t M` | 0 = `--spec` | Cap the draft layer's windows at M tokens |
| `--mtp-window N` | 32768 | How many of the recent KV cells the draft layer attends to (0 = all of them) |
| `--mtp-hnorm pooled\|stream` | `pooled` | One normalization over all streams, or one per stream (llama.cpp's form) |
| `--mtp-q4 all\|proj\|head` | off | Keep the draft layer's projections and/or head as 4-bit copies |
| `--mtp-draft-vocab FILE` | `<mtp>/draft_vocab.bin` | The draft head's token subset |
| `--suffix-draft N` | 3 | Prompt lookup: draft from an earlier repeat of the last N+ tokens when it pays (0 = draft layer only) |
| `--lookup-chain K` | 0 | After the draft layer's guesses, add up to K prompt-lookup drafts continuing them (window grows to at most 8) |
| `--lookup-chain-min M` | 3 | The shortest context match the chain extends on |
| `--spec-split` / `--no-spec-split` | off | Split the verify window across the cards: exact, about 7% slower (the overlap study) |
| `--batch N` / `--slots N` | off | `--serve`: decode up to N requests together in batch slots (2..8), each with its own session ([BATCHING.md](BATCHING.md)) |
| `--batch-mtp` | off | Each batch slot also verifies one draft proposal per window; needs `--mtp` and `--spec` |
| `--batch-groups G` | — | `--batch` under a layer split: the slots in G groups pipelined through the cards |

**Several GPUs**

| Flag | Default | What it does |
|---|---|---|
| `--gpu LIST` | every visible card | Which GPUs to run on, as `nvidia-smi` / PCI indices. The engine sets `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` itself, so a caller never has to export them |
| `--layer-split LIST` | empty = one GPU | Where each later card's layers start, or `auto` ([MULTI_GPU.md](MULTI_GPU.md)) |
| `--split-device LIST` | the next visible cards | Which cards take the later stages. `"0"` puts both stages on one card, sharing everything — the bit-identical comparison of the hand-off |
| `--split-skip-if-fits` | off | With `--layer-split auto`, run on the first card alone when it holds every profiled expert pair plus the whole session, the drafter and the reserve — a split then only adds hand-offs |
| `--pcie-mode auto\|dma\|kernel\|direct` | `auto` | How experts cross the PCIe bus: DMA copy, a kernel, or a direct transfer |
| `--pcie-frac F` | -1 = the model's default | What share of the missed experts is computed on the GPU and pulled over PCIe: 0.2 for a Q2_0 pack, 0.55 for native packs. On one NVIDIA GPU in `--serve`, left unset it is refined during the first decode windows |
| `--peer-device N` | off | A second GPU as an adaptive expert-cache tier, reading rows over P2P; it also computes its experts' prompt rows. Not with `--layer-split` or `--expert-cache-device1..3` ([SECOND_GPU.md](SECOND_GPU.md)) |
| `--peer-reserve-mib N` | 600 | GPU memory the peer tier leaves free on its card |
| `--peer-slots N` | what fits | Expert slots on the peer |
| `--peer-adapt-swaps N` | `--adapt-swaps` | Peer cache swaps per adaptation step |
| `--peer-prefill-rows N` | half of chunk × top-k | Prompt rows per layer the peer computes (0 = the prompt path stays on the primary) |
| `--pipeline-windows 1\|2` | off | `--serve` with a split on two cards: overlap the cards within one conversation — 1 = the prompt's short reads, 2 = decode as well ([MULTI_GPU.md](MULTI_GPU.md#one-conversation-with-both-cards-busy---pipeline-windows-opt-in)) |

**`--serve`: conversations, checkpoints and sessions**

| Flag | Default | What it does |
|---|---|---|
| `--serve` | — | Stay resident and take requests on stdin (what the server does) |
| `--prompt-cache N` | 6 | Keep N conversation checkpoints between requests (~118 MB of RAM each). 0 = read every prompt from the start |
| `--prompt-cache-every N` | 16384 | Also checkpoint every N fresh prompt tokens (0 = off) |
| `--prompt-cache-tail` | off | One extra checkpoint near the prompt's end, at an existing chunk boundary ([PROMPT_CACHE_TAIL.md](PROMPT_CACHE_TAIL.md)) |
| `--prompt-cache-root N` | 2048 | Also checkpoint at the first turn boundary of a prompt read from token 0 — the end of the system prompt, which every chat of the same client shares — when it is at least N tokens (0 = never) |
| `--turn-token ID` | 248045 | The token that opens a chat turn (`<\|im_start\|>`). The last one in a prompt is where the history ends and the new answer begins, which is the checkpoint the next request reuses |
| `--tail-role-token ID` | -1 = off | Leave a short turn of this role just before the last turn out of the checkpoint |
| `--conversation-cache-mib N` | 0 = off | RAM budget for parked conversations |
| `--conversation-cache-slots N` | 4 | At most N parked conversations |
| `--conversation-cache-min-free-mib N` | 2560 | RAM that must stay free when parking, saving or restoring a conversation |
| `--conversation-cache-min-tokens N` | 0 | Park only conversations of at least N tokens, so short side requests don't take the slots a long one needs |
| `--short-read N` | 64 | Read at most N fresh tokens through the decode windows instead of the batched prompt path (0 = off) |
| `--session-min-free-mib N` | 4096 | Disk space a save must leave free (0 = no check) |
| `--session-save-reclaim` | off | Reclaim unpinned host caches if a save needs RAM |
| `--stream-token` | off | Enqueue token work on the session stream (experimental) |
| `--check-logits` | off | Copy and check all logits in the stream-token path |

**Control vectors (steering)**

| Flag | What it does |
|---|---|
| `--control-vector FILE` | Add a control vector to the residual stream (llama.cpp's GGUF format; scale 1) |
| `--control-vector-scaled FILE:SCALE,…` | The same with explicit scales |
| `--control-vector-layer-range A B` | Which layers it follows (inclusive; default 1 .. last) |
| `--cvec-mode add\|project` | `h += scale·v`, or remove the vector's direction from `h` |
| `--cvec-dir per-layer\|single:L` | Each layer's own direction (default), or layer L's direction everywhere |

With `--serve`, requests switch vectors on and off with `cvec=0|1`.

**Images**

| Flag | What it does |
|---|---|
| `--vision` | `--serve` also takes images: embeddings arrive from `strata-vision` (the model's `mmproj` file) as `GENI` requests ([DETAILS.md](DETAILS.md#images-vision)) |

**Measurement, dumps and debugging**

Only useful when you are measuring or comparing something.

| Flag | What it gives you |
|---|---|
| `--gpu-only-full` | Replay pre+post graphs for all layers plus the output head, no CPU pool: **the true per-token GPU floor**. Quote this one |
| `--graph-only` | Replay only the 48 "pre" graphs — **not** the GPU floor; it omits the post graphs and the head |
| `--gpu-stages` | Capture each layer as three graphs and time them from outside; prints and exits |
| `--stage-timing` | Per-stage kernel-count shares. Not a time profile: an uncaptured event interval includes host gaps, so run `--gpu-only-full` first |
| `--stats` | The per-stage breakdown |
| `--sync-every-layer` | Synchronize after every layer |
| `--dump-logits PATH` | One line of raw logits per position |
| `--logits-stride N` | Store every Nth logits row (N > 1 needs `--max-new 1`) |
| `--dump-residual PATH`, `--dump-final-r PATH` | The final residual, for bisecting the output head |
| `--dump-layers PATH` | The residual after every layer: the bisection ladder |
| `--dump-halves PATH` | Both halves' block output and injection per layer |
| `--dump-mixed PATH` | The post-attention residual |
| `--dump-routing PATH` | The routed expert ids and weights per layer per position |
| `--spec-follow PATH` | Benchmarks: emit this continuation instead of the argmax and accept the drafts that match it, so speculation settings compare on the same text |
| `--spec-oracle PATH` | Draft from a token file (the expected continuation) — the exactness test |
| `--spec-corrupt N` | Replace every Nth draft token with a wrong one, to check that a bad draft is rejected |
| `--window-hashes PATH` | Per verify window: index, position, size, a hash of its final residual rows and its argmaxes. Two builds that do the same arithmetic write the same file |

---

## Build options (CMake)

Only for `--build`, or a source build for AMD / Intel / Strix Halo. `setup.py` sets these for you; the linked
documents give the full command lines.

| Option | What it does | Where |
|---|---|---|
| `STRATA_ENABLE_CUDA=ON\|OFF` | Build the CUDA backend | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_ENABLE_HIP=ON\|OFF` | Build the HIP (AMD) backend | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_PORTABLE=ON` | Portable kernels instead of card-specific ones | [OLDER_GPUS.md](OLDER_GPUS.md) |
| `STRATA_ISA_FLOOR` | The lowest CPU instruction set to build for | [INSTALL.md](INSTALL.md#older-cpus-experimental) |
| `CMAKE_CUDA_ARCHITECTURES` | Which NVIDIA architectures to compile for (default 120) | [OLDER_GPUS.md](OLDER_GPUS.md), [NVIDIA_V100.md](NVIDIA_V100.md) |
| `CMAKE_HIP_ARCHITECTURES` | Which AMD GPUs to compile for | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_HIP_ARCHS` | The same, as an environment variable to the Windows HIP build script | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_GGML_DIR` | Where the llama.cpp checkout lives | [AMD_HIP.md](AMD_HIP.md), [INTEL_ARC.md](INTEL_ARC.md) |
| `STRATA_BUILD_TESTS=ON` | Build the parity and benchmark programs | [AMD_HIP.md](AMD_HIP.md), [ORCA_Q4_K_S.md](ORCA_Q4_K_S.md) |
| `STRATA_PREFILL_MMQ=ON` | Compile the GGML MMQ prompt kernels | [AMD_HIP.md](AMD_HIP.md) |
| `STRATA_MMQ_KQUANTS`, `STRATA_ORCA_Q4KS_MMQ=ON` | Extra MMQ quant instances | [ORCA_Q4_K_S.md](ORCA_Q4_K_S.md) |
| `STRATA_VISION_CUDA=ON\|OFF` | Build the image encoder's GPU code | [DETAILS.md](DETAILS.md#images-vision) |

The SYCL (Intel) port is a separate CMake project in `sycl/`: [INTEL_ARC.md](INTEL_ARC.md), [INTEL.md](INTEL.md).

---

## SEE ALSO

- [AI_SETUP.md](AI_SETUP.md) — install and configure Strata for a user, step by step
- [INSTALL.md](INSTALL.md) — what you need, where files go, tuning for your PC
- [MODELS.md](MODELS.md) — sizes, downloads, what fits
- [DETAILS.md](DETAILS.md) — the API, every setting and its measured numbers
- [HOW_IT_WORKS.md](HOW_IT_WORKS.md) — the model, the tiers, guess-and-check
- [MULTI_GPU.md](MULTI_GPU.md), [BATCHING.md](BATCHING.md), [SECOND_GPU.md](SECOND_GPU.md) — more than one card, more than one request
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — which switch fixes which problem
- [MCP_SERVER.md](MCP_SERVER.md) — the same steps as MCP tools
- [paper/Strata-Paper.pdf](paper/Strata-Paper.pdf) — the long version
