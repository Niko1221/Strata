# llama.cpp-Vulkan → Strata protocol bridge

**This is a bridge, not the native Vulkan engine.** It exists so that on this
B390 the GGUF can be *run through Strata* today, while the real Vulkan engine
in `../src` is still being ported. The native engine will replace this binary.

`strata`'s server (`terminal/serve/server.py`) only speaks the `strata --serve`
line protocol. This tiny C++ binary speaks that protocol on stdin/stdout and
does inference with llama.cpp's Vulkan backend (`libggml-vulkan.so`):

```
READY <ctx> stop                      startup
GEN <max_new> [sampling] <id,id,...>  a request (token IDs from Strata's tokenizer)
  PP <read> <total> <ms> <tok/s>
  T <id>                              one per generated token
DONE <n> <prompt> <prompt_ms> <decode_ms> <finish>
VRAM ...                              answered with empty values
STOP / QUIT                           ignored / exit
```

Only `READY`, `T` and `DONE` are required by the server; `INFO`/`PP`/`VRAM` are
optional. See `terminal/tests/tuning/stub_engine.py` for the reference stub.

The prompt IDs come from Strata's own tokenizer (`ST.Tokenizer.encode`), which
is exported from the same GGUF. They were checked against llama.cpp's tokenizer
on this model: the same IDs produce coherent text in both.

## Build

The adapter uses the llama.cpp C API. It must be compiled against headers from
the **same commit** as the libs it links to; on this machine that is commit
`a956a852c` (`/home/g/Code/llama-bench`) and `/home/g/Code/llama.cpp/build/bin`.

```sh
bash engine/vulkan/bridge/build.sh
# override on another PC:
#   LLAMA_CPP_SRC=/path/to/llama.cpp LLAMA_CPP_LIB=/path/to/build/bin bash ...
```

## Run through `strata` (recommended)

`strata launch --backend vulkan` is wired to this bridge. It is experimental
and needs one GGUF path:

```sh
strata launch --backend vulkan \
  --model-path /home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf \
  --port 8095 --yes --no-tui
# check first, changes nothing: add --plan
```

From a source checkout use `python terminal/strata.py launch ...`. The command
builds the bridge if needed, exports the tokenizer, writes the config and
starts Strata's server in the background. The native engine will reuse this
CLI hook when its kernels are ported.

## Run the pieces by hand

```sh
python engine/vulkan/bridge/setup_bridge.py \
  --gguf /home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf \
  --data-dir /home/g/.cache/strata-vulkan-bridge --port 8099

cd terminal && python -m serve.server --engine strata \
  --config /home/g/.cache/strata-vulkan-bridge/config.json
```

`setup_bridge.py` exports the tokenizer from the GGUF with Strata's own
`tools/strata_tokenizer.py` and writes a config pointing `exe` at this binary.
The server then exposes the normal OpenAI-compatible API on the chosen port:

```sh
curl -s http://127.0.0.1:8099/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-35b-a3b-vulkan","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":32}'
```

## Limits

- This path uses **llama.cpp's kernels**, including its own expert handling. It
  does **not** use `strata_vulkan_stream` (the NVMe expert tier). Streaming
  experts in Vulkan mode end-to-end needs the native engine.
- All layers are offloaded with `n_gpu_layers=-1`. On the iGPU that is shared
  memory, so the 21 GB model plus KV cache must fit in system RAM (30 GB here).
- No batching, no drafter/MTP, no session save/restore, no logprob lines yet.
  The server only needs `READY`, `T` and `DONE` for a plain request.
- Sampling is greedy when `temperature=0`, otherwise
  top-k → top-p → temperature → dist(seed), matching llama.cpp's default order.
