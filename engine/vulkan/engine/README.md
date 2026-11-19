# Native Vulkan engine (experimental)

This is the engine behind `strata launch --backend vulkan`. It runs the whole
forward pass itself — dense weights resident, MoE experts streamed from the GGUF
through `ExpertStream` and our router/combine/sampler — and generates tokens
without llama.cpp. The faster `bridge/` adapter drives llama.cpp's Vulkan backend
but does not stream experts. See `docs/VULKAN.md` for the user guide and
`../README.md` for the storage/compute tier this engine is built on.

## Build

Vulkan and C++17 only; no llama.cpp, oneAPI or Level Zero.

```sh
bash engine/vulkan/engine/build.sh
```

## Run directly

Shader paths are relative to the repository root when the engine is started from
there. The eight shader flags are set by `serve.py` and by the `strata launch`
hook; for a direct run the minimal form is:

```sh
printf 'EXPERT 0 0\nVSTAT\nVRAM\nQUIT\n' | \
  ./engine/vulkan/engine/strata-vulkan-engine \
    --model /path/to/model-Q4_K_M.gguf \
    --ctx 4096 --native-shaders engine/vulkan/native/shaders \
    --expert-shaders engine/vulkan/shaders
```

`bash engine/vulkan/engine/test_protocol.sh` runs the smoke test and checks the
replies. It needs the GGUF and the GPU; set `MODEL` to use a different file.

## Run through Strata

```sh
python engine/vulkan/engine/serve.py --port 8095
```

It builds the engine if needed, exports the tokenizer with `bridge/setup_bridge.py`,
writes a config with our shader paths, and starts `serve.server`. In normal use
`strata launch --backend vulkan --model-path /path/to/model.gguf` does all of this
and owns the `launch-vulkan` profile, so `--status`, `--stop` and
`strata connect --check` work like any other backend.

## Protocol

Standard Strata engine commands: `READY`, `GEN`, `PP`, `T`, `DONE`, `VRAM`, `QUIT`.
Three extra commands exercise our tier:

- `EXPERT <layer> <expert>` — runs gate → SiLU×up → down on one real expert with
  a deterministic input and prints the output sum, first value and time in µs.
- `SHEXP <layer>` — runs the shared expert (gate → SiLU×up → down plus the
  scalar sigmoid gate) and prints the same statistics.
- `VSTAT` — streaming cache hits, misses, prefetches and bytes read from the GGUF.

`--lp K` prints Strata's `LP <logprob> <token>:<logprob> ...` line after each
token, and `EVAL <id,id,...>` runs a fixed token sequence through the forward
without sampling. Both are test-only; normal serving does not pass them.
`engine/vulkan/compare_greedy.py` and `compare_forward.py` use them for the
correctness gates described in `docs/VULKAN.md`.

## Verified on the Arc B390 (2026-11-19)

Direct protocol test (`test_protocol.sh`): `READY 4096 stop`,
`EXPERT 0 0 sum=0.124008` on a cold read of three experts,
`SHEXP 0 sum=0.167194 first=0.003059`, `VSTAT hits=0 misses=6 mb=4.1`.

Through `serve.py`, the 21 GB Qwen3.8-35B-A3B GGUF answered "What is 2+2?" with
the same first tokens as the llama.cpp bridge (`271, 248068, 198`). Teacher
forcing agrees on 214 of 214 positions of a 200-token prompt. Native decode is
about 5.5 tok/s in a short warmed window (3.7 tok/s in one longer 32-token run);
the bridge is about 10x faster but does not stream experts. Run-to-run variance
on the B390 is large, so treat these as ranges rather than benchmarks.

## Not done yet

- The native forward is one token at a time, with fp32 activations and host-side
  sampling. There is no prompt batching or MTP.
- Batched submissions exist; timeline semaphores and transfer-queue overlap do
  not. Speculative next-layer prefetch is off by default: the routing corpus
  shows only 2.9% consecutive-layer overlap. Record a trace with `--trace FILE`.
- QSA sparse attention, KV quantization, gated residual and n-gram PLE are for
  Qwen3.8-Flash-Next and are not ported. This model uses dense grouped-query
  attention, which is ported.
- `VRAM` reports `vram_free_mib` from `/proc/meminfo` MemAvailable, capped at the
  Vulkan DEVICE_LOCAL heap. On an integrated GPU that heap is system RAM, so the
  number is shared memory, not dedicated VRAM (issue #515). `strata doctor
  --vulkan` reports the same heap, extensions and subgroup facts.
- `strata launch --backend vulkan` starts this engine by default. Use
  `--vulkan-impl bridge` for the llama.cpp path.
