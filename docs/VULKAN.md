# Vulkan expert streaming (research preview)

This is an experimental path for people with an Intel Arc iGPU who cannot use
oneAPI or Level Zero. It runs a Qwen3.8-35B-A3B GGUF through Strata's own server
with **Vulkan and C++17 only**. It is not a released engine package, and it is not
supported on Windows or macOS. The maintained Intel route is
[Intel Arc on Linux](INTEL_ARC.md), which uses SYCL.

There are two engines on this path:

- The **native engine** runs the whole forward pass itself. Dense weights stay
  resident; the 256 mixture-of-experts (MoE) experts per layer are read from the
  GGUF on the NVMe only when the router asks for them, into a bounded VRAM slot
  pool. The router, combine and sampler are Vulkan kernels. llama.cpp is not
  involved in inference. This is the default for `strata launch --backend vulkan`.
- The **llama.cpp bridge** drives llama.cpp's Vulkan backend through Strata's
  protocol. It is much faster but does not stream experts. Use it with
  `--vulkan-impl bridge`.

Both keep the normal CUDA, HIP, SYCL and Metal launch flow untouched.

## Run it

You need a source checkout, a Vulkan 1.3 driver and one Qwen3.8-35B-A3B GGUF file.
The Terminal installs without administrator rights (see
[Install Strata](INSTALL.md#the-terminal)). Then:

```sh
strata launch --backend vulkan --model-path /path/to/model-Q4_K_M.gguf --plan
strata launch --backend vulkan --model-path /path/to/model-Q4_K_M.gguf
```

The first command shows the plan and changes nothing. The second builds the
engine if needed, writes a config and starts the server through Strata's normal
runtime. It prints the address when the model has loaded (a 21 GB GGUF takes
about a minute). The launch owns a profile called `launch-vulkan` (or
`launch-vulkan-bridge`), so the usual commands work:

```sh
strata launch --status
strata launch --show-key --preset vulkan
strata launch --stop --preset vulkan
strata connect opencode --check
```

By default the server listens on `127.0.0.1` only, as for every local Strata
launch. `--save-preset NAME` stores the GGUF, port and implementation so
`strata launch --preset NAME` starts the same server again.

`--vulkan-impl bridge` picks the faster llama.cpp path:

```sh
strata launch --backend vulkan --vulkan-impl bridge --model-path /path/to/model.gguf
```

The server speaks Strata's normal OpenAI- and Anthropic-compatible API. It
listens on `127.0.0.1` only. `--port` changes the port. There is no VRAM planner
or drafter on this path yet, and `--access lan` or a tunnel is not recommended.

## Check the machine with `strata doctor`

Before a long model download, ask the doctor what Vulkan can actually do here:

```sh
strata doctor --vulkan
strata doctor --vulkan --output json
```

It reports the loader version, each Vulkan device (vendor, API version,
extensions, subgroup size and the size-control range), the memory heaps and
which memory type is `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`, and sizes a
suitable expert cache from `MemAvailable`. It also runs automatically when the
run config names `strata-vulkan-engine`, so `strata doctor --profile PATH` needs
no extra flag.

On an integrated GPU the `DEVICE_LOCAL` heap **is system RAM**. Do not size the
cache from a runtime that reports ~0.2 GB of "VRAM" — that number is the PCI
BAR (issue #515). The engine's `VRAM vram_free_mib` line reports `MemAvailable`
capped at the Vulkan `DEVICE_LOCAL` heap — shared system RAM here, not dedicated
VRAM. The doctor explains this and warns when the configured cache is
too large for available memory, when `--ple-io ram` would lock ~29 GB beside the
experts, or when this user cannot open `/dev/dri/renderD*`.

## What was measured

Measured on an **Intel Core Ultra X7 358H with an integrated Arc B390**, Mesa ANV
Vulkan 1.4, about 30 GB of shared RAM and a YMTC PC411 NVMe. The model is
`Qwen3.8-35B-A3B-Q4_K_M.gguf`, 41 layers, 256 experts per layer. Gate and up
experts are Q4_K; down experts are Q6_K in 21 layers and Q4_K in 20, so the shader
is chosen from each tensor's type.

| Check | Result |
| --- | --- |
| Expert matvec + full gate → SiLU → up → down, all 256 experts, 64 slots, 8 threads | 0 bad rows out of 63,488 on both a Q6_K-down and a Q4_K-down layer |
| Shared expert, layers 0 and 5 | 0/2,048 bad rows; max abs error under 4e-08 |
| Router, combine, BF16 GEMV, sampler, Q8_0 matvec (kernel tests, no model file) | 0 failures each |
| IQ3_XXS / IQ2_XS / IQ1_M matvec (synthetic GGUF, 64 rows × 2048 columns) | 0/64 bad rows each; rel_l1 6.9e-07 / 8.0e-07 / 5.9e-07 |
| AddressSanitizer, all six test binaries | No reports |
| Layers 0–3 of the native forward vs a NumPy CPU oracle | max abs error 3e-07 to 3e-06, 0 bad rows out of 2,048 per layer |
| Greedy prompt "What is 2+2?", native vs llama.cpp bridge | first tokens `271, 248068, 198` identical, token for token |
| Decode speed, native engine, old 64-slot cache | about 1 token/s: 12-token decode 9.5–12.7 s (two runs) |
| Decode speed, native engine, current defaults | about 5.5 token/s: 12-token decode 2.1–2.5 s (two runs, same prompt and GPU) |
| Decode speed, native engine, before the latest fixes | about 1 token/s at 64 slots with uncached host memory and speculative prefetch |
| Decode speed, llama.cpp bridge | about 31 token/s (30.8 tok/s on one chat completion) |
| 30-minute soak, native engine, 16 tokens/request | 555 requests, 0 bad tokens; first tokens unchanged; no resets or memory growth |
| NVMe random reads, 589,824 B (Q4_K), direct io_uring, QD 1 / 8 / 16 / 32 | 1,055 / 4,325 / 4,764 / 4,809 MB/s (YMTC PC411-1TB-D) |
| Decode A/B, 7-token prompt, 32 new tokens, one warmup: native / bridge | 3.7 / 37.9 tok/s (single runs) |

The 30+ token/s figure is the **bridge**, not the native engine. The native
forward now groups kernels into one submission per stage, uses cached host-visible
memory and io_uring reads, keeps a ~1.8 GiB expert cache, and does not speculate
on the next layer's experts (the router trace showed only 2.9% overlap). All of
this was measured on the B390. Host sampling remains, and there is no prompt
batching. `vram_free_mib` is a shared-memory estimate from `MemAvailable`, not a
dedicated-VRAM measurement. On a machine with less shared memory, lower the cache
with `--vulkan-slots` (64 slots is about
55 MiB).

## Check correctness yourself

`engine/vulkan/compare_greedy.py` runs both engines on the same prompt token ids
and compares the tokens they sample and the top logprobs they report. It starts
each engine once and sends every prompt in one session, so the GGUF is loaded once
per engine.

```sh
# Quick smoke: 5 prompts x 5 tokens
python3 engine/vulkan/compare_greedy.py --tokens 5 --prompts 5

# Full gate: 5 prompts x 200 tokens (takes about 15 minutes on the B390)
python3 engine/vulkan/compare_greedy.py --tokens 200
```

A greedy argmax can flip when two tokens are almost equally likely, and that one
flip then cascades into a completely different answer. The harness reports the
verified prefix and the flip, and checks top-1 logprobs and top-3 overlap up to
that point. `--lp-tol`, `--lp-rtol` and `--tie-tol` set the tolerances.

To check the forward itself without decoding cascades, use the teacher-forcing
gate. It feeds the same token sequence to both engines with the `EVAL` verb and
compares the logprobs at every position:

```sh
python3 engine/vulkan/compare_forward.py --tokens 200 --prompt '1, 2, 3, 4, 5,'
```

On that prompt the two forwards agree on all 214 positions. This is the gate that
says whether the native forward matches llama.cpp; `compare_greedy.py` says what
happens when greedy decoding runs on top of it.

Both engine binaries accept `--lp K` to print top-K logprobs as Strata's
`LP <logprob> <token>:<logprob> ...` line, and `EVAL <id,id,...>` to run a fixed
token sequence. Both are only for tests; normal serving does not emit `LP`.

## Build and test from a checkout

```sh
cmake -S engine -B build-vulkan-verify -G Ninja -DSTRATA_ENABLE_VULKAN=ON
cmake --build build-vulkan-verify
./build-vulkan-verify/vulkan/strata-vulkan-router-test
./build-vulkan-verify/vulkan/strata-vulkan-combine-test
./build-vulkan-verify/vulkan/strata-vulkan-bf16-test
./build-vulkan-verify/vulkan/strata-vulkan-sampler-test
./build-vulkan-verify/vulkan/strata-vulkan-q8-test engine/vulkan/shaders/expert_q8_0_gemv.comp.spv
./build-vulkan-verify/vulkan/strata-vulkan-iq-test \
  engine/vulkan/shaders/expert_iq3_xxs_gemv.comp.spv \
  engine/vulkan/shaders/expert_iq2_xs_gemv.comp.spv \
  engine/vulkan/shaders/expert_iq1_m_gemv.comp.spv
python3 engine/vulkan/tools/gen_iq_tables.py   # only after a ggml-common.h change
bash engine/vulkan/engine/build.sh
bash engine/vulkan/engine/test_protocol.sh
bash engine/vulkan/asan_check.sh

# Long-run robustness: NaN/token-0/xe-reset checks (30 minutes on the B390)
python3 engine/vulkan/soak.py --minutes 30 --tokens 16

# Random-read throughput of the GGUF's drive at the engine's queue depths
python3 engine/vulkan/bench_nvme.py --out /tmp/opencode/nvme.json

# Decode-speed A/B: native engine vs llama.cpp bridge, engine-measured timings
python3 engine/vulkan/bench_decode.py --engines native,bridge --tokens 32
```

The stream test needs the GGUF; its command is in `engine/vulkan/README.md`. The
layer-by-layer CPU oracles live in `engine/vulkan/native/`.

## What is not done yet

- The native forward is one token at a time, with fp32 activations and host-side
  sampling. There is no prompt batching.
- QSA sparse attention, KV quantization, gated residual and n-gram PLE are for
  Qwen3.8-Flash-Next and are not ported. The model here uses dense grouped-query
  attention, which is ported.
- IQ2_XS, IQ3_XXS and IQ1_M now have dequant matvec shaders and pass a
  synthetic-GGUF parity test against ggml-compatible CPU references, but no
  real IQ model has been run end to end (the test model uses Q4_K/Q6_K).
- The 30-minute soak passed: 555 requests, 0 bad tokens. Batched submissions
  exist; timeline semaphores and transfer-queue overlap do not. A real
  routing-trace corpus is checked in under `engine/vulkan/traces/`;
  `check_prefetch.py` scores it (2.9% consecutive-layer overlap, so speculative
  next-layer prefetch stays off).

## Development notes (Intel Arc B390)

Module-level notes and the exact build/test commands are in
`engine/vulkan/README.md`. Two points are worth keeping in mind when reading the
numbers above:

- The two engines' forwards agree. Teacher forcing compares them on identical token
  sequences, so a greedy decoding difference is a near-tie argmax flip followed by a
  cascade of different contexts, not a broken kernel. A few positions still differ by
  up to about 0.22 in logprob while sampling the same token. One plausible explanation
  is lower precision inside llama.cpp's Vulkan kernels; that is a hypothesis from
  reading code, not a measurement, and it needs a dedicated test.
- A forward check and a decoder check answer different questions. `compare_greedy.py`
  is the end-user check; `compare_forward.py` is the diagnostic. Both are described
  under [Check correctness yourself](#check-correctness-yourself).

Bugs and measurements from real cards are welcome; follow
[Community benchmarks](COMMUNITY_BENCHMARKS.md) and say what you did not test.
