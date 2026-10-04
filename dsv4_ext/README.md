# dsv4-strata-ext

Additive extension for running DeepSeek-V4-Flash with Strata's "hot experts in VRAM" philosophy.
**Does not modify Strata.** Step 1 is CPU-only C++17 (no CUDA needed), written to build with gcc-10.
Step 2 is the real CUDA device layer (`src/gpu_cuda.cu`): same API, opt-in, see **CUDA** below.

## Build and test

    cd dsv4-strata-ext
    CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 sh build.sh

`build.sh` compiles with `-Wall -Wextra`, runs `test_dsv4_core` (header/config/plan) and `test_ops`
(the ops oracle in `tests/golden.txt`). It generates `src/iq_tables.cpp` on first run from a llama.cpp
checkout (`LLAMA_CPP_DIR=… sh build.sh`, default `~/workspaces/llama.cpp`); llama.cpp is only READ,
nothing in it is modified. Regenerate manually with:

    python3 tools/extract_iq_tables.py ~/workspaces/llama.cpp src/iq_tables.cpp include/dsv4/iq_tables.hpp

(If your filesystem is mounted noexec, build.sh falls back to /tmp/dsv4_build automatically.)

## Inspect your real GGUF (all 3 shards, header only)

    python3 tools/dump_gguf_shapes.py DeepSeek-V4-Flash-UD-IQ1_M-0000{1,2,3}-of-00003.gguf > shapes.txt

## Inference: one-shot

    IDS=$(python3 tools/dsv4_tokenize.py --model M.gguf --chat \
          --messages '[{"role":"user","content":"ciao"}]' --ids)
    ./build-out/dsv4_run --model M.gguf --prompt-ids "$IDS" --n-predict 32 \
        --temperature 0.7 --top-k 64 --top-p 0.9

`--prompt-ids` are token ids. `tools/dsv4_tokenize.py` produces them with the model's own BPE tokenizer
(Strata's `tools/strata_tokenizer.py`, read out of the GGUF metadata) and can apply the model's chat
template with `--chat`. Without `--temperature` the runner is greedy, as Strata's engine default.

## Inference as an API: OpenAI, Anthropic and Strata's web app

`dsv4_run --serve` is a resident engine on stdin/stdout. It speaks the same line protocol as
`strata --serve`, so Strata's HTTP layer and web app sit on top of it unchanged:

    python3 tools/serve_dsv4.py --model DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf --port 8095
    python3 tools/serve_dsv4.py --cuda --model DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf --port 8095

    curl http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"ciao"}],"max_tokens":64}'

    open http://127.0.0.1:8095/          # Strata's Chat / Monitor / About app, same port

`tools/serve_dsv4.py` imports `serve/server.py`, replaces its `StrataEngine` with a subclass that
spawns `dsv4_run --serve`, and calls its `main()`. **Nothing under the Strata checkout is edited.**
It also extracts the tokenizer (`vocab.json`, `merges.txt`, `token_type.json`, `chat_template.jinja`)
from the GGUF into `serve-work/tokenizer/` with Strata's own `tools/strata_tokenizer.py`. In the
CPU-emulated device layer the VRAM size is `DSV4_EMULATED_VRAM_MIB` (default 8192): pass it with
`--engine-env DSV4_EMULATED_VRAM_MIB=22000`. With `--cuda` that variable is ignored: the engine asks the
driver for its real free VRAM, so `--expert-vram-pct` / `--expert-vram-reserve-mib` are what steer the plan.

### Speed: what actually decides the tok/s

The real GGUF is **43 layers x 256 experts, top-6** (`dsv4_plan` prints it): 74.4 GiB of
expert weights, and every token evaluates 258 of them, about **1.8 GB of weights per token**.
Three things decide how fast that goes:

1. **OpenMP must be linked.** `build.sh` probes it and fails loudly if it is missing. Without it
   every `#pragma omp parallel for` in `cpu_mv` / `gpu::matvec` / `experts_hit` is inert and the
   whole model runs on ONE core: ~3.7 s/token instead of ~0.2 s. `ldd build-out/dsv4_run | grep gomp`
   is the check. `--threads N` (0 = all cores) is applied at load.
2. **A profile, or nothing is in RAM.** With `--expert-ram profile` and a `--expert-profile` file
   that does not exist, the engine warns and keeps **zero** experts in RAM: every MISS then reads
   its 6.8 MiB out of the mmap'ed shard. `serve_dsv4.py` prints that warning before starting.
   Build the profile by running once with `--expert-profile-save` (it is written when the engine
   stops, also in `--serve`), or with `--dump-routing` + `tools/make_expert_profile.py`.
3. **The MISS LRU arena** (`--ram-cache-mib`, default 8192). Experts that are neither in the
   profile's RAM copy nor in a VRAM slot are copied once into a per-layer LRU arena instead of
   being re-read from disk on every token. `0` restores the old read-straight-from-disk behaviour.

`--expert-ram all` copies all 74 GiB into RAM: only do it on a machine with that much free RAM,
and remember the mmap'ed shards already want the page cache.

### CUDA: the real device layer

`src/gpu.cpp` (CPU-emulated) and `src/gpu_cuda.cu` (real device) implement the **same** `dsv4::gpu` API
([`include/dsv4/gpu.hpp`](include/dsv4/gpu.hpp)), so exactly one of them is ever compiled into a binary:

    sh build.sh --cuda          # emulated binaries + tests, PLUS build-out/dsv4_run_cuda
    sh build.sh --cuda-only     # only the CUDA binaries (no tests, no emulated binaries)
    DSV4_CUDA_ARCH=sm_80 sh build.sh --cuda-only     # another card

CMake does the same thing with one switch:

    cmake -S . -B build -DDSV4_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j

What is on the device: one dequantise-and-dot kernel per ggml type (one block per output row, weights
decoded on the fly because the weights *are* the traffic), the IQ codebooks in `__constant__` memory, and
the whole MoE HIT path - gate/up matvecs, a fused clamped-SwiGLU that folds in the routing weight, then
the down matvecs summed on the device. `DSV4_EMULATED_VRAM_MIB` no longer applies: `mem_info()` reports
the driver's real free VRAM, so `--expert-vram-pct` plans against it.

Capability gating, not silent zeros: `type_supported()` / `experts_supported()` decide at load time which
tensors get uploaded and which MoE layers run on the device. A `matvec()` or `experts_hit()` that returns
`false` wrote **nothing**, and the caller falls back to the host path (once per layer, with one warning).
Adding a ggml type = one trait in [`include/dsv4/dq_traits.hpp`](include/dsv4/dq_traits.hpp) plus one
`case` in `matvec()`.

The decode math is not duplicated between host and device: `dq_traits.hpp` is compiled twice, with the
`G_*` codebook names pointing at `__constant__` memory under nvcc and at `dsv4::iq::` on the host.
`tests/test_dq_traits.cpp` then compares every trait, element by element, against
`dequant_row()`/`row_dot()` from [`src/dequant.cpp`](src/dequant.cpp) - **on the CPU, no GPU and no nvcc
needed**, which is how a wrong nibble shift gets caught before it becomes garbage in the logits.
Precision note: the host reference accumulates in `double` left-to-right, the kernel in `float` as a tree
reduction, so device and host agree to float rounding, not bit for bit.

**How the VRAM plan gets its numbers.** Both backends report free VRAM *already net of what this model has
put on the card* (`cudaMemGetInfo` for CUDA, its own pool counter for the emulated one), and this engine's
KV cache / activations / prefill buffers are host memory. So `plan_expert_memory()` is called with
`weights = kv = act = prefill = mtp = scratch = 0`: charging them again double-counts and used to abort the
run with `fixed costs (8742 MiB) exceed the usable VRAM (-883 MiB after the reserve)` on a 24 GiB card that
had room. `dsv4_plan` still estimates those costs, because there it is a what-if tool with no device in
front of it. At load the engine prints the real split: `device: CUDA | VRAM a/b GiB used, c GiB free
(weights x GiB, scratch y GiB)`.

Protocol (`dsv4/serve_loop.hpp`):

    -> READY <ctx> stop            INFO key=value ... before it
    <- GEN <max_new> [temperature= top_p= top_k= min_p= penalty_* seed=] <id,id,...>
    -> PP <read> <total> <ms> <tok_s>   T <id> ...   DONE <n> <prompt> <ms> <ms> <finish>
    <- STOP (honoured between tokens)   QUIT

## Verified / not verified

| Area | Status | Evidence |
| --- | --- | --- |
| GGUF header reader (3 synthetic shards) | OK | `test_dsv4_core`: tensors merged, duplicate names refused |
| Config from GGUF keys | OK (synthetic values copied from the real shard-1 dump) | test; wrong arch refused |
| Expert byte inventory (variable quant per layer, MTP layer apart) | OK on synthetic shapes | IQ1_M 5,505,024 B/expert checked by hand |
| VRAM plan, P=0/25/50/75/100, auto, --expert-cache, fixed>free error | OK (arithmetic) | test: monotonic, <= budget, <= n_expert/layer |
| VRAM plan inside the engine | FIXED | `plan_expert_memory()` now gets zero fixed costs: `mem_info()` is already net of this model's allocations (see **CUDA** above). Old code double-counted and aborted with `fixed costs (8742 MiB) exceed the usable VRAM (-883 MiB …)` |
| Run on the REAL GGUF | NOT RUN | needs `shapes.txt` / your machine |
| CMake build, both variants | OK (configure + compile) | `cmake 4.4.3` + `g++-10`: emulated and `-DDSV4_WITH_CUDA=ON` both build clean; `CUDA_FLAGS` shows `arch=compute_89,code=sm_89` |
| gcc-10 / CUDA 12.8 | COMPILES, NOT RUN | `nvcc 12.8.93` + `sm_89`: `dsv4_run_cuda` / `dsv4_plan_cuda` link `libcudart.so.12`; never executed (the GPU was busy) |
| CUDA decode math == `src/dequant.cpp` | WRITTEN, TEST NOT RUN | `tests/test_dq_traits.cpp`: float **bits** per element + 1e-5 dot tolerance over the real shapes; host-only, so it can run with the GPU occupied |
| CUDA kernels on device (matvec, swiglu, `experts_hit`) | NOT RUN | needs a free GPU: `./build-out/dsv4_run_cuda --selfcheck`, then compare logits against `dsv4_run` with `tools/compare_logits.py` |
| `dsv4_plan` executable (real GGUF -> config, per-layer expert bytes, VRAM plan, P sweep) | OK on synthetic shards; NOT RUN on the real GGUF | built with g++ 13.3, no warnings |
| CPU reference ops: router (sqrtsoftplus/sigmoid/softmax, bias, hash), YaRN rope + rotary (fwd/inv), window/compress top-k indices, clamped SwiGLU, rmsnorm | OK vs numpy port of the official `model.py` (`tests/gen_golden.py`, `tests/golden.txt`) | `test_ops`: 24 checks, indices exact, floats <= 4e-6 |
| Hyper-connections (`hc_split_sinkhorn`, `hc_pre`, `hc_post`, `hc_head`) and `sparse_attn` with sink | OK vs numpy port of the official `kernel.py` | `test_ops`: floats <= 1.6e-7 |
| Weight dequant: F32/F16/BF16/F64/I8-64, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_K, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/IQ4_NL/IQ4_XS/MXFP4 | WRITTEN (`src/dequant.cpp`, byte-offset port of `ggml-quants.c`) | compiles clean with g++-10 and g++ 12; **not yet parity-tested against ggml** |
| IQ codebooks (iq1s_grid, iq2xxs/iq2xs/iq2s, iq3xxs/iq3s, ksigns_iq2xs, kvalues_iq4nl) | GENERATED from `ggml-common.h` | `tools/extract_iq_tables.py` -> `src/iq_tables.cpp` |
| GGUF tensor data offsets (`abs_offset`, `general.alignment`, per-shard `data_start`) + bounds check at load | WRITTEN | compiles clean; runtime check not run |
| CPU-emulated device layer (`src/gpu.cpp`: malloc VRAM, H2D/D2H, matvec, `experts_hit`) | WRITTEN | lets the HIT/MISS pipeline run with no GPU; CUDA port replaces this file |
| Loader, forward (HC, CSA/HCA, router, hash), HIT/MISS tier, profile, benchmarks | WRITTEN, NOT RUN | no token output verified yet: run `dsv4_run --selfcheck` on the real GGUF |
| Dequant self-check on the REAL IQ1_M shards (10 types) | OK on your run | `finite=1`, mean ~0, std ~0.025; `output.weight` Q4_K std 0.30 |
| Sampler (temperature, top-k, top-p, min_p, repeat/freq/presence penalties, seed) | WRITTEN (`src/sampler.cpp`) | replaces the hardcoded argmax; greedy still the default |
| `--serve` engine loop + `tools/serve_dsv4.py` (OpenAI /v1, Anthropic /v1/messages, web app) | WRITTEN, NOT RUN | links clean; the protocol matches `serve/server.py:533` line for line |
| Tokenizer strings kept whole (`kMaxStoredStrs` 16 -> 2M) | WRITTEN | 16 entries truncated the 129280-entry vocab: every answer decoded to "" |
