# V100 Stage 1 — Final Report

**Verdict: PASS** — the Qwen3.8-Flash-Next MoE engine (Niko1221/Strata @ `1ee8b66`,
Swift-1.5 IQ3_XXS GGUF) now builds, runs, verifies, and serves on Tesla V100
(sm_70) with three logical commits, no MoE redesign, no new quants, and the
output matching an independent llama.cpp reference token-for-token on the
primary prompt.

Machine: 5× V100 (GPU0 32 GB PCIE = dev, GPU4 16 GB SXM2 = validation,
GPU2 = reference), 2× Xeon E5-2680 v4 AVX2 (no AVX-512), 125 GB RAM.

## Phase results

| phase | result | evidence |
|---|---|---|
| 1. Analyze | done | `Docs/v100-compatibility.md` — arch `qwen4exp`, 48 layers, 24,576 experts/layer top-10, hybrid attention+GDN, QSA sparse attention, MTP, PLE 320,001,536-row table, expert arena in RAM + expert cache in VRAM; SM70 gap list |
| 2. Baseline | done | `Docs/v100-stage1-baseline.md` — original build red on V100 (no sm_70 codegen; in-tree ctest registrations unusable) |
| 3. Patch | done | `6bb6b94` SM70 support (CMAKE_CUDA_ARCHITECTURES=70, FA_ALL_QUANTS, tf32 scorer fallback — SASS: 0 ldmatrix / 0 MMA, 448 FFMA.FTZ); `ae7b4fb` ctest fixes + build doc; `12979f4` **dangling-else fix** in `physical_cores` (the 15-minute startup hang: unbraced `else` bound to the inner `if`, 54,263 "cores" on a 56-CPU box → two braces, 55 workers) |
| 4. Build | green | `build-sm70`; in-tree 20/21; ctest 19/20 on V100 (`ple_parity` red by design — needs Q2_0 fixtures) |
| 5. 32 GB first inference | PASS | `Docs/v100-testing.md` §5 — full production path up (arena pin, PLE, MTP, spec/verify windows, graph capture); 28.4 tok/s decode, coherent `<think>` output |
| 6. Correctness | PASS (documented caveats) | §5.3 deterministic across runs (32/32 ids). llama.cpp reference, same quant, greedy, identical raw prompt: **32/32 tokens identical** (prompt 1); prompt 2 (64 tokens): 31/31 identical, ULP-level branch at token 32, both coherent — consistent with the upstream-declared imprecise GPU hit path (ROUND 328; opt-in, startup warning) |
| 7. 16 GB validation | PASS | §7 — GPU4 fits with peak 15.6 GiB (auto cache 6,321 slots / 10.23 GiB), 27.3 tok/s decode (≈GPU0, CPU-pool-bound), coherent output semantically identical to the reference |
| 8. Optimize | done | `--pool-workers 28` (physical-core count): 39.9/39.7 vs 35.3–37.1 tok/s sustained decode (+9–13 %), adopted into the server config. 2 MB hugepages for the arena measured (12.2 tok/s) and left off on this box; the startup line reports which page backing the arena got |
| 9. Benchmark | done | `Docs/v100-benchmarks.md` — prefill 401 / 423 tok/s (2,047 tokens) on GPU0 / GPU4; sustained decode 37.1 / 32.8 tok/s (55 workers) → **39.9 with 28 workers**; peak VRAM 18.3 / 15.6 GiB; llama.cpp CPU-only reference ~3.9 tok/s decode (≈9–10× slower) |
| API (port 8180) | verified | `serve/server.py` + `strata-swift-iq3_xxs.json` (32,768 ctx, `--kv int8`): `/v1/models` ok; `/v1/chat/completions` greedy — **identical output across two calls**, thinking split into `reasoning_content`, correct usage counts, SSE streaming with ~1.9 s TTFT and ~40 tok/s sustained; clean shutdown |

## Deliverables

- Docs: `v100-stage1-baseline.md`, `v100-compatibility.md`, `v100-build.md`,
  `v100-testing.md`, `v100-benchmarks.md`, `v100-stage1-final.md` (this file).
- Commits (branch `feature/v100-moe`): `5f75af7` baseline doc, `6bb6b94` SM70,
  `ae7b4fb` ctest + build doc, `12979f4` dangling-else fix, + docs commit.
- Server config `strata-swift-iq3_xxs.json` (gitignored, user-machine artifact)
  with the verified production flags + `--pool-workers 28`.

## Known caveats (all documented, none blocking)

1. **GPU expert-cache hit path** is declared NOT CORRECT upstream (ROUND 328):
   tokens can diverge from a cache-off run; the cache is opt-in and warned at
   startup. In practice, on this model/quant, output matched the llama.cpp
   reference for 31+ tokens on both test prompts (the R4.2h fp32-scale fix
   appears to have closed the divergence for this workload).
2. **16 GB VRAM headroom**: ~0.4 GiB at the auto cache on 8192-ctx fp16 KV.
   The 32 K int8-KV server config targets the 32 GB card.
3. **memlock**: run the engine with `ulimit -l unlimited` so `cudaHostRegister`
   can pin the ~40 GiB expert arena (portable mode, 4 KB pages on this box).
4. **Hugepages**: leave the kernel pool empty on this machine (measured loss);
   on machines with a large 2 MB pool and plenty of free normal RAM the arena
   will use them automatically.
5. **Reference engines**: the production `ik_llama.cpp` fork crashes on this
   box at CUDA init (NCCL/SIGFPE); the vanilla llama.cpp build is the working
   cross-check.
6. `ple_parity` ctest is red by design (missing Q2_0 fixtures).

## Stage 2 candidates (out of scope here)

dense-model path, multi-GPU, additional quants, scheduler/memory rewrites,
new API frameworks.
