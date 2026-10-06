# GP100 VMAD DP4A emulation

On Tesla P100, sm_60 lacks DP4A. Four signed byte-select PTX VMAD operations replace native scalar byte extraction and multiplication. The change is gated to `__CUDA_ARCH__ == 600`; other CUDA architectures and HIP keep their previous source path. The pinned llama.cpp MMQ fallback is separate and unchanged.

## Credit and arithmetic

The optimization idea and original P100 measurements are by [shinbunbun](https://github.com/shinbunbun), from [llama-cpp-p100-patches, patch 01](https://github.com/shinbunbun/llama-cpp-p100-patches/blob/939dd95f9f926c87769bd0442fbcf8a40aff2597/patches/01-vmad-dp4a-sm60.patch). This Strata adaptation adds a scoped PTX accumulator so the output register is written only after all inputs are consumed, including compiler-visible equal inputs. It preserves signed-byte multiplication and low 32-bit accumulation without floating-point changes. The license notice for the original implementation appears below.

The original scalar C++ expression is defined only when every intermediate addition fits int32. Quant block sums are bounded. Boundary tests compare against an independent widened/modulo oracle, including wrapping and cancellation. [CUDA 12.9 PTX VMAD semantics](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#scalar-video-instructions-vmad) describe the signed byte selectors; [inline PTX constraints](https://docs.nvidia.com/cuda/archive/12.9.1/inline-ptx-assembly/index.html) describe the operands and scoped registers.

## Verification

The first live comparison used Strata v0.1.39, commit 6f32ec0, and an independent native-only VMAD candidate 2d17ce0. This PR targets newer main, whose pristine DP4A header is unchanged; the quoted engine performance is from the pinned v0.1.39 experiment, not a benchmark of current main.

Two Tesla P100 PCIe 16 GB cards, Gen3 x8/x8, Ryzen 9 7900X, 64 GB RAM, Ubuntu 24.04.5, driver 580.178.04, and CUDA 12.9.86 ran three ABBA blocks per corpus, six trials per arm. Context was 131072, INT8 KV with 32768 positions resident, split 25/23, spec 4 with MTP, zero prompt-cache reuse, and 256 greedy output tokens. Each trial restarted the engine and performed a disjoint 32-token warmup. The corpora contained 19790 and 119628 actual input tokens; the extended corpus matches the pristine baseline's 3100-row request.

| Actual input tokens | Control prompt tokens/s | VMAD prompt tokens/s | Control decode tokens/s | VMAD decode tokens/s | Decode change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 19790 | 428.20 | 428.25 | 37.85 | 41.70 | +10.17% |
| 119628 | 544.35 | 544.50 | 36.55 | 40.00 | +9.44% |

Prompt throughput was effectively unchanged in both corpora. All twelve text and reasoning outputs matched within each corpus; end-to-end logits were not measured by these performance suites. Raw timings and provenance are in the [19790-token result](../bench/results/p100-vmad-2026-10-06.json) and [119628-token result](../bench/results/p100-vmad-extended128k-2026-10-06.json). The extended result supports this near-full-context synthetic request and runtime configuration. Neither result is a benchmark of current main.

A separate [eight-case correctness suite](../bench/results/p100-vmad-correctness-2026-10-06.json) compared diagnostic twins of the same pinned control and candidate, with identical opt-in serve-path instrumentation. It retained the 128K configuration, INT8 KV/resident 32768, split 25/23 and MTP spec 4, and used fresh processes, no warmup, zero prompt-cache reuse, greedy decoding and a 128-token output limit. Actual input counts ranged from 28 to 4214 and output counts from 3 to 21. Text, reasoning, usage and finish reasons matched in all eight pairs. Each pair's complete 248320-entry float32 logit vector at the first verifier window's row zero matched bit for bit, with maximum absolute difference zero. This measures the first generated position, not every generated position.

Seven cases supplied 223 teacher-forced prompt logprob rows per arm; token IDs and serialized scores matched exactly. The retrieval case had no eligible teacher-forced windows, so that metric is unmeasured for it. Separately, both arms passed all eight declared output fixtures: arithmetic, Python, JSON, Spanish, Japanese, retrieval and ordering. These smoke fixtures do not establish general model quality. The result JSON lists each case, actual token counts, observed process binary hashes, source commits, common diagnostic patch hash and raw/source archive hashes. The original performance binaries were retained unchanged; correctness was measured on the diagnostic twins. The earlier missing-dump attempt was rejected and archived. This is pinned v0.1.39 evidence, not a current-main engine correctness result.

Both P100s passed 1,813,290 exact arithmetic variants. Compute Sanitizer reported zero errors. IQ/MMVQ fixtures passed on both cards and both binaries. All 48 real expert layers passed the existing GPU/CPU-float tolerance tests on GPU0. sm_60 probes showed four VMAD.S8.S8 instructions, 12 registers versus 13 for scalar, and no spills. Separately compiled sm_61 and sm_75 probes matched control SASS exactly; no modern GPU device tests were run. The clean current-main PR header also passed CUDA 12.9 compile-only probes on sm_60, sm_61 and sm_75: four VMAD operations, 12 registers and no stack/local storage on sm_60, and byte-identical control/candidate probe SASS on sm_61/sm_75. The full current-main engine was not built or benchmarked. Synthetic arithmetic timing favored VMAD by about 1.87x and 2.09x for one/four chains, independently of model throughput.

The IQ2_XS-labelled file's expert gate/up tensors are IQ2_S in 34 layers, IQ2_XXS in 11 and IQ1_M in 3; all 48 down tensors are Q2_0. Native expert checks use the existing 3% GPU/reference tolerance, not full-model bitwise comparisons.

Sampled configured limits stayed at 200 W, and final requested/enforced limits read back 200 W on both cards. Instantaneous board-power samples peaked at 226.24/225.79 W in the first corpus and 242.30/236.01 W in the extended corpus, so neither a strict instantaneous ceiling nor an energy-efficiency gain is claimed. The clean engine/configuration was restored after each suite. The extended suite's per-trial cleanup confirms no live experimental session members and a free port 8081 before reuse.

## Repeat the focused tests

CPU/offline checks require a C++17 compiler and no CUDA device:

```sh
uv run --no-project python tests/p100/verify_offline.py
```

Compile-only CUDA 12.9 probes and the device test:

```sh
mkdir -p build-p100-vmad
nvcc -std=c++17 -O3 -arch=sm_60 -Iinclude --cubin -Xptxas=-v tests/p100/dp4a_compile_probe.cu -o build-p100-vmad/probe.cubin
cuobjdump --dump-sass build-p100-vmad/probe.cubin
nvcc -std=c++17 -O3 -arch=sm_60 -Iinclude tests/p100/dp4a_cuda.cu -o build-p100-vmad/dp4a_cuda
```

When a P100 is free for tests, the explicit device flag runs parity plus ABBA/BAAB microtiming, emitting raw CSV. Index 0 is relative to `CUDA_VISIBLE_DEVICES`. Run each card separately:

```sh
CUDA_VISIBLE_DEVICES=0 build-p100-vmad/dp4a_cuda --run-on-authorized-p100 > build-p100-vmad/raw-gpu0.csv
CUDA_VISIBLE_DEVICES=1 build-p100-vmad/dp4a_cuda --run-on-authorized-p100 > build-p100-vmad/raw-gpu1.csv
compute-sanitizer --tool memcheck build-p100-vmad/dp4a_cuda --run-on-authorized-p100
```

Synthetic arithmetic tests filter a candidate; actual quant kernels, model correctness and identical full-request benchmarks remain required before accepting a performance change.

## Original implementation license

```text
MIT License

Copyright (c) 2026 shinbunbun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
