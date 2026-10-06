# GP100 VMAD DP4A emulation

On Tesla P100, sm_60 lacks DP4A. Four signed byte-select PTX VMAD operations replace native scalar byte extraction and multiplication. The change is gated to `__CUDA_ARCH__ == 600`; other CUDA architectures and HIP keep their previous source path. The pinned llama.cpp MMQ fallback is separate and unchanged.

## Credit and arithmetic

The optimization idea and original P100 measurements are by [shinbunbun](https://github.com/shinbunbun), from [llama-cpp-p100-patches, patch 01](https://github.com/shinbunbun/llama-cpp-p100-patches/blob/939dd95f9f926c87769bd0442fbcf8a40aff2597/patches/01-vmad-dp4a-sm60.patch). This Strata adaptation adds a scoped PTX accumulator so the output register is written only after all inputs are consumed, including compiler-visible equal inputs. It preserves signed-byte multiplication and low 32-bit accumulation without floating-point changes. The license notice for the original implementation appears below.

The original scalar C++ expression is defined only when every intermediate addition fits int32. Quant block sums are bounded. Boundary tests compare against an independent widened/modulo oracle, including wrapping and cancellation. [CUDA 12.9 PTX VMAD semantics](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#scalar-video-instructions-vmad) describe the signed byte selectors; [inline PTX constraints](https://docs.nvidia.com/cuda/archive/12.9.1/inline-ptx-assembly/index.html) describe the operands and scoped registers.

## Verification

The first live comparison used Strata v0.1.39, commit 6f32ec0, and an independent native-only VMAD candidate 2d17ce0. This PR targets newer main, whose pristine DP4A header is unchanged; the quoted engine performance is from the pinned v0.1.39 experiment, not a benchmark of current main.

Two Tesla P100 PCIe 16 GB cards, Gen3 x8/x8, Ryzen 9 7900X, 64 GB RAM, Ubuntu 24.04.5, driver 580.178.04, and CUDA 12.9.86 ran three ABBA blocks, six trials per arm. Context was 131072, INT8 KV with 32768 positions resident, split 25/23, spec 4 with MTP, zero prompt-cache reuse, actual 19790 input tokens, and 256 greedy output tokens.

| Metric | Control median | VMAD median |
| --- | ---: | ---: |
| Prompt tokens/s |428.20|428.25|
| Decode tokens/s |37.85|41.70|

Decode throughput increased 10.17% for this corpus/configuration. All twelve text and reasoning outputs matched; end-to-end logits were not measured. Raw trial timings are in [the machine-readable result](../bench/results/p100-vmad-2026-10-06.json). This does not establish near-full 128K performance or general model-quality parity.

Both P100s passed 1,813,290 exact arithmetic variants. Compute Sanitizer reported zero errors. IQ/MMVQ fixtures passed on both cards and both binaries. All 48 real expert layers passed the existing GPU/CPU-float tolerance tests on GPU0. sm_60 probes showed four VMAD.S8.S8 instructions, 12 registers versus 13 for scalar, and no spills. Separately compiled sm_61 and sm_75 probes matched control SASS exactly; no modern GPU device tests were run. Synthetic arithmetic timing favored VMAD by about 1.87x and 2.09x for one/four chains, independently of model throughput.

The IQ2_XS-labelled file's expert gate/up tensors are IQ2_S in 34 layers, IQ2_XXS in 11 and IQ1_M in 3; all 48 down tensors are Q2_0. Native expert checks use the existing 3% GPU/reference tolerance, not full-model bitwise comparisons.

Requested and enforced limits remained 200 W. Instantaneous board-power samples peaked above 200 W, so neither a strict instantaneous ceiling nor an energy-efficiency gain is claimed. The clean engine/configuration was restored after testing.

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
