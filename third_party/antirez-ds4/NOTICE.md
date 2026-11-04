# antirez/ds4, MIT

The Metal form of DeepSeek-V4.1's vision tower follows antirez/ds4 (https://github.com/antirez/ds4), commit 890898cf, MIT License.
The licence text is in `LICENSE` beside this file; it is copied unchanged next to the Metal sources in the Mac engine package.

Derived files (the way a DeepSeek-V4 vision tower is split into Metal kernels, which steps round to BF16, the 2-D rotary on q and k,
the aligner's unfold order, a polynomial erf because MSL has none):
- `metal/src/engine/msl/ds4_vision.metal` and `ds4_vision_mpp.metal`, after `metal/deepseek4_vision.metal`, `metal/glm53_vision.metal`, `metal/glm53_bf16.metal`
- `metal/src/engine/ds4_vision_metal.cpp` and `metal/include/metal_compat/ds4_vision_metal.h`, after `ds4_gpu_deepseek4_vision_encode`
  and its helpers in `ds4_metal.m`

The tiled BF16 GEMM, the rest of the host side and the wiring into `src/dsv41/ds4_vision.cu` are Strata's own.

Copyright (c) 2026 The ds4.c authors; (c) 2023-2026 The ggml authors; (c) 2023 DeepSeek.
