// strata-quant-common.h - Strata's name for the quantization block layouts, codebook grids and sign tables of every ggml format its kernels read.
//
// Phase 1A: this file forwards to third_party/ggml/ggml-common.h (llama.cpp's ggml/src/ggml-common.h at 3cf03257f219afbe7334045ff7c6a06ac68c627d, MIT,
// Copyright (c) 2023-2026 The ggml authors, third_party/strata-ggml/LICENSE, with one Strata edit: GGML_COMMON_IMPL_METAL also opens iq1s_grid).
// The real file stays at the old path for now because the Metal runtime reads that file's TEXT at run time, under that name, and the Mac package ships it
// as metal/ggml-common.h; moving the text here (and the old path becoming the forwarder) goes with the packaging change in phase 1B.
// New Strata code can already `#include "strata-quant-common.h"`; every existing `#include "ggml-common.h"` is unchanged.
#include "../ggml/ggml-common.h"
