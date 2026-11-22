# v0.7.1 full code port plan — implementation log

## Overview
User decision 2026-11-22: port ALL researched code into the Q38 tree before the
0.6.4 engine build: Inkling-2 DLL bundling + GGUF reader support, TurboQuant
CUDA/9B KV code (Apache-2.0), ELT ideas (license unknown -> docs only, no code
copy). Then bump engine to 0.6.4 and semantic product version, build, overwrite
install, publish.

## License handling (SOP-Security compliance)
- `zapabob/yasakani` (yskn_inkling2.dll): Apache-2.0. LICENSE, NOTICE and
  THIRD_PARTY_NOTICES.md kept verbatim. DLL SHA-256 verified against BUILDING.md
  pinned value (`b10248cd...078`).
- `zapabob/Turboquant-CUDA`: Apache-2.0 (LICENSE present). Code may be ported
  with attribution.
- `zapabob/elastic-looped-transformer`: NO LICENSE file -> license unknown.
  Per SOP-Security, do NOT copy code; document ideas/measurements only.

## Researched facts
- Inkling-2 GGUF: `general.architecture = inkling2_loop` (GGUF v3, 704 tensors,
  63 KV). ggml-org publishes IQ4_NL 17.6GB / Q8_0 27.8GB. It is a looped MoE
  (24 layers, 6-17 loop 3x = 48 evals, 128 experts/8 active, NVFP4/MXFP8).
- Inkling-2 engine is a separate DLL (`yskn_inkling2.dll`, C API `ink2_step`),
  24GB+ GPU required, cannot run on this 32GB/16GB PC.
- TurboQuant-CUDA measures Qwen3.5-9B KV-cache TurboQuant with captured replay
  (Apache-2.0).
- ELT shares transformer weights over L iterations with per-request L in [1,4].

## Port plan
1. Inkling-2: bundle DLL as a separate (unsigned, local) component with NOTICE;
   add `inkling2_loop` architecture recognition + tensor/geometry metadata to
   gguf_reader (inspect only; generation stays in the DLL).
2. TurboQuant: port the 9B KV measurement/cuda reference with Apache-2.0 header
   and attribution under a research/ directory (not wired into the Q38 build).
3. ELT: document ideas only (no code copy).
4. Bump native engine to 0.6.4 (unsigned) + semantic product version; sync
   docs/data/release.json upstream ref to v0.2.18.
5. Build -> NSIS overwrite install -> tag/release/push.

## Residual risks
- Large amount of unverifiable code (models need 24GB+/64GB+, this PC has 16GB).
- No signing key; 0.6.4 engine and DLL component are unsigned/local.
- Keep fork Q38/DS4/RAM32/9B anchors intact; verify after every step.
