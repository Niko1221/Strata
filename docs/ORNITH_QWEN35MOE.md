# Ornith-1.5 (Qwen35MoE) on Strata

Ornith-1.5-35B-A3B is a Qwen3.5-family mixture-of-experts model. Strata added a separate Qwen35MoE
architecture path for it rather than teaching the existing Qwen4Exp path a second meaning. This page
records the model, the architecture, the artifacts, what is implemented and verified on the RX 7700 XT
(gfx1101), and what remains.

## Status

| piece | state |
|---|---|
| GGUF inspection and the checked-in layout report | done, measured |
| Architecture identity (`ModelKind`) and the Qwen35MoE geometry/tensor guard | done, 16-case test + the real header |
| Native routed experts at 2048/512 with IQ4_XS gate/up + Q4_K down (grouped HIP) | done, GPU parity |
| Single-file (Ornith) cache resolution and `run3.sh` host-side launcher | done, dry-run verified |
| Qwen35MoE execution backend (GDN, dense attention, MoE block, embedding/head, session) | **not implemented** |
| External Qwen3.6 MTP backend, speculative rollback, `run3.sh` end-to-end serve | **not implemented** |

The guard and the expert kernel are what this increment proves; the execution backend is the next
increment. See "Remaining work" at the bottom for the phase map.

## The model

- **Main:** `AtomicChat/Ornith-1.5-35B-A3B-GGUF`, file `Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf` (20.1 GB).
  This is the architecture-aware ~4-bit quant: routed expert gate/up `IQ4_XS`, routed expert down `Q4_K`.
- **Speculative draft:** `EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF`, file `mtpdraft-Q8_0.gguf` (2.0 GB).
  A trained Qwen3.6-derived MTP head, intended to run as a separate draft with stock Ornith GGUFs. Ornith's
  own embedded MTP tensors are NOT the default.
- **Text only.** Images are not handled on this path.

`docs/ornith/gguf-layout.txt` is the full checked-in tensor layout of the main artifact;
`bench/results/2026-10-02-ornith-rdna3/gguf-metadata.json` is the derived geometry in one file.

## Architecture

Ornith-1.5-35B-A3B is a 40-layer hybrid. Every fourth layer is conventional full attention
(`full_attention_interval = 4`); the other 30 are gated delta-net (GDN) recurrent layers. Both families
run the same 256-expert top-8 MoE FFN with a sigmoid-gated shared expert.

Verified from the artifact:

| | value |
|---|---|
| `general.architecture` | `qwen35moe` |
| layers | 40 (30 GDN recurrent + 10 full attention) |
| residual width | 2048 |
| routed experts / selected | 256 / 8 |
| routed expert FF | 512 |
| shared expert FF | 512 |
| attention heads / KV heads | 16 / 2 |
| head width | 256 |
| partial RoPE | 64, sections `[11, 11, 10, 0]`, base 1e7 |
| GDN state | 128, key heads 16, value heads 32, conv 4 |
| GDN inner (value dim) | 4096 |
| qkv projection width | 8192 (= 2·16·128 + 32·128) |
| RMS epsilon | 1e-6 |
| context | 262144 |
| vocab | 248320 |

Routed expert types, every layer: `ffn_gate_exps` and `ffn_up_exps` `IQ4_XS`, `ffn_down_exps` `Q4_K`.
Shared expert `Q8_0`. Router `ffn_gate_inp` and the shared-expert scalar gate `F32`. Embedding and head
`Q8_0`; final norm `F32`.

The reference is `src/models/qwen35moe.cpp` and the Qwen3.5 conversion code in the llama.cpp revision this
repository pins (`third_party/ggml/VERSION.txt`). The guard's shapes are exactly the `create_tensor` shapes
that file builds.

## Why a separate backend

Qwen4Exp (Qwen3.8-Flash-Next) and Qwen35MoE disagree on almost everything structural: four-stream
hyper-connection residual versus an ordinary residual; QSA sparse selection with an indexer versus dense
attention; 512 experts top-10 and 2560 hidden versus 256 experts top-8 and 2048 hidden; a 2:1 attention
split versus 3:1 with GDN; PLE/ngram state versus none. Folding both into one `ModelGeometry` would make
every kernel ambiguous and every field a possible wrong answer.

So the Qwen4Exp code is left as it is, and Qwen35MoE is introduced beside it:

- `include/strata/core/model_kind.hpp` - `ModelKind` (`Qwen4Exp`, `Qwen35Moe`, `Unknown`).
- `include/strata/core/qwen35.hpp` + `src/core/qwen35.cpp` - `Qwen35Geometry`, `detect_model_kind`,
  `qwen35_geometry`, `check_qwen35_tensors`, `check_qwen35_all`.

`check_qwen35_all` validates, at load time, the architecture plus every dimension the kernels depend on,
and the required tensor set per layer family. It refuses, naming the tensor, its shape/type and the
required shape/type:

- a dense attention projection on a recurrent layer, or a GDN tensor on a full-attention layer;
- a Qwen4Exp-only hyper-connection (`hc_*`), QSA indexer (`indexer.*`) or PLE (`ple_*`) tensor under a
  Qwen35 block;
- a norm or router in a type the kernels do not read as `F32`;
- a weight whose contiguous dimension is not a whole number of its format's blocks;
- inconsistent metadata (`value_length != key_length`, a non-dividing interval, non-positive geometry).

`qwen35_layout_test` exercises all of these on a synthetic header and accepts a real artifact header when
given a path. `strata-qwen35-check` is the same binary installed in the runtime image for `run3.sh
--check-only`.

## Routed expert execution

The grouped native expert HIP path gained Q4_K as a DOWN format (`src/kernels/cuda/iq_kernels.cu`,
`STRATA_D_FMTS`). Q4_K was already a gate/up format (Unsloth UD-Q4_K_XL); Ornith is the first checkpoint
whose routed experts use Q4_K down. The down kernel goes through the same `Fmt<12>::dot` the gate/up path
uses, so this is an admission change, not a new kernel - and it is admitted only because the parity test
says so, at the real 2048/512 geometry, against ggml's float reference and ggml-cpu:

```
synthetic iq4_xs /q4_K  cpu rel 2.04e-02  gpu rel 1.24e-02  cpu-gpu 2.28e-02  ok
q4_K down rows, AVX2 multi-token vs ggml vec_dot: 0 of 36 token-sets differ in any bit
gpu decode-once vs per-entry kernels: bitwise equal
```

`native_expert_parity` gained `STRATA_PARITY_H` / `STRATA_PARITY_FF` so the same harness can check another
model's expert geometry, and the CMake tests are now registered for HIP as well as CUDA.

## `run3.sh`

```
./run3.sh                     # Ornith AD-Q4_K-IQ4_XS + external MTP, http://127.0.0.1:9931
./run3.sh --no-mtp            # target-only, spec 0
./run3.sh --mtp /path/draft.gguf
./run3.sh --model-file /path/Ornith-....gguf
./run3.sh --check-only        # GPU + artifact + geometry validation, then exit
./run3.sh --dry-run
./run3.sh --max-context 131072      # default; 262144 also accepted
./run3.sh --spec 4 --expert-cache auto --pool-workers 10
```

It uses the same 10 GiB VRAM contract as `run.sh`/`run2.sh`, a distinct container name
(`strata-ornith-gfx1101`), and a distinct work tree (`/work/packs/ornith-ad-q4-iq4-xs`,
`/work/mtp/ornith-qwen36`, `/work/logs/ornith`). The main GGUF stays in the shared HF cache. It never
touches the Qwen3.8 pack or MTP directories, and it does not modify `run.sh` or `run2.sh`.

The container entrypoint is `docker/entrypoint-ornith.sh`. It resolves the single GGUF and the external MTP
through `docker/hfmodel.py` (generalized so a family describes one or more files), validates the artifact
with `strata-qwen35-check`, builds the engine config and starts the server.

**The launcher is complete and its host side is verified (`--dry-run`, cache resolution, space gate,
`hfmodel` unit tests), but the engine serve step depends on the execution backend below. Until that
backend exists, a real `./run3.sh` will prepare and validate everything and then be refused by the engine
at model load.** No throughput is claimed for it.

## Measured on gfx1101

- Full HIP `ctest`: 67/70 pass (the 3 failures need a local `pack/full/experts.bin` fixture).
- Ornith-dimension Q4_K-down expert parity: 0 failures, down rows bitwise-equal to ggml.
- The Qwen35MoE guard accepts the real Ornith header and reads back the geometry above.

## Remaining work

The phase map from the task, and where this increment stops:

- **Done:** Phase 1 (inspect), Phase 2/3 (architecture identity and guards), Phase 5 (Q4_K down), Phase 15
  (launcher, host side), Phase 18A/B (unit and HIP primitive tests for the above), Phase 26 (this page).
- **Next, in order:** Phase 6 (GDN backend and state), Phase 7 (dense full attention), Phase 8 (ordinary
  residual + 256×8 MoE + shared expert), Phase 9 (embedding/final norm/head), Phase 10 (session), Phase 11
  (prefill), Phase 12 (trained Qwen3.6 MTP), Phase 13/14 (speculative rollback and tuning), then Phase
  16/17/19-25 (cache/pool/context/performance/quality/API), and finally the end-to-end `run3.sh` numbers.
- **Out of scope, explicitly:** DFlash. It is not implemented, not stubbed, and not part of the design.

## Building and testing

```
./build.sh                    # compile the engine in the HIP builder container and package the image
./build.sh --tests            # ... and run the HIP ctest set on the GPU
python tools/ornith_inspect.py --repo AtomicChat/Ornith-1.5-35B-A3B-GGUF \
       --file Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf \
       --out docs/ornith/gguf-layout.txt --header-out /tmp/ornith-header.gguf
strata-qwen35-check /tmp/ornith-header.gguf          # inside the runtime image
```
