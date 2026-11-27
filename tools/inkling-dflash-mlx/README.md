# Inkling DFlash2 draft projections on Apple Silicon

Pack the DFlash2 safetensors checkpoint once with MLX on Apple Silicon.

```sh
python inkling_dflash_mxfp8.py model.safetensors draft-mxfp8.safetensors
```

A draft worker can load the resulting file once and apply a named projection to the four concatenated SQ-10 target features.

```python
import mlx.core as mx
from inkling_dflash_mxfp8 import linear

weights = mx.load('draft-mxfp8.safetensors')
projected = linear(weights, 'fc', target_features)  # [..., 9216] -> [..., 2304]
```

Use the same `linear` function for the attention, convolution-controller, MLP and candidate-selector projections. Normalization weights, base kernels and selector codebooks retain their checkpoint representation. The helper evaluates the packed projections with MLX's MXFP8 kernel and BF16 inputs. The target model verifies the resulting draft tokens. Host feature transport and the verification loop are integration work for the inference engine.

## Generate a candidate block

Create one `DraftBlock(weights)` per request. Call `append(target_features)` with newly committed rows from target visits 5, 23, 41 and the final normalized state. For a restored prefix, pass the absolute position of the first supplied row as `start`. Supply up to the most recent 2,047 committed rows when initializing from a long prompt. `hidden(embeddings)` takes the normalized target input embeddings for the anchor followed by seven mask tokens. `select(hidden, anchor_id, (head_weight, head_scales))` uses the packed target output head and returns seven candidate IDs for target verification.

Run a deterministic comparison of chunked and contiguous input using the packed draft weights. The smoke test generates its own feature rows, input embeddings and small output vocabulary.

```sh
python smoke_block.py draft-mxfp8.safetensors
```

## Run with the Metal target

The single-slot native launcher uses the target's verified tokens and an MLX draft process. Build `strata-inkling` from this branch. For FP8 target KV, also apply the Metal E4M3 implementation from [PR 6983](https://github.com/Niko1221/Strata/pull/6983) before building. The measurements use this combination.

Set `ENGINE` to the compiled binary, `INK_DFLASH_MODEL` to the SQ-10 pack, `INK_DFLASH_ASSETS` to the packed draft directory, and export the latter two variables. From this tool directory, prepare the target layout and output head once.

```sh
INK_DFLASH_LAYOUT=1 "$ENGINE" --serve --native "$INK_DFLASH_MODEL" --draft off --kv fp8 </dev/null 2>"$INK_DFLASH_ASSETS/layout.log"
python pack_head.py
python launch.py "$ENGINE" "$INK_DFLASH_MODEL" "$INK_DFLASH_ASSETS"
```

The launcher uses MLX and NumPy. Submit a greedy text request through the engine's native stdin protocol after `READY`. The target returns `BT` token records followed by `BDONE` with the number of offered and accepted drafts.

## Single-slot streaming HTTP

In the Strata server run configuration, set `exe` to the absolute path of `strata-inkling-dflash` in this directory. Export `DFLASH_ENGINE` (the patched target executable), `DFLASH_PYTHON` (the MLX environment's Python executable), and `INK_DFLASH_ASSETS` (the packed assets directory) before starting the server. Use the normal Inkling arguments with `--native MODEL --parallel 1 --draft off --kv fp8 --max-context 16384`. The external worker supplies the proposals. The launcher forwards model inspection and replaces itself with the native engine so the server manages the engine PID. Closing the engine closes the worker socket.

Use greedy text requests. On streamed responses, read `timings.draft_n` and `timings.draft_n_accepted` from the final JSON event. On a prompt checkpoint hit, the external drafter rebuilds its prefix cache from retained target feature rows. If generation has evicted required rows, the target decodes normally until a complete draft window is available.

For structured directory extraction, set `INK_DFLASH_GREEDY_HEAD=1` in the worker environment to propose the highest-scoring token at each draft position. The target verifies every proposal. Leaving the variable unset uses the checkpoint's candidate selector.

For a breakdown of multi-second GPU submission stalls and request-level measurements, see [Profiling long decode stalls on Metal](request-latency.md).
