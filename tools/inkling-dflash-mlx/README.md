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
