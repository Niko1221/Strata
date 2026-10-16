# Muse Spark numerical fixture

From the repository root, with the supplied Python/NumPy environment:

```sh
python -m tools.muse.fixture --output-dir /tmp/muse-fixture
python -m unittest discover -s tools/muse -p 'test_*.py'
```

This writes a locally generated **F32 GGUF v3** `model.gguf` and
`reference.json`. No downloads, native runtime, GPU, Torch, or external model
weights are used. The fixture has hidden64, head16, heads4/KVheads2, latent16,
FF32, four experts/top2, vocabulary64, two layers, context32, window3. Its local
RoPE/global NoPE pattern is `[true,false]`. The seven forced tokens are
`[1,5,8,3,7,2,9]`; logits have shape `[7,64]`, including every vocabulary entry
for the next token after each forced input. `generated_tokens` is empty.

## What the reference computes

`reference.py` implements an independent batch-causal NumPy FP32 text graph:

- Weightless embedding RMS, weighted pre-attention RMS, Q/K projection and
  per-head weighted RMS (Q scale already folded), NeoX half-split RoPE only on
  local layers, contiguous GQA, causal local/full attention with `1/sqrt(head16)`.
- Per-head attention output RMS, separate sigmoid gate **before** output
  projection, weightless post-attention RMS, folded alpha/beta residual.
- Weighted FFN stream RMS; full-stream sigmoid router. Correction bias changes
  top-k **selection only**; unbiased selected probabilities are normalized with
  `1e-15`, then multiplied by `expert_weights_scale`.
- Latent down, weighted pre-expert RMS, selected SwiGLU/down experts, each
  **input-scaled** post-expert RMS before weighted accumulation, latent up,
  **input-scaled** `post_ffw_norm.weight` RMS, second alpha/beta residual.
- Final weighted RMS, output head, logit scale, then tanh softcap.

The serialized names/keys and already-folded semantics follow the published
Muse GGUF graph (`llama-muse-spark.cpp`), literal metadata/tensor research, and
HF `modeling_muse_spark.py` text norm/router definitions. HF's independently
specified tiny-config rotary/mask flags are not used. Shared
`tools/gguf_writer.py` and `gguf_reader.py` handle the container; tensor directory
shapes are explicitly reversed from NumPy row-major shape for GGUF's
contiguous-axis-first layout. The reference is computed from serialized F32
readback, not from Python's unrounded metadata literals.

Seed 2762 weights are nonzero/nonuniform. Norms, residual gates, expert
magnitudes, and correction bias differ. Latent-up weights deliberately put the
FFN branch near `sqrt(post_norm_epsilon)`: otherwise downstream RMS almost
cancels a missing common router weight scale/normalization. Tests compile
explicitly wrong test-local graph variants, requiring each omission/reordering
to fail the numerical comparison. They also check causal prefixes, the exact
window boundary (three visible tokens including self), scalar norm/RoPE/router
values, denominator epsilon, malformed metadata, deterministic serialization,
and nonfinite/shape/numeric comparison failures.

## Compare with separately produced native output

The parent/integrator runs the native engine separately. This tool never invokes
it:

```sh
python -m tools.muse.compare \
  --reference /tmp/muse-fixture/reference.json --actual /tmp/native.json
```

Default elementwise tolerance is `abs(actual-ref) <= 2e-5 + 2e-5*abs(ref)`.
Override with `--atol` and `--rtol` (finite, nonnegative). Token order and full
matrix shape must match; malformed, ragged, nonnumeric and nonfinite data fail.
Failures exit 1; success exits 0 and prints the maximum absolute error.

This is a synthetic numerical gate, **not** full-checkpoint validation or a
performance measurement. Only local F32 inputs with normalized sigmoid routing
are supported by the oracle loader. Quantized decoding, sharding, native
streaming/KV state, accelerator accuracy and hardware memory bounds belong to
separate integration gates. Exact byte determinism is tested in the same
Python/NumPy environment; different BLAS/NumPy platforms may round differently.
