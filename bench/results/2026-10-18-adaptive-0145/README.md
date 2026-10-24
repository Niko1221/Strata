# Adaptive draft integration with Strata 0.1.45

This update merges upstream [v0.1.45](https://github.com/Niko1221/Strata/releases/tag/v0.1.45), commit `477ac13cbe8e582c20484871ce459d20d82df09d`, into public adaptive head `281220bc69315c1570b4ab824f2fc45867c76a6b`. Existing authorship and provenance are retained. The separately evaluated disk checkpoint bank is not included.

The public feature remains opt-in: acknowledged native RAM/VRAM relief, guarded idle/pressure unload, and supervised terminal-tool leases. The merge preserves adaptive preparation ownership while adding upstream prompt-cache cleanup and answer-room behavior. Native merge resolutions retain the layer-aware expert pointer publication order, cache teardown, adaptive READY fields and cooperative prefill progress alongside .45's prompt chunk planning.

One semantic compatibility fix was needed beyond resolving conflict markers. The local generated-output cap/EOS helper now recognizes .45's optional merged prompt window. Such a window consumes all remaining prompt rows and emits only the last row's prediction; treating earlier prompt predictions as generated tokens could truncate prompt consumption or stop at an earlier EOS prediction. The regression exercises prompt windows of 2–16 rows, the one-token output allowance, earlier/final EOS positions, reusable prefix state and the lookup-chain length invariant. This is a CPU arithmetic/state fixture, not a model-level equivalence result.

Four additional fake-engine tests cover adaptive ownership across .45's answer-room continuation, an explicit reasoning-budget override, answer-room opt-out, and rejected context fitting with prompt-cache cleanup. At an 8,192-token output allowance, a request without an explicit positive reasoning budget uses the upstream 6,144/2,048 reasoning/answer split; an explicit 1,024-token reasoning budget remains 1,024. Those tests do not prove that normal history compression will finish or preserve every fact.

## Independent validation of this public composition

Windows, Python 3.11.9. No model was loaded and no active service was contacted.

| Check | Result |
|---|---:|
| Server unittest discovery, including four new integration cases | 1,298 passed; 13 skipped; 1,311 total |
| External supervisor broker tests | 54 passed |
| Parallel setup tests | 3 passed |
| Browser resource-control tests | 12 passed |
| Standalone native CPU fixtures | 7 passed |

The native fixtures are `conversation_cache_test`, `conversation_memory_test`, `session_save_reclaim_test`, `message_boundary_test`, `conv_cache_test`, `serve_window_test` and `coupled_draft_test`. They were configured with CUDA, HIP, native experts and the general GPU test suite disabled. All commands exited zero. Skipped Python cases are not passes.

Reproduce the Python/browser checks:

```text
python -m unittest discover -s serve -t . -p "test_*.py"
python -m unittest discover -s integrations/hermes-resource-feedback -p "test_*.py"
python -m unittest tools.test_setup_parallel
node --test tools/test_resource_controls.cjs
```

The checked code tree before adding this results directory was `541ac19a27b48a2dfeb88c0c5f3d8d4d19d8024d`. See `results.json` for the parent/release identities and counts. Final commit identity will be supplied by the commit containing this report.

## What this does not qualify

This is a public source/CPU integration result. No CUDA engine build, GPU runtime, HIP/SYCL build, live model, image/video workflow or visible Desktop acceptance was performed on this public composition. Independently tested private compositions contain additional components and cannot be used as evidence that this public head alone passes those workflows. Earlier CPU CI and GPU/media reports retain their exact older revision scope.

Recent local work verified bounded protected compilation, completed-result delivery to a fresh agent owner without repeating the compiler, and external compiler pressure causing model unload and guarded return. Those results include the separate checkpoint bank and operational harness. They are useful lifecycle evidence, not this public merge's runtime validation. One normal-compression attempt still failed when a later summary reached its 8,192-token output allowance; it was rejected before adoption. The earlier successful generation does not make that complete attempt pass.

The PR remains draft. Finite execution workspace and measured rather than OS-reserved capacity remain limitations. This update makes no performance, arbitrary-foreground responsiveness, universal no-OOM, or unrestricted recovery claim. See the existing [provenance](../../../docs/ADAPTIVE_PROVENANCE.md) and [tool lease protocol](../../../docs/TOOL_RESOURCE_LEASES.md) for contributor credit and security/ownership boundaries.
