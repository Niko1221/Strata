# Experimental DFlash2 runtime

This branch extends the standalone DFlash drafter from PR #1426 at `22da2204`,
rebased onto Niko main at `296b59d2`.
The DFlash2 commit contains only the extension; the prerequisite remains in its separate PR.
DFlash2 is disabled at build time by default.
There is no trained Flash-Next DFlash2 checkpoint in this change. Synthetic tests check the operations and
artifact contract; they do not establish draft acceptance, end-to-end output equality or a speed improvement.

## Implementation reference

The implementation was checked against:

- [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/a9d27ac693ff0a1d3cba449c912c2b891904a4c7),
  especially `src/models/dflash.cpp` and `common/speculative.cpp`.
- [Z Lab's MIT-licensed reference](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py).
- The [llama.cpp DFlash2 change](https://github.com/ggml-org/llama.cpp/pull/27342).

Strata's existing `feat/dflash` implementation at `22da2204` supplies feature capture, private draft KV,
parallel attention, the shared target embedding/head and target verification. Its DeepSpec artifact layout
remains supported. DFlash2 adds the operations below; weights from a plain DFlash checkpoint cannot supply them.

Each attention and MLP sublayer projects its normalized input to two sets of dynamic coefficients.
The first set convolves that input; the second convolves the sublayer output before its residual addition.
For a channel `c` in group `g`, at block position `i`:

```text
out[i,c] = sum_t (base[side,t,c] + dynamic[i,side,t,g]) * input[i-t,c]
```

Positions before the block start contribute zero. No convolution state persists across proposal blocks.
The output coefficients come from the original normalized sublayer input, not from its output.

The selector retains the top-K unary head candidates at each noise position. It starts from the verified
anchor token, then greedily chooses each successor using:

```text
score(candidate) = unary[candidate]
                 + dot(predecessor_codebook[previous] * hidden_projection(hidden),
                       successor_codebook[candidate])
```

The chosen token becomes the next position's predecessor. This is the same path as the greedy CPU walk over
llama.cpp's selector lattice. Strata computes the selected edges directly on the GPU and copies the final IDs
back. The initial top-K scan and path walk are serial correctness implementations; no speed claim is made.
Equal unary scores prefer smaller token IDs. Near-tied logits may differ from another backend through rounding.

DFlash2 noise rows are `[anchor, mask, ...]`. Row zero is not emitted: a trained block of eight rows produces
at most seven proposals. Plain DeepSpec DFlash still emits its row-zero prediction. Returned gate probabilities
are conditional softmax confidences over selector candidates. The verifier treats proposals as point masses,
including on sampled requests; this change does not implement sampling from the DFlash2 proposal distribution.
Only the target-confirmed prefix is committed by Strata's existing verifier.

## Build and use

```sh
cmake -S . -B build-dflash2 -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_ENABLE_DFLASH2=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-dflash2 --target strata
```

Choose the CUDA architecture for your GPU. Once a compatible trained checkpoint exists:

```sh
python tools/dflash_gguf.py /path/to/model.safetensors \
  --config /path/to/config.json -o /path/to/dflash2.gguf
build-dflash2/strata generate --pack /path/to/target-pack --native /path/to/model-shard1.gguf \
  --dflash /path/to/dflash2.gguf --spec 8 --dflash-block 7 \
  --spec-min-p 0 --tokens-file /path/to/prompt-token-ids.txt --max-new 128
```

Use the engine's existing generation arguments for your target pack. There is no automatic DFlash2 download
or installer choice. The installer and calibration are unchanged from the DFlash prerequisite branch.

The converter recognizes `architectures: ["DFlash2DraftModel"]` and reads the architecture fields from
`dflash_config`. It accepts BF16 safetensors, including both published selector-codebook naming forms.
The GGUF architecture remains `dflash`, as in llama.cpp. Required extra keys are:

| Key | Meaning / supported values |
| --- | --- |
| `dflash.conv_kernel_size` | Causal taps, 1–8 |
| `dflash.conv_group_size` | Positive channel group size dividing the hidden width |
| `dflash.selector_rank` | Transition rank, 1–1024 |
| `dflash.selector_top_k` | Candidates per position, 1–64, at most the vocabulary size |
| `dflash.block_size` | Noise rows including anchor, 2–8 |
| `dflash.sample_from_anchor` | Must be false |
| `dflash.target_layer_inputs` | Optional Strata marker; absent means llama.cpp layer-input indices |

HF `target_layer_ids` identify layer outputs. The Strata converter retains those IDs and writes
`dflash.target_layer_inputs=false`; generation advances them to the next attention-input boundary.
A llama.cpp GGUF already contains input-boundary IDs, which generation uses directly. Taps are validated
against the target. This interpretation still needs validation against a trained Flash-Next DFlash2 checkpoint,
including Flash-Next's HC contraction.

Extension tensors use llama.cpp names: `blk.N.{attn,ffn}_conv_base`,
`blk.N.{attn,ffn}_conv_proj.weight`, `selector_predecessor.weight`, `selector_successor.weight`,
and `selector_hidden.weight`. Raw HF aliases are also accepted. Base kernels retain all three dimensions
`[hidden, kernel, 2]` in GGUF order. Extension tensors must remain BF16; `tools/dflash_quantize.py` preserves
their bits while allowing Q8_0/Q5_0/Q4_0 backbone matrices.

The runtime retains the existing Flash-Next fast-path geometry: hidden 2560, five layers, five feature taps,
24 Q heads / two KV heads, head dimension 256 and MLP width 7680. It requires full non-causal attention,
SiLU, RMS epsilon 1e-6 and the full shared target head. It rejects private/pruned heads, DSpark extras,
sliding windows and non-default embedding, attention or logit scaling. A dense Qwen3.8-27B checkpoint is not
a Flash-Next checkpoint and must not be substituted. Single-GPU serial requests only; draft-aware conversation
snapshots are not implemented. SYCL builds reject DFlash2 explicitly. HIP compilation and execution remain
unvalidated in this environment.

## Verification

The synthetic checks require no model download:

```sh
cmake -S . -B build-dflash2 -DSTRATA_BUILD_TESTS=ON
cmake --build build-dflash2 --target dflash2_parity dflash_load_test
ctest --test-dir build-dflash2 -R 'dflash2_parity|dflash_load_test' --output-on-failure
python tools/test_dflash2_gguf.py
```

The Python export/quantization checks need NumPy and llama.cpp's `gguf-py` dependency, like the existing
DFlash quantization tools. Tests cover both convolution sides, group sizes, padding and block isolation;
selector top-K ties, predecessor-dependent reranking, confidence and invalid-score propagation; artifact
shapes, missing weights, transactional reopen and unsupported metadata; and bit-exact BF16 export and
extension preservation during quantization. The small loader fixtures test parsing, not the fixed runtime
geometry. The flag-off loader test confirms that DFlash2 is refused unless compiled in.

Before reporting usable Flash-Next support, a compatible trained checkpoint must pass intermediate-tensor
comparison against its reference, feature alignment across acceptance and rejection, and target-only versus
DFlash2 generation checks on identical prefixes. Performance needs a same-day baseline on the same hardware.
No such generation or performance results are claimed here.

### Checks run on 2026-10-11

The extension is tested on top of the DFlash prerequisite at `dcd07e9b` (merged with main at `7695859a`).
Linux, RTX 4070 Ti SUPER 16 GB, CUDA toolkit 13.4, GCC 14.2, CUDA architecture 89:

- CUDA engine build with `STRATA_ENABLE_DFLASH2=ON`: passed after extracting the extension.
- Eight CTest checks (DFlash2 parity, DFlash loader/policy/taps/RoPE/owned KV, batched MMVQ and BF16 projection): passed after extracting the extension.
- Converter/quantization group: six Python tests passed, including three new DFlash2 tests.
- HIP compilation and execution remain unvalidated; no ROCm installation is available.
- DFlash2 + SYCL is explicitly rejected by configuration.

The earlier combined branch on Niko's main at `61b3fb5d` passed its CUDA build, eight relevant CTests,
292 server tests and 17 calibration tests. Those server and calibration results were obtained on the
combined main-based tree, not on this extracted extension. The installer and calibration are not changed here.

No trained-drafter generation comparison or end-to-end byte equality check was run. The synthetic checks
and inspection of the inactive feature branches do not replace that check.

### Rebase checks on 2026-10-15

Rebased the extension onto DFlash `22da2204`, which includes Niko main `296b59d2`
and the optional original BF16 draft head. The DFlash2 selector still requires the
full shared target head; the original-head option remains available for plain DFlash.
Linux, RTX 4070 Ti SUPER 16 GB, CUDA 13.4, GCC 14.2, architecture 89:

- CUDA engine build with DFlash2 enabled: passed.
- Nine CTests (DFlash2 parity, DFlash loader/policy/taps/RoPE/owned KV, BF16 head,
  batched MMVQ and BF16 projection): passed.
- Loader with DFlash2 disabled: passed, including refusal of DFlash2 artifacts.
- DFlash converter, quantization and original-head tools: eight Python tests passed.
- Setup suite: 597 tests, passed with one skipped.
- Plain DFlash with the original BF16 head on an IQ3_XXS target: the 16 generated
  token IDs matched the same-day target-only run (prompt IDs `1,2,3`, greedy,
  context 1024, expert cache 1000). This checks the inherited plain-DFlash path,
  not generation with a trained DFlash2 checkpoint.

HIP remains unvalidated because ROCm is unavailable. The SYCL compiler is also
unavailable; DFlash2 continues to reject SYCL at configuration time. No trained
DFlash2 generation or performance comparison was run.
