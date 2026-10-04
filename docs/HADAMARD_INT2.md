# Strata Hadamard-INT2 GGUF extension

This document specifies the experimental Hadamard-INT2 encoding written by
[`tools/convert_hadamard_int2_gguf.py`](../tools/convert_hadamard_int2_gguf.py).
It is a Strata-specific GGUF extension. Standard GGUF tools do not know its
custom type ID. This Strata fork reads the converted routed experts through its
native expert pack path.

## Encoding

The GGUF type ID is **144** (`HADAMARD_INT2`). Each block stores 128 values in
34 bytes:

| Byte range | Contents |
| --- | --- |
| 0–1 | One little-endian IEEE FP16 non-negative scale `s` |
| 2–33 | 128 two-bit codes, four codes per byte, least-significant code first |

For code `c` in `0..3`, the decoded transformed value is `s * levels[c]`, where
`levels = [-1, -1/3, +1/3, +1]`. The block size is 128 values, so the encoding
uses 2.125 bits per value, including its scale. GGUF dimension 0 must be a
multiple of 128. Blocks follow the flattened GGUF row order, with dimension 0
as the contiguous input-channel dimension.

The scale is fitted independently for each block. Starting from the largest
absolute coefficient, the converter assigns each coefficient to its nearest
codebook level `q_i`, then fits the non-negative least-squares scale

```text
s = max(0, sum_i(omega_i * w_i * q_i) / sum_i(omega_i * q_i^2))
```

It alternates assignment and scale fitting for eight iterations by default;
`omega_i=1` without an importance matrix. It then stores the scale as FP16 and
assigns codes again using the stored scale. Ties select the lower code index.

Every output shard carries this metadata schema:

| Key | GGUF value type | Required value |
| --- | --- | --- |
| `strata.had2.version` | `u32` | `1` |
| `strata.had2.block_size` | `u32` | `128` |
| `strata.had2.bits` | `u32` | `2` |
| `strata.had2.codebook` | `array<f32>` | `[-1, -1/3, +1/3, +1]` |
| `strata.had2.rotation` | `string` | `normalized_sylvester_fwht_splitmix64_input_sign` |
| `strata.had2.scale` | `string` | `nonnegative_fp16_per_128_values` |
| `strata.had2.packing` | `string` | `four_2bit_codes_per_byte_lsb_first` |
| `strata.had2.tensor_scope` | `string` | `blk.*.ffn_{gate,up,down}_exps.weight` |
| `strata.had2.seed` | `u64` | Converter-selected unsigned 64-bit seed |

The converter reads GGUF v3. It preserves existing metadata and the declared
tensor-data alignment, then adds these keys. Type ID 144 and these metadata
keys are Strata extensions; they are not registered GGUF quantization types.

## Orthogonal transform

The transform is applied to every 128-channel block of each target weight row.
For absolute input-channel index `i`, it derives a deterministic sign from
SplitMix64:

```text
z = seed + (i + 1) * 0x9E3779B97F4A7C15             (mod 2^64)
z = (z xor (z >> 30)) * 0xBF58476D1CE4E5B9          (mod 2^64)
z = (z xor (z >> 27)) * 0x94D049BB133111EB          (mod 2^64)
z = z xor (z >> 31)
sign[i] = +1 if (z & 1) is 1, otherwise -1
```

Each weight block is transformed as `H_128(D w)`, where `D` is the diagonal
matrix of those signs and `H_128` is the normalized Sylvester Hadamard matrix
(`H H^T = I`). During inference, the corresponding activation block must be
transformed as `H_128(D x)`. The weight and activation transforms use the same
seed and absolute channel indices; applying only one side is incorrect.

## Converter

The converter changes tensors matching
`blk.N.ffn_gate_exps.weight`, `blk.N.ffn_up_exps.weight`, and
`blk.N.ffn_down_exps.weight`. All other tensor payloads are copied as-is. It
accepts F16, BF16, or F32 source tensors and rejects already quantized target
tensors. Target dimension 0 must be a positive multiple of 128. Before writing
output, it requires a complete gate/up/down triplet for every layer, matching
matrix dimensions, and a consistent expert count. Target tensors have three
dimensions in GGUF order: gate and up use `[n_embd, n_ff, n_expert]`, and down
uses `[n_ff, n_embd, n_expert]`. The layer indices must start at zero and be
contiguous.

For one GGUF file:

```sh
python tools/convert_hadamard_int2_gguf.py input.gguf output.gguf --seed 0
```

For split GGUF files, pass any input shard and an output directory:

```sh
python tools/convert_hadamard_int2_gguf.py model-00001-of-00004.gguf converted/ --seed 0
```

The converter finds the other shards beside the input and writes matching
filenames in the output directory. Existing output files require
`--overwrite`. Converted weight rows are streamed in chunks; use
`--rows-per-chunk` to reduce peak memory.

Pack the converted file with Strata's normal packer. For split GGUF files, pass
the converted first shard; `iq_pack.py` reads the other shards from the same
directory and records type 144 and its seed in `native_experts.txt` v5:

```sh
python tools/iq_pack.py --gguf converted/model-00001-of-00004.gguf --out packs/hadamard-int2
```

Run the pack with the same `--pack` and `--native` options as other native
expert packs, pointing `--native` at the converted GGUF (first shard for a
split model). The engine checks the GGUF format metadata against the pack
before serving expert rows.

An optional NPZ file can provide one vector of non-negative input second
moments for each tensor, with the tensor name as its NPZ key:

```sh
python tools/convert_hadamard_int2_gguf.py input.gguf output.gguf \
  --imatrix rotated_second_moments.npz
```

Each vector has one value per input channel and must be measured in the rotated
basis produced by the same Hadamard transform and seed. Entries are optional;
target tensors without an entry use uniform weights. This diagonal weighting
is an approximation to activation error, not a guarantee of better model
quality.

Each output gets GGUF metadata under `strata.had2.*`, including the format
version, block size, codebook, transform description, tensor scope, and seed.
A JSON report records each converted tensor's weight MSE, normalized MSE, and
importance-weighted MSE. Those values measure weight reconstruction only; they
do not measure perplexity, task accuracy, or generated-text quality.

## Runtime support and limits

This fork recognizes type 144 in its GGUF reader and native expert pack. CUDA
decode uses the Hadamard-INT2 grouped kernels; the prompt path decodes the
weights to FP16 and applies the same transform to activations. The CPU native
expert path uses a scalar reference dot product. The supported model geometry
is this build's `n_embd=2560`, `n_ff=640` geometry, and all routed gate, up,
and down expert tensors in the pack must use Hadamard-INT2 together.

The converter's report measures weight reconstruction error only. No
end-to-end perplexity, task-quality, or speed results are claimed for this
format. Standard GGUF tools and other llama.cpp builds cannot load the custom
type without corresponding type-144 support.
