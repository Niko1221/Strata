# Experimental Qwen3.6 on Intel Arc

An opt-in SYCL source build can run Qwen3.6-35B-A3B with hybrid CPU/GPU
experts. Build with `STRATA_SYCL_QWEN35=ON` and use a prepared GGUF directly.
Released packages and automatic model setup still use their existing support
matrix. The default source build continues to report `qwen4exp`.

The decoder is Strata's Qwen3.6 path ported to SYCL. For pictures, the existing
CPU `strata-vision` helper encodes a model-matched projector and sends SVE1
embeddings to the decoder. Hybrid execution keeps an expert arena in RAM and
uses a bounded GPU expert cache. Plan RAM using the [Qwen3.6 model guide](QWEN36.md).

## Build and check

From the repository root in a [oneAPI environment](INTEL.md), run the following
for a Battlemage B580 build. `STRATA_GGML_DIR` names a local checkout of the
ggml dependency if the repository does not already have it.

```sh
cmake -S engine/sycl -B build-q36-sycl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_SYCL_AOT=bmg-g21 -DSTRATA_SYCL_QWEN35=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_OWN_CPU_QUANT=OFF -DSTRATA_SYCL_PARITY=ON \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp \
  -DSTRATA_Q36_PARITY_GGUF=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
cmake --build build-q36-sycl --target strata strata-q36 \
  strata_q36_router_parity strata_q36_kernel_parity strata_q36_attention_values_parity \
  strata_q36_decode_legacy_parity strata_q36_addrms_parity strata_q36_expert_parity
build-q36-sycl/strata --families
# qwen4exp qwen35moe
ctest --test-dir build-q36-sycl -R '^strata_q36_' --output-on-failure
```

## Run a prepared model

Start with the full Unsloth `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` and an 8192-token
context. The following arguments select a small hybrid configuration for a
B580 12 GB. `--mtp` is the number of draft tokens.

```sh
build-q36-sycl/strata --native /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --max-context 8192 --prefill 128 --expert-vram 6 --expert-slots 64 \
  --cpu-threads 6 --mtp 2 --no-graph --serve
```

For the HTTP server, set `exe` to the absolute path to this `strata` binary,
`arch` to `qwen35moe`, and `args` to the model arguments, excluding `--serve`.
For vision, add `--vision` and configure the existing `vision` helper with the
same model GGUF, its 2048-dimensional `mmproj-F16.gguf`, and `gpu: false`.
The [model guide](QWEN36.md) documents the serve protocol and vision comparisons.

The first scope is single-GPU Linux hybrid execution on B580 with graph
capture off. Other Intel devices, graph capture, different quantizations and
the default CPU quantization backend need separate qualification.
