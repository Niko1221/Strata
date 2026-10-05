# Measured source build and reproduction

The measurements use Strata 0.1.39-sycl at
`6f32ec070f23ced9f50e704d854d775da52591ab`, ggml/llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, and Intel oneAPI DPC++ 2026.1.0.
This is a SPIR-V/JIT Release build, not AOT. Compiler options include `-O3`,
`-DNDEBUG`, `-std=c++20`, `-fsycl`, subgroup 32, per-kernel device code split,
`-fp-model=precise`, sequential MKL, and correctly rounded FP32 divide/sqrt.
The measured binary SHA256 is
`19efe17ce46adcebb99b586e9d12cde4fe9b0ddbdf0f585ebed173ef6c992ba0`.
Paths and build environment can affect a rebuilt binary's hash.

Six local patches are attached as reproduction data, with no modifications to
the repository's engine in this results PR:

1. Synchronize the SYCL host thread-affinity interface.
2. Synchronize NativeDense's layer-range load interface.
3. Free test-only SYCL-pinned allocations with the matching allocator.
4. Avoid the commit graph in the eager path (eager is **off** in this report).
5. Refuse NO_HOST startup if any missing expert lacks the pinned mirror.
6. Drain queues and explicitly release the pinned USM expert mirror before
   native QUIT exits. The final linked binary recompiles this shutdown-owning
   translation unit against the unchanged original static objects.

All six patches were applied to a clean copy of the pinned source. The resulting
SYCL source hashes match the measured sources, including final generate.cpp
`5d9b07450349a481848533b4fbc2ec9e79f5d4854d6ad379cc3a3de9521cd998`.
The following full rebuild is a reproduction recipe; this benchmark campaign
reused the previously qualified binary and did not rebuild or tune the host.

With compatible oneAPI/compiler/MKL already installed, in a separate checkout:

```sh
git checkout 6f32ec070f23ced9f50e704d854d775da52591ab
export REPORT=/absolute/path/to/this/report
export STRATA_ROOT="$PWD"
git apply "$REPORT"/patches/*.patch
git clone https://github.com/ggml-org/llama.cpp.git /absolute/path/to/llama.cpp
git -C /absolute/path/to/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
source /opt/intel/oneapi/setvars.sh
python3 -m venv .venv-b65
.venv-b65/bin/pip install -r requirements.txt
export PATH="$STRATA_ROOT/.venv-b65/bin:$PATH"
cmake -S sycl -B build-sycl -G Ninja \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_GGML_DIR=/absolute/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-sycl -j4
```

Model preparation uses the original full 512-expert IQ2_XS, **not** Coder:

- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at
  `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, both files under `IQ2_XS/`.
- Draft tensors: `Qwen/Qwen3.8-Flash-Next` at
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`, upstream 31-tensor hash
  verification, Q2_0 expert packing. `prepare-mtp.py` forbids fallback to main.
- `tools/iq_pack.py` creates the dense/native-expert pack and exports the
  model's tokenizer/template. Full fixed `data/expert-profile.bin` is retained;
  no custom expert ranking or draft vocabulary is used.
- See `artifacts.json` for filenames, byte sizes and recorded hashes. Large
  weights/packs were previously hashed and root-sealed; their size/mtime/inode
  and ownership were rechecked before this campaign. The measured binary,
  profile and launch environment were freshly SHA256-checked.

Place the pinned GGUF shards under `$BENCH_ASSETS/models/IQ2_XS/`, then:

```sh
export BENCH_ASSETS=/absolute/path/to/benchmark-assets
python tools/iq_pack.py \
  --gguf "$BENCH_ASSETS/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf" \
  --out "$BENCH_ASSETS/pack"
python "$REPORT/prepare-mtp.py" --source "$STRATA_ROOT" --assets "$BENCH_ASSETS"
```

Run under your own bounded exclusive-GPU supervisor, with adequate pinned-memory
limits and a RAM-backed directory. The measured machine had swap disabled and
crash capture suspended during generation; independent cleanup restored normal
services and crash capture afterward. This script does not configure those
machine policies or stop other services for you.

```sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_CACHE_PERSISTENT=0
export SYCL_PROGRAM_COMPILE_OPTIONS=-cl-fp32-correctly-rounded-divide-sqrt
export STRATA_MIRROR_MIB=16384 STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1
export STRATA_WARM_GRAPHS=0 STRATA_DBG_NAN=1
export STRATA_STAGER_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset STRATA_VERIFY_EAGER STRATA_DECODE_TIMING STRATA_PLE_TRACE
sudo install -d -m700 -o "$(id -un)" /run/strata-community
python "$REPORT/benchmark.py" --source "$STRATA_ROOT" \
  --profile "$REPORT/profile.json" --ram /run/strata-community \
  --out /absolute/path/to/new-results --runs 3
```

Native logs are RAM-backed and should be deleted only after clean native exit and independent
delayed-fault checks. The script stores numeric/hash receipts, not output text.
To regenerate summary/CSV/allowlisted timing logs:

```sh
python "$REPORT/summarize.py" /absolute/path/to/new-results /absolute/path/to/summary
```
