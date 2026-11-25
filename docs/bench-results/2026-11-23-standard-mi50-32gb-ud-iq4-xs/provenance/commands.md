# Exact commands for this report

Paths are relative to the lab root (`<lab>`) and the Strata source checkout (`<lab>/src`). Python is
`<lab>/.venv/bin/python` (packages listed in `system.txt`). The engine is the source build in
`provenance/BUILD.json`, binary `4767692562180e5d865bc1f5311fcdabffccbb72b0d7f45841f61a96ce0e20de`.

## 1. Model

Downloaded from `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
`f5d175de2cf7f720d45d6064019a81fcf056e986`, folder `UD-IQ4_XS/`, four shards with the SHA-256 in
`results.json` / `hashes.txt`.

## 2. Pack

```
cd <lab>/src
<lab>/.venv/bin/python terminal/tools/iq_pack.py \
  --gguf  <lab>/models/Qwen3.8-Flash-Next-GGUF__UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
  --out   <lab>/packs/ud_iq4_xs --compat-bf16
```
The tool follows the split into shards 2-4 itself; `native_experts.txt` is v4 (per-role shard column,
layer 35 has gate/up/down in different shards). Hashes: `hashes.txt`.

## 3. MTP draft layer

```
cd <lab>/src
<lab>/.venv/bin/python terminal/tools/mtp_fetch.py fetch  --out <lab>/mtp   # pinned de4b8e4d..., per-tensor SHA-256
<lab>/.venv/bin/python terminal/tools/mtp_fetch.py verify --out <lab>/mtp
<lab>/.venv/bin/python terminal/tools/mtp_pack.py --src <lab>/mtp --experts q2_0 --out <lab>/mtp/mtp-q2_0.gguf
<lab>/.venv/bin/python terminal/tools/mtp_rt.py  --gguf <lab>/mtp/mtp-q2_0.gguf --out <lab>/mtp/rt
cp terminal/data/draft_vocab_latin.bin <lab>/mtp/rt/draft_vocab.bin
```
Manifest: `mtp-manifest.json` (dtype, shape, source shard, byte range, SHA-256 per tensor).

## 4. Engine build

Needs `/opt/rocm` 7.2.4, clang 22, CMake, a C++20 host compiler. Two host-side workarounds on this
distribution (GCC 16 headers + HIP `__noinline__`, and a CUDA-only subproject):

```
cat > <lab>/hip_host_fix.h <<'EOF'
#if defined(__has_include)
#  if __has_include(<hip/hip_runtime.h>)
#    include <hip/hip_runtime.h>
#    ifdef __noinline__
#      undef __noinline__
#    endif
#  endif
#endif
EOF

cd <lab>/src
cmake -S engine -B <lab>/build-906 -G "Unix Makefiles" \
  -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DSTRATA_BUILD_INKLING2=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-include <lab>/hip_host_fix.h"
cmake --build <lab>/build-906 -j8 --target strata
```

## 5. Power cap and server

```
lact cli power-limit set 130          # hwmon power1_cap = 130000000 uW; LACT config in lact-config.yaml

cd <lab>/src
HIP_VISIBLE_DEVICES=1 setsid nohup <lab>/.venv/bin/python terminal/serve/server.py \
  --engine strata --config <lab>/configs/q4final.json --port 8090 --offline --no-api-key
```
`HIP_VISIBLE_DEVICES=1` makes the MI50 device 0 (the RX 5700 XT is device 0 otherwise). Config argv is
`provenance/run-config.json`; the started argv is in `engine-log-excerpt.txt`.

## 6. Standard benchmark (this report)

```
strata bench run --url http://127.0.0.1:8090 \
  --model-revision f5d175de2cf7f720d45d6064019a81fcf056e986 \
  --source-revision c6f6427d9de9016fd92d13e29740f65625261df7 \
  --quantization UD-IQ4_XS \
  --build-options "STRATA_HIP_GFX906=ON,STRATA_BUILD_INKLING2=OFF,ROCm 7.2.4,clang 22,GCC16 header workaround" \
  --background "MI50 32GB at 130W via LACT; RX 5700 XT drives the display; no other GPU workload; desktop session running" \
  --out <lab>/bench-results/2026-11-23-standard-mi50-32gb-ud-iq4-xs
```
It starts with a freshly loaded engine; do not run a second measured pass against the same engine.

## 7. Supplementary long-context numbers (not protocol)

```
cd <lab>
<lab>/.venv/bin/python \
  src/docs/bench-results/2026-10-04-community-mi50/benchmark_mi50.py \
  --root <lab>/src/terminal --pack <lab>/packs/ud_iq4_xs --url http://127.0.0.1:8090 \
  --targets 4096,32768,120000 --runs 1 --out <lab>/runs/<arm>/proto

<lab>/.venv/bin/python src/terminal/tools/needle_bench.py --url http://127.0.0.1:8090 \
  --lengths 128k --depths 10,50,90 --out <lab>/runs/<arm>/needle128k.json
```
The 256K numbers use the same config with `--max-context 262144` (server fit reduces
`--kv-resident` to 4,096 cells); the 240K request is the same script with `--targets 240000`.
