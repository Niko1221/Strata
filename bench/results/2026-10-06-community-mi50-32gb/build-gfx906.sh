#!/bin/bash
# Build do Strata para gfx906 (MI50) - engine separada em build-906/ (nao toca em <strata-dir-old>)
set -e
cd <strata-dir>
if [ ! -f third_party/llama.cpp/ggml/CMakeLists.txt ]; then
  curl -fL -o third_party/llama.cpp.zip https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.zip
  rm -rf third_party/_unpack && mkdir third_party/_unpack
  unzip -q third_party/llama.cpp.zip -x "*/tools/ui/*" -d third_party/_unpack
  mv third_party/_unpack/llama.cpp-* third_party/llama.cpp
  rm -rf third_party/_unpack third_party/llama.cpp.zip
fi
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/compat-libxml2-noble:$LD_LIBRARY_PATH
cmake -S . -B build-906 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=<strata-dir>/third_party/llama.cpp \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
nice -n 10 ninja -C build-906 -j 8 strata
echo BUILD_DONE
