# Strata Lab: P2P-Turbo (Autonomous Dual-GPU Pipeline)

This branch (`lab/p2p-turbo`) implements hardware-synchronized GPU-to-GPU autonomous decoding for asymmetric and symmetric multi-GPU setups running MoE models (such as `Swift-Qwen3.8-Flash-Next` / `Qwen3.8-Flash-Next`).

---

## Key Innovations

1. **Direct GPU-to-GPU PCIe Flag Handoff**:
   - Primary GPU (Head/Draft) writes completion flags directly to the peer GPU's bar via PCIe without round-tripping through the host CPU.
   - Eliminates kernel launch latency and CPU worker thread wake-up stalls.

2. **Autonomous Hardware Decode Window**:
   - Speculative drafting (MTP) and expert computation run pipelined across both GPUs.
   - Asymmetric PCIe bandwidth tolerance: tested and benchmarked on PCIe Gen4 x16 (RTX 5080) paired with PCIe Gen4 x2 (RTX 4060 Ti, only ~3.2 GB/s bandwidth).

3. **Unified VRAM Expert Allocation**:
   - Reclaims display reserve (`--vram-reserve-mib 384` when using iGPU).
   - Automatically distributes 17,812 routed experts (72.5% of the model) across dual 16 GB GPUs (6,904 on GPU 0 + 10,908 on GPU 1).

---

## Build Instructions

### Windows (Visual Studio 2022/2026 + CUDA 12.8 / 13.x)

```cmd
cmake -G Ninja -S . -B build ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES="89;120" ^
  -DSTRATA_NATIVE_EXPERTS=ON ^
  -DSTRATA_BUILD_TESTS=OFF

cmake --build build --target strata -j 8
```

---

## Recommended Run Configuration

To reproduce the benchmark results on dual 16 GB GPUs (e.g., RTX 5080 + RTX 4060 Ti):

```bash
./build/strata.exe \
  --pack "<path_to_pack>/swift-iq2_xs" \
  --native "<path_to_model>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf" \
  --ple-gguf "<path_to_model>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf" \
  --expert-profile "<path_to_profile>/expert-profile.bin" \
  --expert-cache auto \
  --prefill auto \
  --spec 8 \
  --spec-min-p 0.70 \
  --mtp "<path_to_mtp>/rt" \
  --max-context 65536 \
  --kv-resident 32768 \
  --kv q4_0 \
  --resident-experts \
  --vision \
  --vram-reserve-mib 384 \
  --peer-device 1 \
  --peer-reserve-mib 128 \
  --peer-adapt-swaps 0 \
  --pool-workers 7 \
  --resident-budget-gib 10
```

### Parameter Breakdown

| Flag | Recommended Value | Purpose |
|---|---|---|
| `--peer-device` | `1` | Enables Secondary GPU as autonomous P2P peer |
| `--spec` | `8` | Speculative draft length (MTP) |
| `--spec-min-p` | `0.70` | Acceptance threshold for speculative drafts |
| `--vram-reserve-mib` | `384` | Reclaims headroom for Head GPU (when display runs on iGPU) |
| `--peer-reserve-mib` | `128` | Reclaims headroom for Secondary GPU (headless compute) |
| `--peer-adapt-swaps` | `0` | Disables dynamic swap oscillation on asymmetric PCIe lanes |
| `--pool-workers` | `7` | Thread pool sizing matching 8-core CPU (Ryzen 7 9800X3D) |
| `--resident-budget-gib`| `10` | RAM resident budget for unplaced experts |

---

## Benchmark Results (RTX 5080 + RTX 4060 Ti PCIe x2)

| Metric | Official Strata v0.2.7 | Basalt v0.5.0 | **Strata `lab/p2p-turbo`** |
|---|---|---|---|
| **Synthetic Decode** | 196.9 tok/s | ~257.0 tok/s | **286.1 tok/s** (peak 295.4) |
| **Real 12K Context Decode** | 162.2 tok/s | 161.9 tok/s | **195.5 tok/s** (peak 255.2) |
| **Prefill TTFT (12K Prompt)** | ~14 - 17 s | 14 - 17 s | **2.55 s** (717.4 tok/s) |
