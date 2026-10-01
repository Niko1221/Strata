# Experimental Strix Halo source build (gfx1151)

This is a manual, opt-in HIP source-build path, not installer support. `gfx1151`
remains in CMake's unvalidated list: one machine and one short model smoke are
not broad hardware or production validation. Other integrated GPUs remain
rejected. No architecture override is needed.

## Tested environment

- AMD Ryzen AI Max / Strix Halo machine, native `gfx1151`, wave32.
- System ROCm 7.2.3, HIP 7.2.53211, Clang 22, Linux.
- Firmware exposes 48 GiB to the GPU and approximately 46.7 GiB to Linux.
- Existing local model services were temporarily paused for the successful run
  and restored afterwards. Coexistence with those workloads is NOT validated.
- Base revision: `c499bd102e7a4135c0de389dcfe38c399759ccc8` (post-v0.1.31 main).

## Build

Use an already installed, coherent system ROCm. This change adds no wheel index
or driver installation. Paths below are examples; keep build output separate.

```sh
cmake -S . -B build-hip-gfx1151 \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.3/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.3 \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON
cmake --build build-hip-gfx1151 -j2
python3 -m unittest tools.test_hip_arch_gate tools.test_setup_amd
```

The regular setup script still rejects integrated GPUs. Do not add gfx1151 to
its supported list merely to bypass that check: automatic residency, memory
fit estimates, existing-config launch and runtime installation need separate
review. HIP free-memory reporting and DRM counters differed on this host;
neither alone establishes a safe shared-memory allocation budget.

## Model preparation and smoke

The tested artifact was the two-shard GSQ-RCO Coder IQ1_M model. It has **256**
experts per layer; the shipped 512-expert profile is incompatible. Use the
actual tensor metadata to choose the PLE shard: this artifact places
`per_layer_token_embd.weight` in **shard 2**, not shard 1.

Prepare the native pack with `tools/iq_pack.py`, and fetch, verify, pack and
export the MTP using `tools/mtp_fetch.py`, `tools/mtp_pack.py` and
`tools/mtp_rt.py` as documented in [ORCA.md](ORCA.md). Point
`STRATA_GGUF_PY` at this build's pinned llama.cpp `gguf-py`, e.g.
`build-hip-gfx1151/_deps/strata_llamacpp-src/gguf-py`.

Generate a dimension-matched, uncalibrated starting profile (not a measured
routing-frequency profile):

```sh
python3 tools/make_profile.py --no-base --n-expert 256 --out coder-profile.bin
```

`TOKENS` below is a text file of comma-separated token IDs. The test used the
artifact's exported chat template with `enable_thinking=False`, then its own
`strata_tokenizer.Tokenizer` with special-token parsing enabled. The prompt
was: `What is the capital of Germany? Answer with the city name only.`
The encoded prompt contained 26 tokens and round-tripped exactly.

The tested invocation (set the paths for your local artifacts) was:

```sh
OMP_NUM_THREADS=2 build-hip-gfx1151/strata \
  --pack "$PACK" --native "$SHARD1" --ple-gguf "$SHARD2" \
  --mmap-experts --expert-cache 128 --pool-workers 2 \
  --prefill 16 --spec 2 --mtp "$MTP_RT" \
  --max-context 256 --max-new 16 --tokens-file "$TOKENS" \
  --expert-profile coder-profile.bin --stats
```

There is no `generate` positional subcommand, despite the help banner wording.
This diagnostic invocation did not use `--stop-eos`, so it continued after the
first answer through role markers. Normal CLI use should set `--stop-eos`.

## Actual results and limits

- Full HIP build completed; 13 architecture/setup Python tests passed.
- `hip_intrinsics` passed signed dot4/overflow, byte permutation, packed integer
  boundaries and wave32 checks. `strata-device --selftest` passed its 64 MiB
  arena checks. A separate small HIP pinned-host/device transfer probe passed.
- Across CTest and the IQ-fixture rerun: **51 passed, 3 failed, 2 skipped**.
  The failures require unavailable reference artifacts: `ple_parity` needs its
  compatible reference pack/block fixtures; `expert_parity` and `pool_test`
  need `pack/full/experts.bin`. These are not represented as passing tests.
  RDNA4-only prompt attention and GPU-specific hipBLASLt tuning were skipped.
  The IQ fixture rerun needed `PYTHONPATH` pointing to the pinned `gguf-py`.
- Real model invocation returned exit 0. First decoded answer: **Berlin**,
  followed by `<|im_end|>` and further role markers because EOS stopping was
  not enabled. Decode: 16 tokens in 2047.8 ms, **7.81 tokens/s**; prefill:
  25 tokens in 2350.8 ms. This is one short, cold, low-cache smoke, not a
  throughput or quality benchmark. No server/API or long-context test yet.
- Concurrent-service attempts triggered host reclaim/compaction and more than
  512 MiB additional swap, so an external watchdog stopped them. The successful
  isolated run retained that guard and an 8 GiB host-memory floor; it did not
  disable swap or alter system VM settings. The independent restoration timer
  and normal cleanup restored the paused services, whose health checks passed.

Do not use `--expert-cache auto` or automatic resident-expert sizing based on
this report. Start with explicit limits and host-memory/pressure monitoring.
This patch does not make the installer, vision, all APUs, multi-user serving,
or coexistence with existing GPU workloads supported.
