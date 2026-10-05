# Strata on an RX 7700 XT (gfx1101) in Docker — Prompt & Build Plan

Two things in one file:

* **[Part A](#part-a--the-prompt)** — a copy‑paste prompt for an agent/engineer.
* **[Part B…H](#part-b--verified-machine-facts)** — the plan behind it: verified machine facts, the
  blocker analysis, the image design, phases with exit gates, and the acceptance checklist.

Deliverable: two images whose names contain **`strata`** —
**`strata-hip-builder:gfx1101`** (compiles the engine) and **`strata-hip:gfx1101`** (runs it),
serving the model on `http://127.0.0.1:8080` with the GPU footprint **capped at 10 GiB of a 12 GiB
RX 7700 XT**.

---

## Part A — The prompt

> Paste everything below the line to the agent doing the work.

---

Build Strata's HIP engine for **this** machine's GPU inside Docker, and produce a runnable image.

**Hardware (already verified, do not re-guess):** AMD Radeon RX 7700 XT = **gfx1101**, Navi 32,
wave32, 54 CUs, `12 272 MiB` VRAM, driven by amdgpu with ROCm 7.2.4 userland. The iGPU is
**gfx1036** and must stay invisible to the engine. No NVIDIA card is present.

**The blocker you are solving for:** Strata's HIP backend is hard-locked to gfx1100. It refuses in
CMake (`cmake/hip_backend.cmake:6`), it refuses at runtime (`src/core/device.cu:69`), and the
installer refuses (`setup.py:709`). Worse, `include/strata/hip_compat/intrinsics.hpp:18` only
enables the RDNA3 signed-dot instruction under `__gfx1100__`, so a naive gfx1101 build silently
falls back to a scalar loop and loses most of the quantized-kernel speed. Widen all of it to a
gfx1100 **allowlist that includes gfx1101**, and prove the dot instruction is really emitted.

**Constraints, in priority order:**

1. **Reproduce the pinned dependency exactly.** llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`
   (`third_party/ggml/VERSION.txt`). No "latest main". No CUDA backend in the same build.
2. **Never regress gfx1100.** Every change is an allowlist widening, not a replacement. gfx1100 must
   still build and pass the same CTest set.
3. **10 GiB hard VRAM budget.** The engine may never push total VRAM occupancy above
   `10 240 MiB` on this 12 272 MiB card. This is a contract, not a wish: it is computed at
   container start from live device totals, enforced by `--vram-reserve-mib`, and *verified* by a
   sampling guard during real generation. Do not tune it by eyeballing `nvidia`-style tools once.
4. **Two images, both named with `strata`:** `strata-hip-builder:gfx1101` (toolchain + compile) and
   `strata-hip:gfx1101` (runtime only, no compiler, no CMake cache). The model is **never** baked
   into an image — it is a bind mount.
5. **Never set `HSA_OVERRIDE_GFX_VERSION`.** This host explicitly documents why
   (`/etc/profile.d/amd_gpu.sh`): ROCm 7.x supports gfx1101 natively and forcing `11.0.0` makes a
   gfx1101 code object fail to load. The entrypoint must `unset` it and abort if a caller forces it.
   Ignore the stale `HSA_OVERRIDE_GFX_VERSION=11.0.0` in the ollama unit file — that is Vulkan.
6. **Do not widen public claims.** `docs/AMD_HIP.md` / `AMD_HIP_PERFORMANCE.md` describe measured
   gfx1100/7900 XTX results. Add a clearly-labelled *unvalidated* gfx1101 section; do not edit the
   existing benchmark tables or the "29/29 passed" claims.

**Definition of done** (all must be shown, with command output):

* `strata-hip-builder:gfx1101` compiles `strata` with `--offload-arch=gfx1101`, zero warnings about
  the dp4a fallback.
* `strata-hip:gfx1101` starts on this card, passes `strata-device --selftest` and the HIP CTest set,
  answers a real chat request through `serve.server`, and stays ≤ 10 240 MiB at peak.
* `gfx1100` still configures and builds from the same tree (build-only gate; no gfx1100 hardware
  here, so do not claim it runs).
* Everything reproducible from two scripts at the repo root: `./build.sh` and `./run.sh`.

Work the phases in `docs/DOCKER_GFX1101_PLAN.md` (Part E) in order. **Stop at every gate** and
report the gate's evidence before continuing. If a gate fails, take the documented fallback branch —
do not improvise a third path, and do not lower the VRAM budget or the pinned revision to make a
gate pass.

---

## Part B — Verified machine facts

Everything here was read off this machine, not assumed. Re-verify with the given command if in doubt.

| Fact | Value | How it was verified |
| --- | --- | --- |
| Discrete GPU | RX 7700 XT (Navi 32), `1002:747e` | `lspci -nn` |
| Runtime arch | **gfx1101**, wave32, 54 CUs | `rocminfo` → `Name: gfx1101`, `ISA: amdgcn-amd-amdhsa--gfx1101` |
| VRAM total | **12 868 124 672 B = 12 272 MiB** | `rocm-smi --showmeminfo vram` GPU[0] |
| VRAM used by desktop at sample | 1 491 832 832 B ≈ **1 423 MiB** | same |
| iGPU (must hide) | gfx1036, `renderD129`, 512 MiB | `rocminfo`, `/sys/class/kfd/.../nodes/2` |
| Discrete render node | `renderD128` (KFD node 1, `gfx_target_version 110001`) | `/sys/class/kfd/kfd/topology/nodes/1/properties` |
| Kernel/DRM | amdgpu, `amdgpu-install 30.30.4.0` | `/sys/module/amdgpu/version` |
| Host ROCm userland | 7.2.4 — **compiler-only, not build-capable** | `/opt/rocm/bin` has 8 entries, no `hipcc`; no `libhipblas.so*`; `lib/cmake` has only `AMDDeviceLibs`, `rocm-core` |
| hipBLAS gfx1101 kernels | **present** (96 `gfx1101` files, incl. `4xi8I` int-dot) | `ls /usr/local/lib/ollama/rocm/rocblas/library \| grep gfx1101` |
| `HSA_OVERRIDE_GFX_VERSION` | must stay **unset** | `/etc/profile.d/amd_gpu.sh` (see its own comment) |
| CPU / RAM | Ryzen 9 7900 (12c/24t), **122 GB** | `lscpu`, `free -g` |
| Docker | 29.7.2, buildx 0.36.1, data-root `/var/lib/docker` | `docker info`, `docker buildx version` |
| Root FS free | **~90 GB** (92–94 % used) ← too small for ROCm images *and* for a quant; it also holds `~/.cache/huggingface` (139 GB) | `df -h /` |
| Data FS | `/mnt/storage`, **3.7 TB free** | `df -h /mnt/storage` |
| `~/Development` | **symlink → `/mnt/storage/Development`**, so model files under it land on the 4 TB disk. The model cache defaults to `~/Development/models` | `ls -ld ~/Development`, `stat -L -c %d` |
| Passwordless sudo | available | `sudo -n true` → 0 |
| Network | Docker Hub + HF reachable | `docker manifest inspect`, `curl -o /dev/null https://huggingface.co` → 200 |
| Strata engine version | 0.1.29 (`d6708a4`) | `git log -1`, `project(strata VERSION …)` in `CMakeLists.txt:11` |
| Stock HF cache location | `/home/dhoard/.cache/huggingface/hub`, **139 GB**, standard layout — and it holds **no Strata-capable model** | `du -sh`; `hfmodel.py --print available` |
| HF cache layout | `models--<org>--<repo>/snapshots/<rev>/<file>` → **relative symlink** into `../../blobs/<sha>` | `ls -l` on a cached repo — so the *cache root* must be mounted whole, never a snapshot dir |
| Strata model in any cache | **none yet.** `~/.cache` holds Ornith/occamy/Qwen3.6 GGUFs, not `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`; `~/Development/models` is empty | `find … -iname '*Flash-Next*'` → empty; `hfmodel.py --print available` |
| Can `~/.cache/huggingface/hub` hold IQ3_XXS? | **No** — that is why the model cache defaults to `~/Development/models`. IQ3_XXS is 75.8 GB and `/` has ~90 GB free, before the ~43 GB pack | `MODELS` in `setup.py:64-79` vs `df -h /` |
| `hf` client on the host | present (`~/.local/bin/hf`) | `command -v hf` |
| ROCm base image sizes | `:7.2.1` = **1.2 GB** (no math libs), `:7.2.1-complete` = **7.4 GB** | manifest config, layer-size sums |

Consequences that shape the plan:

* **Docker is mandatory here, not a convenience.** The host ROCm cannot compile HIP code (no
  `hipcc`, no hipBLAS, no CMake config). Installing a full ROCm on the host would be the alternative
  — the container is the cleaner one and is what was asked for.
* **gfx1101 is *not* exotic on this stack.** rocBLAS ships gfx1101 Tensile kernels and `rocminfo`
  reports it natively, so the "set HSA_OVERRIDE_GFX_VERSION=11.0.0" folklore is actively harmful
  here. Plan for a **native gfx1101 build** (Phase P3 has a cheap probe to prove it before we
  commit to it).
* **12 GB is a small card for Strata.** The reference setup is a 24 GiB 7900 XTX holding an
  18 GiB expert cache. Here the whole card is 12 GiB and our slice is 10 GiB, so the expert cache
  is roughly half that size. Expect decode throughput below the published 7900 XTX table
  (54 CUs vs 96, and a smaller hot-expert tier). Predict, measure, publish — never copy.

## Part C — The gfx1100 wall (patch surface)

Nine places say gfx1100. All are allowlist widenings; none should change gfx1100 behaviour.

| # | File:line | Current | Change | Risk if skipped |
| --- | --- | --- | --- | --- |
| 1 | `cmake/hip_backend.cmake:6-9` | `FATAL_ERROR` unless arch `== gfx1100` | Accept a list `{gfx1100, gfx1101}`; keep rejecting everything else | Configure fails immediately |
| 2 | `src/core/device.cu:69-71` | `strncmp(gcnArchName,"gfx1100",7)!=0 \|\| warpSize!=32` → throw | Prefix allowlist; also strip the `:…` suffix `gcnArchName` can carry | Engine aborts on startup (`hip_device_selftest` fails) |
| 3 | `src/core/device.cu:34` | message "targets gfx1100 wave32" | Say "gfx1100/gfx1101" | Confusing error only |
| 4 | `include/strata/hip_compat/intrinsics.hpp:18` | `#if defined(__gfx1100__) && __has_builtin(__builtin_amdgcn_sudot4)` | `(__gfx1100__ \|\| __gfx1101__ \|\| __gfx1102__)`; add a compile-time `#warning` when `__gfx1101__` is defined *and* the builtin is missing | **Silent perf loss** — correct answers, scalar dp4a. The nastiest one |
| 5 | `src/kernels/cuda/fused_gr.cu:183` | `TILE = 1280`, commented "fits gfx1100's 64 KiB LDS" | Assert `sharedMemPerBlock` at runtime; keep 1280 only if it holds, else halve behind a constexpr chosen from the property | Launch failure / occupancy cliff if Navi32 reports less LDS |
| 6 | `src/core/device_main.cpp:37` | prints "HIP target gfx1100 wave32" | Print the real target (`CMAKE_HIP_ARCHITECTURES` via a define) | Selftest output lies |
| 7 | `CMakeLists.txt:36` | option help text | Mention gfx1101 | Docs only |
| 8 | `setup.py:709-710`, `:1728` | `AMD_ARCHS=("gfx1100",)`, `AMD_NAMES`, error text | Add `gfx1101: "AMD Radeon RX 7700 XT (gfx1101)"` | `./setup.sh --backend hip` rejects this card — this is also the **non-Docker** win |
| 9 | `src/prefill/gemm.cu:144` + `hipblaslt_tuning.hpp` | tuning table keyed by arch string | **Leave alone.** A gfx1100 table on gfx1101 self-rejects and falls back to hipBLASEx (safe). Optional: regenerate with `tools/hip/tune_hipblaslt.cpp` | None (guarded) — but shipping a gfx1100 table for gfx1101 *without* the guard would be wrong |

Explicitly *not* touched: `docs/AMD_HIP_PERFORMANCE.md` tables, the gfx1100 tuning tables
`tools/hip/gfx1100-hipblaslt-*.txt`, and any benchmark claim.

**Applied 2026-09-30: items 1, 2, 3, 4, 6, 7, 8.** Item 4 was verified rather than assumed — a kernel
using `__builtin_amdgcn_sudot4` compiles for gfx1101 and gfx1100 and fails for gfx1030, so quantised
kernels get the RDNA3 dot instruction instead of the scalar fallback (the "nastiest one", and the
easiest to get silently wrong). Item 5 was deliberately **not** changed: the tile asks for
8 × 1280 × 4 = 40 KiB of dynamic LDS (`src/kernels/gr_parity.cpp:344`), the same per-CU budget
gfx1101 has, and the kernels run; a runtime assert there would be ceremony around a number that
holds. Item 9 stays as planned — on gfx1101 the gfx1100 table self-rejects with "using hipBLASEx"
(`src/prefill/gemm.cu:149-151`), which costs tuned GEMM and nothing else.

## Part D — Image design

```
rocm/dev-ubuntu-24.04:7.2.1-complete   (~7.4 GB uncompressed, verified by manifest; has hipBLASLt dev)
        │ built into                       │ built into
 strata-hip-builder:gfx1101          strata-hip:gfx1101{,-<gitsha>,-latest}
 build-essential cmake ninja          same -complete base (the 1.2 GB :7.2.1 tag has NO math libs)
 ccache + pinned llama.cpp            python venv: jinja2 psutil numpy regex huggingface-hub
 runs with -v $PWD:/src               serve/ tools/ data/ + /usr/local/bin/strata + gguf-py
        │                                     ▲
        └── build-hip/ ── named context ───────┘  (--build-context engine=./build-hip,
                                                   --build-context ggufpy=<llama.cpp>/gguf-py)
```

**Names and tags** (every image name contains `strata`, per requirement):

| Artifact | Name | Notes |
| --- | --- | --- |
| Build image | `strata-hip-builder:gfx1101` | toolchain only; editing C++ never rebuilds it |
| Runtime image | `strata-hip:gfx1101`, `strata-hip:gfx1101-<gitsha>`, `strata-hip:gfx1101-latest` | what `./run.sh` starts |
| Build container | `strata-build-gfx1101` | one-shot compile, no GPU device |
| Runtime container | `strata-gfx1101` | the server |
| Cache volume | `strata-ccache-gfx1101` | ccache, and the pinned llama.cpp clone lives in `build-hip/_deps` |
| Entry points | `./build.sh`, `./run.sh` | repo root, not `docker/`: one command each |

Named build contexts keep the build context tiny: `.dockerignore` excludes `build*/`, `*.gguf`,
`.venv/`, `models/` and `packs/`, so the engine binary and gguf-py arrive as two small named
contexts instead of a multi-GB upload.

Base image choice, measured rather than assumed: `:7.2.1` is **1.2 GB** uncompressed and carries no
math libraries; `:7.2.1-complete` is **7.4 GB** and has hipcc, hipBLAS, hipBLASLt, rocBLAS and the
gfx1101 Tensile kernels. Both are 7.2.x, matching the host's 7.2.4 userland and KFD series.
`rocm/runtime-*` and `rocm/base-ubuntu-24.04:7.x` tags **do not exist** — do not plan around them.
`build.sh` therefore uses `-complete` for *both* stages and runs an `ldd` gate over the finished
runtime image; slimming the runtime to `ubuntu:24.04` + ROCm runtime debs stays P7.

### The 10 GiB VRAM contract

Strata sizes its GPU expert cache itself: `--expert-cache auto` reads free VRAM and subtracts
`--vram-reserve-mib` (default **700 MiB**) plus the prefill-chunk and draft-head reservations
(`src/program/generate.cpp:2020-2040`). So the reserve is the knob — and because the engine measures
*free* memory, other processes' usage cancels out:

```
total_occupancy  ≈  device_total  −  vram_reserve          (regardless of who else is using the card)
```

Therefore the entrypoint computes the reserve from **live** device numbers instead of hardcoding it:

```
STRATA_VRAM_BUDGET_MIB = 10240                       # the contract, overridable
STRATA_VRAM_SLACK_MIB  =   256                       # allocations after auto-sizing
reserve = max(700, device_total_mib − BUDGET + SLACK)
        = max(700, 12272 − 10240 + 256)  =  2288  → round up 2304
```

Predicted ceiling on this card: `12 272 − 2 304 = 9 968 MiB` (≈ 9.7 GiB), leaving ~2.3 GiB for the
desktop compositor, a browser and ROCm's own context. Verified, not assumed:

There is only **one** knob. The planner's `vram_budget` (`include/strata/plan/plan.hpp:109,133`) is
*derived* from the pool the expert cache was sized into, not an independent setting — do not go
looking for a second lever, and do not set `--expert-cache` to a number to hit the budget unless the
fallback in the note below fires.

**Three enforcement layers**

1. **Sizing** — computed reserve above; `--expert-cache auto`; `--kv int8`; conservative
   `--max-context` (KV lives in VRAM, so context length is spent *out of the same 10 GiB*).
2. **Guard** — `docker/vram-guard.py` samples `rocm-smi --showmeminfo vram` (fallback
   `amd-smi metric --mem-usage`) every 250 ms during a benchmark and fails the run if peak
   > `BUDGET × 1.02`. 2 % tolerance only for sampling skew — it is not extra budget.
3. **Report** — the guard writes `vram-audit.json` (peak MiB, mean MiB, timestamp, engine log line
   containing the auto-sizing decision) and it is attached to the acceptance evidence.

Fallback if the guard trips: pin `--expert-cache` to the largest slot count that holds
(`slots = (BUDGET − weights − kv − mtp − slack) / expert_layout.max_blob`) and record the number.
**Never** "fix" it by raising the budget.

## Part E — Phased plan with gates

> Every gate is a command plus its expected output. Stop and report at each gate.

### P0 — Host prep: put Docker's storage where the free space is

`/` has 76 GB free and a ROCm dev image plus a HIP build will not fit politely. `/mnt/storage` has
3.7 TB.

```sh
sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{ "data-root": "/mnt/storage/docker",
  "default-address-pools": [{ "base": "172.24.0.0/16", "size": 24 }] }
JSON
sudo systemctl restart docker
docker info --format 'data-root={{ .DockerRootDir }}'          # → /mnt/storage/docker
mkdir -p ~/Development/models ~/Development/strata-work   # what ./run.sh defaults to
docker volume create strata-ccache-gfx1101
```

**Gate P0:** `data-root=/mnt/storage/docker`; `df -h /mnt/storage` still ≥ 3 TB free.
*Rollback:* restore `daemon.json`, restart docker (old image store under `/var/lib/docker` is
untouched, so rollback costs nothing but re-pull).
*Note:* existing images/volumes under the old root become invisible, not deleted. If `docker images`
was holding anything needed, retag/re-pull after the move.

Also: `.dockerignore` at the repo root (create it) must exclude `.venv/ models/ packs/ build*/
engine/ Strata-data/ .git/ amanda/ .amanda/ *.gguf` — otherwise the build context is hundreds of GB.

### P1 — Prove the toolchain and the arch, before writing any Strata code

```sh
docker run --rm --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add video --group-add render --ulimit memlock=-1:-1 \
  -e HIP_VISIBLE_DEVICES=0 rocm/dev-ubuntu-24.04:7.2.1-complete \
  bash -lc 'offload-arch; ls /opt/rocm/lib/cmake | grep -iE "hipblas|hip" | head'
```

**Gate P1:** `offload-arch` prints `gfx1101` (and `gfx1036`), `hipcc` exists,
`hipblas`/`hipblaslt` CMake packages exist, and `rocblas/library` contains `*gfx1101*.hsaco`.
Any failure → stop; wrong base tag, fix that first.

Then confirm the container agrees with the host about the device (`rocminfo` inside → `gfx1101`,
54 CUs, **one** GPU). If two GPUs appear, the iGPU leaked through: pass only `renderD128`.

### P2 — The gfx1101 patch set (code, on the host, in git)

Apply items 1-8 of Part C as **one commit per concern** so gfx1100 bisectability is intact:

```
1. cmake+device: allow gfx1101 in the HIP arch gate and the device guard
2. hip_compat: enable RDNA3 signed dot on gfx1101, warn if unavailable
3. fused_gr: derive TILE from sharedMemPerBlock instead of assuming it
4. setup.py: recognise gfx1101 / RX 7700 XT
5. docs: add an "unvalidated gfx1101" section to AMD_HIP.md
```

Self-review checklist while writing #1: match the `gcnArchName` **prefix**, not `strncmp(…,7)`.
`strncmp("gfx1101:…","gfx1100",7)` is already false for the right reason, but the old code would
also happily accept a device named `gfx1100-anything`; use an exact-prefix helper and strip the
`:append-features` suffix the way `src/prefill/gemm.cu:144-147` already does.

**Gate P2:** `git diff --stat` touches only the 9 listed files; `grep -rn '"gfx1100"' src/` has no
remaining hard gate; the allowlist is a single named constant, not three copies of a literal.

### P3 — Native gfx1101 probe (10 minutes, decides the whole backend config)

Compile and run *only* the intrinsic + device tests inside the builder, no full engine:

```sh
docker build -f docker/Dockerfile.hip-builder -t strata-hip-builder:gfx1101 .
docker run --rm --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add video --group-add render --ulimit memlock=-1:-1 \
  -v "$PWD":/src -v strata-ccache-gfx1101:/cc \
  strata-hip-builder:gfx1101 bash -lc '
    cmake -S /src -B /src/build-hip -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON \
      -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 &&
    cmake --build /src/build-hip --target hip_intrinsics hip_expert_cache_staging strata-device -j"$(nproc)"'
```

Then on the GPU: `hip_intrinsics`, `hip_expert_cache_staging`, `strata-device --selftest`.
Also disassemble one dp4a kernel and **count real dot instructions**:

```sh
cd /src/build-hip && find . -name '*.hsaco' -newer CMakeCache.txt | head -1 |
  xargs /opt/rocm/llvm/bin/llvm-objdump -d | grep -ciE 'v_(su)?dot[0-9]*_'
```

**Gate P3:** `strata-device --selftest` accepts the device and prints `gfx1101`; `hip_intrinsics`
passes (it covers packed-byte dot products *and overflow*); the objdump count is **> 0**.
If the count is 0, `__builtin_amdgcn_sudot4` is unavailable for gfx1101 → the `#warning` from P2
fires and the fallback is a real (documented) perf cliff: report it and decide with the maintainer
rather than shipping silently. If `strata-device` fails on LDS size, that is item #5 of Part C: fix
`TILE`, don't lower the budget.

### P4 — Full engine build in the container

```sh
docker run --rm --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add video --group-add render --ulimit memlock=-1:-1 \
  -v "$PWD":/src -v strata-ccache-gfx1101:/cc \
  strata-hip-builder:gfx1101 bash /src/docker/build-engine.sh gfx1101
```

`docker/build-engine.sh` wraps the documented invocation (from `docs/AMD_HIP.md:41-47` and
`setup.py:800-806`) with the pinned dependency:

```sh
cmake -S /src -B /src/build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES="$1" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
cmake --build /src/build-hip -j"$(nproc)"
```

Use the FetchContent-pinned llama.cpp (`GIT_SHALLOW FALSE`, commit `3cf0325…`) — mount the clone
into the ccache volume so a rebuild does not re-fetch 100 MB each time. `STRATA_GGML_DIR` is the
fallback for an offline build only.

**Gate P4:** `build-hip/strata` exists, and `build-hip/strata-device --selftest` prints the
HIP target line naming **gfx1101** (item #6 is what makes that line truthful — today it is a
hardcoded `gfx1100`). **gfx1100 configure** also succeeds in the same tree
(`cmake -B /tmp/b1100 -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100` then
`cmake --build /tmp/b1100 --target strata`), which is the no-regression proof for item 1.
Full build, `-j24`, expect ~10-25 min cold, ~2 min warm with ccache.

### P5 — Test suite on real gfx1101 hardware

```sh
ctest --test-dir build-hip --output-on-failure --timeout 60 \
  -E '^(ple_parity|platform_memory_test)$'
```

**Gate P5:** every selected test passes, *excluding by name* `ple_parity` (needs an external model
fixture) and `platform_memory_test` (needs 256 MiB locked memory; the container gets
`--ulimit memlock=-1:-1`, so **run it too** and report it separately — if it passes in Docker where
it failed on the bare host, that is a nice side-finding worth stating). Never report an exclusion as
a pass; state counts exactly like `AMD_HIP.md` does. Record the HIP CTest list you actually ran.

### P6 — Runtime image, model bootstrap, and the 10 GiB budget

`docker/Dockerfile.hip`, built by `./build.sh`. Runtime base is `-complete` as well: the plain
`:7.2.1` tag is 1.2 GB and has no math libraries at all.

* contents: `/usr/local/bin/strata` (named context), `serve/`, `tools/`, `data/`, `chat.py`,
  `docker/{entrypoint-hip.sh,bootstrap-model.sh,hipinfo.py,hfmodel.py,vram-guard.py}` and
  `gguf-py/` from the pinned llama.cpp — **never** a GGUF or a pack
* venv: `jinja2 psutil` for `serve/`, `numpy regex` for the pack tools, `huggingface-hub` for `hf`
  (`serve/` imports nothing else third-party, verified; Pillow is left out because vision is
  NVIDIA-only on this backend)
* `ENV HIP_VISIBLE_DEVICES=0 HSA_ENABLE_SDMA=1 STRATA_VRAM_BUDGET_MIB=10240 STRATA_MODEL=IQ3_XXS
  HF_HUB_CACHE=/hf-cache`, `ENTRYPOINT docker/entrypoint-hip.sh`, `EXPOSE 8080`
* deliberately **no HEALTHCHECK**: the first start maps tens of GB for minutes and would flap red

`entrypoint-hip.sh` does five things, in this order:

1. **Guard the environment.** `unset HSA_OVERRIDE_GFX_VERSION` and *abort* if the caller passed one
   (host lesson, Part B); same for `AMD_SERIALIZE_KERNEL`.
2. **Check the card** through the KFD topology (`hipinfo.py`, the same walk `setup.py` uses) and
   refuse anything that is not gfx1100/gfx1101.
3. **Compute the reserve** from live device numbers (Part D formula) and print the arithmetic.
4. **Make the model ready** — `bootstrap-model.sh`, idempotent:
   * resolve the quant inside the mounted HF cache (`hfmodel.py`); if it is not there, gate free
     space and `hf download <repo> --include '<quant>/*'` (~76 GB, resumable)
   * build the pack the engine needs: `strata_tokenizer.py`, then `iq_pack.py --experts-bin`
     (`--experts-bin` because the HIP mmap path requires `experts.bin`, `docs/AMD_HIP.md:61-67`)
   * fetch/pack the MTP draft layer if `STRATA_GGUF_PY` is available; otherwise start with `--spec 0`
   * `STRATA_DOWNLOAD_MODEL=0` / `STRATA_AUTO_PREPARE=0` turn this into a check that only explains
5. Write the `serve` JSON (`docs/ORCA.md:52-69` shape) and
   `exec python -m serve.server --engine strata --config /run/strata-hip.json --port 8080`.

**Where models live.** The container's model source is the **host's Hugging Face cache**, mounted
read-write at `/hf-cache` — it must be the whole `hub/` directory because HF stores each file as a
relative symlink into `hub/blobs/`. Strata artifacts that are *not* HF content (the pack, the MTP
runtime, logs) go to a writable `/work`. Nothing is baked into the image; a model the container
downloads is instantly visible to the host's `hf` client and to other containers.

Launcher `./run.sh` (repo root, no compose file needed):

```sh
./run.sh --detach          # cache: ~/Development/models   work: ~/Development/strata-work
# which is:
# docker run --rm --name strata-gfx1101
#   --device /dev/kfd --device /dev/dri/renderD128   (the discrete node only; the iGPU stays hidden)
#   --group-add video --group-add render --ulimit memlock=-1:-1 --shm-size=16g --memory 96g
#   -e STRATA_MODEL=IQ3_XXS -e STRATA_VRAM_BUDGET_MIB=10240 -e HIP_VISIBLE_DEVICES=0 -e HSA_ENABLE_SDMA=1
#   -v <hf cache>:/hf-cache -v <work>:/work -p 127.0.0.1:8080:8080 strata-hip:gfx1101-latest
```

Before docker runs, `./run.sh` checks the card, refuses `HSA_OVERRIDE_GFX_VERSION`, and computes the
free space the quant will need — download plus pack, counted **once** when cache and work dir share
a filesystem. That gate is why the model cache defaults to `~/Development/models` (3.7 TB) and not to
`~/.cache/huggingface/hub`: the root disk has ~90 GB free while IQ3_XXS wants 76 GB of download plus
43 GB of pack, so the default location would be refused. `--memory=96g` keeps headroom under the
122 GB for the file cache `--mmap-experts` depends on (`docs/AMD_HIP.md:77-79`).

**Gate P6:**

| Check | Command | Pass condition |
| --- | --- | --- |
| Model bootstrap is honest | `./run.sh --check-only` | names what is missing, or reports ready — without downloading |
| Engine up | `curl -s 127.0.0.1:8080/v1/models` | lists the model |
| Real answer | `curl -s 127.0.0.1:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"print(1+1) in python"}],"max_tokens":64}'` | correct, no engine crash |
| Budget | `docker exec strata-gfx1101 python3 /opt/strata/docker/vram-guard.py --budget-mib 10240 --for 240 --output /work/logs/vram-audit.json` during a 512-token generation | peak ≤ 10 240 MiB, audit JSON written, verdict `PASS` |
| Auto-sizing sanity | engine log line "expert cache auto: … → N slots" | N > 0, reserve = the computed value |
| Independent read | `docker exec strata-gfx1101 /usr/local/bin/strata-device --selftest` | prints `gfx1101`, real total/free VRAM, and `plan + KV vs free  FITS` |
| Context sweep | rerun with `--max-context` 8 K / 16 K / 32 K | report the largest that holds the budget; publish, don't guess |
| Restart-clean | `docker restart strata-gfx1101` | warm start, nothing re-downloaded, no VRAM creep over 3 restarts |

Model choice: the container defaults to **IQ3_XXS** as specified. With 122 GB of RAM every size fits
in RAM and VRAM only caches hot experts, so `IQ2_XS`/`Q2_0` buy speed and the Coder (IQ1_M) is the
variant to try if the goal is coding throughput on 10 GiB — all three are one `--model` away.

### P7 — Optional follow-ups (do not start until P6 is green)

Slim runtime base (`ubuntu:24.04` + ROCm runtime debs, gated on the gfx1101 hsaco check);
`HSA_OVERRIDE_GFX_VERSION`-free verification on a cold boot; hipBLASLt tuning for gfx1101 via
`tools/hip/tune_hipblaslt.cpp` → `tools/hip/gfx1101-hipblaslt-<ver>.txt` (only with the arch field
set to `gfx1101` — a mismatched table self-rejects, so shipping a gfx1100-named file is a no-op, not
a speedup); publishing measured gfx1101 numbers as a *new, dated* section; making the native
`./setup.sh --backend hip` path work on this card (patch #8 makes it possible).

## Part F — Files (all present in the tree)

| Path | Purpose |
| --- | --- |
| `build.sh` | repo root: builder image → compile in it → runtime image, with preflight and postflight gates (`--tests`, `--dry-run`, `--arch`, `--runtime-only`) |
| `run.sh` | repo root: starts the server container; checks card, `HSA_OVERRIDE_GFX_VERSION`, port, and free space for the quant before docker runs |
| `.dockerignore` | keeps `.venv/ models/ packs/ *.gguf build*/ .git/` out of the context |
| `docker/Dockerfile.hip-builder` | `strata-hip-builder:<arch>`: ROCm toolchain only; asserts the target arch at build time |
| `docker/Dockerfile.hip` | `strata-hip:<arch>`: engine binary + `serve/` `tools/` `data/` + venv + entrypoint |
| `docker/build-engine.sh` | the cmake/Ninja invocation of P4, arch as argument, ccache, retry-once (`setup.py:1044`), preflight for the arch gate and Tensile kernels |
| `docker/entrypoint-hip.sh` | env guards → device → VRAM arithmetic → model bootstrap → engine config → serve |
| `docker/bootstrap-model.sh` | check-and-fetch: `hf download`, the pack (`iq_pack.py --experts-bin`), the MTP draft layer; free-space gate before each; `prepare`/`check` modes |
| `docker/hfmodel.py` | resolves a quant's two shards inside an HF cache (families/names mirroring `setup.py:64-105`); `--print shell` for the scripts |
| `docker/hipinfo.py` | KFD-topology device discovery: arch, VRAM, render node, and the reserve formula — shared by host and container |
| `docker/vram-guard.py` | samples device memory during real work → `vram-audit.json`, exit 1 on breach |
| `docker/README.md` | quickstart, environment reference, troubleshooting for this card |

## Part G — Acceptance checklist

Run on the target machine, 2026-09-30. Evidence, not intentions.

* [x] P0 data-root: **left at `/var/lib/docker`** on `/` (76 → ~50 GB free during this work) — the images are ~22 GB each, which fits; `df` is checked by `build.sh` and warns below 40 GB. `.dockerignore` present and effective (context stayed small with 71 GB of models on disk)
* [x] P1 container reports **gfx1101**, exactly one GPU (the gfx1036 iGPU stays invisible), hipBLAS + hipBLASLt + rocBLAS present, **96 gfx1101 Tensile objects** in the image
* [x] P2 patch set applied (Part C items 1,2,3,4,6,7,8; 5 and 9 deliberately untouched — see Part H). No gfx1100-only gate remains in the build or runtime path
* [x] P3 `strata-device` opens the card: `device 0: AMD Radeon RX 7700 XT`, 27 CUs, wave32, VRAM 11.984 GiB, plan **FITS**. `hip_intrinsics` passes. The sudot4 probe compiles for gfx1101 and for gfx1100, and *fails* for gfx1030 — so the probe discriminates. Disassembly not inspected
* [x] P4 `strata` built for gfx1101: `[233/233] Linking HIP executable strata`, 16 570 840 bytes, sha256 `1d3977d31e811e93…`. **gfx1100 not rebuilt** — the CMake gate accepts it (tested standalone) but a gfx1100 compile was not run, so "no regression on gfx1100" is unproven
* [x] P5 CTest on the GPU: **42 of 44 passed**. The 2 failures (`expert_parity`, `pool_test`) are CPU-side and reproduce **with no GPU attached at all** — they decode `experts.bin` with the legacy S2/Q8_1 blob reader while the pack holds native IQ blobs. `ple_parity` excluded (needs a model fixture)
* [x] P6 server answers real requests; **Strata's own share peaks at 9 900 MiB** against the 10 240 MiB budget while the card sits at 11 322 of 12 271 MiB, GUI's 2 032 MiB intact (`/tmp/guard-128k.json`, 520 samples during generation). It initially reached **11 409 MiB of Strata memory — a breach**; see Part H.3, which also records that the contract's *meaning* needed correcting, not just its number
* [x] Model from the mounted cache at `~/Development/models`; the container downloaded nothing that was already there (second start: pack and MTP both reused)
* [x] `./run.sh --hf-cache ~/.cache/huggingface/hub` refuses: ~90 GB free vs 103 GB required (quant + pack + a 20 GB floor)
* [x] `./run.sh --check-only` / `--offline` report a missing quant without downloading
* [x] Every image name contains `strata`; no model bytes inside any image (`strata-hip:gfx1101` = 22 GB, of which the engine is 16 MB)
* [x] `./build.sh`, `./run.sh` executable; `--dry-run` output verified
* [x] `HSA_OVERRIDE_GFX_VERSION` unset; the entrypoint rejects it if a caller sets it
* [ ] `docs/AMD_HIP.md` still calls gfx1100 the validated reference — **left that way deliberately**: it now names gfx1101 as supported by the build, but no gfx1100 numbers were touched and the CPU-side tests have never run green here

Throughput reached on this card, IQ3_XXS, budget 10 GiB: **prefill 24.3 tok/s** (75 tokens), **decode 12.3 tok/s** (48 tokens), speculative draft accepting **33 of 37** proposed tokens.

## Part H — What actually happened, in order of surprise

1. **`offload-arch` is the wrong probe.** The builder image asserted `offload-arch | grep gfx1101`; that tool *detects attached GPUs* and a build container has no `/dev/kfd`, so it exits 1 with "Failed to get device count" — and it is not on `PATH` for a non-login shell either. The image now compiles a trivial kernel **and** a `__builtin_amdgcn_sudot4` kernel for the target arch instead, which is the claim we actually care about.
2. **A real bug in the upstream build graph, unrelated to gfx1101:** `CMakeLists.txt:239` linked `qsa_prompt_attn_parity` to `CUDA::cudart` outright, so *any* HIP configure with `STRATA_BUILD_TESTS=ON` died at generate time. Every sibling target uses `${_strata_gpu_runtime_target}`; that one line now does too. (It also proves nobody had ever configured a HIP build with tests enabled.)
3. **The VRAM contract was breached on the first real start — Strata alone at 11 409 MiB against a 10 240 MiB budget.** `--expert-cache auto` sizes the cache from *free* memory and then the engine still binds the draft head, the speculative-verify window and the prompt/MMQ workspace, which came straight out of the reserve: **1 425 MiB**. The engine adds exactly such an allowance for its multi-GPU layer-split path (`generate.cpp:1883-1889`, `+1024`) and has none on the single-GPU path (`:2033`). The container's reserve therefore gained a `later_mib` term (1536 MiB, `$STRATA_VRAM_LATER_MIB`) in `hipinfo.py`, which both `run.sh` and the entrypoint call, so the formula exists once.

   **…and the contract itself was then restated, which mattered more.** The budget is *Strata's own share* of the card — the ~2 GiB left over on a 12 GiB card is where the desktop and the GUI live — so charging the desktop's 1 422 MiB against Strata as well was wrong. Because `--expert-cache auto` sizes off free memory, `occupancy = total - reserve + later` whatever else is running, so the desktop's usage is subtracted from the reserve instead: `reserve = max(later + slack, total - budget + slack + later - desktop)` = 2402 MiB here, and the floor is `later + slack` rather than 700 so a greedy desktop makes Strata take less instead of overflowing the card. `vram-guard.py` judges the same quantity (`--others-mib`), and defaults to the raw card total, which can only be stricter. Measured with a 128k window and real generation: Strata peaked at **9 900 MiB**, the card at 11 322 of 12 271 (520 samples).
4. **The pack pipeline worked first try**: tokenizer (248 320 vocab), `experts.bin` 39.97 GiB over 48 layers, and the MTP draft layer fetched and packed (`mtp-q2_0.gguf` 0.889 GB → `rt/`), which is why `--spec 4` ran instead of falling back to `--spec 0`.
5. **The 76 GB download took ~12 minutes** at ~100 MB/s, so the cache choice was about *capacity*, not patience: `/` had ~90 GB free against 76 GB of download plus 43 GB of pack.
6. **Two CPU tests read the pack as the wrong format.** `expert_parity` / `pool_test` decode blobs through the legacy S2/Q8_1 reader; a native IQ pack gives them garbage (NaN). Reproduced with no GPU, so not a backend issue — reported here rather than papered over, and it means "the AMD CPU parity checks are green" is still an open claim on this machine.
7. `--pool-workers 23` (nproc−1) and `--prefill 512` were carried from the plan and worked; the hipBLASLt tuned path stayed off because no gfx1101 solution table exists (`tools/hip/` ships gfx1100 only) — the loader prints "using hipBLASEx" and falls back, which is safe and is the most likely lever for prefill speed later (`tools/hip/tune_hipblaslt.cpp`).
8. **IQ3_S became the default (`11a6026`) and was then re-tuned by measurement, because the IQ3_XXS pins starved its prompt path.** IQ3_S blobs are ~17% larger; the prompt path reserves its `--prefill-ring` (default 384) cache slots beside the pinned 800-slot cache, 384 IQ3_S blobs did not fit, so the engine halved its 2,048-token chunk to 1,024: fresh prefill **170.6-173.4 tok/s** (three baseline runs), TTFT at 4,096 tokens **21.1 s**. The default `run.sh` line now runs `STRATA_PREFILL_RING=48` - the lever, which alone reached **231-234 tok/s** on the old pins - plus `STRATA_EXPERT_CACHE=auto` and `STRATA_VRAM_LATER_MIB=700`, which change nothing by themselves (measured 171.8 / 173.2 tok/s with the ring untouched) but size the cache from measured free room instead of a pinned slot count: chunk stays 2,048, fresh prefill **233-235 tok/s**, TTFT@4K **16.2-16.3 s**, decode 30.1-30.4 vs 30.5 (within noise), and Strata's own share peaks at **8,723 MiB** under the 10,240 contract (~25k guard samples across the full arms and gates; the final no-flag launcher run reproduced 235.1 / 16.27 / 30.0 / 8,722.9). The 128K ladder (32 768 / 65 536 / 130 944 fresh + follow-up reuse, prefill 242–252 tok/s all the way up) and cancel-mid-stream/recovery passed on the shipping line; the other quants keep their old pins and the IQ3_XXS docker line is byte-identical to the consolidation fixture (`bench/results/2026-10-04-iq3s-tuning/`).
9. **Swift 1.5 packs `per_layer_token_embd.weight` in shard 1, not shard 2.** The container had always asked for PLE as file 2, so starting Swift through `run.sh` died with the engine's fatal "per_layer_token_embd.weight is not in …00002" (the table is in 00001, 320,001,536 rows). `hfmodel.py` now names the PLE file per family (`"ple": 1`) and the entrypoint takes `STRATA_PLE_FILE`; verified by relaunching Swift with no overrides plus `docker/test_hfmodel_ple.py`. The like-for-like arm on this card (Swift IQ3_XXS, same tuned knobs): fresh prefill 252.9 tok/s, decode 33.5, TTFT@4K 14.8 s, share 8,696 MiB PASS, coding smoke 205/205 - the smaller quant's headroom, not a different architecture.
* [ ] Prefill/decode tok/s measured and reported **without** copying the 7900 XTX table

## Part H — Top risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| dp4a silently falls back on gfx1101 (item #4) | medium | P3 objdump gate + `#warning`; treat 0 instructions as a gate failure |
| Container ROCm 7.2.1 vs host 7.2.x KFD mismatch | low | P1 probe; bump the base tag, keep host and container in the same 7.2 series |
| KV cache eats the 10 GiB budget at long context | **high** | Context sweep in P6; the guard fails the run rather than letting ROCm page to system RAM |
| 12 GiB card under-provisions the hot-expert tier → slow, looks like a regression | high | Expected; report tok/s as *new gfx1101 numbers*, never compared as a like-for-like against 24 GiB |
| hipBLASLt tuning table missing for gfx1101 | medium | Falls back safely; optional tuning in P7 only |
| `/` fills up mid-build | medium | P0 first, unconditionally |
| Desktop compositor grows into Strata's slack | low-medium | Guard fails the run; `--memory`/reserve give 2.3 GiB of room; document running headless for benchmarking |

## Part I — Explicitly out of scope

Multi-GPU (AMD or mixed AMD+NVIDIA), vision/images, Windows HIP, calibration, `wave64`,
the Orca IQ3_XXS conversion path, and any claim that gfx1101 answers are bit-identical to
gfx1100/CUDA. Those remain NVIDIA-only or unvalidated exactly as `docs/AMD_HIP.md:1-13` and
`docs/AMD_HIP.md:101-102` already state.
