# syntax=docker/dockerfile:1
#
# Strata: Qwen3.8-Flash-Next on NVIDIA GPUs (RTX 30/40/50, 12+ GB VRAM; two or
# three cards can share one model, 8 GB each - docs/MULTI_GPU.md). A Tesla P40
# (Pascal, sm_61) can be built for too, with -DSTRATA_EXPERIMENTAL_PASCAL=ON and
# a CUDA 12.x base image - see the CUDA_IMAGE/CUDA_ARCHITECTURES notes below.
#
# The engine is compiled during docker build, so the first container start only
# downloads the model (~70 GB) and starts the server. docker build has no GPU,
# so the CUDA architectures are fixed here instead of read from nvidia-smi: the
# engine is a fat binary with a cubin per listed arch, and the runtime picks the
# one matching your card. Narrow CUDA_ARCHITECTURES to your card for a faster
# build; a card outside the set needs a rebuild with its own arch.
#
# Build:
#   docker build -t strata .
#   docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .        # RTX 40 only
#
# Run (host needs an NVIDIA driver >= 580 and nvidia-container-toolkit):
#   docker run --rm --gpus all \
#     -p 8080:8080 \
#     --ulimit memlock=-1 \
#     -v strata-data:/data \
#     -e MODEL=IQ2_XS \
#     strata
#
# Setup choices are env vars, read by docker-entrypoint.sh: FAMILY, MODEL, CONTEXT,
# VISION (no | yes | cpu), KV (int8 | q4_0 | k8v4), GPU (one card) or GPUS ("0,2"
# or "all", with LAYER_SPLIT), LOW_RAM (auto | on | off), HOST, PORT, API_KEY.
#
# Only the model files, the prepared pack, the MTP layer and the install config
# live in the /data volume; the engine is part of the image. Strata loads 32-62 GB
# into RAM, so a capped container needs -e LOW_RAM=on: setup.py reads the RAM from
# /proc/meminfo, which here is the host's total, not the container's limit. Add an
# API key before exposing the port to a network: -e API_KEY=<secret>. Pass
# -e REINSTALL=1 to change the model settings later.
#
# --gpus all on a host with two usable cards: setup takes both (the layer split is
# its recommended default). Pin one card with -e GPU=0, or name them with
# -e GPUS=0,2. A volume set up for one card switches to the pair on its first start
# on a two-card host unless GPU or GPUS pins it. LOW_RAM=on runs on one card.

# CUDA 13.0 is the default base image (it is what an RTX 50 needs: an engine built with 12.8 crashed on sm_120,
# issues #220 / #224).  A Tesla P40 is Pascal (sm_61) and CUDA 13.0 REMOVED offline compilation for Maxwell,
# Pascal and Volta, so a P40 build needs the last 12.x image instead:
#   docker build -t strata --build-arg CUDA_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04 \
#                    --build-arg CUDA_ARCHITECTURES=61 .        # Tesla P40
ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

# STRATA_EXECV=1: setup.py replaces itself with the server, so the server is PID 1
# and docker stop's SIGTERM reaches it (see setup.start). Normal Linux starts, which
# don't set it, keep spawning the server as a child.
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 STRATA_EXECV=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 \
        python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata
COPY . .

# RTX 20 (75), RTX 30 (86), RTX 40 (89), RTX 50 (120), plus 80 for A-series. CMakeLists
# refuses anything below 75 unless STRATA_EXPERIMENTAL_PASCAL is on, which the build step below sets
# automatically when 61 (or 62) is in this list - pair it with CUDA_IMAGE=nvidia/cuda:12.9.1-*.
ARG CUDA_ARCHITECTURES=75;80;86;89;120
ARG BUILD_VISION=1

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir \
        numpy jinja2 regex pyyaml tqdm requests cmake ninja pillow psutil \
    && chmod +x setup.sh docker-entrypoint.sh

# llama.cpp at the pinned commit, then the engine and the image encoder, built
# exactly the way setup.py builds them. BUILD.json is what setup.py reads to
# decide whether an engine is current: source=local with a matching src hash
# means the first start reuses it instead of recompiling.
RUN .venv/bin/python - <<'PYEOF'
import json, os, pathlib, shutil
import setup

llama = setup.get_llama_cpp()
nvcc, _ = setup.find_nvcc()
arch = os.environ.get("CUDA_ARCHITECTURES", "75;80;86;89;120").strip().strip('"').replace(",", ";")
vision = "gpu" if os.environ.get("BUILD_VISION", "1") == "1" else "none"

# 61/62 in the arch list means a Pascal card, and Pascal needs CMake's own opt-in (the runtime CC floor and the
# fp32 cuBLAS path both come from it).  CUDA 13 cannot compile it at all, so refuse the pair here rather than let
# nvcc fail on a target it no longer has.
archs = [int(a.split("-")[0]) for a in arch.split(";") if a.split("-")[0].isdigit()]
defs = ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DSTRATA_GGML_DIR={llama}"]
if any(a in setup.PASCAL_ARCHS for a in archs):
    if max(archs) >= 120:
        raise SystemExit("CUDA_ARCHITECTURES mixes Pascal (61/62) and RTX 50 (120); no CUDA toolkit compiles for "
                         "both (13.0 dropped Pascal, 12.x crashed on sm_120 - issues #220/#224). Build one image "
                         "per card generation.")
    print("Pascal arch in CUDA_ARCHITECTURES: adding -DSTRATA_EXPERIMENTAL_PASCAL=ON "
          "(needs a CUDA 12.x base image: CUDA_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04)")
    defs.append("-DSTRATA_EXPERIMENTAL_PASCAL=ON")

setup.cmake_build(setup.ROOT, setup.ROOT / "build", "strata", defs, None, "build-strata.bat")
if vision != "none":
    setup.cmake_build(setup.ROOT / "tools" / "vision", setup.ROOT / "build-vision", "strata-vision",
        [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON",
         f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}"], None, "build-vision.bat")

eng = setup.ROOT / "engine"
eng.mkdir(exist_ok=True)
shutil.copy2(setup.ROOT / "build" / setup.EXE, eng / setup.EXE)
if vision != "none":
    shutil.copy2(setup.ROOT / "build-vision" / "bin" / setup.VEXE, eng / setup.VEXE)
bindir = pathlib.Path(nvcc).parent
meta = {"source": "local", "version": setup.source_version(),
        "archs": archs, "vision": vision,
        "cuda_dirs": [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()],
        "src": setup.source_hash(setup.ENGINE_SOURCES),
        "vision_src": setup.source_hash(setup.VISION_SOURCES) if vision != "none" else None}
(eng / "BUILD.json").write_text(json.dumps(meta, indent=1))
PYEOF

# the cmake trees are build-time only; the engine itself is what the container needs
RUN rm -rf build build-vision

VOLUME ["/data"]
EXPOSE 8080

# /health is answered before the API key gate, so it works with or without one.
# The port only opens after the model loads (1-3 minutes, longer on a first run),
# so the start period is generous: a too short one marks a still-loading container
# unhealthy and a restart policy would kill it mid-download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
