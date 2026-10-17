# syntax=docker/dockerfile:1
#
# Strata in a container: the CUDA engine, the Python server and the tokenizer tool, built
# from pinned sources, run against a model bind-mounted read-only.
#
# WHY THIS EXISTS NEXT TO THE REPOSITORY'S OWN Dockerfile.  That one compiles the engine at
# build time but still starts setup.py inside the container, so the first start downloads the
# model (~70 GB), re-inspects the host PC and writes its install config onto the volume.
# Here every step the installer would take is taken once, at build time, from a pinned input,
# and baked into the image.  A container start then touches no network at all: the tokenizer
# and the engine's run config are generated from the mounted model by docker/entrypoint.py.
#
# PINNED: the base image tags, the llama.cpp commit (fetched BY SHA and verified after the
# checkout, so a moved branch cannot change the build) and every Python package through
# requirements.txt.  NOT PINNED: apt's package versions - the base tag fixes the suite
# (Ubuntu 24.04) but apt's own versions move over time.  That is the one non-reproducible
# input, and it is deliberate: pinning apt versions makes the build fail once Ubuntu rotates
# a package out of the archive.
#
# Build (the script picks the CUDA arch from nvidia-smi for you):
#   docker/build.sh
#   docker build -f docker/Dockerfile -t strata:local --build-arg CUDA_ARCHITECTURES=89 .
#
# Run (the script resolves the model, mounts it read-only and reports what the API serves):
#   MODEL_FILE=/path/to/model.gguf docker/run.sh serve
#   MODEL_FILE=/path/to/model.gguf docker/run.sh engine --bench-pp 512
#   docker run --rm --gpus all -p 127.0.0.1:8080:8080 \
#     -v /path/to/model.gguf:/models/model.gguf:ro -v strata-data:/data strata:local

ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.0.0-runtime-ubuntu24.04

# ===================================================================== builder
FROM ${CUDA_DEVEL_IMAGE} AS builder

# llama.cpp, whose ggml the engine builds against.  setup.py's LLAMA_CPP_COMMIT.
ARG LLAMA_CPP_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
# A cubin per listed arch, baked into the engine; the runtime picks the one its card needs.
# Strata refuses anything below sm_75 (RTX 20).  RTX 40 = 89, RTX 30 = 86, RTX 50 = 120.
ARG CUDA_ARCHITECTURES=89

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake git ninja-build python3 \
 && rm -rf /var/lib/apt/lists/*

# llama.cpp at the pinned commit.  Fetched by SHA and checked afterwards: `git rev-parse`
# returning the pinned commit is what makes this immutable, since git addresses content by
# hash.  .git is dropped - the checkout IS the commit, which is the point of pinning it.
RUN git init -q /opt/llama.cpp \
 && cd /opt/llama.cpp \
 && git remote add origin https://github.com/ggml-org/llama.cpp \
 && git fetch -q --depth 1 origin "${LLAMA_CPP_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${LLAMA_CPP_COMMIT}" \
 && rm -rf .git \
 && echo "llama.cpp pinned at ${LLAMA_CPP_COMMIT}"

# ARGs are not in the process environment; the BUILD.json step below reads these.
ENV STRATA_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    STRATA_LLAMA_CPP_COMMIT=${LLAMA_CPP_COMMIT}

WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake/ ./cmake/
COPY include/ ./include/
COPY src/ ./src/
COPY third_party/ ./third_party/
# tests/ even though the test suite is off: CMakeLists.txt registers a few targets in the
# CUDA block that STRATA_BUILD_TESTS does not gate (dma_batch_parity, among them), and CMake
# needs every referenced source file to exist before it will generate at all.  Only the
# `strata` target below is built, so none of them are compiled.
COPY tests/ ./tests/

# The engine, built the way setup.py's --build path builds it: CUDA on, tests off, ggml from
# the llama.cpp checkout above (STRATA_GGML_DIR).  No image encoder: the Qwen3.6-35B-A3B
# family has no vision path (setup.py warns the same way).  The CPU side is compiled for the
# CPU that builds the image, so build on the machine that will run it - the engine warns at
# start if the host has no AVX2.
RUN cmake -S /src -B /src/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_CUDA=ON \
      -DSTRATA_BUILD_TESTS=OFF \
      -DSTRATA_GGML_DIR=/opt/llama.cpp \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
 && cmake --build /src/build --target strata --parallel "$(nproc)" \
 && mkdir -p /out/engine \
 && cp /src/build/strata /out/engine/strata \
 && if [ -f /src/build/libstrata_kq.so ]; then cp /src/build/libstrata_kq.so /out/engine/; fi

# `strata --families` names the model families the engine can run.  Asked here so a
# mis-built engine fails the build instead of the first container start.  Tolerated when it
# cannot answer (this stage has no GPU).
RUN /out/engine/strata --families || echo "note: --families did not answer at build time"

# BUILD.json: serve/server.py reads it for the version it shows, and its presence marks the
# engine as a local build (`source: local`), which is what setup.py looks at to decide
# whether a compiled engine is current.  It does NOT carry the source hash setup.py computes;
# nothing in the runtime path reads that.
RUN python3 - <<'PY'
import json, os, pathlib, re
src = pathlib.Path("/src/CMakeLists.txt").read_text(encoding="utf-8")
ver = re.search(r"project\(strata VERSION ([0-9][0-9.]*)", src).group(1)
archs = [int(a.split("-")[0]) for a in os.environ["STRATA_CUDA_ARCHITECTURES"].split(";")
         if a.split("-")[0].isdigit()]
meta = {"source": "local",
        "version": ver,
        "backend": "cuda",
        "archs": archs,
        "vision": "none",
        "lib_dirs": ["/usr/local/cuda/lib64"],
        "llama_cpp": os.environ["STRATA_LLAMA_CPP_COMMIT"]}
pathlib.Path("/out/engine/BUILD.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print("engine", ver, "cuda archs", archs)
PY

# ===================================================================== runtime
FROM ${CUDA_RUNTIME_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    STRATA_DATA=/data \
    STRATA_MODEL=/models/model.gguf \
    HOST=0.0.0.0 \
    PORT=8080 \
    CONTEXT=32768

# libgomp1 / libatomic1: the engine's CPU expert pool needs both.  curl: the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl libatomic1 libgomp1 python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata

# requirements.txt pins every package (and its dependencies) exactly.  pip itself comes from
# the base image's ensurepip, so no unpinned pip is fetched either.
COPY requirements.txt ./requirements.txt
RUN python3 -m venv .venv \
 && .venv/bin/pip install --no-cache-dir -r requirements.txt

# serve/ is the API and the web app; tools/ holds the tokenizer exporter and its GGUF reader
# (serve/server.py puts ROOT and ROOT/tools on sys.path and imports both).
COPY serve/ ./serve/
COPY tools/ ./tools/
COPY docker/entrypoint.py /usr/local/bin/strata-entrypoint
RUN chmod +x /usr/local/bin/strata-entrypoint

# The engine binary, libstrata_kq.so (the K-quant prompt kernels) and BUILD.json.
COPY --from=builder /out/engine/ ./engine/

# The model is never in the image; it is mounted read-only at $STRATA_MODEL.
VOLUME ["/data"]

EXPOSE 8080

# The port only opens after the model loads: minutes, and longer on a cold page cache.  A
# too-short start period would mark a still-loading container unhealthy and a restart policy
# would then kill it mid-load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# exec'd from the entrypoint, so the server is PID 1 and `docker stop`'s SIGTERM reaches it
# (the server stops the engine in turn) instead of being swallowed by a shell.
ENTRYPOINT ["/opt/strata/.venv/bin/python", "/usr/local/bin/strata-entrypoint"]
