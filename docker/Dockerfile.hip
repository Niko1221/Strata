# syntax=docker/dockerfile:1.7
# strata-hip: the runnable image - the gfx1101/gfx1100 engine, the OpenAI-compatible server, and the
# model bootstrap.  Built by ../build.sh; run with ../run.sh.
#
# Manual build (./build.sh does this with the right contexts and tags):
#   cmake --build build-hip                     # first, in strata-hip-builder
#   docker build -f docker/Dockerfile.hip \
#     --build-context engine=./build-hip \
#     --build-context ggufpy=<build-hip>/_deps/strata_llamacpp-src/gguf-py \
#     -t strata-hip:gfx1101 .
#
# The two named contexts keep the build context small: .dockerignore excludes build*/ and every GGUF,
# so nothing but source, serve/, tools/ and data/ is uploaded.  'engine' is the compiled binary,
# 'ggufpy' llama.cpp's python gguf-py from the SAME pinned checkout that was built against
# (3cf03257f219afbe7334045ff7c6a06ac68c627d), which tools/mtp_rt.py needs via tools/_paths.py:16.
#
# Base: the -complete tag is required for the runtime too.  Verified sizes: :7.2.1 is ~1.2 GB and
# carries no math libraries; :7.2.1-complete is ~7.4 GB and has libhipblas/libhipblaslt/librocblas
# plus the gfx1101 Tensile kernels.  ./build.sh runs an ldd gate over this image to prove it.
ARG RUNTIME_BASE=rocm/dev-ubuntu-24.04:7.2.1-complete
FROM ${RUNTIME_BASE} AS strata-hip

ARG STRATA_VERSION=0.1.29
ARG STRATA_HIP_ARCH=gfx1101
LABEL io.strata.vram-budget="1" \
      org.opencontainers.image.title="strata-hip" \
      org.opencontainers.image.description="Strata: Qwen3.8-Flash-Next on one AMD RDNA3 card, 10 GiB VRAM budget" \
      org.opencontainers.image.version="${STRATA_VERSION}" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv ca-certificates libnuma1 procps \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH} VIRTUAL_ENV=/opt/venv PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# jinja2 + psutil: serve/ (template, and the Monitor's CPU/RAM).  numpy + regex: the pack tools.
# huggingface-hub: the 'hf' client the model bootstrap uses.  Pillow is deliberately absent: vision
# is NVIDIA-only on this backend.
RUN pip install --no-cache-dir jinja2 psutil requests numpy regex huggingface-hub

WORKDIR /opt/strata
COPY serve/           ./serve/
COPY data/            ./data/
COPY tools/           ./tools/
COPY chat.py          ./
COPY docker/entrypoint-hip.sh docker/bootstrap-model.sh ./docker/
COPY docker/hipinfo.py docker/vram-guard.py docker/hfmodel.py ./docker/

COPY --from=engine  strata  /usr/local/bin/strata
COPY --from=engine  strata-device /usr/local/bin/strata-device
COPY --from=ggufpy  /       /opt/strata/gguf-py/
RUN chmod 0755 /usr/local/bin/strata /usr/local/bin/strata-device /opt/strata/docker/*.sh /opt/strata/docker/*.py

# The container contract.  HIP device 0 only (the iGPU must stay invisible), and the 10 GiB VRAM
# budget that entrypoint-hip.sh turns into --vram-reserve-mib for THIS card.
ENV STRATA_VRAM_BUDGET_MIB=10240 \
    STRATA_VRAM_SLACK_MIB=256 \
    HIP_VISIBLE_DEVICES=0 \
    HSA_ENABLE_SDMA=1 \
    STRATA_MODEL=IQ3_S \
    STRATA_HF_CACHE=/hf-cache \
    HF_HUB_CACHE=/hf-cache \
    STRATA_WORK=/work \
    STRATA_HOST=0.0.0.0 \
    STRATA_PORT=8080 \
    STRATA_GGUF_PY=/opt/strata/gguf-py \
    STRATA_EXPERT_CACHE=auto \
    STRATA_PREFILL=2048 \
    STRATA_POOL_WORKERS=0 \
    STRATA_VRAM_LATER_MIB=700 \
    STRATA_PREFILL_RING=48 \
    STRATA_VRAM_RUNTIME_RESERVE_MIB=1024
# The IQ3_S tuning above is the one measured on the gfx1101 host
# (bench/results/2026-10-04-iq3s-tuning/); ./run.sh passes its own per-quant values explicitly,
# so these ENV defaults are what a container started without them gets.
# Not a HEALTHCHECK on purpose: the first start maps tens of GB and takes minutes, which would flap
# red for no reason.  ./run.sh --check asks /v1/models instead, once the engine says it is up.
EXPOSE 8080
ENTRYPOINT ["/opt/strata/docker/entrypoint-hip.sh"]
