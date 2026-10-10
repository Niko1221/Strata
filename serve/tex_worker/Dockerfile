FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587
# Pin the TeX distribution and base image; record all transitive package versions.
# The resulting image ID is the comparison runtime lock.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       texlive-full=2022.20230122-3 dvisvgm ghostscript python3 ca-certificates \
    && dpkg-query -W > /opt/tex-packages.lock \
    && rm -rf /var/lib/apt/lists/*
COPY worker.py /opt/strata-math/worker.py
RUN chmod 444 /opt/strata-math/worker.py
USER 65534:65534
ENV TEXMFVAR=/tmp/texmf-var TEXMFCONFIG=/tmp/texmf-config
CMD ["sleep", "infinity"]
