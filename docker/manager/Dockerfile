# Swap the base image for a mirror when docker.io is unreachable:
#   docker compose build --build-arg BASE_IMAGE=docker.1ms.run/library/python:3.12-slim manager
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Likewise for PyPI, e.g. https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX=https://pypi.org/simple

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN pip config set global.index-url ${PIP_INDEX} \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && apt-get update && apt-get install -y --no-install-recommends curl tzdata \
 && rm -rf /var/lib/apt/lists/*

COPY app.py /app/app.py
COPY static /app/static

ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1 PORT=8088
EXPOSE 8088
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fs "http://127.0.0.1:8088/health" || exit 1
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8088 --timeout-keep-alive 600"]
