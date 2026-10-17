# Strata Docker Deployment Bundle

Run the [Strata](https://github.com/Niko1221/Strata) inference engine
(Qwen3.8-Flash-Next, 125B MoE) on a consumer NVIDIA GPU, and expose it as an
OpenAI-compatible API behind a web management console.

**This directory is self-contained**: apart from the engine source
([Strata](https://github.com/Niko1221/Strata), needed at build time), everything —
management console, gateway, orchestration, scripts and documentation — lives here.

---

## Quick start

```bash
cd docker

# 1. Generate a TLS certificate (self-signed, 10 years)
./scripts/gen-cert.sh

# 2. Prepare configuration
cp .env.example .env
vim .env          # at minimum set CUDA_ARCHITECTURES and MODEL

# 3. Prepare the engine build context (see "About the build context" below)
./scripts/prepare-build.sh /path/to/Strata

# 4. Build and start (the first run downloads the model, tens of GB)
docker compose up -d --build

# 5. Watch until ready
docker compose logs -f strata
```

Once running:

| Entry point | URL |
| --- | --- |
| Management console | `https://<host>:38888` |
| Inference API | `https://<host>:38888/v1` |
| Health check | `https://<host>:38888/health` |

The console asks you to **create an administrator account** on first visit
(username + password).

### Create API keys

```bash
# Creates an admin key and a user key, saved to certs/admin.key and certs/user.key
./scripts/init-keys.sh
```

- **admin key** — management console and inference API
- **user key** — inference API only (`/v1/*`), suitable for handing out to users

You can also create, disable and delete keys any time from the **Keys** tab of the
console. A full key is shown exactly once, at creation time.

---

## Architecture

```
External clients / browser
        │  HTTPS (self-signed certificate)
        ▼
┌───────────────────────┐
│ nginx      :38888     │  the only externally reachable port
│  • TLS termination     │
│  • API key / session   │  auth_request → manager
│    authorization       │
│  • SSE / WebSocket     │
└───────────┬───────────┘
            │
      ┌─────┴──────┐
      ▼            ▼
┌───────────┐ ┌───────────────┐
│ manager   │ │    strata     │  inference engine (GPU)
│ :8088     │ │    :8080      │  internal only
│ metrics   │ └───────────────┘
│ keys/users│
│ control   │
│ web shell │
└───────────┘
```

| Service | Role |
| --- | --- |
| `strata` | The compiled CUDA inference engine. The model stays resident in VRAM; exclusive GPU access |
| `manager` | FastAPI service: system metrics (SQLite), inference metrics, API key store, user sessions, container lifecycle, web shell, static console |
| `nginx` | TLS termination, authorization and forwarding. Neither `manager` nor `strata` publishes a host port |

### Directory layout

```
docker/
├── docker-compose.yml      three-service orchestration
├── Dockerfile              engine image (compiles the CUDA engine at build time)
├── docker-entrypoint.sh    engine container entrypoint
├── Dockerfile.nginx        gateway image
├── nginx.conf              TLS + authorization + forwarding rules
├── .env.example            configuration template
├── manager/                management console (FastAPI + single-page UI)
│   ├── app.py
│   ├── Dockerfile
│   └── static/             UI (uPlot + xterm, vendored, no CDN dependency)
├── scripts/
│   ├── gen-cert.sh         generate a self-signed certificate
│   ├── init-keys.sh        bootstrap API keys
│   └── prepare-build.sh    assemble the engine build context (source + Dockerfile, no secrets)
├── certs/                  certificate and key output (not committed)
├── tests/                  i18n coverage check (see tests/README.md)
├── CONTRIBUTING.md         notes for upstream contributions, known limitations
└── .build-context/         generated at build time (not committed)
```

---

## Configuration (`.env`)

### Hardware

| Variable | Default | Description |
| --- | --- | --- |
| `CUDA_ARCHITECTURES` | `89` | Which GPU architectures to compile. Building only your card's architecture is much faster |
| `GPU` / `GPUS` | _empty_ | Pin a single GPU (`0`) or several (`0,1`) |
| `LOW_RAM` | `auto` | Set to `on` when system RAM is constrained |

CUDA architecture reference:

| GPU | Value |
| --- | --- |
| RTX 20-series | `75` |
| RTX 30-series | `86` |
| RTX 40-series | `89` |
| RTX 50-series | `120` |
| A-series | `80` |

Separate multiple architectures with semicolons, e.g. `"86;89;120"`.

### Model

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL` | `IQ3_S` | `Q2_0` (fastest) / `IQ2_XS` / `IQ3_XXS` / `IQ3_S` (best quality) / `IQ1_M` (Coder, runs in 32 GB) / `UD-Q4_K_XL` (experimental) |
| `CONTEXT` | `262144` | Context window, up to 256K. Lower it (e.g. `131072`) if VRAM is tight |
| `KV` | `q4_0` | KV cache quantization: `int8` (higher quality) / `q4_0` (half the memory) |
| `VISION` | `yes` | Image input: `no` / `yes` / `cpu` |

> About VRAM: the KV cache and the expert cache share the same memory. A larger
> context squeezes the expert cache, which shows up as **slower first requests**;
> throughput returns to steady state once the cache warms. `q4_0` halves KV memory.

### Data and network

| Variable | Default | Description |
| --- | --- | --- |
| `STRATA_DATA_DIR` | named volume `strata-data` | Where models and configs live. Set a host path to use a bind mount |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace mirror; leave empty to use the default |
| `STRATA_HTTPS_PORT` | `38888` | HTTPS port |
| `KEEP_DAYS` | `7` | Metric retention in days |
| `SAMPLE_INTERVAL` | `10` | System metric sampling interval (seconds) |

After changing model or context settings: `docker compose up -d`
(downloaded files are reused; only the model reloads, ~1-2 minutes).

---

## Client integration

```text
Base URL : https://<host>:38888/v1
Auth     : Authorization: Bearer <API key>
Model ID : see GET /v1/models (e.g. qwen3.8-flash-next-iq3_s)
```

### Certificate trust

Clients do not trust self-signed certificates by default:

```bash
# Linux / WSL
sudo cp certs/server.crt /usr/local/share/ca-certificates/strata.crt
sudo update-ca-certificates

# Windows (PowerShell, current user)
certutil -addstore -user Root certs\server.crt

# macOS
sudo security add-trusted-cert -d -k /Library/Keychains/System.keychain certs/server.crt
```

Node.js applications additionally need:

```bash
export NODE_EXTRA_CA_CERS=$(pwd)/certs/server.crt
```

> Skipping verification (`curl -k`, `verify=False` in Python) works for quick tests,
> but is **not recommended on untrusted networks** — the API key travels in clear text.

### Examples

```bash
curl https://localhost:38888/v1/chat/completions \
  -H "Authorization: Bearer $(cat certs/user.key)" \
  -H "Content-Type: application/json" \
  -d '{"model":"<MODEL_ID>","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

```python
import requests

key = open("certs/user.key").read().strip()
r = requests.post(
    "https://localhost:38888/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json={"model": "<MODEL_ID>",
          "messages": [{"role": "user", "content": "Hello"}]},
    verify="certs/server.crt",
)
print(r.json()["choices"][0]["message"]["content"])
```

### Firewall (LAN access)

```powershell
# elevated PowerShell
New-NetFirewallRule -DisplayName 'Strata' -Direction Inbound -Protocol TCP -LocalPort 38888 -Action Allow -Profile Private
```

---

## Operations

```bash
docker compose ps                    # component status
docker compose logs -f strata        # engine log (tokens/s, expert cache hit rate)
docker compose logs -f manager       # console and authorization log
docker compose restart strata        # reload the model (~1-2 minutes)
docker compose down                  # stop (data is preserved)
docker compose down -v               # stop and delete volumes (⚠️ model must be re-downloaded)

# after updating the engine source
./scripts/prepare-build.sh /path/to/Strata   # refresh the build context
docker compose build strata && docker compose up -d
```

### Console features

| Tab | Contents |
| --- | --- |
| System | GPU utilization / VRAM / temperature / power, CPU and memory cards, plus history charts (1h / 6h / 24h / 7d) |
| Metrics | Time-to-first-token, decode tok/s, prefill tok/s, P50/P95, per-request table |
| Chat | Streaming chat with image attachment, live TTFT and generation speed; requests are recorded automatically |
| Keys | Create (admin/user), disable and delete API keys |
| Settings | Runtime parameters, model rebuild (change model / context / GPU) |
| Logs | Container and access logs, pushed live |
| Terminal | Shell inside the `strata` container — you must click **Connect**, and can disconnect at any time |

Metrics are recorded for requests that pass through the manager proxy
(`/v1/chat/completions`, `/v1/completions`, `/v1/messages`). Point your
application's base URL at `https://<host>:38888/v1` to have them counted.

### Console language

The console is **English by default**. When the browser prefers Chinese it
switches automatically, and the button in the header (EN / 中文) overrides the
choice; the selection is remembered in `localStorage`.

Strings that JS renders at runtime go through `t()`; the rest are translated by
walking the DOM on load. When adding UI text, add a matching entry to the `I18N`
dictionary and run `node tests/i18n-coverage.js` to confirm nothing was missed.

---

## Performance reference

RTX 4090 (24 GB), `IQ3_S`, 256K context:

| Metric | Value |
| --- | --- |
| Generation speed | ~100 tok/s in steady state (the first request after a restart is slower) |
| Prompt throughput | ~4.4K tok/s reading a fresh 32K-token document |
| VRAM usage | ~23.7 / 24.5 GB |
| Concurrency | Single-slot engine: concurrent requests queue; throughput saturates near 190 tok/s |

---

## Troubleshooting

**Build is slow.** The first build compiles the CUDA engine (30-90 minutes).
Set `CUDA_ARCHITECTURES` to your GPU's single value to shorten it considerably.

**docker.io pull timeouts.** Override the base image and PyPI index in `.env`:

```bash
BASE_IMAGE=docker.1ms.run/library/python:3.12-slim
PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

**Browser certificate warning.** The certificate is not trusted. Use the
commands printed by `scripts/gen-cert.sh`, or verify the service itself with
`curl -k` first.

**Certificate error over the LAN.** The address you connect to is not in the
certificate SAN. Regenerate:

```bash
FORCE=1 ./scripts/gen-cert.sh <hostname> <LAN-IP>
docker compose restart nginx
```

**First start takes long.** The model (tens of GB) must be downloaded and the
expert pack prepared. Follow `docker compose logs -f strata`. Re-running resumes
where it stopped.

**Out of memory at startup.** Lower `CONTEXT` (e.g. `65536`), switch to a smaller
`MODEL` (`IQ2_XS` / `IQ1_M`), or enable `LOW_RAM=on`.

---

## License

- Engine and this bundle: [Strata](https://github.com/Niko1221/Strata), MIT License
- Model: Qwen3.8-Flash-Next by the Qwen team; parts carry their own licenses

`certs/server.key` is a private key and is excluded by `.gitignore` — do not commit it.
