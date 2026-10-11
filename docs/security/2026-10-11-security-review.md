# Security Review: StrataQwen3.8-Flash-Next-

> **Status:** Initial static review; findings remain open. No fixes or live exploit reproductions are claimed.
>
> **Reviewed revision:** [`7695859aa13afb248bdb4d36f9b2e95707db079d`](https://github.com/kcmrshll9-ux/StrataQwen3.8-Flash-Next-/commit/7695859aa13afb248bdb4d36f9b2e95707db079d)  
> **Review date:** October 11, 2026  
> **Method:** Codex Security static source review with independent baseline, architecture, and focused investigations. This is an AI-assisted review, not an exhaustive audit or security certification.

This report records four source-backed findings: three medium-severity issues and one low-severity issue. Severity depends on the stated deployment and import prerequisites. The default native installation is local; the Docker path has different exposure defaults.

The full review follows, including the threat model, source evidence, counterevidence, recommended fixes, and coverage gaps. The final section turns the findings into a practical improvement checklist and records the intended Jerry deployment context.

## Scope

Broad static security review of main at 7695859aa13afb248bdb4d36f9b2e95707db079d (2976 tracked files). Prioritized production HTTP/browser/MCP/tokenizer/import/deployment boundaries and current-tree secret patterns.

- Scan mode: repository
- Target kind: git_revision
- Revision: 7695859aa13afb248bdb4d36f9b2e95707db079d
- Inventory strategy: repository
- Included paths: .
- Excluded paths: none
- Artifacts reviewed: AGENTS.md, SECURITY.md, docker-entrypoint.sh, requirements.txt, serve/mcp.py, serve/runconfig.py, serve/web/monitor.js, tools/strata_tokenizer.py, tools/gguf_reader.py, ref/load.py, include/strata/artifact/gguf_reader.hpp, src/artifact/dequant.cpp, docs/monitoring/prometheus.yml, docs/monitoring/servicemonitor.yaml, serve/server.py (security-relevant sections), serve/frontend.py (security-relevant sections), serve/web/app.js (rendering sections), setup.py (download/configuration sections), tools/strata_mcp.py (architecture/control sections), src/program/generate.cpp (control-vector/protocol/session sections), src/core/conversation_file.cpp (restore/parser sections), src/core/native_dense.cpp (payload checks), src/core/native_head.cpp (payload checks), src/kernels/ngram.cpp (payload checks), Dockerfile, docs/INSTALL.md (Docker deployment)
- Scan context: Independent baseline and architecture reviews plus focused HTTP/native investigations; generated threat model. Three concurrent worker slots used within available capacity. This is a first-pass source review with partial file-level coverage.

Limitations and exclusions:
- Coverage is partial, not 2976 fully reviewed files. Most of the tree consists of benchmark artifacts (2058 files) and native/SYCL implementation.
- No runtime exploit reproduction, deployment/network exposure verification, build or GPU tests.
- No dependency advisory database/CVE check or Git-history secret scan; current snapshot only.
- No authentication bypass, XSS, live credential validity, code execution or information disclosure claimed without supporting evidence.

### Scan Summary

| Field | Value |
| --- | --- |
| Scan outcome | completed |
| Reportable findings | 4 |
| Severity mix | medium: 3, low: 1 |
| Confidence mix | high: 4 |
| Coverage | partial |
| Validation mode | Static source validation; local Python standard-library inspection; no application or exploit execution |

Canonical artifacts: `scan-manifest.json`, `findings.json`, and `coverage.json`. This report is a deterministic projection of those files.

## Threat Model

Strata is an operator-run Windows/Linux model service. Python HTTP APIs and browser UI render/tokenize requests and communicate over pipes with a C++ CUDA/HIP/SYCL engine; setup downloads/imports model files and engines. Configured vision and MCP subprocesses extend operator authority (AGENTS.md:3; serve/server.py:591; serve/server.py:1983; serve/mcp.py:93). Native startup defaults to loopback; Docker first setup defaults to 0.0.0.0:8080 and an empty key (serve/server.py:6193; docker-entrypoint.sh:24-26). Architecture review was independent and source-backed; it does not establish complete audit coverage.

### Assets

- Operator-readable files, inherited process/environment credentials, executable/model integrity, prompts/answers/session snapshots, authentication/program configuration, GPU/RAM/disk availability and configured external MCP account capabilities. Setup writes repository-root strata-\<tag\>.json, including executable, args, cwd, tokenizer and optional API key (setup.py:5685; setup.py:5727; setup.py:5753). Web config saving copies the full config into adjacent .bak and stages .tmp; private filesystem ACLs remain host-owned (serve/runconfig.py:164). Shared defaults reside in adjacent \<config-stem\>.shared-settings.json (serve/server.py:6387). Monitoring is opt-in and in-memory; slot persistence is opt-in (serve/server.py:6326; serve/server.py:6339).

### Trust Boundaries

- HTTP callers -\> operator process: optional constant-time Bearer/x-api-key authentication on every POST and sensitive GET, separate no-key Host check and browser Origin/CORS controls (serve/server.py:4827; serve/server.py:4991; serve/server.py:4905; serve/server.py:5850; serve/server.py:4703; serve/server.py:4722; serve/server.py:4783). Settings, load/unload, slots and MCP require JSON plus same/trusted browser Origin; absence of Origin is accepted for nonbrowser clients, not a distinct local-user authorization gate (serve/server.py:5200; serve/server.py:5249; serve/server.py:5352). Web config editing has a fixed allowlist excluding network/key/MCP/program keys (serve/runconfig.py:19; serve/runconfig.py:81; serve/runconfig.py:121). Images cross to operator-readable local files or HTTP fetches: foreign browser local images and UNC paths are refused; HTTP download capped to 32MiB/60 seconds (serve/server.py:2088; serve/server.py:2111; serve/server.py:5258). Vision chooses a random no-space directory from system temp/cwd/SystemDrive; hash-derived .img/.sve files, .img cleanup and 64 cached embedding entries do not imply /data storage (serve/server.py:1983; serve/server.py:2183; serve/server.py:2199; serve/server.py:2207). Configured stdio MCP inherits full environment plus configured env and cwd with operator OS rights; HTTP MCP sends configured headers/tool arguments to operator-configured URL (serve/mcp.py:93; serve/mcp.py:105; serve/mcp.py:271; serve/mcp.py:648). before_load string commands use a shell, argv lists do not; only trusted config enables them (serve/server.py:3005; serve/server.py:6338). GGUF cursor/version checks and model tensor bounds helpers govern imported native files, with standalone consumers separately audited; session restore uses checksums/version and no-follow/reparse-point rejection (include/strata/artifact/gguf_reader.hpp:235; include/strata/artifact/gguf_reader.hpp:421; include/strata/artifact/gguf_reader.hpp:538; src/core/conversation_file.cpp:174; src/core/conversation_file.cpp:349; src/core/conversation_file.cpp:1206). Installer crosses pinned llama source/upstream executable bytes into code execution: digests verify when available, but unavailable digest/explicit skip proceeds and release publishers remain trusted (setup.py:166; setup.py:1715; setup.py:2820; setup.py:2833). Explicit stdio tools/strata_mcp.py clients can install/start/stop and read permitted normalized repository/data roots; typed arguments and confirm=true gate installation but connecting client owns human consent (tools/strata_mcp.py:440; tools/strata_mcp.py:451; tools/strata_mcp.py:1193; tools/strata_mcp.py:1612).

### Attacker Capabilities

- Reachable HTTP clients supply messages, schemas/tool definitions, images and control requests subject to configured API key/browser gates. Foreign webpages have constrained Origin/Host/content type and cannot ordinarily omit browser Origin. Operator-configured MCP peers supply metadata/results under granted OS/account authority; tool visibility is not arbitrary executable configuration authority. Malicious artifact suppliers can provide GGUF/control-vector files that an operator chooses to import. Do not assume attacker control of operator config, release publishing or private filesystem. Native default requires local access or explicit exposure for remote attacks; no actual public/multi-tenant deployment supplied.

### Security Objectives

- Authenticate sensitive HTTP state and controls when keys configured; maintain separate browser-origin protection; prevent web config from enabling programs/MCP/network changes. Preserve file/environment confidentiality across vision and tool flows. Bound untrusted request memory and connection lifetime so API usage or incomplete connections cannot persistently exhaust host resources. Keep slot filenames plain platform-safe names and restore without links, validating serialized session integrity (serve/server.py:2754; serve/server.py:2779; serve/server.py:2884; src/core/conversation_file.cpp:359). Slot effective path derives from CLI relative to server cwd or config relative to config.cwd, not automatically data root (serve/server.py:6339). Validate all native tensor payloads before pointer reads. Preserve source/executable integrity while acknowledging upstream publisher trust. AGENTS.md:10 requires a key before exposing beyond localhost.

### Assumptions

- Source-only current commit 7695859aa13afb248bdb4d36f9b2e95707db079d; no model execution or exploit reproduction. Native local defaults differ from Docker; Docker keyless nonloopback startup only warns (docker-entrypoint.sh:24-26; serve/server.py:6416). Docker later startup picks CONFIG, qualifying persisted link or family/model config under /data/config; HOST/API_KEY changes alone do not update existing security config without REINSTALL/CONFIG because final setup repeats port/GPU only (docker-entrypoint.sh:35; docker-entrypoint.sh:56; docker-entrypoint.sh:91; docker-entrypoint.sh:102-106). SECURITY.md's 'own page' MCP guarantee is browser protection; no-Origin JSON nonbrowser requests also pass (serve/server.py:5200; serve/server.py:5249). Normal setup new model data default is checkout-parent/Strata-data/models and packs/\<tag\>, overridden by --data-dir/settings/models-dir and existing discovered packs; Docker model data default /data (setup.py:3805; setup.py:4914; setup.py:5519; docker-entrypoint.sh:19). Config backups can retain keys, protected by host ACLs rather than independently private writer permissions. Vision temporary cache does not derive from /data. Image URL private destinations are not filtered, but no distinct inference-only untrusted-client contract is established. Published executable digest verification is conditional; HTTPS/default publisher trust is not evidence of a malicious release. Independent architecture findings were verified against material consumer source anchors.

## Findings

| Finding | Severity | Confidence | Detailed write-up |
| --- | --- | --- | --- |
| [Incomplete HTTP connections can exhaust server threads before authentication](#finding-1) | medium | high | inline below |
| [Docker defaults publish an unauthenticated model API](#finding-2) | medium | high | inline below |
| [Rejected prompts permanently consume tokenizer cache memory](#finding-3) | medium | high | inline below |
| [Imported control-vector GGUF can trigger an out-of-bounds read](#finding-4) | low | high | inline below |

### Confidence Scale

| Label | Meaning |
| --- | --- |
| high | Direct evidence supports the finding with no material unresolved blocker. |
| medium | Evidence supports a plausible issue, but material runtime or reachability proof remains. |
| low | Evidence is incomplete and the item is retained only for explicit follow-up. |

<a id="finding-1"></a>

### [1] Incomplete HTTP connections can exhaust server threads before authentication

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | Repository inheritance and Python standard-library source establish indefinite pre-auth reads and one thread per connection; no live exhaustion test was run. |
| Category | Unbounded HTTP connection resource consumption |
| CWE | CWE-400 |
| Affected lines | serve/server.py:5622-5629, serve/server.py:4595-4598, serve/server.py:4703-4710, serve/server.py:5886-5891 |

#### Summary

A peer can open many connections and leave request lines or headers incomplete. Each connection holds a handler thread before the API-key gate, allowing resource exhaustion even when authentication is configured.

#### Root Cause

The listener uses `ThreadingHTTPServer` without a concurrency bound and `BaseHTTPRequestHandler` without an initial read deadline. Accepted peers therefore allocate a thread that can block indefinitely in request/header parsing before Host or API-key validation. The five-second drain timeout applies after handling and does not cover this boundary.

**Listener accepts clients** — `serve/server.py:5886-5891`

Server construction exposes the handler to every peer reaching the configured listener.

```
def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    svc.host_names = host_names_for(host, svc.allowed_hosts, svc.trusted_origins)
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    httpd.svc = svc
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
```

**Unbounded thread-per-connection server** — `serve/server.py:5622-5628`

ThreadingHTTPServer delegates to Python's thread-per-connection mixin; this class adds no admission cap.

```
class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"
    # socketserver listens with a backlog of 5: a burst of 30-40 clients at once got "connection reset by peer" on the
    # first ones (measured on 4 x R9700, also with one request at a time); the requests wait in the server, not the kernel
    # (STRATA_HTTP_BACKLOG overrides it)
```

**Initial parsing has no deadline** — `serve/server.py:4583-4598`

Handler delegates initial request reading without defining a timeout; body draining begins only after that call completes.

```
def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        record = None                                       # #332: this request's monitor record, if kept
        watch_done = None                                   # #430 #431: stops this request's disconnect watcher
        body_read = False                                   # whether a handler took this request's body
        DRAIN_SECONDS = 5                                   # the longest an unread body is read and dropped

        def log_message(self, fmt, *args):
            pass

        def handle_one_request(self):
            self.answer_started = False
            super().handle_one_request()
            self._drain_body()
```

**Host validation follows header parsing** — `serve/server.py:4703-4710`

super().parse_request must finish before this Host check, and endpoint key validation occurs still later.

```
        def parse_request(self):
            """Without an API key, every request (any method) first passes the Host check: DNS rebinding protection
            (host_allowed).  With a key a rebinding page cannot authenticate, so the check is skipped: tunnels and
            proxies that pass their own name on keep working."""
            if not super().parse_request():
                return False
            host = self.headers.get("Host")
            if svc.api_key or host_allowed(host, svc.host_names, "*" in svc.allowed_hosts):
```

#### Validation

The local `ThreadingMixIn.process_request()` starts a new thread for every accepted request; `StreamRequestHandler.setup()` only installs a socket timeout when the handler's timeout is non-None. This repository never sets that initial timeout. `handle_one_request()` reads request/header data before route authentication.

Validation method: Static source trace plus local Python standard-library inspection

**Unbounded thread-per-connection server** — `serve/server.py:5622-5628`

ThreadingHTTPServer delegates to Python's thread-per-connection mixin; this class adds no admission cap.

```
class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"
    # socketserver listens with a backlog of 5: a burst of 30-40 clients at once got "connection reset by peer" on the
    # first ones (measured on 4 x R9700, also with one request at a time); the requests wait in the server, not the kernel
    # (STRATA_HTTP_BACKLOG overrides it)
```

**Initial parsing has no deadline** — `serve/server.py:4583-4598`

Handler delegates initial request reading without defining a timeout; body draining begins only after that call completes.

```
def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        record = None                                       # #332: this request's monitor record, if kept
        watch_done = None                                   # #430 #431: stops this request's disconnect watcher
        body_read = False                                   # whether a handler took this request's body
        DRAIN_SECONDS = 5                                   # the longest an unread body is read and dropped

        def log_message(self, fmt, *args):
            pass

        def handle_one_request(self):
            self.answer_started = False
            super().handle_one_request()
            self._drain_body()
```

**Host validation follows header parsing** — `serve/server.py:4703-4710`

super().parse_request must finish before this Host check, and endpoint key validation occurs still later.

```
        def parse_request(self):
            """Without an API key, every request (any method) first passes the Host check: DNS rebinding protection
            (host_allowed).  With a key a rebinding page cannot authenticate, so the check is skipped: tunnels and
            proxies that pass their own name on keep working."""
            if not super().parse_request():
                return False
            host = self.headers.get("Host")
            if svc.api_key or host_allowed(host, svc.host_names, "*" in svc.allowed_hosts):
```

Limitations:
- No live connections or resource-exhaustion reproduction.
- Deployment firewall/proxy configuration not supplied.

#### Dataflow

TCP accept -\> new handler thread -\> superclass request/header read -\> indefinitely retained thread/socket

- **Source:** Unauthenticated TCP peer

- **Sink:** Handler request/header read

- **Outcome:** Process/OS connection and thread resource exhaustion

**Listener accepts clients** — `serve/server.py:5886-5891`

Server construction exposes the handler to every peer reaching the configured listener.

```
def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    svc.host_names = host_names_for(host, svc.allowed_hosts, svc.trusted_origins)
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    httpd.svc = svc
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
```

**Unbounded thread-per-connection server** — `serve/server.py:5622-5628`

ThreadingHTTPServer delegates to Python's thread-per-connection mixin; this class adds no admission cap.

```
class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"
    # socketserver listens with a backlog of 5: a burst of 30-40 clients at once got "connection reset by peer" on the
    # first ones (measured on 4 x R9700, also with one request at a time); the requests wait in the server, not the kernel
    # (STRATA_HTTP_BACKLOG overrides it)
```

**Initial parsing has no deadline** — `serve/server.py:4583-4598`

Handler delegates initial request reading without defining a timeout; body draining begins only after that call completes.

```
def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        record = None                                       # #332: this request's monitor record, if kept
        watch_done = None                                   # #430 #431: stops this request's disconnect watcher
        body_read = False                                   # whether a handler took this request's body
        DRAIN_SECONDS = 5                                   # the longest an unread body is read and dropped

        def log_message(self, fmt, *args):
            pass

        def handle_one_request(self):
            self.answer_started = False
            super().handle_one_request()
            self._drain_body()
```

#### Reachability

No API key is needed. Remote attackers need a nonloopback binding/published port or forwarding arrangement that permits incomplete connections; default direct localhost deployment limits the path to local peers.

#### Severity

**Medium** — An unauthenticated peer with TCP access can retain unbounded handler threads/sockets and deny service. Direct startup is loopback by default, so remote reachability requires a network binding or port publication. API keys do not mitigate parsing-time exhaustion.

An ingress proxy with enforced connection limits and header deadlines lowers exposure; confirmed direct public deployment increases likelihood.

#### Remediation

Bound accepted active connections and enforce finite request-line/header deadlines before authentication, with a separate bounded body-read policy. Clear or adapt timeouts once streaming inference begins so long SSE responses remain supported.

Tests:
- Incomplete headers expire without accumulating threads.
- Clients without keys cannot consume unlimited parsing resources.
- Long valid SSE responses survive the ingress deadline.

<a id="finding-2"></a>

### [2] Docker defaults publish an unauthenticated model API

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | Entrypoint defaults, first-setup propagation, authentication fail-open behavior and the published-port example are explicit in source; no deployed container was inspected. |
| Category | Insecure deployment defaults |
| CWE | CWE-1188 |
| Affected lines | docker-entrypoint.sh:24-26, docker-entrypoint.sh:76-78, docker-entrypoint.sh:102-106, docs/INSTALL.md:100-101, serve/server.py:4827-4835, serve/server.py:6416-6418 |

#### Summary

Following the Docker run example publishes an API with no key: HOST defaults to 0.0.0.0 and API_KEY to empty. Any peer that can reach the published port can use inference and sensitive controls; the server warns but continues. Changing API_KEY on an existing data volume alone does not update the saved configuration without REINSTALL or CONFIG.

#### Root Cause

The Docker entrypoint combines a nonloopback listener with an empty authentication key and propagates those defaults into setup. The documentation publishes that listener without a host-address restriction. `_authorized()` treats no key as unrestricted access and startup only warns. Saved settings persist across volume reuse, so later environment changes alone are not enforcement.

**Run example publishes port without key** — `docs/INSTALL.md:100-101`

The example maps host port 8080 without a loopback address or API_KEY.

```
3. Run (the first start downloads the ~70 GB model, then starts; later starts go straight to serving):
   `docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -v strata-data:/data strata`
```

**Container chooses network listener and empty key** — `docker-entrypoint.sh:23-26`

First-start configuration defaults differ from direct server localhost defaults.

```
VISION="${VISION:-no}"          # no | yes | cpu (the image encoder on the CPU)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-}"
```

**Defaults enter persisted run config** — `docker-entrypoint.sh:76-78`

Setup receives host and key during first installation.

```
  set -- --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" --vision "$VISION" \
    --data-dir "$STRATA_DATA" --host "$HOST" --api-key "$API_KEY" \
    --port "$PORT" --no-start --low-ram "$LOW_RAM"
```

**Empty key disables API authentication** — `serve/server.py:4827-4835`

The configured empty key makes authorization succeed for every request reaching an endpoint.

```
        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if key_matches(given, svc.api_key):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False
```

**Exposure emits only a warning** — `serve/server.py:6415-6418`

The network/no-key condition prints a warning after startup instead of refusing service.

```
            print("       from other devices: http://<this PC's IP address>:" + str(a.port) + "/", flush=True)
        if not svc.api_key:
            print("       WARNING: no API key - anyone on your network can use this model. Add \"api_key\": \"...\" "
                  "to the config (clients send it as their API key; the web page asks for it)", flush=True)
```

**Existing config governs later security settings** — `docker-entrypoint.sh:102-106`

Later starts repeat port/GPU flags but omit host/key overrides; operators must update the selected saved configuration.

```
set -- --port "$PORT"
if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
exec .venv/bin/python setup.py "$@"
```

#### Validation

HOST=0.0.0.0 and API_KEY empty feed first setup. setup.py:5725-5728 writes nonempty supplied values and server uses configured key; absent key opens authorization. The published-port example supplies no key. Docs at INSTALL.md:126 separately recommend setting one, and at :110 explain REINSTALL for stored settings; this is counterevidence to an undocumented bypass, not safe defaults.

Validation method: Static entrypoint/configuration/handler trace

**Container chooses network listener and empty key** — `docker-entrypoint.sh:23-26`

First-start configuration defaults differ from direct server localhost defaults.

```
VISION="${VISION:-no}"          # no | yes | cpu (the image encoder on the CPU)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-}"
```

**Defaults enter persisted run config** — `docker-entrypoint.sh:76-78`

Setup receives host and key during first installation.

```
  set -- --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" --vision "$VISION" \
    --data-dir "$STRATA_DATA" --host "$HOST" --api-key "$API_KEY" \
    --port "$PORT" --no-start --low-ram "$LOW_RAM"
```

**Empty key disables API authentication** — `serve/server.py:4827-4835`

The configured empty key makes authorization succeed for every request reaching an endpoint.

```
        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if key_matches(given, svc.api_key):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False
```

**Run example publishes port without key** — `docs/INSTALL.md:100-101`

The example maps host port 8080 without a loopback address or API_KEY.

```
3. Run (the first start downloads the ~70 GB model, then starts; later starts go straight to serving):
   `docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -v strata-data:/data strata`
```

**Existing config governs later security settings** — `docker-entrypoint.sh:102-106`

Later starts repeat port/GPU flags but omit host/key overrides; operators must update the selected saved configuration.

```
set -- --port "$PORT"
if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
exec .venv/bin/python setup.py "$@"
```

Limitations:
- Host firewall/container network not inspected.
- Direct native startup remains loopback by default.
- MCP and monitoring require separate opt-in configuration.

#### Dataflow

Docker defaults -\> setup run config -\> server listener and empty key -\> anonymous inference/state/control

- **Source:** Published container endpoint

- **Sink:** API routes

- **Outcome:** Unauthenticated use and control of the model service

**Run example publishes port without key** — `docs/INSTALL.md:100-101`

The example maps host port 8080 without a loopback address or API_KEY.

```
3. Run (the first start downloads the ~70 GB model, then starts; later starts go straight to serving):
   `docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -v strata-data:/data strata`
```

**Container chooses network listener and empty key** — `docker-entrypoint.sh:23-26`

First-start configuration defaults differ from direct server localhost defaults.

```
VISION="${VISION:-no}"          # no | yes | cpu (the image encoder on the CPU)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-}"
```

**Empty key disables API authentication** — `serve/server.py:4827-4835`

The configured empty key makes authorization succeed for every request reaching an endpoint.

```
        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if key_matches(given, svc.api_key):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False
```

#### Reachability

Requires container deployment with host publication or another reachable container network path; firewall may block external access. Browser Host/Origin checks do not authenticate nonbrowser clients. No claim of a live exposed service.

#### Severity

**Medium** — The documented Docker run command publishes port 8080 while the container defaults to 0.0.0.0 and no key. Network peers reaching that port can use inference and state/control APIs. Actual remote reachability depends on host firewall/routing; configured MCP or monitoring increases conditional impact.

Loopback-only host publication or verified API-key enforcement eliminates anonymous network access; public exposure and privileged configured tools increase impact.

#### Remediation

Make documented Docker publication loopback-only by default and require an explicit nonempty key for network deployments. Validate effective saved-config authentication before listening; make current HOST/API_KEY overrides apply consistently or fail clearly when persisted settings disagree.

Tests:
- Fresh default container cannot expose anonymous API beyond loopback publication.
- Reused volume with a newly supplied key enforces that key or fails clearly.
- Documented network example includes authentication.

<a id="finding-3"></a>

### [3] Rejected prompts permanently consume tokenizer cache memory

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | Direct source shows persistent insertion before context checks and no byte limit/eviction; resource thresholds were not measured at runtime. |
| Category | Persistent untrusted-input memory exhaustion |
| CWE | CWE-400 |
| Affected lines | tools/strata_tokenizer.py:232-233, tools/strata_tokenizer.py:215-222, serve/server.py:3465, serve/server.py:3419-3429, serve/server.py:3515-3529, serve/server.py:6212 |

#### Summary

Repeated distinct long alphabetic prompt pieces populate the persistent tokenizer cache even when the request receives a context-window error. Entry counts are capped, but key lengths and token-list memory are not, so rejected requests can exhaust server RAM.

#### Root Cause

`Service.prepare()` renders and tokenizes caller messages through a long-lived custom tokenizer before enforcing context limits. `_encode_plain()` caches each regex piece and token list using only `PIECE_CACHE_MAX` as an entry-count bound. Arbitrarily long pieces below the HTTP body limit are retained, and context rejection does not roll back insertion.

**Prompt tokenization precedes validation** — `serve/server.py:3461-3465`

Service.prepare sends caller messages to encode_prompt before checking available context.

```
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context.  `force` (forced_call): without thinking the reply starts with it, so it ends the
        prompt; with thinking, Service.run writes it once the thinking is over."""
        fetched = self._note_unreadable_tool_images(messages)
        ids = self.encode_prompt(messages, tools, kwargs)
```

**Caller text reaches tokenizer** — `serve/server.py:3419-3429`

Rendered message content reaches self.tok.encode on both literal-handling branches.

```
    def encode_prompt(self, messages, tools, kwargs) -> list[int]:
        """The request's prompt: the template rendered and tokenized.  #537: a <think> / </think> written inside a
        message's text is encoded as the text it is, not as the model's reasoning markers (the template's own are).
        A control token's text (<|im_start|>, <|im_end|>, <|endoftext|>, ...) inside a message is text as well: only
        the control tokens the template writes are control tokens."""
        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)
        prompt = self.render_prompt(marked, marked_tools, kwargs)
        if not changed:
            return self.tok.encode(prompt, parse_special=True)
        prompt, plain = unmark_think_literals(prompt, self.literals)
        return self.tok.encode(prompt, parse_special=True, plain=plain)
```

**Long input runs form one piece** — `tools/strata_tokenizer.py:56-59`

The letter/mark plus quantifier permits an arbitrarily long alphabetic piece within the request body limit.

```
QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
```

**Count-only persistent cache** — `tools/strata_tokenizer.py:215-234`

Each new piece and its token-ID array are retained if the dictionary has fewer than 200000 entries; neither their byte size nor total retained memory is bounded.

```
    PIECE_CACHE_MAX = 200_000    # pre-tokenizer pieces remembered (an agent resends its whole history every turn)

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        # A piece's ids depend on the piece alone, so repeated pieces (most of a resent conversation) are looked up
        # instead of merged again.  The ids are the ones _bpe gives: this only skips the work.
        cache = self._piece_ids
        for piece in self._re.findall(text):
            got = cache.get(piece)
            if got is None:
                mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
                got = []
                for tok in self._bpe(mapped):
                    i = self.ids.get(tok)
                    if i is None:
                        raise KeyError("BPE produced a token outside the vocabulary: %r" % tok)
                    got.append(i)
                if len(cache) < self.PIECE_CACHE_MAX:
                    cache[piece] = got
            out.extend(got)
```

**Context rejection comes after insertion** — `serve/server.py:3514-3530`

Over-context requests raise ValueError only after tokenization, leaving cached entries in the tokenizer.

```
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
        room = ctx - CTX_SLACK - len(ids)
        if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                 f"({ctx}); requests are never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                 f"({ctx}); requests are never truncated. Send a smaller "
                                 f"max_tokens (at most {max(0, room)} here), or add \"fit_max_tokens\": true to the "
                                 "model's strata-<model>.json to shorten it to the room left (#545)")
            max_new = max(1, room)          # --fit-max-tokens: a shorter completion beats a 400
        if images:
```

**Long-lived production tokenizer** — `serve/server.py:6210-6212`

Normal startup constructs the custom tokenizer used across requests.

```
        merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tpath / "token_type.json").read_text())
        tok = ST.Tokenizer(tokens, merges, types)
```

#### Validation

The regex admits long letter runs, production startup constructs ST.Tokenizer, both encode_prompt branches use it, and prepare rejects excessive token counts after cache insertion. Search found no cache eviction, clearing or byte budget. The 256MiB body cap and 200000-entry cap still permit impractically large retained memory.

Validation method: Independent baseline discovery and parent static source validation

**Long input runs form one piece** — `tools/strata_tokenizer.py:56-59`

The letter/mark plus quantifier permits an arbitrarily long alphabetic piece within the request body limit.

```
QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
```

**Count-only persistent cache** — `tools/strata_tokenizer.py:215-234`

Each new piece and its token-ID array are retained if the dictionary has fewer than 200000 entries; neither their byte size nor total retained memory is bounded.

```
    PIECE_CACHE_MAX = 200_000    # pre-tokenizer pieces remembered (an agent resends its whole history every turn)

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        # A piece's ids depend on the piece alone, so repeated pieces (most of a resent conversation) are looked up
        # instead of merged again.  The ids are the ones _bpe gives: this only skips the work.
        cache = self._piece_ids
        for piece in self._re.findall(text):
            got = cache.get(piece)
            if got is None:
                mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
                got = []
                for tok in self._bpe(mapped):
                    i = self.ids.get(tok)
                    if i is None:
                        raise KeyError("BPE produced a token outside the vocabulary: %r" % tok)
                    got.append(i)
                if len(cache) < self.PIECE_CACHE_MAX:
                    cache[piece] = got
            out.extend(got)
```

**Context rejection comes after insertion** — `serve/server.py:3514-3530`

Over-context requests raise ValueError only after tokenization, leaving cached entries in the tokenizer.

```
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
        room = ctx - CTX_SLACK - len(ids)
        if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                 f"({ctx}); requests are never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                 f"({ctx}); requests are never truncated. Send a smaller "
                                 f"max_tokens (at most {max(0, room)} here), or add \"fit_max_tokens\": true to the "
                                 "model's strata-<model>.json to shorten it to the room left (#545)")
            max_new = max(1, room)          # --fit-max-tokens: a shorter completion beats a 400
        if images:
```

**Long-lived production tokenizer** — `serve/server.py:6210-6212`

Normal startup constructs the custom tokenizer used across requests.

```
        merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tpath / "token_type.json").read_text())
        tok = ST.Tokenizer(tokens, merges, types)
```

Limitations:
- No model run or RAM measurement.
- Requires inference access; API key required when configured.

#### Dataflow

HTTP messages -\> prepare -\> encode_prompt -\> Tokenizer._encode_plain -\> persistent _piece_ids -\> context rejection

- **Source:** Inference request message content

- **Sink:** Long-lived tokenizer dictionary

- **Outcome:** Cumulative RAM exhaustion

**Prompt tokenization precedes validation** — `serve/server.py:3461-3465`

Service.prepare sends caller messages to encode_prompt before checking available context.

```
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context.  `force` (forced_call): without thinking the reply starts with it, so it ends the
        prompt; with thinking, Service.run writes it once the thinking is over."""
        fetched = self._note_unreadable_tool_images(messages)
        ids = self.encode_prompt(messages, tools, kwargs)
```

**Caller text reaches tokenizer** — `serve/server.py:3419-3429`

Rendered message content reaches self.tok.encode on both literal-handling branches.

```
    def encode_prompt(self, messages, tools, kwargs) -> list[int]:
        """The request's prompt: the template rendered and tokenized.  #537: a <think> / </think> written inside a
        message's text is encoded as the text it is, not as the model's reasoning markers (the template's own are).
        A control token's text (<|im_start|>, <|im_end|>, <|endoftext|>, ...) inside a message is text as well: only
        the control tokens the template writes are control tokens."""
        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)
        prompt = self.render_prompt(marked, marked_tools, kwargs)
        if not changed:
            return self.tok.encode(prompt, parse_special=True)
        prompt, plain = unmark_think_literals(prompt, self.literals)
        return self.tok.encode(prompt, parse_special=True, plain=plain)
```

**Count-only persistent cache** — `tools/strata_tokenizer.py:215-234`

Each new piece and its token-ID array are retained if the dictionary has fewer than 200000 entries; neither their byte size nor total retained memory is bounded.

```
    PIECE_CACHE_MAX = 200_000    # pre-tokenizer pieces remembered (an agent resends its whole history every turn)

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        # A piece's ids depend on the piece alone, so repeated pieces (most of a resent conversation) are looked up
        # instead of merged again.  The ids are the ones _bpe gives: this only skips the work.
        cache = self._piece_ids
        for piece in self._re.findall(text):
            got = cache.get(piece)
            if got is None:
                mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
                got = []
                for tok in self._bpe(mapped):
                    i = self.ids.get(tok)
                    if i is None:
                        raise KeyError("BPE produced a token outside the vocabulary: %r" % tok)
                    got.append(i)
                if len(cache) < self.PIECE_CACHE_MAX:
                    cache[piece] = got
            out.extend(got)
```

**Context rejection comes after insertion** — `serve/server.py:3514-3530`

Over-context requests raise ValueError only after tokenization, leaving cached entries in the tokenizer.

```
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
        room = ctx - CTX_SLACK - len(ids)
        if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                 f"({ctx}); requests are never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                 f"({ctx}); requests are never truncated. Send a smaller "
                                 f"max_tokens (at most {max(0, room)} here), or add \"fit_max_tokens\": true to the "
                                 "model's strata-<model>.json to shorten it to the room left (#545)")
            max_new = max(1, room)          # --fit-max-tokens: a shorter completion beats a 400
        if images:
```

#### Reachability

An inference API caller can trigger this on normal custom-tokenizer startup. A configured key restricts callers; direct loopback default limits remote exposure. Over-context rejection does not undo the effect.

#### Severity

**Medium** — An inference client can repeatedly retain long distinct input strings/token arrays until the server exhausts RAM, including requests rejected for exceeding context. The cache's 200000 entry cap does not provide a practical byte cap; a key limits which clients can trigger it.

A byte budget and maximum cacheable piece size remove this path; externally reachable keyless service increases likelihood.

#### Remediation

Use a byte-budgeted LRU cache with maximum individual piece/token-list sizes. Skip caching large pieces and ensure rejected requests cannot leave disproportionate persistent cache state.

Tests:
- Repeated distinct over-context prompts keep retained cache bytes below budget.
- Long individual pieces are tokenized but not cached.
- Repeated valid chat pieces still reuse bounded cache entries.

<a id="finding-4"></a>

### [4] Imported control-vector GGUF can trigger an out-of-bounds read

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | The file-controlled offset is used for N float reads without a payload check; the finding is static and no malformed artifact was executed. |
| Category | Native tensor payload out-of-bounds read |
| CWE | CWE-125 |
| Affected lines | src/program/generate.cpp:1505-1506, src/program/generate.cpp:2058-2071, src/program/generate.cpp:1485-1493, include/strata/artifact/gguf_reader.hpp:450-451, include/strata/artifact/gguf_reader.hpp:373 |

#### Summary

When an operator imports an attacker-provided control-vector GGUF, a tensor with a valid F32 shape but an offset outside the file reaches unchecked float reads. This can crash the engine during loading.

#### Root Cause

`--control-vector` and `--control-vector-scaled` load standalone `GgufFile` artifacts. Parsing accepts file-supplied tensor offsets; the loader checks architecture, type and element count but not the offset plus required bytes. `tensor_data()` returns a raw mapping-relative pointer and `src[j]` dereferences it outside the mapped artifact. Other model consumers' payload checks are not applied here.

**Operator imports supplied GGUF path** — `src/program/generate.cpp:2058-2071`

CLI options add selected artifact paths to the loader's control-vector list.

```
        else if (a == "--control-vector") o.cvec_files.push_back({next("--control-vector"), 1.0f});
        else if (a == "--control-vector-scaled") {
            // FILE:SCALE, comma-separated; the LAST colon splits, so a Windows path (C:\...) keeps its drive
            std::stringstream list(next("--control-vector-scaled"));
            std::string item;
            while (std::getline(list, item, ',')) {
                const size_t colon = item.rfind(':');
                char* end = nullptr;
                const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
                if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                    std::fprintf(stderr, "--control-vector-scaled: expected FILE:SCALE, got '%s'\n", item.c_str());
                    return 2;
                }
                o.cvec_files.push_back({item.substr(0, colon), sc});
```

**Standalone GGUF file opened** — `src/program/generate.cpp:1485-1493`

The loader checks the architecture tag but uses GgufFile rather than a bounds-validating model consumer.

```
    for (const auto& [path, scale] : o.cvec_files) {
        try {
            strata::GgufFile f(path);
            const strata::MetaValue* arch = f.get("general.architecture");
            if (arch == nullptr || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            const strata::MetaValue* hint = f.get("controlvector.model_hint");
```

**Tensor offset originates in artifact** — `include/strata/artifact/gguf_reader.hpp:449-465`

Parser retains the supplied offset; its EOF check validates only data_start.

```
            t.type = c.read<uint32_t>();
            t.offset = c.read<uint64_t>();
            tensors_.push_back(std::move(t));
        }
        uint64_t align = 32;
        if (const MetaValue* a = get("general.alignment"))
            if (a->u) align = a->u;
        alignment_ = align;
        data_start_ = (c.pos() + align - 1) / align * align;
        if (data_start_ > size_) {
            // A file with tensors needs its data section.  One without (a split model's metadata-only first
            // shard) may end at its header, before the aligned start - unsloth's UD-Q5_K_XL shard 1 ends 6 bytes
            // short, and llama.cpp's gguf-py reads it.  Nothing is read there; the clamp keeps
            // `file_size() - data_start()` (the payload checks downstream) at 0 instead of wrapping.
            if (n_tensors != 0) throw std::runtime_error("GGUF: data section starts past EOF");
            data_start_ = size_;
        }
```

**Raw unchecked tensor accessor** — `include/strata/artifact/gguf_reader.hpp:373`

tensor_data adds the stored offset to the mapping without checking payload size.

```
    const uint8_t* tensor_data(const TensorInfo& t) const { return base_ + data_start_ + t.offset; }
```

**Type/element checks omit payload bounds** — `src/program/generate.cpp:1496-1507`

After checking F32 and element count, the loader reads N floats from the unchecked pointer.

```
            int found = 0;
            for (const strata::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
```

#### Validation

A controlvector-tagged GGUF can declare direction.1 with N F32 elements and a payload offset beyond EOF. The loader's checks accept the metadata and then read at that offset. GgufModel.in_bounds at gguf_reader.hpp:538-542 exists but standalone GgufFile is not protected by it.

Validation method: Independent native audit and parent source trace

**Tensor offset originates in artifact** — `include/strata/artifact/gguf_reader.hpp:449-465`

Parser retains the supplied offset; its EOF check validates only data_start.

```
            t.type = c.read<uint32_t>();
            t.offset = c.read<uint64_t>();
            tensors_.push_back(std::move(t));
        }
        uint64_t align = 32;
        if (const MetaValue* a = get("general.alignment"))
            if (a->u) align = a->u;
        alignment_ = align;
        data_start_ = (c.pos() + align - 1) / align * align;
        if (data_start_ > size_) {
            // A file with tensors needs its data section.  One without (a split model's metadata-only first
            // shard) may end at its header, before the aligned start - unsloth's UD-Q5_K_XL shard 1 ends 6 bytes
            // short, and llama.cpp's gguf-py reads it.  Nothing is read there; the clamp keeps
            // `file_size() - data_start()` (the payload checks downstream) at 0 instead of wrapping.
            if (n_tensors != 0) throw std::runtime_error("GGUF: data section starts past EOF");
            data_start_ = size_;
        }
```

**Raw unchecked tensor accessor** — `include/strata/artifact/gguf_reader.hpp:373`

tensor_data adds the stored offset to the mapping without checking payload size.

```
    const uint8_t* tensor_data(const TensorInfo& t) const { return base_ + data_start_ + t.offset; }
```

**Type/element checks omit payload bounds** — `src/program/generate.cpp:1496-1507`

After checking F32 and element count, the loader reads N floats from the unchecked pointer.

```
            int found = 0;
            for (const strata::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
```

Limitations:
- No malformed artifact execution.
- No demonstrated disclosure or code execution.
- Requires operator import; not an HTTP upload path.

#### Dataflow

CLI artifact path -\> GgufFile tensor offset -\> load_control_vectors -\> tensor_data -\> src\[j\]

- **Source:** Imported GGUF tensor directory

- **Sink:** Native float dereference

- **Outcome:** Invalid read and possible engine crash

**Operator imports supplied GGUF path** — `src/program/generate.cpp:2058-2071`

CLI options add selected artifact paths to the loader's control-vector list.

```
        else if (a == "--control-vector") o.cvec_files.push_back({next("--control-vector"), 1.0f});
        else if (a == "--control-vector-scaled") {
            // FILE:SCALE, comma-separated; the LAST colon splits, so a Windows path (C:\...) keeps its drive
            std::stringstream list(next("--control-vector-scaled"));
            std::string item;
            while (std::getline(list, item, ',')) {
                const size_t colon = item.rfind(':');
                char* end = nullptr;
                const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
                if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                    std::fprintf(stderr, "--control-vector-scaled: expected FILE:SCALE, got '%s'\n", item.c_str());
                    return 2;
                }
                o.cvec_files.push_back({item.substr(0, colon), sc});
```

**Tensor offset originates in artifact** — `include/strata/artifact/gguf_reader.hpp:449-465`

Parser retains the supplied offset; its EOF check validates only data_start.

```
            t.type = c.read<uint32_t>();
            t.offset = c.read<uint64_t>();
            tensors_.push_back(std::move(t));
        }
        uint64_t align = 32;
        if (const MetaValue* a = get("general.alignment"))
            if (a->u) align = a->u;
        alignment_ = align;
        data_start_ = (c.pos() + align - 1) / align * align;
        if (data_start_ > size_) {
            // A file with tensors needs its data section.  One without (a split model's metadata-only first
            // shard) may end at its header, before the aligned start - unsloth's UD-Q5_K_XL shard 1 ends 6 bytes
            // short, and llama.cpp's gguf-py reads it.  Nothing is read there; the clamp keeps
            // `file_size() - data_start()` (the payload checks downstream) at 0 instead of wrapping.
            if (n_tensors != 0) throw std::runtime_error("GGUF: data section starts past EOF");
            data_start_ = size_;
        }
```

**Type/element checks omit payload bounds** — `src/program/generate.cpp:1496-1507`

After checking F32 and element count, the loader reads N floats from the unchecked pointer.

```
            int found = 0;
            for (const strata::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
```

#### Reachability

An artifact supplier must persuade the operator to use a malicious file through the explicit control-vector option. The importer is the security boundary; no direct remote HTTP trigger was found.

#### Severity

**Low** — A malicious artifact selected by the operator can crash native startup through unchecked payload access. This requires an explicit file import; no remotely writable file/upload path, disclosure or code execution was established.

A remote attacker-controlled import path or demonstrated memory disclosure would increase severity; accepting only integrity-verified trusted artifacts lowers likelihood.

#### Remediation

Before reading a standalone tensor, validate its nonzero expected payload size using overflow-safe arithmetic and require offset \<= remaining payload and bytes \<= remaining payload-offset. Prefer a shared checked accessor for standalone and multi-shard consumers.

Tests:
- Out-of-file, truncated and overflowing control-vector payloads return errors before dereference.
- Valid control vectors retain existing output.

## Reviewed Surfaces

| Surface | Risk Area | Outcome | Notes |
| --- | --- | --- | --- |
| HTTP initial parsing and connection admission | Availability | Reported | Validated http-connection-exhaustion; localhost default narrows exposure but configured keys do not cover pre-auth reads. |
| Persistent production tokenizer cache | Availability | Reported | Validated tokenizer-cache-memory; over-context rejection happens after insertion. |
| Docker authentication defaults and persisted security settings | Network authentication | Reported | Validated docker-keyless-default; docs mention keys/REINSTALL but run example exposes no-key default. |
| Standalone control-vector GGUF import | Native memory safety | Reported | Validated control-vector-oob; low severity because operator import required. |
| HTTP route authentication, browser Origin and editable config | Authorization/CSRF | No issue found | Sensitive GET branches and every POST gate configured keys; _own_page requires JSON and same/trusted Origin. No-Origin is intentionally nonbrowser, not a new local-user auth boundary. /config allowlists exclude key/network/MCP/program fields. serve/server.py:4827,4991,5200,5249,5352; serve/runconfig.py:19,81,121. Handler routes inspected; server file not fully audited. |
| Browser chat/MCP/monitor rendering | XSS | No issue found | Inspected chat markdown escapes source and permits only http(s) links; MCP fields escape HTML; monitor uses textContent. Persisted chat goes through same renderer. serve/web/app.js:8,493-556,619-636; serve/web/monitor.js fully reviewed. app.js reviewed in relevant sections, not fully. |
| Vision file paths and HTTP image destinations | Local file access/SSRF | Rejected | Foreign-browser local-file input refused across Chat/Responses/Anthropic; UNC paths refused. count_tokens only renders/tokenizes and does not fetch images. Operator/trusted API local-image/URL support is explicit; private HTTP destinations lack filtering, but no separate untrusted inference-only boundary established. serve/server.py:2096-2125,5258-5268,5342,5515,5567,5580; frontend.py:164-193. |
| Configured MCP process and tool dispatch | Command execution/tool authority | No issue found | Full serve/mcp.py review found operator-owned command/args/env/cwd/URLs; model tools resolve configured registered names. Client tool collisions excluded; max_rounds bounds execution. Transport results buffer before max_result_chars truncation; configured trusted-peer hardening observation, not independent vulnerability. serve/mcp.py:93-110,284-295,494; serve/server.py:4232-4236,5352-5371. |
| Native dense/head/PLE and session import controls | Malformed native artifacts | No issue found | Inspected sibling native payload paths use subtraction-based bounds and dimension/product checks (src/core/native_dense.cpp:238-267; src/core/native_head.cpp:41-46; src/kernels/ngram.cpp:288-313). Session parser checks counts, remaining bytes, geometry/caller limits and snapshots before device writes; generate.cpp:9922-9941 and token IDs:10178-10179. Session subsystem/GPU kernels not fully audited. Unchecked src/artifact/dequant.cpp reads are STRATA_DEQUANT_SELFTEST, separate from production control-vector path. |
| Current-tree credential and private-key pattern scan | Secret exposure | No issue found | Repository-wide recognizable AWS/GitHub/OpenAI/HF token and PEM-private-key searches found no supported secret. Inspected assignment-style hits were placeholders/test values or environment/Secret references. Pattern scan is not full line-by-line review and cannot exclude arbitrary-format secrets; history excluded by workflow. |
| Installer prebuilt/source download trust | Artifact integrity | No issue found | Relevant setup.py download/checksum sections inspected. Engine published digest mismatch rejects replacement, but unavailable digest and STRATA_SKIP_SHA256 allow warned fallback. Defaults use HTTPS/upstream release publisher trust and pinned llama commit. No independent active-attacker bypass established. Still points to upstream Niko1221/Strata, not this fork (setup.py:174-179,166,2820-2845); SECURITY.md/reporting links also upstream. Installer not fully audited. |

## Open Questions And Follow Up

- Verify the effective API key and host publication for any reused Docker volume before deployment.
  - Follow-up prompt: Verify Docker API_KEY enforcement for fresh and reused /data/config volumes at commit 7695859aa13afb248bdb4d36f9b2e95707db079d.
- Review remaining native/SYCL parser and kernel sections, and installed dependencies/advisories, before treating the project as fully audited.
  - Follow-up prompt: Audit remaining native input and GPU/SYCL memory-safety boundaries, plus dependency advisories, at commit 7695859aa13afb248bdb4d36f9b2e95707db079d.
- Broad static attack-surface review completed, but not a full file-by-file audit: remaining sections of serve/server.py, serve/frontend.py, serve/web/app.js, setup.py, tools/strata_mcp.py, native program/session/runtime files, CUDA/HIP/SYCL kernels and benchmark/test tooling remain unaudited. Full source-file coverage is not claimed.
  - Follow-up prompt: Review deferred unit remaining-source-audit and close its stated proof gap.
- No actual deployment configuration, model/GPU execution, live exploit reproduction, installed dependency inventory or advisory-database CVE audit was performed.
  - Follow-up prompt: Review deferred unit deployment-dependency-validation and close its stated proof gap.

## Remediation and Jerry Deployment Planning

### Remediation Checklist

All items below are proposed work, not completed fixes.

- [ ] **HTTP ingress:** Bound accepted connections and request/header read time. Verify that incomplete connections expire before authentication and valid long-running streaming responses still work.
- [ ] **Docker authentication:** Use loopback-only host port publication for local use; require an effective nonempty API key for LAN use. Verify fresh and reused data volumes so saved configuration cannot silently defeat a newly supplied key.
- [ ] **Tokenizer memory:** Use a byte-budgeted cache and maximum cacheable piece size. Confirm that repeatedly rejected prompts cannot produce unbounded retained memory.
- [ ] **Control-vector imports:** Validate complete tensor payload ranges before native reads, with overflow-safe arithmetic and regression coverage for truncated/out-of-file payloads.
- [ ] **Remaining coverage:** Review the unaudited native/SYCL and installer sections and perform an installed-dependency advisory check.
- [ ] **Fork ownership:** Review upstream project links in `SECURITY.md`, installer release URLs, and contribution guidance before establishing this fork's own release and reporting process. Upstream references should remain where they intentionally identify dependencies.

### Known Deployment Context

The project owner calls the intended PC **Jerry**. Jerry is connected to the current PC through a LAN cable. The owner wants to improve this project and customize it for Jerry.

The owner supplied the inventory on October 11, 2026. Read-only SSH inspection and the repository's `python3 setup.py --check --gpus 0,1` subsequently confirmed the OS, CPU, both GPUs, RAM, storage and driver. No model inference benchmark has been run.

### Jerry Hardware Inventory

| Component | Owner-supplied inventory |
|---|---|
| Operating system | Ubuntu 26.04.1 LTS, x86_64 |
| CPU | AMD FX-8350 Eight-Core Processor; 8 cores / 8 threads |
| GPUs | Two NVIDIA GeForce RTX 3060 cards |
| VRAM | **2 × 12 GiB**; separate memory on each GPU |
| System RAM | 30 GiB usable; approximately 28 GiB available when reported |
| Main filesystem (`/`) | 457 GiB total; 195 GiB available when reported |
| Swap | 8 GiB; unused when reported |

**VRAM accounting:** Jerry has two independent 12 GiB GPU memory pools. We must budget allocations per card. Multi-GPU layer splitting may distribute work and data between them, but we must not represent the system as one GPU with a single shared 24 GiB pool.

Available RAM, free disk space and swap usage are point-in-time observations and should be checked again before installation or a benchmark.

### Configuration Decisions and Verification

| Item | Status or next check |
|---|---|
| CPU instruction set | Verified AVX and SSE4.2; no AVX2. Setup selects the experimental AVX build |
| NVIDIA driver and CUDA compatibility | Driver 595.99.02; CUDA compiler 13.1.115 present. Native build compatibility still needs validation |
| GPU topology and PCIe links | Both GPUs report x16 links with board-limited PCIe Gen2 capability; inter-GPU topology PHB. Measure transfer speed under load |
| Model host versus client roles | Confirmed: this PC hosts Harry, the Hermes agent; Jerry hosts model inference over LAN |
| LAN addressing and reachable hostname | Existing SSH alias `jerry` works; Ethernet reports 1 Gb/s. Actual inference endpoint remains to be configured |
| Storage type and sustained read speed | Main filesystem is on a Samsung 860 EVO SATA SSD. Sustained read speed not benchmarked |
| Model variant and context size | Setup check selects Coder IQ1_M in low-RAM mode; context and per-card split remain benchmark decisions |
| Backend | CUDA, compute capability 8.6 on both cards; experimental CPU-compatible source build required |

The repository's [installation guide](../INSTALL.md#older-cpus-experimental) says its ready-made engine requires AVX2 and describes an experimental locally compiled path for older CPUs. Its [Docker entrypoint](../../docker-entrypoint.sh) also warns about AVX2 requirements. Jerry's flags and setup check now confirm that AVX2 is absent. Setup explicitly selects `STRATA_ISA_FLOOR=avx` and requires a source build. The GPUs do not remove the engine's CPU requirements.

The [multi-GPU guide](../MULTI_GPU.md) describes layer splitting across cards. We should validate Jerry's actual topology and budget each 12 GiB card separately before choosing split settings. No throughput, model-fit or stability claim is made for Jerry.

Read-only SSH inspection and a hardware-only setup check have been performed. A checkout of this fork at the reviewed commit is prepared on Jerry. No model download, engine build, inference measurement, firewall change, or service configuration has been performed. The confirmed topology is **Harry/Hermes on this PC → Jerry's model API over LAN**. The priority is balanced coding/tool quality and speed. See the [Jerry performance plan](../JERRY_PERFORMANCE_PLAN.md) for the candidate profile and benchmark criteria.

### Acceptance Criteria for LAN Readiness

- [x] Confirm host/client roles: Harry/Hermes runs on this PC; Jerry hosts inference.
- [ ] Configure and verify the effective authenticated inference endpoint.
- [ ] Resolve the review findings or document specific enforced mitigations with validation evidence.
- [ ] Verify unauthorized API calls are rejected and authorized inference succeeds.
- [ ] Restrict network reachability to the intended LAN peers; record the effective firewall or ingress configuration.
- [ ] Keep keys and machine-specific run configuration outside source control. Share non-secret configuration templates only.
- [ ] Select hardware settings from inventory and measured runs; document throughput, memory use and stability rather than assuming them.
- [ ] Verify that configuration persists correctly after restart, including when a Docker data volume is reused.

This section is a planning baseline. It does not claim that Jerry is configured or ready for deployment.
