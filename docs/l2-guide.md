# L2 Session Persistence Guide

## Prerequisites

- Linux (the file I/O layer uses POSIX APIs)
- LZ4 development package (`liblz4-dev` on Debian/Ubuntu)
- Strata built from source with conversation tests enabled:
  ```bash
  cmake -S . -B build -DSTRATA_BUILD_CONVERSATION_TESTS=ON
  cmake --build build --target conversation_file_test conversation_cache_l2_test
  ctest --test-dir build -R conversation --output-on-failure
  ```

## Server configuration

Enable session files and the v2 format:

```json
{
  "slot_save_path": "/path/to/private/sessions",
  "prompt_cache": 64,
  "parallel": 1
}
```

Environment variables:

```bash
export STRATA_SESSION_V2=1          # Enable v2 format writes
export STRATA_SESSION_BLK=1048576   # Block size (default 1 MiB)
export STRATA_SESSION_CHK=8         # Checkpoint grid size
export STRATA_SESSION_CHK_HEAD=3    # Head anchors
export STRATA_V2_READAHEAD=0        # Serial reads (increase to 2-4 for NVMe)
```

**Important:** `--layer-split`, `--peer-device`, and `--batch` are not
supported. Use a single-slot configuration.

## Quick start

### 1. Save a conversation

```bash
curl -X POST 'http://127.0.0.1:8080/slots/0?action=saveinc' \
  -H 'Content-Type: application/json' \
  -d '{"filename": "my-session.bin"}'
```

Response:
```json
{
  "n_saved": 150000,
  "n_written": 1073741824,
  "incremental": true,
  "n_reused": 7,
  "n_kept": 8,
  "timings": {"save_ms": 450.2}
}
```

If the file doesn't exist or the conversation diverged, this falls back
to a full save automatically.

### 2. Check cache availability (probe)

```bash
curl -X POST 'http://127.0.0.1:8080/v1/cache/probe' \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Continue our conversation."}
    ]
  }'
```

Response:
```json
{
  "reusable_tokens": 48000,
  "pool_entries": 2,
  "live_lcp": 48000,
  "prompt_tokens": 48120,
  "parked_best": 48000,
  "parked_serial": 3
}
```

### 3. Preload as PARKED

```bash
curl -X POST 'http://127.0.0.1:8080/v1/cache/preload' \
  -H 'Content-Type: application/json' \
  -d '{
    "filename": "my-session.bin",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Continue our conversation."}
    ],
    "pin": true
  }'
```

This reads the session file into the PARKED cache **without** replacing
the active session. The next normal request will automatically borrow
from it. If a matching checkpoint prefix is found, only that prefix is
read (`partial: true`).

### 4. Send a normal request

After preloading, send requests normally through `/v1/chat/completions`.
The engine's existing cache logic takes or borrows from the PARKED entry.

### 5. Clean up (unpin)

```bash
curl -X POST 'http://127.0.0.1:8080/v1/cache/unpin' \
  -H 'Content-Type: application/json' \
  -d '{"serial": 3}'
```

## API reference

### Session file operations

| Action | Endpoint | Description |
|--------|----------|-------------|
| Full save | `POST /slots/0?action=save` | Write whole session |
| Incremental save | `POST /slots/0?action=saveinc` | Reuse unchanged segments |
| Restore active | `POST /slots/0?action=restore` | Replace active session |
| Restore parked | `POST /slots/0?action=restorep` | Read into PARKED cache |
| Verify | `POST /slots/0?action=verify` | Structural integrity check |

### Cache operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/cache/probe` | Read-only: reusable token count |
| POST | `/v1/cache/preload` | Read session file as PARKED |
| GET | `/v1/cache/stats` | Pool stats (budget, entries, bytes) |
| GET | `/v1/cache/list` | Per-entry inventory |
| POST | `/v1/cache/pin` | Protect entry from eviction |
| POST | `/v1/cache/unpin` | Release pin |
| POST | `/v1/cache/drop` | Remove entry by serial |
| POST | `/v1/cache/touch` | Move entry to newest |

## Configuration options

| Variable | Default | Description |
|----------|---------|-------------|
| `STRATA_SESSION_V2` | unset (off) | `1` enables v2 format writes |
| `STRATA_SESSION_BLK` | 1048576 | Block size in bytes (64 KiB – 16 MiB) |
| `STRATA_SESSION_CHK` | 1 | Checkpoint grid size (>1 enables grid) |
| `STRATA_SESSION_CHK_HEAD` | 3 | Head anchor count (with grid) |
| `STRATA_V2_READAHEAD` | 0 | Parallel block readahead workers |
| `STRATA_L2_STATE_DIR` | unset | Mini-state directory (optional) |
| `STRATA_L2_STATE_SINK_TOKENS` | 262144 | Mini-state sink threshold (0 = aligned) |
| `STRATA_L2_WRITER_BUDGET_MIB` | 2048 | Async writer queue budget (0 = sync) |
| `STRATA_L2_WRITER_JOIN_MS` | 5000 | Same-file pending wait limit |

## Troubleshooting

| Error | Meaning |
|-------|---------|
| 501 | Session files not enabled or unsupported configuration |
| 400 | Invalid filename, identity mismatch, or corrupt file |
| 404 | File not found |
| 503 | Insufficient RAM or model not loaded |
| 507 | Disk or PARKED budget exhausted |

### Common issues

**Old files refused after upgrade:** Expected. Configuration fingerprint
changes when engine version or settings change. Generate new snapshots.

**Restore slower than cold prefill:** Possible for short conversations
or slow storage. Use `probe` to check expected reuse before preloading.

**`partial: false` on preload:** No matching checkpoint prefix was found.
The engine fell back to a full-file read. Check that the prompt shares
a sufficient prefix with the saved conversation.

**Writer pending never reaches 0:** Check `/slots` for commit status.
A failed generation stays in the failed state; the next save will retry
as a full save.

## Security notes

- Keep `slot_save_path` private (mode 0700). Session files contain
  model state and should be treated as sensitive.
- Bind the server to localhost unless you need remote access.
- The reference controller has no authentication. Do not expose it
  to untrusted networks.
- File hashes detect corruption, not malicious tampering. Only restore
  files written by your own engine.

## Running the benchmark

```bash
# Terminal 1: start the engine with session persistence
python3 serve/server.py --config my-config.json

# Terminal 2: start the reference controller
STRATA_ENGINE_URL=http://127.0.0.1:8080 python3 l2/reference_controller.py

# Terminal 3: run the rotation benchmark
python3 l2/bench_rotation.py --url http://127.0.0.1:8091 \
  --restart-cmd "systemctl restart my-engine"
```

The benchmark reports per-request reuse rates, latency percentiles,
and failure count across build/rotate/fork/restart phases.
