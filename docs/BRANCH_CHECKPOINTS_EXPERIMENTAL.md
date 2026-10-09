# Experimental branch history and execution checkpoints

History determines what can be continued. Checkpoints determine how quickly.
Enable this explicitly with `--experimental-branch-checkpoints` and
`--responses-store-path DIR`. It implies Responses persistence and requires a
serial native engine. The ordinary serving path remains unchanged when disabled.

## Branching and retention

`GET /v1/responses/RESPONSE_ID/history` returns immutable item nodes and their
canonical `before`/`after` branch points. Before a message means its parent;
after a message means that node. Adjacent end/start boundaries are identical.
`token_offset: null` reserves future within-message branching. Generation headers
belong to execution framing, not additional history nodes.

Start a response, then branch before one of its assistant messages:

```python
import requests

api = "http://127.0.0.1:8080/v1"
first = requests.post(api + "/responses", json={
    "input": "Write a Python function that adds two numbers.",
    "max_output_tokens": 128,
}).json()
history = requests.get(api + "/responses/" + first["id"] + "/history").json()
answer = next(n for n in history["nodes"] if n["item"].get("role") == "assistant")
alternative = requests.post(api + "/responses", json={
    "branch_from": {"response_id": first["id"], "node_id": answer["id"], "side": "before"},
    "input": [], "max_output_tokens": 128,
}).json()
```

Use `side: "after"` to include the selected message. Ordinary
`previous_response_id` continuation remains supported. A branch adds new nodes;
it does not edit its source. Input-item pagination respects the selected cut.

Bookmark with `POST /v1/responses/RESPONSE_ID/bookmark {"protected": true}`;
set false to remove protection. Unprotected records expire after 30 days without
successful user continuation. GETs, cache inspection and copying do not renew
activity. An hourly sweep and request-time maintenance remove expired references.
Ancestors required by surviving branches remain internally available. History is
never moved to the checkpoint archive. SQLite can reuse freed pages; deletion is
not a promise of secure erasure or immediate shrinking of its database file.

Recorded tool results and assistant messages are replayed as input, never rerun.
HTTP image references are embedded once before admission. The store also records
the prepared prompt IDs and generated token IDs. Framing settings belong to the
lineage. Legacy records or a changed execution identity require an explicit
`migrate_history: true` continuation, creating a new response/branch; migration
does not claim to reproduce an old unavailable implementation.

## Checkpoint configuration

These JSON configuration options apply only to the experimental path:

| Setting | Default | Meaning |
|---|---:|---|
| `checkpoint_budget_mib` | 32768 | Local checkpoint block budget |
| `history_reserve_mib` | 4096 | Free-space reserve protected from cache writes |
| `checkpoint_max_snapshot_mib` | 4096 | Conservative bound for staging one native session |
| `checkpoint_archive_path` | unset | Existing directory on the optional SSD/RAID pool |
| `checkpoint_archive_budget_mib` | 0 | Pool checkpoint budget; zero disables demotion |
| `checkpoint_chunk_mib` | 0 | Whole-file objects; positive values enable fixed-size shared blocks |
| `checkpoint_policy` | `utility` | Experimental policy; `fifo` and `lru` are comparison controls |
| `responses_store_max_mib` | 1024 | Separate durable history/receipt capacity |

Disable the native `--conversation-cache-disk-mib` FIFO tier for this experiment.
Set native `--session-min-free-mib` at least as high as `history_reserve_mib`.
The native writer preflights the actual session size; the frontend reserves room
for staging and admission before asking it to save. Cache admission can be skipped
without preventing generation. RAM and live-engine reuse remain independent.

Choose an archive directory dedicated to this server, beneath the pool's mounted
filesystem. The manager copies, verifies, flushes, then publishes the destination
before releasing local blocks. Missing/corrupt checkpoint data falls back to
history replay. Archive performance is not assumed to beat prefill.

Native session files remain opaque, preserving the engine's recurrent and MTP
state. Identity includes actual artifact hashes, tokenizer/template, executable,
execution arguments and relevant environment; the native reader also verifies its
own model/config/state-layout identity. Hashing large models increases startup
time. Bookmarks protect conversation history and select the execution checkpoints
eligible to survive restart. They do not force checkpoint admission or prevent
ordinary capacity eviction during runtime. A separate runtime checkpoint pin
does not make an unbookmarked conversation survive the restart cleanup.

Graceful shutdown drains active HTTP handlers, then removes checkpoints without
a bookmarked owner (or a required ancestor of one). Startup repeats this cleanup
under the history owner lock, including staging files left by a crash. Shared
blocks required by retained checkpoints remain. Cache demand statistics and
decision logs reset at this boundary. Authoritative history, replay settings,
execution fingerprints, and bookmark records keep their normal retention rules.
Continuing an unbookmarked chat after restart therefore replays its history.
An unavailable archive is cleaned when it becomes available again.

## Selection and eviction

For each demand position, compare history replay against every compatible saved
prefix, including restore overhead. Replaying a suffix currently uses a linear
per-token estimate learned from successful engine prefill timings; it is an
estimate, not a claim of constant attention cost at long context.

For each feasible deletion set, compute expected additional reconstruction
seconds divided by physical bytes reclaimed. Forecast demand uses a one-day
exponential half-life and a conservative cold prior. Singletons, shared-block
owner sets up to eight snapshots, and bounded unions of dependency groups are
evaluated. This is a heuristic, not an exhaustive optimizer. Recalculate after
every action. In-flight leases and explicit pins are excluded.

When an archive is configured, compare demotion with deletion, accounting for
transfer cost and estimated archive restoration. Successful restores replace
the initial cost estimate. Shared blocks are counted once. An individual deletion
that reclaims no bytes is not considered progress; a feasible group may reclaim
those bytes. Successful-use statistics and decisions apply to the current server
lifetime. Clearing that catalog does not delete authoritative history.

## Validation and current limits

Run `python -m unittest serve.test_branch_checkpoints serve.test_branch_http
serve.test_responses_experimental` and `python tools/bench_checkpoint_policy.py
--output policy.json`.

The fixed-seed policy comparison uses three equally sized checkpoint slots and
modeled reconstruction costs. It exercises the actual catalog and filesystem:

| Trace | FIFO seconds | LRU seconds | Utility seconds |
|---|---:|---:|---:|
| Hot prefix plus scan | 146.9 | 47.9 | 47.9 |
| Demand shift | 186.6 | 67.8 | 67.8 |
| Held-out mixed requests | 229.2 | 208.5 | 79.8 |

These are simulation results, not GPU throughput or measured user latency.
Tests cover an exhaustive two-snapshot dependency example, zero-byte individual
eviction, incompatible fingerprints, reserve pressure, corruption, interrupted
archive copying, expiration, protected ancestry, HTTP boundaries and restart.

The experimental bridge indexes the actual live token IDs and retained internal
checkpoint boundaries from native session format v1. In particular, MTP may commit
more tokens than the HTTP stream emits; those IDs are read from the saved metadata,
not guessed from the response. Older boundaries omitted by native SAVE are not
indexed as if their recurrent state had been retained. A requested earlier boundary is always replayable, but may need more
prefill until a matching checkpoint has been created. Physical block sharing is
optional; its benefit depends on the native session's byte layout.

The [plain-server 32K probe](measurements/branch-checkpoints-32k/README.md)
includes three fresh/reuse samples and one restart/deletion example each, with
raw responses and a chart. Hardware calibration of a real RAID tier and repeated
latency distributions remain separate validation work. Do not infer p95/p99 or a
universal speedup from these samples. Keep this experimental until larger workloads
and broader storage-pressure traces have been measured.
