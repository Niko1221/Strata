# NVMe KV cache in the web interface — design

**Status: design only. Nothing here is implemented.** This is the C11 item of
`docs/nvme-kv-cache-design.md` §7 / §9.2 ("Metrics: the three failure classes, promotes, refusals and dump results
are stderr-only; `serve/telemetry.py` parses nothing NVMe-related") turned into a buildable design, plus the
web page that consumes it.

The tier itself is settled and merged (`--kv-nvme`, `--kv-delta`); this document is about **seeing it** while it
runs: what a page may show, where each fact actually lives, what the engine has to start saying, and what the page
must not claim.

---

## 1. The questions the page answers

The user-facing questions, and the honest one-word answers available today:

| question | today | after this design |
|---|---|---|
| is the cache on, and in which mode? | only in the config's `args` | a tab that exists only when it is on |
| how much cache space is used / available? | one stderr line at startup | gauge + per-class bytes + free space on that filesystem |
| how many sessions are warm? | nothing | two different numbers, named separately (§6) |
| did this request promote, and from which tier? | `DONE`'s `reused` conflates RAM and disk | a `src=` per request, and a Monitor column |
| what did the cascade cost this turn? | nothing measured | ms and bytes written, per turn and cumulative |
| evictions / GC? | one stderr line for a sweep | counted events, with bytes |
| refusals and transfer failures? | stderr prose only | the three §5 classes, as UI states |
| is the tier inert? | a once-per-process stderr line | a badge naming the reason |

**Design rule inherited from the tier:** a plain engine runs nothing new. The Cache tab does not exist in the DOM
until the server knows the tier is configured; the Monitor tab is otherwise unchanged.

---

## 2. Where each fact lives (three sources, three costs)

### 2.1 Config facts — already in the serve process, free

`serve/server.py:469 engine_args()` builds the engine's command line from `cfg["args"]`, so the serve process
already knows `--kv-nvme DIR`, `--kv-nvme-max GB` and `--kv-delta N` without asking the engine anything. This
gives: enabled, store directory, byte cap, tier family (v3-only or delta), and — from `gpu_list(cfg)` /
`--layer-split` — whether the tier is **inert by §3** before a single request runs.

No engine change. This is also the only source that survives the engine being dead, which matters: §5.2 makes a
transfer failure kill the engine, and the page must still explain what happened.

### 2.2 Store facts — the serve process reads the directory

The store is a plain directory the serve process can walk itself:

```
<dir>/kv-<pid>-<seq>.bin                 v3 snapshots
<dir>/delta/log-<pid>-<seq>.manifest     delta conversation heads
<dir>/delta/chunks/<16-hex>.bin          sealed chunks
<dir>/delta/states/<16-hex>.bin          per-turn State records
<dir>/delta/**/.tmp-<pid>-<seq>          crash residue: the scan ignores the class, the sweep reclaims it
```

(The `.tmp-` prefix is the delta tier's own convention — `kv_delta.cpp:116-145` writes every record to
`.tmp-<pid>-<seq>` and renames it into its content-addressed name, so a delta record under its real name is
complete. A **v3 snapshot has no temp name**: `kv_nvme.cpp:730` writes `kv-<pid>-<seq>.bin` directly, so a torn
v3 file is caught by its footer at promote time, not by its filename. The scan must therefore count `.tmp-*`
residue as delta-only, and must not assume a v3 file on disk is a valid one.)

From one walk (106 snapshots + ~650 delta records is nothing; the production store is 106 files / 199 GiB) the
serve side gets: counts and bytes per class, newest and oldest mtime (the age of the warmest and the coldest
stored prefix), `.tmp` residue, and `shutil.disk_usage(dir)` — free space on the filesystem the store lives on,
which the engine never reports and which is the actual "how much cache space is available" answer (§ cap is a
policy, free space is physics).

Reading 8 bytes of each `kv-*.bin` / `log-*` header (magic + version) also gives the §5.3 stale-store fact
directly — "N snapshots of format version 2 in DIR" — as a number on the page instead of a startup sentence.

**The divergence must be shown, not hidden.** A disk scan sees files the engine's `open()` refused and skipped
(stale version, foreign geometry, foreign weight fingerprint). Those files are on disk and are **not** promotable.
The page therefore reports two different quantities and names them: *on disk* and *promotable by this engine*.
Collapsing them into "cache size" would be the exact ambiguity §5.3 says it must not be.

### 2.3 Event facts — only the engine knows, so the engine must say them

Promotes, dump cost, refusals, evictions and sweep bytes exist only inside the engine's serve loop
(`src/program/generate.cpp:3761` promote, `:4208` cascade, `:4237` cap) and inside the two stores
(`src/platform/kv_nvme.cpp:789 enforce_cap`, `src/platform/kv_delta.cpp:1113 sweep`, `:1155 kv_delta_enforce_cap`).
Today they are prose on stderr. A log reader is what §7 calls "the only way to tell the failure classes apart";
this design replaces it with a protocol line.

**Why not extend `DONE`.** `DONE` is parsed positionally (`serve/server.py:234 _parse_done`, and the oracle
scripts read its fields), and its field order is already a compatibility surface. A separate line is additive:
`StrataEngine.generate()`'s read loop (`serve/server.py:322-339`) ignores anything that is not `T`/`PP`/`DONE`/`ERR`,
so an **older server skips the new line and keeps working**, and a newer server against an older engine simply gets
no cache events. That is the same forward-compatibility trick `INFO` already uses (`generate.cpp:3430`,
"what the server's Monitor tab shows (servers before 0.1.8 skip unknown lines until READY)").

---

## 3. The engine protocol addition: the `KV` line

One line per request, printed **after** the cascade and **before** `DONE` (so it can carry the store's totals
straight after this turn's write), only when `have_kvstore`:

```
KV src=delta resume=4107 promote_ms=1840 promote_bytes=1010893312 staging_bytes=1010893312 dump_ms=412 \
   dump_bytes=1184923648 evict=1 evict_bytes=154000384 sweep=0 sweep_bytes=0 refused=0 transfer=0 \
   entries=104 entries_bytes=213674598400 delta_entries=1 delta_bytes=1181116416 cap=107374182400 \
   checkpoints=3 live=4131 total_dump_bytes=… total_promote_bytes=… total_refused=… total_transfer=… total_evict_bytes=…
```

plus one startup line, `KV start=1 entries=… entries_bytes=… delta_entries=… delta_bytes=… cap=…`, printed after
`READY` (the store scan is a real cost and `READY` must not wait for it) and read as **store state**, not a
request event.

Space-separated `key=value`, no value containing a space — the `INFO` line's own convention, so the same parser
shape works. **No paths in the line**: the serve side already knows the directory (§2.1), and a path is the one
value that breaks a whitespace-split parser.

Fields, and where each one comes from:

| field | meaning | source in the current code |
|---|---|---|
| `src` | `none` (cold re-prefill) / `ram` (live session or a RAM checkpoint) / `nvme` (v3 snapshot) / `delta` (manifest) | the serve loop already computes exactly this: `from_live`, `from_nvme`, `best->kind` (`generate.cpp:3737`, `:3815`) |
| `resume` | tokens this request did not read | already `DONE`'s field 8; repeated here so the cache record is self-contained |
| `promote_ms` | wall time of the `restore()` call | wrap `deltastore.restore` / `kvstore.restore` (`:3776-3778`) |
| `promote_bytes` | bytes the tier read: `NvmeEntry::bytes` for v3; manifest + chunks + State for delta | `NvmeEntry::bytes` exists for both; the delta store already sums them for `total_` |
| `dump_ms` | the cascade's server occupancy at `DONE` — after the reply streamed, before the next request | the `dump()` calls **and** the cap call (`:4227-4237`); the device sync that opens the block is outside the window |
| `dump_bytes` | bytes this turn's cascade wrote (delta: new chunks + State + manifest; v3: the snapshot written, 0 when the dump was an idempotent skip) | the delta writer already sizes each record (`delta_chunk_payload_bytes`, `delta_state_payload_bytes`); the v3 path takes the file size it just wrote |
| `evict` / `evict_bytes` | entries the LRU cap dropped this turn | `kv_nvme.cpp:789` and `kv_delta.cpp:1155` both know; today they say nothing |
| `sweep_bytes` | orphan records the sweep reclaimed | `kv_delta.cpp:1113` already computes `bytes` and prints it; it just has to return it |
| `refused` | `invalid` promotes this request (§5.1) | `generate.cpp:3806` |
| `transfer` | `transfer_failed` this request (§5.2) — the line is printed **before** the `ERR` line and the `return 1`, so the server sees the class before the engine dies | `generate.cpp:3791` |
| `entries` / `entries_bytes` | the v3 store's promotable entries and bytes, straight after this turn's cascade | `KvNvmeStore::size()` / `total_bytes()` |
| `delta_entries` / `delta_bytes` | same for the delta tier | `KvDeltaStore::size()` / `total_bytes()` |
| `cap` | the byte cap in force (0 = unlimited) | `o.kv_nvme_max_gb` |
| `checkpoints` | RAM-tier checkpoints alive at the end of the request | already printed in the per-request stderr line (`checks.size()`) |
| `live` | tokens the live session holds | `live.size()` |
| `staging_bytes` | the whole-file buffer the restore staged (the accepted C10 cost) | v3: the snapshot's bytes; delta: the size of the v3 image it assembled (`delta_restore`'s buffer, surfaced by `KvDeltaStore::last_image_bytes()`) — a different number from the entry's byte count. 0 when nothing was promoted or the tier refused before assembling |

Two implementation notes the design has to state, because they are where this would otherwise go wrong:

1. **Capture it on the pump thread, not in the request loop.** The request loop's early-stop drain discards
   everything that is not `DONE` (`serve/server.py:340-355`), so a cancelled request's cascade - which really
   happened - would lose its line. Reading `KV` on `_pump` instead (`server.py:190`) solves that and also handles
   the startup line, which the engine prints *after* `READY` (the stores open at `generate.cpp:3471`, `READY` is
   `:3468`): `_pump` sees both in order and routes them - store state vs this request's event. "No `KV` line for a
   request that ran" is still *unknown*, never *cold*.
2. **Cumulative totals ride the same line.** Add `total_dump_bytes`, `total_promote_bytes`, `total_refused`,
   `total_transfer`, `total_evict_bytes` to the line (the engine keeps them in one struct for the life of the
   process). The serve side then takes the **last seen** cumulative values as its totals, so a line it could not
   parse costs one event row, never the running totals. This is the difference between a dashboard and a lie.

The engine-side counters are a small struct (`TierCounters`) beside the two stores at `generate.cpp:3198`, filled
by the serve loop and by the two stores' cap/sweep paths. The stores' own APIs grow minimally: `enforce_cap` and
`sweep` return their counts and bytes **as well as** printing (the stderr lines stay - they are the operator's
log, and `tools/short_tests.py` asserts on them).

---

## 4. serve side: `serve/kvcache.py`

A new module beside `telemetry.py`, same shape and same rule ("nothing here can stop the server").

```
class KvCache:
    def __init__(self, engine_args: list[str], log_path: str | None)
        # parses --kv-nvme / --kv-nvme-max / --kv-delta; --layer-split => inert reason
        # enabled=False when there is no --kv-nvme: every method becomes a no-op
    def observe(self, kv: dict, done: dict)      # one request's KV line + its DONE
    def store_state(self, kv: dict)              # the engine's store fields, incl. the start=1 line
    def scan(self, force=False) -> dict          # the directory walk, throttled to one per ~5 s
    def summary(self) -> dict                    # cheap: goes in /metrics every second
    def detail(self) -> dict                     # expensive: goes in /cache only
```

State it keeps:

- `events`: `deque(maxlen=500)` — the same shape as `Service.history` (`server.py:546`), newest last, each entry
  `{time, kind, src, tokens, bytes, ms, finish}` where `kind ∈ promote, dump, refuse, transfer, evict, sweep`.
- `totals`: cumulative, taken from the engine's cumulative fields (§3 note 2), plus serve-side ones the engine
  cannot know (turns served, turns with a `KV` line, engine restarts that followed a transfer failure —
  `Service.run` already notices a dead engine at `server.py:732-737`).
- `series`: `deque(maxlen=60)` per sparkline, sampled once a second from the telemetry thread (see §5): store
  bytes, per-turn write MB, per-promote read MB, warm count.
- `scan` result cached with its timestamp; `detail()` re-walks only when it is stale.

Wiring (three small touches to `server.py`, no restructuring):

1. `main()` builds `svc.cache = KvCache(engine_args(cfg), cfg.get("log"))` next to where `svc.gpu_index` is set
   (`server.py:1552`), and `StrataEngine` gets a back-reference so a `KV` line can reach it.
2. `StrataEngine._pump` (`server.py:190`) reads `KV` lines off the engine's stdout and routes them (store state
   vs this request's event, §3 note 1); they never enter the request's line queue, and `generate()` is otherwise
   untouched.
3. `Service.run`'s `finally` block (the one that appends the history record at `server.py:791`) adds `"cache": ...`,
   `svc.cache.observe(engine.last_kv, last)`, and adds the disk-tier resume tokens to `self.totals`
   (`reused_from_disk`), so the existing totals line can say how much of the reuse came off NVMe.
4. `Telemetry(extra=...)` gains the cache series (`server.py:590`), so the sparkline sampling and the hardware
   sampling share one thread and one clock — the Monitor already does this for `tok_s`.

`/metrics` gains a `cache` key from `svc.cache.summary()` (`server.py:604`), and `do_GET` gains one route
(`server.py:1121`):

```
GET /cache     -> 200 {enabled, inert_reason, dir, cap_bytes, mode, disk_free_bytes,
                       on_disk{...}, promotable{...}, entries[...], events[...], totals, series,
                       stale{count, version}, warnings[...]}
```

`/cache` is behind the same `_authorized()` gate as `/metrics` (`server.py:1111`). It is read-only: **no POST
route, no "clear the store", no "forget this conversation"** in v1. The serve protocol has no command for it
(`generate.cpp:3544-3552` accepts only `QUIT`/`STOP`/`GEN`/`GENI`), and a store mutation from the web layer would
have to re-derive the tier's whole drop/sweep contract from the wrong side of the pipe. That is a separate
decision with its own design, not a button.

---

## 5. The page

A fourth tab, `Cache`, between Monitor and About, built from components that already exist
(`st-card`, `metric-card` + sparkline, `st-gauge`, `st-progress`, `st-table`, `st-badge`, `dl.facts`), so it looks
like the Monitor tab and costs no new CSS beyond a few rows.

```
┌ Cache ──────────────────────────────────────────────────────────────────────────────┐
│ ● NVMe cache on · delta tier · /local/strata/kvstore · cap 100 GB      [inert: …]  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ ┌ Store used ┐ ┌ Written/turn ┐ ┌ Read/promote ┐ ┌ Warm prefixes ┐ ┌ Promoted ┐    │
│ │ 199.3/100GB│ │ 115.8 MB     │ │ 964 MB        │ │ 106           │ │ 8 of 12  │    │
│ │ ▁▂▃▄▅ spark│ │ ▁▁▂▁ spark   │ │ ▂▁▃ spark     │ │ ▃▃▃ spark     │ │ turns    │    │
│ └────────────┘ └──────────────┘ └───────────────┘ └───────────────┘ └──────────┘    │
├───────────────────────────────────┬─────────────────────────────────────────────────┤
│ Cache state                       │ Cache events                                    │
│  ◔ store fill 72% of cap          │ Time  Event          Tier   Tokens   MB     ms  │
│  ▸ v3 snapshots   104 · 198.7 GB  │ 14:22 promote        delta   4,107   964  1,840 │
│  ▸ delta chunks    644 · 187 GB   │ 14:22 cascade        delta   217 new 116    412 │
│  ▸ delta states    106 · 11.9 GB  │ 14:19 cascade        delta   1,480 new 118    398│
│  ▸ manifests       106 · 0.3 GB   │ 13:58 evict (LRU)    v3      –       147     –  │
│  ▸ disk free on /local  412 GB    │ 13:40 refused → re-read  v3  –        –      –  │
│  ▸ RAM tier: 3 checkpoints, 4,131 │ 09:14 TRANSFER FAILURE → engine stopped (… )   │
│    tokens live                    │                                    Show all (87) │
│  ▸ promote staging: 964 MB held   │                                                │
│    at once (~2 GB RAM transient)  │                                                │
├───────────────────────────────────┴─────────────────────────────────────────────────┤
│ Stored prefixes (what is promotable right now)                                      │
│ Age      Tokens   Tier    Records        Bytes      Last used                      │
│ 3 min    41,301   delta   26 chunks + 1  1.62 GB    this session                   │
│ 1 h      58,513   v3      1 snapshot     0.94 GB    another conversation           │
│ 2 d      12,110   v3      1 snapshot     0.19 GB    …                              │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Store directory  /local/strata/kvstore                                             │
│ Format version   3 (this build) · 0 files of another version on disk              │
│ Chunk BLOCK      lcm(page_size, idx_block) = 1,024 tokens                         │
│ Weight set       7f3ac1… (delta tier only)                                        │
│ Not cached       layer-split sessions · boundaries past the drafter ring          │
└───────────────────────────────────────────────────────────────────────────────────┘
```

Polling: the tab renders from the `cache` block already inside `/metrics` (1 s, no extra request), and fetches
`/cache` only while the tab is visible, every 2 s. The directory walk therefore costs nothing when nobody is
looking — which is the point of splitting `summary()` from `detail()`.

Tab visibility: `render()` shows/hides the tab button from `m.cache.enabled` (`app.js:173`). A server whose engine
has no `--kv-nvme` renders exactly today's three tabs.

Two smaller additions, both cheap and both high-value:

- **Monitor's request table gains one column**, "Cache", showing `src` as a badge (`RAM` / `NVMe` / `disk` /
  `cold`). Today the table's `Reused` column cannot tell a 4,107-token RAM resume from a 4,107-token promote,
  which is the single most useful fact the tier produces.
- **About gains a "NVMe cache" card** (the `facts` list): directory, cap, tier family, format version, and the
  two honest limits — "layer-split sessions are not cached" and "a promote stages the whole snapshot in RAM
  (~2× its size)". About is where a user reads the caveats; the Cache tab is where they watch them.

---

## 6. Metric definitions (so the numbers mean what they say)

**"Warm" is two different things and the page names both.**

- *Promotable prefixes* = `entries + delta_entries` — stored prefixes a new request could resume from. These are
  **snapshots, not conversations**: §7 of the design doc is explicit that there is no cross-restart conversation
  identity, so one conversation that ran five turns holds five prefixes. The column is labelled "stored prefixes"
  for exactly this reason; calling it "sessions" would overstate it.
- *RAM tier* = `checkpoints` + `live` — what the running engine can resume without touching disk.

**Store fill** = `(entries_bytes + delta_bytes) / cap`, with the *disk* figure beside it: `disk_free_bytes` from
`shutil.disk_usage(dir)`. The cap is a policy the user set; free space is what actually stops the tier. Showing
only the cap would let a 100 GB cap on a 120 GB volume look healthy at 99 GB.

**The engine's byte totals are cap accounting, not a footprint** (found while building step 1). The delta store's
`total_` sums every `chunks/` + `states/` file once — shared records included — and *then* adds each entry's own
`bytes` (its manifest + its State + its chunks), so a chunk shared by three manifests is counted four times
(`kv_delta.cpp:921-970`, deliberately: "the cap must not lie about the disk"). Consequence for the page: the
engine's `delta_bytes` and the directory walk's bytes are **different quantities measuring different things**, and
the page must label them — *cap accounting* (what the LRU compares against the cap) vs *on disk* (what the volume
actually holds). Reconciling them into one "cache size" number would be wrong in both directions: the accounting
over-counts shared chunks, and the walk includes records the engine refuses to promote.

**Written per turn** = `dump_bytes` (and its mean over the last 60 turns). **Read per promote** = `promote_bytes`.

**Overhead**, stated as three separate costs rather than one number:
- *server occupancy at `DONE`*: `dump_ms` — after the answer streamed, before the next request can start;
- *TTFT cost of a promote*: `promote_ms` — the price paid instead of a re-prefill;
- *RAM transient*: `staging_bytes` (the whole-file buffer, the accepted C10 cost) plus the engine's own RSS from
  telemetry (`psutil` process RSS, a new series), so the ~2 GB transient §6 of the design doc measured is visible
  as a bump rather than a footnote.

**Time saved** is derived, and is labelled as derived:
`saved_s ≈ resume_from_disk / fresh_prefill_tok_s − promote_ms`, where `fresh_prefill_tok_s` is measured from this
server's own cold turns (`prompt_ms` over `prompt_tokens − reused` where `src == none`) — not a constant. Where
there is no cold turn to measure yet, the page shows tokens-not-read only. The design doc's own numbers
(110 s cold → 1.3-1.9 s warm) are the shape of the payoff; the page should compute its own from this machine.

**Endurance** is shown as cumulative bytes written since the engine started, with the honest ratio from §10.2 of
the design doc — `(total_tokens × 16 KB + 118 MB) / (new_tokens × 16 KB + 118 MB)`, i.e. ~6.5× at the production
average turn, **not** the handoff's 33×. The running-state term is not a rounding error and must not be dropped
from an endurance claim.

---

## 7. Failure classes and inert states, as UI states

The three classes of `strata::core::ConversationRestore` map one-to-one onto badge tones, because the page is
where an operator will first look:

| class | badge | what the page says |
|---|---|---|
| `restored` | neutral/green | `promote · delta · 4,107 tokens · 964 MB · 1.84 s` |
| `invalid` | warn | "a stored snapshot was refused (<reason>); the prompt was read instead and that snapshot was deleted. The answer was served normally." |
| `transfer_failed` | danger | "a stored snapshot could not be applied to the GPU. The engine stopped rather than risk continuing on a half-restored session; the next request started it again. The snapshot is still on disk." + the engine's own log line + a link to `docs/nvme-kv-cache-design.md` §5.2 |

The `transfer_failed` row is the one place the page must be careful: the serve side learns of it from the `KV`
line printed before the `ERR` line, then from `EngineDied` and the restart it performs (`server.py:732-737`). The
page says *the engine was restarted*, only after it actually has been — never as a promise.

Inert states, each with its own reason string (all three already exist in the code as stderr lines):

- **layer split active** — detected from the config before any request (`--layer-split` / several GPUs): "stored
  snapshots are neither dumped nor promoted (the envelope carries the primary stage only)".
- **stale store** — N files of another format version on disk, 0 promotable: both options stated, as §5.3 does
  ("re-dump them with the binary that wrote them" / "remove them and let the store rebuild"), because what it
  must not be is an ambiguity.
- **boundary past the drafter ring** — per-turn: the delta tier refused and the v3 whole-snapshot fallback wrote
  it. A warn row in the event table, not an error: it is a documented limit (§5.15), not a fault.

---

## 8. What the page must not show

- **No conversation content.** Entries are keyed by token ids; the page shows lengths, ages, byte sizes and
  record counts — never ids, never decoded text, never image hashes beyond a count. The store directory name is
  shown (the user configured it); individual file names are not, because `kv-<pid>-<seq>.bin` is noise.
- **No cross-process history.** The engine's totals are per process; a restart resets them. The serve side keeps
  its own totals for as long as the server lives and says "since the server started", the same wording
  `renderTotals` already uses (`app.js:190`).
- **No numbers the tier cannot produce.** No hit rate for the KV tier (the tier has no notion of a partial hit —
  a promote is all-or-nothing at a prefix boundary), no "cache misses" (a miss is just a prompt being read, which
  the Monitor already shows), no predicted endurance date.
- **No actions** (§4).

---

## 9. Phased plan, each phase independently shippable and gated

The build order, file by file with signatures, gates and sizes, is `docs/nvme-kv-cache-web-plan.md`. The phases
below are the same work at the level of a design review.

**Phase 1 — the engine says it (C++ only).**
`TierCounters` in the serve loop; `enforce_cap` / `sweep` return `{count, bytes}`; the `KV` line printed before
`DONE`; cumulative totals on the same line. Gates: `kv_nvme_host_test` gains assertions that the stores report
eviction/sweep counts (it already asserts the entries and bytes); `tools/short_tests.py` asserts the `KV` line
appears on every tier-on turn and that its `src` matches the stderr prose it replaces; `tools/nvme_steps123_test.sh`
asserts the eviction turn reports `evict=1`; `tools/nvme_failure_contract_test.sh` asserts `transfer=1` arrives
**before** the `ERR` line and the exit.

**Phase 2 — serve reads it (`serve/kvcache.py`, `server.py`).**
Config parse + `observe` + `scan` + `summary`/`detail` + `GET /cache` + the `cache` block in `/metrics`. Gates in
`serve/test_server.py`: a fake engine emitting `KV` lines (the harness already has `MockEngine`, `server.py:63`);
a scratch store directory with synthetic `kv-*.bin` / `delta/` files, asserting the walk's per-class bytes, the
stale-version count, the `.tmp` residue count, and that `--kv-nvme` absent means `enabled: false` and no `/cache`
data. Also: a test that a `KV` line from an engine the server does not understand cannot break the request path
(an unknown line must still be ignored).

**Phase 3 — the page (`serve/web/index.html`, `app.js`, `app.css`, `sprite.svg`).**
The tab, the five metric cards, the two cards, the two tables, the facts block; the tab hidden when disabled;
`/cache` polled only while visible. Gate: extend the web test class in `test_server.py` (`test_page_and_files`,
`test_metrics`) to assert the tab markup exists only with the tier on, and that `/cache` needs the API key when
one is set.

**Phase 4 — the smaller surfaces and the record.**
The Monitor "Cache" column, the About card, the engine-RSS series in `telemetry.py`, and the design doc's §7
bullet closed (C11 settled: "a log reader is no longer the only way"). `setup.py` gets a `--kv-nvme` / cap / delta
prompt so the tier is reachable without hand-editing `args` — today it is only reachable that way, which is why
the page must handle "configured but invisible" gracefully.

---

## 10. Open decisions (for the maintainer, not settled here)

1. **`KV` line vs a `STATS` request command.** A request/response command (`STATS` → `KVSTAT ...`) would let the
   page ask for store state on demand without waiting for a turn, and would be the only way to get fresh numbers
   while the engine is idle. It costs a stdin protocol addition and a FIFO interaction. This design takes the
   push line because it needs no new stdin path; if the pull command is wanted, the same `TierCounters` serve it,
   so Phase 1 is not wasted either way.
2. **Where the disk scan lives.** serve-side (this design: works while the engine is dead, no engine change, but
   duplicates the tier's own view) vs engine-side (one source of truth, but blind while the engine is restarting —
   which is precisely the §5.2 case the page exists to explain). The design chooses serve-side and *labels* the
   difference ("on disk" vs "promotable by this engine") rather than pretending the two agree.
3. **Per-conversation identity.** The "Stored prefixes" table would be far more useful grouped by conversation,
   and the tier cannot group: §7 lists "a conversation identity" as the missing thing that would enable
   cross-restart supersession. If that identity is ever added, this table is its first consumer.
4. **Store actions.** "Forget this prefix" / "drop the stale store" are the two obvious next buttons; both need a
   serve→engine command and both touch the drop/sweep contract. Explicitly out of v1 (§4, §8).
5. **Multi-engine / split stores.** One serve process, one engine, one store directory is the current shape. If a
   second server ever shares a store, the scan's "one writer per store" assumption (§5.15) breaks and the page
   would need to say which process it is describing.
</content>
