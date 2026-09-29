# NVMe KV cache web page — implementation plan

**Status: plan. No step is done.** Companion to `docs/nvme-kv-cache-web-design.md` (the design; read it first —
this file is the build order, not a restatement of it).

Eight steps, each one a commit-sized change that compiles, passes its own gate, and can be reviewed alone. Steps
1-2 are C++ (engine), 3-5 are Python (serve), 6-7 are the web app, 8 is setup + docs. Only steps 2 and the Phase-1
gates need a GPU; everything else is host-only.

```
1 store counters ──┐
                   ├─ 2 KV line (engine) ──┐
                                           ├─ 4 server wiring ── 5 /cache ── 6 Cache tab ── 7 Monitor/About
                       3 serve/kvcache.py ─┘                                   └─ 8 setup + docs
```

---

## Step 1 — the two stores report what they did (C++, no behavior change)

Today `enforce_cap` and `sweep` know their counts and print prose; `KvDeltaStore::dump` already computes this
turn's write bytes (`appended`, `src/platform/kv_delta.cpp:1036-1048`) and prints them at `:1075` without ever
returning them. Step 1 makes
those facts returnable without changing a single decision the tiers make.

**`include/strata/platform/kv_nvme.hpp`** — one small struct, shared by both tiers (it lives here because
`kv_delta.hpp` already includes this header):

```cpp
/// WHAT A TIER DID, reportable without a log line (docs/nvme-kv-cache-web-design.md §3).
/// Every field is a fact the tier already knew and only ever printed.
struct TierActivity {
    bool skipped = false;          // an idempotent dump: nothing was written, recency refreshed
    uint64_t written = 0;          // bytes this call wrote (delta: only the chunks THIS turn sealed)
    int64_t dropped = 0;           // entries this call superseded
    uint64_t dropped_bytes = 0;
    int64_t evicted = 0;           // LRU cap evictions caused by this call
    uint64_t evicted_bytes = 0;
    int64_t swept = 0;             // orphan records reclaimed (delta only)
    uint64_t swept_bytes = 0;
};
```

**`KvNvmeStore`** (`kv_nvme.hpp` / `kv_nvme.cpp`):
- `void enforce_cap()` → `TierActivity enforce_cap()` (`kv_nvme.cpp:789`): counts the entries it removes. Its
  behavior — including "the last entry is kept even over the cap" — is unchanged.
- `dump(...)` gains a trailing defaulted out-param, `TierActivity* act = nullptr`: `written = e.bytes`
  (`:744`), `skipped = true` on the exact-match skip (`:704-712`), `dropped/dropped_bytes` from the supersede
  (`:712-731`), plus the `enforce_cap()` result it already calls at `:750`. Defaulted, so every existing caller
  and fixture compiles untouched.

**`KvDeltaStore`** (`kv_delta.hpp` / `kv_delta.cpp`):
- `void sweep()` → `TierActivity sweep()` (`:1113`): it already computes `swept` and `bytes`; return them, keep
  the stderr line.
- `dump(..., TierActivity* act = nullptr)`: `written = appended` (the value it already sums at `:1036-1048` and
  prints at `:1075` as `nvme delta: appended N chunks (X MiB)` — so the counter is checkable against a line the
  oracles can already grep), `skipped` on the exact-match head refresh
  (`:991-998`), `dropped` on a real supersede only (`:1053-1062`) — the same "only on a real supersede" rule the
  byte bookkeeping already applies, so a fork never reports a drop it did not make.
- `void kv_delta_enforce_cap(...)` → `TierActivity kv_delta_enforce_cap(...)` (`:1155`): evictions from both
  entry lists plus the sweep's numbers.

**Gate** (`src/platform/kv_nvme_host_test.cpp`, `src/platform/kv_delta_host_test.cpp`, ctest under
`STRATA_BUILD_CONVERSATION_TESTS=ON`, `CMakeLists.txt:659-676`):
- the existing cap-eviction cases now assert `evicted == 1` and `evicted_bytes == the evicted entry's bytes`;
- the existing sweep case asserts `swept`/`swept_bytes` equal the orphan count the fixture created;
- a delta dump after a fork asserts `dropped == 0` (the P2-2 rule, asserted on the activity struct too, not just
  on the byte totals);
- an idempotent re-dump asserts `skipped && written == 0`;
- the byte-identity oracle (`§5.2`'s `cmp`) still passes — this step must not move one payload byte.

~120 lines, no format change, no serve-loop change. **Risk: none** — every decision is unchanged, only reported.

---

## Step 2 — the serve loop's counters and the `KV` line (C++)

**`src/program/generate.cpp`**, beside the two stores (`:3198-3200`):

```cpp
strata::platform::TierCounters tc;   // per-request facts + the process's cumulative totals
```

`TierCounters` (in `kv_nvme.hpp`) is the per-request fields of design §3 plus `total_*` cumulative fields, so a
line the server never saw cannot corrupt the totals.

Fill it at the three places that already know:

| where | anchor | what to add |
|---|---|---|
| promote | `:3761-3825` | `src` from `from_live` / `from_nvme` / `best->kind` (the three values already exist at `:3737`, `:3815`); `promote_ms` = `steady_clock` around the `restore()` call at `:3776-3778`; `promote_bytes = best->bytes`; `staging_bytes` = the same for v3, and for delta the assembled image size (`delta_restore`'s `buf`, `kv_delta.cpp:779-872` — add a trailing defaulted `uint64_t* image_bytes = nullptr` out-param there and surface it through `KvDeltaStore::restore` / `last_image_bytes()`); `refused` at `:3806`; `transfer` at `:3791` |
| cascade | `:4208-4241` | `dump_ms` around the `dump()` calls at `:4227-4231` **and** the cap call at `:4237` (design §6 defines it as the cascade's occupancy, so it is dump + cap; the `cudaDeviceSynchronize()` that opens the block is outside the measured window); the `TierActivity` out-params from step 1; the `kv_delta_enforce_cap` return at `:4237` |
| end of request | `:4244` (before `DONE`) | the `KV` line, only when `have_kvstore` |

The line, printed with `std::printf` + `fflush` **before** `DONE`, space-separated `key=value`, no value
containing a space, **no paths**:

```
KV src=delta resume=4107 promote_ms=1840 promote_bytes=1010893312 staging_bytes=1010893312 dump_ms=412 \
   dump_bytes=1184923648 evict=1 evict_bytes=154000384 sweep=0 sweep_bytes=0 refused=0 transfer=0 \
   entries=104 entries_bytes=213674598400 delta_entries=1 delta_bytes=1181116416 cap=107374182400 \
   checkpoints=3 live=4131 total_dump_bytes=… total_promote_bytes=… total_refused=… total_transfer=… total_evict_bytes=…
```

Two ordering facts this step has to get right:

1. **`transfer=1` must be printed before the `ERR` line and the `return 1`** (`:3791-3799`). The engine is about
   to die; the server's only chance to record which class failed is this line. So the transfer branch prints its
   `KV transfer=1 …` first, then the existing `ERR`, then returns 1.
2. **The startup store scan happens after `READY`** (`READY` at `:3468`, the stores open at `:3471-3496`). Do not
   reorder it — the scan plus the open-time sweep is a real cost and delaying `READY` delays the browser opening.
   Instead, the startup line is `KV start=1 entries=… entries_bytes=… delta_entries=… delta_bytes=… cap=…` and
   the server treats `start=1` as **store state, not a request event** (step 4 handles this in `_pump`, where the
   line is read in order regardless of which request is in flight).

**Gate** (GPU oracles, all already exist and already run with the tier on):
- `tools/nvme_steps123_test.sh`: `KV start=1 entries=` appears once, one `KV src=` line per `DONE` line, and the
  cap turn asserts `evict=[1-9]… evict_bytes=[1-9]`.
- `tools/nvme_failure_contract_test.sh`: the `transfer_failed` case asserts `transfer=1` arrives **before** the
  `ERR` line and before exit 1; the `invalid` case asserts `refused=1`, `transfer=0` and `src=none`.
- **`tools/short_tests.py` cannot be a `KV` gate** — found while executing step 2, and a correction to this plan.
  The server gives the engine's **stderr** to the log file (`server.py:159`) and keeps **stdout** on a pipe, so a
  log-tail grep never sees a `KV` line. The two shell oracles above capture engine stdout directly and are the
  live-server gate; the short suite's cache assertions belong to step 5, where they read the facts off `/metrics`.
- `tools/nvme_delta_p0_test.sh` and `tools/nvme_p0_test.sh`: unchanged verdicts (the line must not change a byte
  or a promote decision — the state-hash oracles are the guard).
- **No-tier regression**: run `tools/needle_bench.py` (tier off) and diff the engine's stdout against the pre-step
  binary: `DONE` and `INFO` byte-identical, no `KV` line.

~150 lines. **Risk: low, but it is the only step that touches the serve loop.** The contract to hold: the line is
printed after the cascade and before `DONE`, and nothing about resume, dump, cap or the failure classes changes.

---

## Step 3 — `serve/kvcache.py` (new module, no wiring yet)

The module mirrors `telemetry.py`'s shape and its rule: *nothing here can stop the server* — every filesystem and
parse path is wrapped, every failure degrades to `None`/`{}`.

```python
def parse_kv(line: str) -> dict          # "KV k=v k=v" -> dict; ints/floats typed, unknown keys kept
                                         # unit-tested against the exact strings step 2 prints

class KvCache:
    def __init__(self, engine_args: list[str], log_path: str | None = None)
        # --kv-nvme DIR / --kv-nvme-max GB / --kv-delta N / --layer-split  (the FINAL args, i.e. engine_args()'s
        # output, so a split added by the server is seen)
        # no --kv-nvme -> self.enabled = False and every method is a no-op returning {} / None
    def store_state(self, kv: dict)      # the start=1 line and every request line's store fields
    def observe(self, kv: dict, done: dict) -> None      # one request: event rows + totals + series
    def scan(self, force: bool = False) -> dict          # the directory walk, throttled to one per SCAN_S = 5 s
    def summary(self) -> dict            # cheap, goes in /metrics every second
    def detail(self) -> dict             # summary + the scan's per-class and per-entry tables
    def series(self) -> dict             # deques(maxlen=60) for the sparklines, sampled by the telemetry thread
```

`scan()` walks `<dir>` and `<dir>/delta{,/chunks,/states}`:
- counts and bytes per class: `kv-*.bin`, `log-*.manifest`, `chunks/*.bin`, `states/*.bin`, `.tmp-*` residue;
- 8 header bytes per `kv-*.bin` and per `log-*` → magic + **format version**, so the §5.3 stale-store fact is a
  number (`stale: {count, version}`), and foreign-geometry files are counted as *on disk, not promotable*;
- newest / oldest mtime per class (the age of the warmest and coldest stored prefix);
- `shutil.disk_usage(dir)` → `disk_free_bytes`, `disk_total_bytes`;
- a top-N table for "Stored prefixes": `{age_s, tokens, tier, records, bytes, kind}` — **lengths and sizes only,
  never ids, never file names** (design §8).

`observe()` appends to `events: deque(maxlen=500)` (the same shape as `Service.history`, `server.py:546`) with
`kind ∈ promote | cascade | refuse | transfer | evict | sweep`, and takes the **cumulative** totals from the
line's `total_*` fields rather than summing what it happened to see (a cancelled request's line is still seen —
step 4 captures in `_pump` — but a malformed one is not).

**Gate** (`serve/test_server.py`, new `KvCache` class, no HTTP, no engine):
- `parse_kv` on a real step-2 string, on a line with an unknown key, on a truncated line, on `KV start=1 …`;
- a scratch store directory built from synthetic files (correct magic/version, one version-2 file, one `.tmp-*`,
  one foreign file) → the scan's per-class counts/bytes, the stale count, the residue count, the mtime ages;
- `--kv-nvme` absent → `enabled is False`, `summary() == {}`, `detail() == {}`;
- a directory that does not exist / is unreadable → no exception, `scan()` reports it as a warning row;
- `observe()` totals survive a missing line and a garbage line.

~350 new lines. **Risk: none to the running server** — nothing calls it yet.

---

## Step 4 — wiring it into `serve/server.py`

Five small edits, no restructuring:

1. **`StrataEngine.__init__` (`:149-186`)**: `self.last_kv = None`, `self.store_kv = {}`, `self.cache = None`
   (the back-reference the service sets), and `self.in_request = False`.
2. **`StrataEngine._pump` (`:190-194`)** — capture here, **not** in `generate()`:

   ```python
   def _pump(self):
       for line in self.proc.stdout:
           if line.startswith("KV "):
               self._kv_line(line)          # store state, or this request's event
               continue                     # never reaches the request's line queue
           self.lines.put(line)
   ```

   This is the right place, and it fixes two problems at once: the startup `start=1` line arrives after `READY`
   and is still read in order, and the early-stop **drain path** (`:340-355`) discards queue entries — a `KV`
   line parsed by `_pump` is captured even for a cancelled request. `_kv_line` sets `self.last_kv` when
   `self.in_request` (set at the top of `generate()`, cleared at its `finally`), else merges the store fields into
   `self.store_kv`. `generate()` itself is otherwise untouched, and an engine that prints no `KV` line behaves
   exactly as before.
3. **`Service.__init__` (`:534-553`)**: `self.cache = None` (set by `main()`), and `self.totals` gains
   `"reused_from_disk": 0`.
4. **`Service.run`'s `finally` (`:781-815`)**: read `getattr(self.engine, "last_kv", None)`, add
   `"cache": {"src": …, "promote_ms": …, "promote_bytes": …}` to the history record (`:791`), add the disk-tier
   tokens to `self.totals["reused_from_disk"]`, and call `self.cache.observe(kv, last)` when a cache is attached.
   A request with no `KV` line records `"cache": None` — *unknown*, not *cold* (design §3 note 1).
5. **`main()` (`:1552`, beside `svc.gpu_index`)**: `svc.cache = KvCache(engine_args(cfg), cfg.get("log"))`, and
   `engine.cache = svc.cache` for the strata engine. `start_telemetry`'s `extra` lambda (`:590-596`) gains the
   cache series, so the sparkline sampling shares telemetry's thread and clock.

**Gate** (`serve/test_server.py`):
- a `CacheEngine(MockEngine)` fake (the harness's pattern, `:24`, `:286`) that sets `last` and `last_kv` like
  `StrataEngine` does → `svc.metrics()["requests"][0]["cache"]["src"]`, `totals["reused_from_disk"]`, and the
  event rows in `metrics()["cache"]["events"]`;
- `metrics()` gains the `cache` key and the existing `test_metrics` key list is extended — a tier-off server
  reports `{"enabled": false}` and every old assertion still passes;
- a test that a `KV` line cannot break the request path: feed `_pump` a `KV` line, a garbage `KV` line and a
  `KV` line with a huge value, and assert the request still completes and `DONE` is still parsed.

~120 lines. **Risk: medium-low** — it touches the request path's `finally` block and the pump thread. The pump
change is the one to review hardest: a `KV` line must never be forwarded into `self.lines`, or it would be read
as a token line by an older… no — it is *not* forwarded, so a server that does not parse it would hang waiting for
`DONE`? **No**: `_pump` is this server's own code, and the same version parses and drops it. The compatibility
direction that matters is *new engine + old server*, where the old server's `_pump` forwards the line and its
`generate()` ignores it in the `if/elif` chain (`:322-339`) — which is why step 2 also keeps `DONE` unchanged.

---

## Step 5 — `GET /cache` and the `cache` block in `/metrics`

- `Service.metrics()` (`:604-634`) returns `"cache": self.cache.summary() if self.cache else {"enabled": False}`
  — cheap fields only: `enabled`, `mode` (`v3` / `delta`), `inert_reason`, `dir`, `cap_bytes`, `promotable`
  (from the engine's own `entries*` fields), `totals`, `events[-12:]`, `warnings`.
- `do_GET` (`:1121-1195`) gains, next to `/mcp`:

  ```python
  if path == "/cache":
      if self._authorized():
          self._json(200, svc.cache.detail() if svc.cache else {"enabled": False})
      return
  ```

  Read-only. No POST route (design §4, §8): the serve protocol has no store-mutation command
  (`generate.cpp:3544-3552` accepts `QUIT`/`STOP`/`GEN`/`GENI` only), and a web-side store mutation would have to
  re-derive the drop/sweep contract from the wrong side of the pipe.

**Gate** (`test_server.py`'s `WebApp` class, `:466-523`): `/cache` returns 200 with `enabled: false` for the
mock-engine service; `/cache` needs the key when one is set (the `test_metrics_need_the_key_when_one_set`
pattern, `:519`); a service with a `KvCache` pointed at a scratch store returns the per-class tables; `/cache` is
not in the `/web/` traversal allow-list problem set (it is a JSON route, and `test_only_the_app_files_are_served`
stays green).

~60 lines. **Risk: none.**

---

## Step 6 — the Cache tab (`serve/web/index.html`, `app.js`, `app.css`, `sprite.svg`)

- **`index.html`**: a fourth `<button class="st-tab" data-tab="cache">` (a new `#i-cache` sprite glyph — a disc
  with a stack, drawn in the sprite's existing 24×24 stroke style) and a `<section class="view view--cache"
  id="view-cache" hidden>` with the layout of design §5: the status strip, five `metric-card`s (the same
  `METRICS` array pattern, `app.js:111-130`), the `Cache state` / `Cache events` pair (reuse `monitor-row`,
  `ctx-card`'s gauge + `bar-row` + `st-progress`, and `st-table`), the `Stored prefixes` table, and the facts
  block (`dl.facts`).
- **`app.js`**:
  - `showTab` (`:74-84`): add `"cache"` to the tab list and `if (tab === "cache") loadCache();`
  - `render(m)` (`:173-188`): `if (m.cache && m.cache.enabled) show the tab button; else hide it` — a tier-off
    server renders today's three tabs, and the tab appears without a reload because `/metrics` already carries
    `cache.enabled`.
  - `loadCache()` on its own 2 s timer, started only while the tab is visible and stopped when it is not — the
    directory walk costs nothing when nobody is looking.
  - `renderCache(c)`: the five cards (with `spark()` on the series from `summary()`), the gauge
    (`stroke-dasharray`, the same arithmetic as `ctx-fill`, `:259-262`), the bar rows, the two tables, the facts,
    and the badge tones for the three failure classes (design §7's table, verbatim wording).
  - Reuse `fmt`, `kfmt`, `gb`, `esc`, `facts()`, `spark()`, `setMetric` — no new formatting helpers beyond a
    duration formatter (`ms` → `1.84 s`).
- **`app.css`**: `.cache-strip`, `.cache-grid`, `.cache-table` rows — a few dozen lines, all built from existing
  tokens (`tokens.css`), no new colors.

**Gate**: `test_page_and_files` extended — the cache markup exists in `index.html`, `/web/sprite.svg` still serves,
and a new assertion that the tab button carries `data-tab="cache"`. Plus a manual pass on both themes at the
narrow breakpoint (the Monitor's `monitor-row` already collapses; the Cache page must too).

~450 lines (mostly markup + render). **Risk: none to the engine or the API.**

---

## Step 7 — the smaller surfaces

- **Monitor's request table gains a "Cache" column** (`index.html`'s `req-body` header, `app.js:277-292`): a
  `st-badge` reading `RAM` / `NVMe` / `disk` / `cold` from `r.cache.src`, `–` when the request had no `KV` line.
  This is the highest value-per-line change in the whole plan: today `Reused` cannot tell a RAM resume from a
  promote.
- **About gains an "NVMe cache" card** (`renderAbout`, `app.js:314-358`): directory, cap, tier family, format
  version + stale count, weight fingerprint (short hex), and the two honest limits — layer-split sessions are not
  cached, and a promote stages the whole snapshot in RAM.
- **`serve/telemetry.py`**: add the engine's own process RSS (`psutil.Process(engine_pid).memory_info().rss`) as
  a `rss_used` series, and include it in the sampled keys (`_loop`, `:233-245`). That is how the ~2 GB promote
  transient §6 of the design doc measured becomes visible as a bump instead of a footnote. Falls back to nothing
  when `psutil` is absent.

**Gate**: `test_metrics` extended for the new column's data (`requests[0]["cache"]`) and the `rss_used` key when
`psutil` is present; the existing hit-rate column and its tooltip are untouched.

~120 lines. **Risk: none.**

---

## Step 8 — reachability and the record

- **`setup.py` (`:1401-1420`, the `args` builder)**: an optional NVMe cache prompt — store directory (default
  `<root>/kvstore`), cap GB (default 100), delta tier on/off (default on) — appended as
  `--kv-nvme DIR --kv-nvme-max N --kv-delta 1`, with the split case warned: setup already knows `cfg["gpu"]` is a
  list when a layer split is chosen (`:1431-1437`), and under a split the tier is inert by design, so setup must
  say so rather than write flags that do nothing. Today the tier is reachable only by hand-editing `args`, which
  is why the page must handle "configured but invisible" gracefully either way.
- **Docs**: close the C11 bullet in `docs/nvme-kv-cache-design.md` §7 and §9.2 (it currently points at this
  design as "not built"), and add the page's metric definitions to that doc's §6 test record so the numbers on
  the page and the numbers in the oracles are the same numbers.

---

## Test matrix (which claim is proven where)

| claim | oracle | needs a GPU |
|---|---|---|
| the stores report eviction/sweep/write counts correctly | `kv_nvme_host_test`, `kv_delta_host_test` | no |
| the `KV` line appears on every tier-on turn and its `src` agrees with the promote | `tools/short_tests.py` (19/19 with `--kv-delta 1`) | yes |
| `transfer=1` arrives before the `ERR` line and the exit | `tools/nvme_failure_contract_test.sh` | yes |
| the cap turn reports `evict=1` | `tools/nvme_steps123_test.sh` | yes |
| the line changed no byte and no decision | `tools/nvme_p0_test.sh`, `tools/nvme_delta_p0_test.sh` (state-hash + `cmp` oracles) | yes |
| a tier-off server is byte-identical (no `KV` line, same `DONE`/`INFO`) | `tools/needle_bench.py` stdout diff | yes |
| `parse_kv` / the scan / the totals | `serve/test_server.py` (`KvCache` class) | no |
| `/metrics` and `/cache` payloads, key gating | `serve/test_server.py` (`WebApp` class) | no |
| the tab exists only when the tier is on | `test_page_and_files` + a manual pass | no |

## What must not regress

- `DONE`'s field order and `INFO`'s fields (parsed positionally at `server.py:234` and read by the oracles).
- Every resume, dump, supersede, cap, sweep and failure-class decision in the two tiers — steps 1 and 2 add
  reporting only. If a step changes a decision, it is the wrong step.
- A tier-off server: no new tab, no `/cache` data, no `KV` line, no extra filesystem work.
- The web app's existing three tabs, the Monitor's eight cards and its request table.
- Startup time: the store scan stays after `READY` (step 2's ordering note), so the browser opens when it always
  did.

## Rough sizes

| step | files | ~lines | GPU needed |
|---|---|---|---|
| 1 | `kv_nvme.hpp/.cpp`, `kv_delta.hpp/.cpp`, both host tests | 220 | no |
| 2 | `generate.cpp`, `kv_nvme.hpp`, 4 oracle scripts | 250 | yes |
| 3 | `serve/kvcache.py` (new), `test_server.py` | 500 | no |
| 4 | `serve/server.py`, `test_server.py` | 180 | no |
| 5 | `serve/server.py`, `test_server.py` | 80 | no |
| 6 | `index.html`, `app.js`, `app.css`, `sprite.svg` | 470 | no |
| 7 | `index.html`, `app.js`, `telemetry.py`, `test_server.py` | 150 | no |
| 8 | `setup.py`, 2 docs | 120 | no |

Steps 1-2 are the only ones with engine risk and they are also the only ones that unblock everything else; if the
plan is cut in half, cut steps 6-8 and ship 1-5 — `/metrics`'s `cache` block alone already answers "is it on, how
full is it, what happened last turn" for anyone reading JSON, and steps 3-5 are what make the page honest.
