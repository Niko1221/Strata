# Request-owned metrics (local candidate, not deployed)

This is the server-side part of execution-plan step 2. It does not change model
weights, kernel arithmetic, context, vision, native scheduling, or draft policy.
No GPU inference or production restart was performed while developing it.

## Why

The batch server previously combined another admission's `engine.last` with the
finishing request's output count, and one completion cleared the shared `busy`
flag. The old server reproducibly reports one in-flight request while two are
held at independent generation barriers. Global rate samples were reset by
arrivals and used counters from different requests.

## Ownership

- `serve/request_stats.py`: one `GenerationStats` per `Service.run` generation.
  `ObservedGeneration` binds it during each iterator operation, including close,
  and resets the context before returning control to the caller. Interleaving two
  generators on the same thread is safe; this is not thread-local `last` storage.
- DONE describes a solo segment or the BGEN admission's first token. BADM says
  whether batch decoding follows. BDONE supplies that slot's total output count
  (including the first token) and its post-admission wall time.
- Solo promotion and reasoning-budget continuation accumulate their own engine
  segments. Original input/cache counts come from the first segment; work done
  re-reading generated text is separate (`engine_prompt_read_total`).
- Batch admission expert-cache hit counts are NOT full-request expert counts;
  those fields are unknown rather than copied into the completed batch record.
- A missing terminal acknowledgement means `engine_complete=false` and no full
  engine timing/rate, not a fabricated GEN-1 rate. Closed requests' published
  history is frozen; a later background drain does not rewrite another record.
- `engine.last` remains only a legacy diagnostic. Serial mock/third-party engines
  can still supply it; concurrent engines never use that compatibility fallback.

## Endpoints and scopes

- `/metrics.active_requests`: content-free per-generation id, HTTP request_id,
  original input count, output count, phase, slot id/generation, progress,
  control/slot wait, maximum observed token gap and completion metadata.
- `/metrics.live`: aggregate **server-consumed output** rate over a bounded time
  window. It decays during stalls and does not reset when another request starts.
  Multiple requests do not share an arbitrary prompt length/output budget.
- `/v1/status.activity`: actual registered generation count, with `unit=generation`.
  An HTTP MCP request can contain several generations; this is not an HTTP count.
- `X-Strata-Request-ID`: server-generated, content-free correlation id on OpenAI
  and Anthropic responses, streaming and non-streaming. It matches request_id in
  metrics and, when enabled, the API monitor. It is not copied from client headers.
- `decode_scope=engine_segments_including_batch_stalls`: sum of measured engine
  segment times, including other requests' admission pauses while in batch; it
  excludes this request's between-segment control waits and is NOT GPU-active time.
- `prompt_scope=admission_including_cache_work`: existing DONE.prompt_ms includes
  parking/restore/etc. It is NOT a new measurement of pure prefill.
- `first_token_s`, `server_decode_wall_s`, `max_token_gap_s` are measured at the
  Service.run consumer, not at GPU completion or client receipt. The first starts
  AFTER HTTP loading/tokenization/vision preparation. Do not call it client TTFT.
- Total decode_ms sums overlapping per-request durations. The dashboard no longer
  divides total output by that sum and labels it aggregate throughput. Unknown
  engine work is not displayed as proven uncached reads.
- Requests with an asynchronous cancellation drain can already be gone from the
  HTTP generation map while a native slot remains occupied. Metrics reports
  `draining`; unload refuses occupied slots/control, and queue ownership stays
  with the original drain even if a later process replaces the engine's queues.

No prompt/answer content is added to metrics. Existing opt-in API content capture
is unchanged. Per-request transient status tails retain the existing bounded
behavior, are not in metric records and are cleared after completion.

## Protocol edge cases covered

A validation ERR before DONE is terminal: do not wait for a nonexistent BADM.
But a BGEN slot-copy failure AFTER DONE emits ERR followed by BADM <slot> 0;
that marker must be drained, or the next request consumes stale control data.
Both paths have CPU regression coverage. An unacknowledged cancelled slot is not
silently freed on a drain timeout, which could route late output to its successor.

## Validation

Run `python -m unittest discover -s serve -t .`, `node --check serve/web/app.js`, and `git diff --check`.
The new tests exercise seven-way reverse completion, interleaved native protocol lines, promotion, cancellation,
iterator context isolation, slot generations, late drains, HTTP correlation, rolling rates and incomplete statistics.
They use synthetic inputs and clocks; they are not GPU performance measurements.

This patch targets the multi-sequence-batching branch underlying upstream PR #559.
It does not introduce batch scheduling into main, or change the model, context size or precision.
Content-free request identifiers correlate diagnostic records; prompts and responses are not stored by this collector.
The separate opt-in API monitor retains its existing behavior.
