# Local Output History

The dashboard's History tab lists newly recorded API requests. Select a request to
read its Output, Thinking, Tools, or Details. Search and status filters apply to
saved requests. Older requests and long text have explicit pagination; Download
exports all saved text for a request, while Copy copies the displayed text.

Each request also shows its source under From. Strata Web identifies itself explicitly; other
apps can send `X-Strata-App: My Editor` (or `X-Client-Name` / `X-App-Name`).
Otherwise, recognized editor product names in User-Agent become the label.
Generic SDK/browser user agents and requests recorded before this feature show
Unknown. These are self-reported labels, not verified identities. Only the label
and which identification field supplied it are saved, never raw request headers.

The model shown in History is the actual model serving the request. A client's
compatibility alias is retained separately as `requested_model` in Details;
for example, Claude Code can request a Claude-named alias while Strata runs Qwen.
This does not change the model name or any output returned through the API.

Enable `"output_history": true` in the model's `strata-<model>.json`, or use the
Output history switch under About's model settings, then restart Strata. It is
off by default. Existing requests from
before recording was enabled cannot be recovered.

The database is `<config-name>.history.sqlite3` beside the model configuration
(or `output.history.sqlite3` when started without a configuration). SQLite's
associated WAL and SHM files may also exist while running. History survives
restarts; requests left active at shutdown appear as Interrupted on the next
start. There is no automatic deletion or text-length retention limit.

The recorder supports Chat Completions, Messages, and Responses, with or without
streaming. It saves the thinking, answer text and tool names/arguments emitted
through these APIs, plus token usage, timing, status and available generation
metadata. It does not record request headers, credentials, or complete input
prompts. Generated text and tool arguments may themselves contain sensitive
information: the database is local plaintext, not an encrypted vault. Do not
commit or publicly share it.

Recording does not change model prompts, sampling, output limits or API events.
Disk writes use a bounded background queue. An unavailable disk or an overloaded
writer produces a visible recording warning and may leave an incomplete record;
it does not fail the model request. Parser-held text cannot be recorded before
the API emits it. Output history is not a raw engine-token trace.

`GET /api/history` uses the server's existing authorization. The public gateway
treats it as a dashboard route, not an editor inference API route. It stays
unavailable unless dashboard sharing is explicitly enabled, and then requires
the dashboard login or gateway key. See [the setup guide](LOCAL_FEATURES.md).
