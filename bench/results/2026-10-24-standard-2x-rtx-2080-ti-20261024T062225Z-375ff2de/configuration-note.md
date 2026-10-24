# Configuration note for this pilot run

`config.json` in this directory says `"engine_options": null` because the engine was already serving
when the runner connected (`--url http://127.0.0.1:8080`), so the server could not report its own start
arguments. The exact argument list is in the sibling community report:

    ../2026-10-24-community-2x2080ti-dual-socket/config.json

In short: two RTX 2080 Ti at 180 W, layer split `gpu [0, 1]`, `--batch 4 --batch-groups 2`,
`--prompt-cache-every 4096`, `--resident-budget-gib 55`, `--vram-reserve-mib 700`, INT8 KV,
`--max-context 131072`, vision served by a second binary. Track `custom`, as the runner recorded.

The engine was already warm: the cache-warming requests of that report ran before this protocol run, so
`request_counts` below does not describe a freshly loaded engine. That is the "server that has already
answered requests" case `docs/BENCHMARK_PROTOCOL.md` describes; the runner records it rather than hiding
it.

`engine.log` is the serving engine's own log for this session, with the local account, model directory
and mount point replaced by `<home>`, `<data>`, `<checkout>` and `<mount>`. Its substantive content is
unchanged; no API key or credential appears in it.
