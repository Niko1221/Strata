# Full TeX refresh, 2026-10-11

Refreshed to upstream main `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`
and shared math core `253ea8cf`. No fixed compiler CPU/wall budget, host RPC
timeout or TeX browser conversion deadline remains. Browser-only adapters still
default to fifteen seconds. The shared core parses and schedules; the adapter renders.

Windows Python 3.13.15, Playwright browsers and Docker Desktop Engine 29.8.2
were used with the local Linux-container worker. The updated worker overlays the
previous archived image, retaining its installed package versions. Exact image,
worker hash and container isolation are recorded in `runtime-no-budget.json`.
The image is 7,040,120,419 bytes uncompressed; this is not a download-size estimate.

Validation:

- 41 endpoint, origin/authentication, body handling and security tests passed.
- Real worker: seven valid equations and six malformed/restricted inputs passed.
- Chromium, Firefox and Playwright WebKit passed the complete browser suite.
- Regression checks cover explicit adapter deadline opt-out, default browser
  deadlines, serialization after deadline, and keeping a cancelled compiler RPC
  pending until the compiler finishes, then discarding its cancelled result.
- Screenshots were inspected; source copy/export/persistence, macro isolation,
  stale results, UTF-8 size limits and narrow display geometry passed.

The infinite recursive TeX macro fixture is explicitly skipped. Without a time
limit, it requires operator cancellation by stopping/restarting the worker
container and occupies the compiler until then. Memory, PID, filesystem,
input/output and SVG geometry limits remain enabled.

Timings are single acceptance runs on a working laptop. Chromium's initial chat
overlapped the tail of the standalone worker corpus, so its cold timing includes
contention and is not a controlled comparison. Firefox and WebKit ran serially.
Valid compilations exceeding five seconds succeeded in both the standalone corpus
and WebKit. Earlier five-second-policy records remain historical evidence.

Reproduce with the optional worker configured:

```text
python -m unittest serve.test_tex_math serve.test_security serve.test_chunked_body -q
python tools/test_tex_worker.py worker-results.json
python tools/test_web_math.py --browser chromium --output chromium
python tools/test_web_math.py --browser firefox --output firefox
python tools/test_web_math.py --browser webkit --output webkit
```

Run compiler/browser suites one at a time. WebKit coverage is not a physical
Safari/iPhone test; source alt text is not semantic MathML.
