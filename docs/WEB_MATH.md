# Chat math comparison

This branch supplies TeX Live 2022 with XeLaTeX and dvisvgm. The other comparison branches supply MathJax 4.1.3 and a full TeX Live
compiler. All start at upstream main `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` and use the same parser,
settings, cache and acceptance corpus. No GPU engine or chat API changes are needed.

Open Chat's sampling drawer, enable **Render math**, and Apply. This browser remembers the choice. Rendering
starts off. Turning it off uses the existing Markdown formatter and loads no engine assets. Turning it back on
also retries initialization and previously failed equations.

Inline math accepts `\(x_1\)` and `$x_1$`; display math accepts `\[...\]` and `$$...$$`, including multiline
equations. Dollar parsing is deliberately conservative: `$5$`, `$5 and $10`, escaped dollars, multiline single
dollars, and closing dollars followed by a digit stay literal. Use backslash delimiters for ambiguous input.
Code stays literal. Math is protected before Markdown emphasis, links and table cells are formatted.

Only assistant answer text is typeset. Reasoning, user messages and tool payloads keep their existing presentation.
Copy, Markdown export, saved history and model requests use the original source. Malformed or unsupported equations,
unfinished streamed equations, and unavailable engines remain readable source. No repair request is sent to the model.

## Adapter contract

Each branch replaces `serve/web/math-adapter.js`; engine details stay out of the application and parser.

```ts
initialize(): Promise<void>
render(source: string, options: {display: boolean; signal: AbortSignal}): Promise<
  {ok: true; element: HTMLElement} |
  {ok: false; reason: "invalid" | "unsupported" | "unavailable" | "limit"}
>
```

The adapter returns an unattached element. Shared code clones it into the current equation placeholder; a result
from an older message paint cannot replace newer content. Rendering is serialized. Up to 256 results are cached
by exact source and display mode; an equation is limited to 16 KiB of UTF-8. Queued work evicted from the cache
or invalidated by a settings change is skipped. Initialization is limited to ten seconds. An asynchronous conversion returns source fallback after fifteen
seconds; if an engine ignores cancellation, subsequent conversions wait for it to finish to avoid overlapping
macro state. Browser timers cannot interrupt synchronous JavaScript; engine macro limits remain necessary. Macro definitions do
not carry across equations. Complete equations render while the answer streams, without retypesetting unchanged
source on each token.

The adapter calls POST /web/math/render on the same server. It accepts {source, display}, returns either
{ok: false, reason} or {ok: true, svg, width, height}, with dimensions in em. Existing API authentication,
trusted-origin checks and JSON content-type checks apply. Requests are limited to 64 KiB before reading.

The optional worker runs in a read-only container with no network or host mounts, dropped capabilities, a
512 MiB memory limit, one CPU, 64 processes and a 128 MiB temporary filesystem. XeLaTeX runs with shell escape
disabled and restricted file reads/writes. Choosing XeTeX avoids a Lua interpreter in generated equations.
Each compilation and SVG conversion together get five seconds; all temporary files are removed. SVG output is
limited to 2 MiB and filtered to vector geometry and internal references, then displayed as an image.
The accessible description is the equation source; it lacks the semantic MathML of the browser adapters.

Install Docker with Linux-container support on the compiler host. Build/start are explicit:

```text
python tools/tex_worker.py build
python tools/tex_worker.py start
```

Set STRATA_TEX_CONTAINER=strata-math-tex before starting Strata (PowerShell: $env:STRATA_TEX_CONTAINER='strata-math-tex';
POSIX shell: export STRATA_TEX_CONTAINER=strata-math-tex). STRATA_TEX_DOCKER can select a Docker executable;
DOCKER_HOST can select a configured daemon. Without a worker, equations remain source. No download or installation
occurs at server startup. Stop the worker with python tools/tex_worker.py stop.

The Dockerfile pins the Debian base digest and texlive-full package version, records all installed package
versions in /opt/tex-packages.lock, and starts the worker using the built image ID. Transitive Debian packages
come from the current Bookworm repositories: archive the built image for exact reproduction. The complete
distribution is intentionally included for this comparison; no complete-document compilation UI is exposed.

## Reproduce validation

Install test dependencies in a virtual environment, separate from Strata's normal requirements:

```text
python -m pip install playwright jinja2
python -m playwright install chromium firefox webkit
python tools/test_web_math.py --browser chromium --output /path/to/results
python tools/test_web_math.py --browser firefox --output /path/to/results
python tools/test_web_math.py --browser webkit --output /path/to/results
python -m unittest serve.test_security serve.test_chunked_body -q
```

The browser suite starts a mock server without a model. It blocks external network requests and checks equations,
streaming splits, Markdown, currency, code, narrow layouts, themes, accessibility markup, source persistence,
macro isolation, stale results, size limits and unavailable engines. It writes a screenshot and timing JSON when
an output directory is supplied. Playwright WebKit is engine coverage, not a test on a physical iPhone or Safari.
Semantic markup checks are not a manual screen-reader audit.

Run python -m unittest serve.test_tex_math -q for endpoint and transport failure tests. With the worker
configured, python tools/test_tex_worker.py /path/to/results.json checks real compilation, restricted file
reads, disabled Lua commands and the timeout. Run browser tests one at a time: the compiler accepts only one
job at a time, and concurrent clients receive a readable source fallback.

See `WEB_MATH_COMPARISON.md` for the measured results and their limits.
