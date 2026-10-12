# Chat math comparison

This branch supplies MathJax 4.1.3. The other comparison branches supply MathJax 4.1.3 and a full TeX Live
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

MathJax emits CommonHTML and assistive MathML. Its component, safety extension, assistive extension, font data,
WOFF2 fonts and licenses are bundled, with no CDN. Only base, AMS and newcommand packages are enabled; generated
source cannot load arbitrary extensions. The safe extension filters URLs and styling. Speech and explorer workers
are disabled after startup; semantic MathML supplies the accessible representation. Macro expansion is limited
to 1,000. Definitions are restored after each equation using the pinned version's definition maps.

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

`python tools/vendor_math.py mathjax` reproduces the pinned assets from npm and verifies SHA-512 registry integrity.
`PROVENANCE.json` records the version, archive URL and integrity. End users never need npm, Node or this script.

See `WEB_MATH_COMPARISON.md` for the measured results and their limits.
