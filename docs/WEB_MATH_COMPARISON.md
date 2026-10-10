# Web math: three implementations

These branches address [LaTeX support #1769](https://github.com/Niko1221/Strata/issues/1769) and
[math rendering #1780](https://github.com/Niko1221/Strata/issues/1780). Neither issue supplied an equation example.
They share the same upstream main baseline, parser, settings, application integration, cache and acceptance tests.
The adapter and dependencies change; chat messages and model requests do not.

| Branch | Rendering | Added engine footprint | Initial vendor transfer in the corpus |
|---|---|---:|---:|
| `feat/web-math-katex` | Browser HTML + MathML | about 0.56 MB | 345,581 bytes |
| `feat/web-math-mathjax` | Browser CommonHTML + assistive MathML | about 3.20 MB | 1,138,821 bytes |
| `feat/web-math-tex` | Isolated XeLaTeX → PDF → filtered SVG image | about 4.37 GB | no vendor assets; SVG responses per equation |

Sizes use decimal units. Browser footprints include all vendored fonts, licenses and provenance, not Strata's
existing fonts or the small shared scripts. The TeX number is Docker's uncompressed built-image size, not a
download size; it excludes Docker itself. The image includes `texlive-full`, even though the UI compiles only
equation fragments. The normal Strata install does not download any TeX dependencies.

## Features and limits

| Capability | KaTeX | MathJax | Full TeX adapter |
|---|---|---|---|
| Runs entirely in the browser | Yes | Yes | No; optional Docker worker |
| Offline assets after installation | Bundled JS/CSS/fonts | Bundled JS/font data | Local compiler image |
| Output | HTML + semantic MathML | CommonHTML + assistive MathML | Filtered SVG image with source alt text |
| Packages | KaTeX-supported syntax | Configured TeX packages; no require/autoload | Fixed amsmath, amssymb, mathtools, unicode-math preamble |
| Full document or arbitrary package UI | No | No | No; equation fragments only |
| Unsupported input | Original source | Original source | Original source |
| Operational setup | None beyond Strata | None beyond Strata | Explicit image build and isolated worker startup |

All three support the same inline/display delimiters, isolate macro definitions between equations, preserve
chat source and use the same opt-in setting. Installing a full TeX distribution does not expose arbitrary
document compilation or every installed package through this adapter.

## Measurements

Measured on 2026-10-10 with Playwright 1.63.0. Browser tests ran on Windows with an Intel i7-8650U, at a
390 × 844 viewport. The TeX worker ran on `middle-child`, Linux with an Intel i7-2600, limited to one CPU and
512 MiB, using Docker 28.5.1. Windows TeX measurements include a native Windows Docker client connected through
a persistent SSH tunnel to the remote daemon. Earlier measurements with a new SSH process per equation were
variable and are not the table's comparison path.

| Chromium, seven valid equations | KaTeX | MathJax | Full TeX, Windows → Linux |
|---|---:|---:|---:|
| Uncached equation median | 11.5 ms | 31.0 ms | 3,417 ms |
| Uncached equation range | 4.6–17.8 ms | 18.0–147.8 ms | 3,303–3,789 ms |
| Three initial equations settled | 509 ms | 1,643 ms | 9,992 ms |
| Cached result lookup | below 1 ms | below 1 ms | below 1 ms |

The initial-equation measurement starts when placeholders are visible after enabling the setting. It includes
initialization and serialized rendering. Cached lookup measures the cached result, not copying DOM or painting it.
These are single-run acceptance-test timings on a working laptop, not controlled statistical benchmarks.
Other browsers and runs vary; Unicode font-data loading is particularly visible in MathJax.

Linux-local TeX results, compiler security cases, and individual browser results are in `web-math-results/`.
The Linux-local path removes Windows and SSH overhead. Keep those numbers separate from browser-library timings.

## What was checked

- All three adapters passed the common suite in Chromium, Firefox and Playwright WebKit on Windows.
- The corpus covers fractions, roots, scripts, integrals, sums, matrices, cases, aligned equations and Unicode.
  Malformed input, an unknown command and an infinite macro loop preserve source.
- Checks include every streaming split of representative equations, code, currency, Markdown tables and links,
  UTF-8 size limits, macro isolation, missing engines, stale results, keyboard activation and narrow layouts.
- Copy, Markdown export, saved history and subsequent model requests retain the original source. Changing math
  settings while streaming preserves the element used by the live stream.
- External requests are blocked during browser tests. Assets and fonts load from the Strata server.
- Screenshots were inspected; equation geometry and completed SVG image loading are also asserted.
- The TeX server passed 37 endpoint, transport, request-origin, authentication and body-handling tests.
- The isolated compiler rejects file reads and Lua commands, and stops an infinite TeX macro loop at five seconds.

KaTeX and MathJax provide semantic MathML. The TeX image has the source as its accessible description; it does not
provide equivalent mathematical navigation. The tests inspect markup, not a manual screen-reader session.
Playwright WebKit is engine coverage, not a physical Safari/iPhone test. Docker Desktop 4.94.0 was installed
on Windows using its per-user WSL 2 backend; the local Docker Engine 29.8.2 passed hello-world. Native full-TeX
image build and renderer validation are still pending. The recorded TeX results below use the Windows
server/client with the Linux container worker, not a native Docker Desktop worker.

The full TeX image uses a pinned Debian base digest, pinned TeX Live distribution package, installed-package lock
file, and an exact image ID when started. Transitive Debian dependencies can change on rebuild; archive the
tested image for exact reproduction. Setup, limits and commands are documented in `WEB_MATH.md`.

## Recommendation

Use KaTeX for the initial production proposal: it is the smallest implementation and handled every valid case
in this corpus with lower measured latency. Keep the interface independent of it. MathJax provides a viable
alternative when a concrete unsupported expression justifies its larger bundle. The full TeX branch demonstrates
the additional setup, latency and accessibility cost of invoking a real compiler for chat math.

The corpus does not establish support for every expression an LLM can produce. Each engine retains its support
limits, and all three preserve unsupported source instead of guessing a correction or changing model output.
