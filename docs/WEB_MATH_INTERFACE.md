# Shared web math foundation

`feat/web-math-common` contains the delimiter parser, source-preserving chat integration, opt-in setting,
cache, renderer deadlines, static asset serving, adapter contract, fixtures and common browser acceptance suite.
It contains no rendering library or compiler dependencies. Its placeholder adapter returns unavailable;
this branch is a development foundation, not a standalone feature proposal.

The KaTeX, MathJax and full TeX branches add their rendering adapter and dependencies. Keep the KaTeX PR
against main self-contained so it provides a working feature when merged. Compare each implementation
against this foundation to see the engine-specific additions. Run the common browser suite on an implementation
branch, where a real adapter is available.

Each adapter implements `initialize(): Promise<void>` and
`render(source, {display, signal}): Promise<{ok:true, element:HTMLElement} | {ok:false, reason:string}>`.
The element is unattached. The shared layer clones it into a current placeholder and ignores stale results.
Source remains unchanged in history, copying, Markdown export and subsequent model requests.

Initialization is bounded to ten seconds; asynchronous conversion returns source fallback at fifteen seconds.
If an engine ignores cancellation, the shared queue waits for it to finish before starting another conversion.
Timers cannot interrupt synchronous JavaScript, so adapters must also bound macro expansion.
Input is limited to 16 KiB of UTF-8 and the cache to 256 exact source/display pairs.
Macro definitions must not leak between equations. All assets must load from the Strata host.
