# Shared web math core

Adds an opt-in Render math setting, delimiter parsing and source-preserving chat
integration. The placeholder adapter reports unavailable and leaves equations as
source. Rendering engines are separate additions.

Adapters provide `initialize(): Promise<void>` and
`render(source, {display, signal}): Promise<{ok:true, element:HTMLElement} | {ok:false, reason:string}>`.
Return an unattached element; the core clones it only into a current placeholder.
History, copying, Markdown export and API requests keep the original source.

Initialization has a 10-second deadline, asynchronous rendering 15 seconds.
Conversions stay serialized if an adapter ignores cancellation. Adapters must
bound synchronous work and isolate macros between equations. Input is limited
to 16 KiB UTF-8; the cache holds 256 source/display pairs. Assets must come from
the Strata host under `/web/vendor/`.
