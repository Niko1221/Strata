P = "F:/Strata-dev/docs/DETAILS.md"
src = open(P, encoding="utf-8").read()

old = """kernels (the SYCL tree mirrors the fields); Metal/CPU engines report no capability and keep the old paths. The
The GBNF feature surface is exactly the vendored llama.cpp converter's (commit `3cf03257`). A continuation pass (a thinking wrap-up, a close retry, a forced call, a batch slot handed
back to solo) resends the prompt with what it generated and names the tail with `gseed=<n>`:
the engine replays those tokens through the grammar so the state (and mask) continue where the
last pass left off - without the seed the wrap-up's `</think>` would never reach it and the
rest of the reply would sample from the open reasoning mask. A tail that does not fit the
grammar is refused (400), never silently unconstrained.
Measured cost on the RTX 4090 rig (Swift IQ3_S, engine 0.1.47 build, 2026-10-24, `bench/results/2026-10-24-1332-grammar-4090`):
one 400-token answer at effort none ran at a median of 84.8 tok/s plain vs 82.6 tok/s
constrained (-2.6%); a session's first constrained request pays a one-time mask-kernel
warm-up (26.9 tok/s once, 89.0 and 82.6 after); with 6 concurrent slots the aggregate fell
from 172.1 to 140.1 tok/s (-18.6%) - the per-slot mask upload rides every generated token.
The same run answered 50/50 JSON-schema requests inside their schemas and parsed 30/30
tool calls with 0 failures (reasoning_effort low, temperature 0.6, arrays bounded as real
schemas bound them)."""
new = """kernels (the SYCL tree mirrors the fields); Metal/CPU engines report no capability and keep the old paths. The
GBNF feature surface is exactly the vendored llama.cpp converter's (commit `3cf03257`).

A continuation pass (a thinking wrap-up, a close retry, a forced call, a batch slot handed back to solo) resends
the prompt with what it generated and names the tail with `gseed=<n>`: the engine replays those tokens through
the grammar so the state (and mask) continue where the last pass left off - without the seed the wrap-up's
`</think>` would never reach it, and the rest of the reply would sample from the open reasoning mask. A tail
that does not fit the grammar is refused (400), never silently unconstrained.

Measured cost on the RTX 4090 rig (Swift IQ3_S, engine 0.1.47 build, 2026-10-24,
`bench/results/2026-10-24-1332-grammar-4090`): one 400-token answer at effort none ran at a median of
84.8 tok/s plain vs 82.6 tok/s constrained (-2.6%); a session's first constrained request pays a one-time
mask-kernel warm-up (26.9 tok/s once, 89.0 and 82.6 after). With 6 concurrent slots the aggregate fell from
172.1 to 140.1 tok/s (-18.6%) - the per-slot mask upload rides every generated token. The same run answered
50/50 JSON-schema requests inside their schemas and parsed 30/30 tool calls with 0 failures
(reasoning_effort low, temperature 0.6, arrays bounded as real schemas bound them)."""
assert old in src
src = src.replace(old, new, 1)
open(P, "w", encoding="utf-8").write(src)
print("rewrapped")
