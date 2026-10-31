### Constrained decoding (GBNF grammars)

The engine can restrict generation, token by token, to the text a GBNF grammar accepts. Grammars are
per-request and opt-in; a request that names no grammar runs exactly as before (the sampler mask is a null
pointer: no allocation, no upload, byte-identical sampling).

- `response_format` (above) is the schema path: `json_object` runs under the vendored `json.gbnf`,
  `json_schema` is converted through `grammar_schema`.
- A bare `grammar` parameter on `POST /v1/chat/completions` is the client's own GBNF text (up to 1 MiB). It
  requires a grammar-capable engine (400 without one) and refuses to combine with tools/MCP.
- Tool calls (OpenAI, Responses and Anthropic routes) ride the grammar when the engine is capable and MCP is
  off: the grammar spells the chat template's exact call serialization, so a constrained model cannot write a
  broken call, invent a tool, or drop a required parameter. `tool_choice` "required" and a named function allow
  only calls; "auto" adds a prose branch whose grammar refuses the literal call opener `\n<tool_call>`:
  the opening `\n` is the template's, so prose may start a call only to finish it. Parallel calls and the
  `\n</think>` prologue are carried. The forced-prefix
  mechanism is skipped on this path: the grammar opens the call. MCP requests stay on the prompt path (the
  hosted tools' schemas are not validated here) and the reply declares the method under
  `strata.structured_output.method`.

Registration: the server sends each grammar text once per engine generation (`GRAMMAR <sha16> <len> <text>`),
keyed by the SHA-256 of the text; the engine keeps at most 64 parsed grammars and a mask cache per grammar.
A restart clears both sides. The grammar vocabulary is read from the model GGUF itself; a server that cannot
read it says so at start and keeps the prompt paths.

Limits: constrained requests draft with the model drafters and stay token-for-token grammatical. The MTP
drafter (`--mtp`) and `--batch-mtp` ride the masked windows: each window row samples from the mask of the
state its draft prefix leaves (the engine builds that chain into the mask rows before the window; a draft the
grammar refuses truncates the chain, the acceptance breaks at or before its row, and the tail rows are
discarded - they never emit). The drafter itself draws unmasked. A request that raises the draft floor
(`spec_min_p`) is still refused with a grammar, and suffix/n-gram drafting (`--spec`'s prompt lookup) is
silently off for a constrained request (its proposals do not come from a model the chain can score). On a
layer split the pipelined windows decode a constrained request serially (the chain rides the serial windows,
as penalties do). DFlash2 (`--draft dflash2`) drafts only in the CLI run in this build, so serving never
meets it. The mask lives in the CUDA/HIP sampler kernels (the SYCL tree mirrors the per-row reads; its
program side has no grammar wiring); Metal/CPU engines report no capability and keep the old paths. The GBNF
feature surface is exactly the vendored llama.cpp converter's (commit `3cf03257`).

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
(reasoning_effort low, temperature 0.6, arrays bounded as real schemas bound them).

With the MTP drafter on (the same rig and model, engine args `--spec 2 --mtp <rt>`,
`bench/results/p1-mtp-grammar-4090`, same day): plain 121.9 tok/s median vs constrained 123.1 - the
drafting eats the mask cost (within the run-to-run spread), 50/50 JSON-schema and 30/30 tool calls held with
the chain active, and the first constrained request still paid the one-time mask-kernel warm-up (30.2 tok/s
once). Single-stream plain vs plain+MTP the same day (`bench/results/p0-mtp-ab`): 94.1 vs 121.1 tok/s
median (+28.7%) - the speed a constrained request now shares in. With `--batch 2 --batch-mtp` two slots
decoded together, a constrained slot's window drafts accepted 52 of 70, and 6/6 concurrent replies
(constrained and plain mixed) validated (`bench/results/p2-batch-mtp-smoke`).

### API request monitor
