# G3 native preflight and generation

G3 keeps the G2 length-delimited `GENG1` frame and adds one exact trailing
command, `CHECKG`. The `INFO` capability becomes `grammar=gbnf-v2`; Python
rejects v1, absent and `none` capabilities before writing a grammar frame.
The qualified native mode is unchanged: target-only, one GPU, text, standard
end controls. Restart must negotiate its own capability, never inherit a stale
value from the previous process.

```text
GENG1 13
root ::= "ok"
CHECKG
```

The engine compiles through its one immutable compiler/cache, constructs the
same initial matcher as generation and checks its first legal-token mask.
It replies with either one `ERR <bounded message>` or:

```text
GRAMMAR_OK bytes-v1-fnv1a64:<native vocabulary diagnostic>
```

The diagnostic hashes the number of tokens, each emitted byte string with its
length, and sorted end-control IDs. Other special/control IDs have empty byte
strings. Numbers are eight-byte little-endian values. Python derives the same
table from the existing tokenizer and compares it before generation. This
catches configuration drift; it is not an authentication or adversarial-hash
guarantee. Tokenizer/model artifacts remain trusted server configuration.

Preflight neither reads a prompt nor advances KV, recurrent, sampling or
generation state. It holds the existing Service FIFO, then releases it.
Generation takes its ordinary FIFO turn and sends a fresh complete grammar
frame with `GEN` in one binary write. Compilation is immutable; matcher progress
is private to that generation. A cache eviction between check and generation
may require recompilation and cannot transfer another request's progress.

Both adapters validate supported fields, answer-only template boundary,
token/output limits, native capability, tokenizer identity and grammar syntax
before success headers. If a pipe times out during preflight or returns an
unexpected reply, the server ends that desynchronized process so a late response
cannot be consumed by another request. Native compiler/matcher cooperative
limits remain separate from the 30-second pipe deadline.

Native framing keeps the G2 bounds and fatal behavior: malformed/truncated
length, delimiter or command terminates the pipe without reinterpreting data.
Syntax/resource errors in a well-framed source return ERR and leave the next
request usable. No public inspector, file path or second command service is
introduced. Production frames use binary writes on Windows too, avoiding
TextIO newline conversion inside the exact-byte payload.

After admission, existing T/DONE/ERR output and cancellation/draining remain in
charge. Raw constrained answer bytes enter the incremental UTF-8 decoder and
semantic content events without marker parsing. The Responses assembler and
Chat serializer remain protocol views. Only native selection advances grammar
state; neither HTTP nor diagnostics runs a second matcher.
