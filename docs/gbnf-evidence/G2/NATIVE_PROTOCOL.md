# Native grammar input at G2

The existing `INFO` line advertises `grammar=gbnf-v1` only when the optional
backend is built and the engine uses the qualified target-only profile:
one GPU, text mode and the two normal end controls, 248044 and 248046.
Other configurations report `grammar=none`. Clients must check this before
sending a grammar frame. Existing GEN/GENI/STOP/QUIT commands remain available.

One constrained request has this byte framing:

```text
GENG1 13
root ::= "ok"
GEN 8 1,2,3
```

The first line contains a decimal grammar length, from 1 through 8192. Next
come exactly that many UTF-8 bytes, one LF byte, then one complete ordinary
GEN line. Length counts bytes, not Unicode characters. Grammar newlines,
quotes and text such as `STOP` are inert payload. The native grammar validator
rejects raw NUL and invalid UTF-8. No filename or shell expansion is involved.
The example's prompt IDs are illustrative; the live probe supplies the real
chat template/tokenizer output.

The reader uses the existing stdin thread and queue. Headers/command lines
are bounded to 4 MiB, pending commands to eight. Windows stdin uses binary
mode so frame lengths are exact. A bad length, unsupported frame version,
missing delimiter, truncated payload or incomplete/non-GEN trailing command
produces ERR and terminates the pipe. It never attempts to recover by treating
remaining payload bytes as commands. Grammar syntax/resource errors after a
valid frame produce an ERR before prompt processing; the next request can run.

Frames are request-local and must be sent atomically by the admission owner.
`tools/grammar_native_probe.py` uses one process sequentially and prefixes the
existing engine generator's GEN write. This is test-only orchestration. G3
provides the production shared argument/write and pre-header validation.

Every constrained request creates a fresh matcher, even when its compilation
or prompt is cached. Only selected output advances it. No matcher state is
stored with model checkpoints. The ordinary T/DONE/ERR stream and STOP/drain
lifecycle are retained. Budget and cancellation do not claim successful
grammar completion. Runtime selection failures end the engine rather than
reuse uncertain model state or drop the constraint.
