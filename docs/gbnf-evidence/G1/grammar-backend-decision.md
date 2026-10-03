# G1 backend decision

Use one native C++ backend: XGrammar v0.2.8, commit
`97787376faee5ed8466cfad57c99855e4ce2f6aa`, with Strata's explicit cooperative
resource guards (`xgrammar-0.2.8-strata-budget1`). The
[upstream release](https://github.com/mlc-ai/xgrammar/releases/tag/v0.2.8)
and [pinned grammar syntax](https://github.com/mlc-ai/xgrammar/blob/97787376faee5ed8466cfad57c99855e4ce2f6aa/docs/defining_structures/ebnf_grammar.md)
are source references. Selection follows the actual native corpus and byte-table
tests in [the report](REPORT.md), not just the upstream compatibility description.

## Build and licenses

`tools/prepare_xgrammar.py` downloads two commit archives with fixed SHA-256
hashes: XGrammar and its pinned DLPack headers at
`bbd2f4d32427e548797929af08cfe2a9cbb3cf12`. It checks archive paths, file types and
expanded size, retains licenses, checks each source patch anchor and records
original/patched hashes in `PIN.json`. CMake checks the selected backend and
those patched files. Preparation is an explicit build step. Neither CMake nor
the server downloads anything. The resulting native static library needs C++17,
threads, DLPack headers and the bundled picojson header; no Python bindings,
TVM, XGrammar GPU extension or additional inference engine is used.

Upstream's bundled CMake configuration enabled Python bindings despite the
cache argument used in the initial evaluation. That configure failure is
retained. Strata defines the same native source target directly, excluding TVM
bindings and avoiding upstream global compiler flags. The backend distribution
also contains JSON/Lark converters; Strata calls only `CompileGrammar` with the
`root` rule. Those other frontends have no Strata feature or endpoint here.

XGrammar and DLPack use Apache-2.0; picojson uses its two-clause BSD notice.
The prepared source retains the original notices. Binary redistributors must
include the [license material](../../licenses/GBNF-DEPENDENCIES.txt).

## Accepted dialect and byte identity

The initial dialect accepts named rules, alternatives, concatenation, groups,
double-quoted literals and escapes, Unicode character classes, `?`, `*`, `+`,
comments, epsilon and recursive references. Rule attributes, repetition ranges,
macros, regex syntax, lookahead and token-literal extensions are rejected.
There must be a `root` rule. The admission scan checks bounds; XGrammar is the
only grammar parser and matcher. Backend syntax errors retain their location.

The bridge reads existing `gpt2` / `qwen35` pack artifacts. XGrammar decodes the
GPT-2 byte alphabet; Strata's prompt tokenizer stays unchanged. Every normal
token's bytes are compared with the existing Python tokenizer. Control,
user-defined and unused tokens are excluded from text masks; each actual end
ID is a separate transition allowed only at acceptance. A token may span
productions, and a Unicode character may span tokens.

The compiler owns one immutable vocabulary and a bounded cache keyed by the
full source. The backend version, root and dialect are fixed. Checkpoints keep
the exact compiled object and reject another grammar or tokenizer. Stable FNV
diagnostic names include both source and vocabulary; they are not security
identities, cache authorization or serialized replay credentials.

## Resource and ownership policy

Source is limited to 8192 UTF-8 bytes, 128 rules, 256 alternatives and 32 nested
groups. Native recursion is capped at 128. Backend guard patches count grammar
expansion, automaton growth, parser queue work and vocabulary iteration. They
bound expression data, states, queue growth and accumulated compiled masks.
Compilation has five million work units and a cooperative 2.5-second deadline;
matcher calls have a one-second deadline and a cumulative two-million-unit
sequence budget. These are checked in native work and at operation exit, not a
hard real-time process watchdog or an exact allocator-wide memory quota.

Compiled artifacts above 16 MiB are rejected; the cache defaults to eight
entries and 64 MiB. Compiler threads and upstream caches are disabled. Vocabulary
storage and temporary allocations have separate admission bounds. Active
matchers hold their immutable artifact even if it is evicted; private forks
must remain bounded by their caller. Matching keeps at most 8192 committed
tokens / 65536 output bytes. No disk operation or Python callback occurs per
token. Exhaustion fails the sequence; it never removes the grammar.

The accepting-prefix and terminal states differ. A legal EOS ends the matcher;
an accepting prefix can still admit more text. Illegal tokens do not advance
state. Fork/checkpoint/replay concern grammar state only, never model KV or
recurrent state. G2 owns integration with actual model commitment.
