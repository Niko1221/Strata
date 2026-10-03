# Native GBNF live demonstration

**Pass: 14 HTTP cases**, using the real Coder IQ1_M model on the authorized
llm-49 RTX 4090. Both target-only (`--spec 1`) and ordinary MTP (`--spec 4`,
suffix lookup off) enforce the raw grammar through the existing native engine.
No model output was scripted. The public [probe](../../../tools/grammar_live_demo.py)
is byte-identical to the probe used by the passing run.

- Branch: `work/gbnf`.
- Worktree: `C:/Users/dflanag3/Documents/fleet/strata-native-gbnf`.
- Tested production commit: `c91260ccd3a929ae51696c9dff1f972d6d56adfe`.
- Run: `demo-20261003T134434Z-7d87ed`, 2026-10-03 13:44:36 to 13:46:11 UTC.
- Native SHA-256: `b335d98cf16d08795d2e089e12484c3c7212c7b5b85755676b76aab113ba7bbe`.
- Probe SHA-256: `db72dd53029c5cc620817c7c44dd3d730d811d5c21ccaf547e3e4f4749311fd6`.
- Before execution, 300 checked-out source files were hashed and matched on the
  inference host. The binary matches the earlier [G5 receipt](../G5/G5-receipt.json).

## What the test shows

The prompt asks for exactly `The sky is purple and the number is 999.`. With no
grammar, that is the actual output. With the following grammar, the native
decoder instead emits `color=blue;count=1` followed by a newline:

```text
root ::= "color=" color ";count=" count "\n"
color ::= "red" | "green" | "blue"
count ::= [1-3]
```

| Check | Target-only | MTP |
|---|---|---|
| Ordinary Responses request | Requested sentence, completed | Requested sentence, completed |
| GBNF Responses JSON | `color=blue;count=1`, completed | Same, completed |
| GBNF Responses SSE | Same text as final response | Same text as final response |
| Unicode SSE | `café 🐈`, completed | `café 🐈`, completed |
| One-token limit | `c`, incomplete/max_output_tokens | `c`, incomplete/max_output_tokens |
| Undefined grammar rule | HTTP 400 JSON before generation | HTTP 400 JSON before generation |
| GBNF Chat Completions JSON | `color=blue;count=1`, stop | Same, stop |

The stream checks enforce increasing sequence numbers, stable item references,
exactly one terminal event and equality of text deltas with final text. The invalid
grammar check verifies that generation history and native completion counters
remain unchanged. Native traces show `grammar=gbnf-v2`, matcher advancement and
MTP retained-window decisions; the MTP constrained request accepted one draft.

## Evidence

- [Readable requests and results (.txt)](GBNF_DEMO.txt).
- [Complete result with both configurations](demo-20261003T134434Z-7d87ed/run/result.json).
- [Exact MTP request](demo-20261003T134434Z-7d87ed/run/mtp/02-gbnf-json.request.json)
  and [response](demo-20261003T134434Z-7d87ed/run/mtp/02-gbnf-json.response.json).
- [Raw SSE](demo-20261003T134434Z-7d87ed/run/mtp/03-gbnf-sse.sse.txt).
- [Target-only engine log](demo-20261003T134434Z-7d87ed/run/target/engine.txt)
  and [MTP engine log](demo-20261003T134434Z-7d87ed/run/mtp/engine.txt).
- [Source/binary manifest](demo-20261003T134434Z-7d87ed/manifest.json),
  [actual invocation](demo-20261003T134434Z-7d87ed/invocation.json),
  [console](demo-20261003T134434Z-7d87ed/console.txt) and
  [cleanup](demo-20261003T134434Z-7d87ed/cleanup.json).

Each request and its raw response or SSE file is included in the run directory.
The passing process exited 0, both engines stopped and the GPU had no remaining
compute process. Credentials were generated in memory/environment variables and
are absent from the captured requests. The HTTP listener bound only to loopback.
No package installs, model downloads or external inference calls were performed.

The first attempt's Windows capture process could not print the cat emoji using
cp1252. The closed pipe caused the following MTP request to fail. Its
[result](capture-failure/result.json), [console](capture-failure/console.txt),
[failed response](capture-failure/mtp-failure.response.json) and
[final cleanup](capture-failure/cleanup.json) remain available. It is not counted
as a passing run. The capture was changed to UTF-8 and the probe's unexpected-error
diagnostics were improved; Strata production code did not change.

## Reproduce on an idle Linux/CUDA host

The probe is a Linux GPU integration test, not a CPU unit test. It loads an
existing native binary/model, runs both modes sequentially and closes its own
HTTP server and engine in `finally`. It uses the same dependencies as the existing
native API probes, including the optional Responses requirements. Its recorded
NVIDIA checks do not qualify HIP or Windows native execution.

With the exact recorded source/binary and local config paths available, run from
the checkout:

```sh
python tools/grammar_live_demo.py \
  --source "$PWD" \
  --manifest docs/gbnf-evidence/live-demo/demo-20261003T134434Z-7d87ed/manifest.json \
  --config /path/to/target.json \
  --config /path/to/mtp.json \
  --out /path/to/fresh-evidence
```

The two full configs are recorded in `run/result.json` under `modes[].config`.
Replace model, tokenizer, expert-profile and executable paths with the installed
assets on your host; they are provenance, not links to downloadable artifacts.
Use target-only and ordinary MTP configs with the same grammar-enabled executable.
The historical manifest deliberately rejects changed source or binary hashes.

For a new build, record a new manifest rather than editing the historical one.
For example, from the checkout, set `DEMO_NATIVE_EXE` to its absolute executable
path and generate a manifest outside the tracked evidence:

```sh
python - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path

root = Path.cwd()
git = lambda *args: subprocess.check_output(['git', *args], text=True).strip()
paths = git('ls-files', 'serve', 'src', 'include', 'CMakeLists.txt',
            'tools/target_only_probe.py', 'strata_tokenizer.py').splitlines()
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest = {
    'source_commit': git('rev-parse', 'HEAD'),
    'branch': git('branch', '--show-current'),
    'native_sha256': digest(os.environ['DEMO_NATIVE_EXE']),
    'source_hashes': {path: digest(root / path) for path in paths},
    'probe_sha256': digest(root / 'tools/grammar_live_demo.py'),
}
out = root / 'build' / 'grammar-demo-manifest.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(out)
PY
```

Pass that new path to `--manifest`, retain the configs and outputs, and describe
the new build/environment in its own receipt. Its results are separate evidence.

## Limits and next gate

This fresh run qualifies the stated cases in target-only and ordinary MTP modes.
It does not rerun coupled/suffix qualification, test a GBNF-disabled native build,
or exercise a native Codex tool loop. JSON Schema/JSON-object enforcement, Lark
custom tools and grammar combined with tools/reasoning remain excluded. G6 stays
deferred. Cold and warm request times have different cache states and are not a
performance comparison.

This update adds the probe and evidence to the branch without changing generation.
The [publication receipt](publication-receipt.json) pins the evidence commit. The remaining review
gates are the final feature-off native build/run and the previously identified
Responses accounting/replay follow-ups before a merge-ready claim.
