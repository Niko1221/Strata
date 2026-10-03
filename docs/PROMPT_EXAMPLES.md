# Test models before adding examples

The [measured Windows/Ubuntu checkpoint](gbnf-evidence/prompt-examples/REPORT.md)
records real Codex tasks, the failed empty-call baseline and its one-example retry.

Run the zero-example prompt first. If it fails, the qualification harness can
start a fresh attempt with one example, then two, then three. Each attempt keeps
its own request, response and result. A hinted success does not replace a failed
baseline. The server does not retry, repair calls, or add hidden demonstrations.
Here, zero means zero **added task demonstrations**. The model pack's fixed
tool-format instructions, including its generic XML syntax skeleton, remain in
every variant. Record the installed template hash when comparing models; replacing
that native template is a separate variable.

Start with the readable [empty-call example](codex/prompt-examples/windows-tools-get_goal.txt),
[PowerShell example](codex/prompt-examples/windows-tools-exec_command.txt), and
[Ubuntu Bash example](codex/prompt-examples/ubuntu-tools-exec_command.txt).
The committed profiles include [Windows baseline](codex/prompt-examples/windows/codex-instructions-0.txt),
[Windows with three examples](codex/prompt-examples/windows/codex-instructions-3.txt),
[Ubuntu baseline](codex/prompt-examples/ubuntu/codex-instructions-0.txt), and
[Ubuntu with three examples](codex/prompt-examples/ubuntu/codex-instructions-3.txt).

This removes the automatic empty-call hint used at the
[earlier qualification checkpoint](gbnf-evidence/schema-inventory/REPORT.md).
The protocol adapter still validates output normally. Examples cannot enable an
unsupported capability, increase the context limit, or guarantee a model succeeds.

## Inspect the prompts

Generate the complete paired fixture set with the same code used by the probes:

```text
python tools/responses_prompt_examples.py --out <new-directory>
```

The export contains 266 request variants with readable `examples.txt` files:
12 captured Codex tools on each of Windows/PowerShell and Ubuntu/Bash, with
zero through three examples, both captured non-strict and synthetic strict
variants; all 30 JSON schemas with zero or one example; and
seven output modes with zero or one example (text, SSE, visible reasoning,
summary request, JSON object, GBNF, and GBNF with reasoning). Eight matched Codex
instruction files cover both operating systems and all four counts.

Examples are synthetic and labeled. Tool examples include native Qwen XML and
the corresponding Responses function-call item, preserving namespace and separate
item/call IDs. Schema examples deliberately use one known valid sample, rather
than counting the same value three times. The baseline schema task already names
the requested value; this measures formatting and schema compatibility, not
independent problem solving. Reasoning examples contain invented demonstration
text, not extracted hidden model reasoning or a fabricated encrypted payload.
SSE events and summary items remain server views; the model does not author them.

## Native tool and schema probes

```text
python tools/responses_codex_tools_probe.py --base-url http://127.0.0.1:8095/v1 --model qwen3.8-flash-next --catalog-dir docs/codex --platform windows --hint-on-failure --out <new-directory>
python tools/responses_codex_tools_probe.py --base-url http://127.0.0.1:8095/v1 --model qwen3.8-flash-next --catalog-dir docs/codex --platform ubuntu --examples 1 --out <another-new-directory>
python tools/responses_schema_inventory_probe.py --base-url http://127.0.0.1:8095/v1 --model qwen3.8-flash-next --hint-on-failure --out <another-new-directory>
```

Set `STRATA_API_KEY` in the process environment. `--examples 0` is the default;
`--hint-on-failure` must start from zero. Tool probes accept `--tool get_goal`
to select a fixture, and both probes accept `--reasoning none|medium`. Keep
the reasoning/sampling settings fixed when comparing example counts. The tool
probe generates real calls but **mocks the results and executes no tools**.

## Real Codex CLI on Windows and Ubuntu

Use the same existing authenticated Strata server, with its API monitor enabled
for the qualification receipt. Codex runs on the client OS and edits only a fresh
test project created by the harness. The LAN server performs inference.

PowerShell on Windows:

```powershell
python tools/responses_native_coding_probe.py --codex C:\path\to\codex-0.160.0.exe --base-url http://127.0.0.1:8095/v1 --hint-on-failure --out C:\tests\codex-windows-new
```

Bash on Ubuntu, using the separately pinned Linux binary:

```bash
python3 tools/responses_native_coding_probe.py \
  --codex /path/to/codex-0.160.0-linux/bin/codex \
  --codex-sha256 12eb3e81114588aca3b7998f4f19e8997b056aca08e57a7ca7c8a3ec8c652aad \
  --base-url http://127.0.0.1:8095/v1 --hint-on-failure \
  --out /path/to/tests/codex-ubuntu-new
```

Each attempt uses an isolated Codex home, fresh project and saved instruction
file. The client must read/edit/test, recover from a deliberately missing file,
leave test/sentinel bytes intact and pass an independent verifier. Requests and
typed streams are recorded without authorization headers. Interrupted runs do
not count as passes. No encrypted replay bytes are decrypted by the harness.

For an interactive session, set the documented `model_instructions_file` to the
exported `windows/codex-instructions-0.txt` or
`ubuntu/codex-instructions-0.txt` in your existing profile. Restart a fresh
session with the `-1`, `-2`, or `-3` file if you want explicit examples. The
zero-example profile is deliberately small and has no inherited prose examples;
it differs from the historical full `local-model-instructions.txt` profile.
Do not attribute a change between those two base profiles solely to examples.

The shell examples read a filename containing spaces/brackets, write exact UTF-8
bytes, and verify the result. They use PowerShell on Windows and Bash/python3 on
Ubuntu. Other tools have their own examples in the export; a coding prompt never
gets dozens of examples appended to it. The limit is three total per prompt.

See the [official Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
for `model_instructions_file`; this is client prompting, not a new API field.
