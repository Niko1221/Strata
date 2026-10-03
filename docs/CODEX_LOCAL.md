# Run Codex CLI with your own Strata model

This guide is for **`CC-David-CC/Strata-a5500`, branch `work/gbnf`**. It includes
Responses, native GBNF and JSON support. An upstream release engine or the earlier
Responses-only branch is not sufficient for Codex's automatic JSON title requests.

You run **Codex on the computer containing your project**. It reads files, edits
them and executes commands there. An Ubuntu NVIDIA server runs the model. The
client computer does not need a GPU or a model download.

```text
Your Windows or Ubuntu PC                    Ubuntu GPU server
Codex CLI -> SSH tunnel -> /v1/responses -> Strata -> local Qwen model
   |
   +-- reads/edits your project and runs its tests on your PC
```

The tested combination is **Codex CLI 0.160.0** on x86-64 Windows/Ubuntu clients,
Qwen3.8-Flash-Next **Coder IQ1_M**,
32K context and int8 KV on an RTX 4090 Ubuntu server. Real Windows and Ubuntu
clients each passed the [read/edit/verify qualification](gbnf-evidence/prompt-examples/REPORT.md).
Other client versions, GPU backends and models need their own qualification.
This is a manual setup, including one native build; it is not a one-click package.

1. [Prepare the Ubuntu model server](#1-prepare-the-ubuntu-model-server).
2. [Build this branch's native engine](#2-build-the-native-engine).
3. [Start the model server](#3-start-the-model-server).
4. [Connect from Windows](#4-connect-from-windows-powershell), or use the
   [Ubuntu client alternative](#ubuntu-client-alternative).
5. [Try a read/edit/test task](#5-try-a-small-coding-task).

## 1. Prepare the Ubuntu model server

**Run this section on the GPU host**, locally or in an SSH session. Use a host
with enough free RAM, VRAM and disk for the model; see [hardware requirements](INSTALL.md#what-you-need).
The commands below use a fresh folder named `Strata-codex`. Keep an existing
installation if you have one; its model files can be reused.

```bash
set -e
sudo apt-get update
sudo apt-get install -y git curl build-essential python3-venv
git clone --branch work/gbnf --single-branch https://github.com/CC-David-CC/Strata-a5500.git Strata-codex
cd Strata-codex
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-responses.txt -r requirements-json.txt
```

**If you already have a working Coder installation on this host**, skip the model
download below. Find its complete `strata-coder-iq1_m.json` config and keep its
absolute path for step 3. A file containing only sampling settings is not a model
config. The config needs `args`, `tokenizer`, `cwd` and the model's existing paths.

**For a first installation**, run:

```bash
./setup.sh --yes --family coder --model IQ1_M --context 32768 --kv int8 --vision no --no-start
```

This downloads/prepares the model and writes `strata-coder-iq1_m.json`. Allow about
80 GB of disk for the model installation, plus room for build tools. `--no-start`
leaves the GPU server stopped while you prepare the new engine. The installer also
downloads a standard engine; step 3 will select the branch build instead.

## 2. Build the native engine

Stay in `Strata-codex` with `.venv` activated. You need the **NVIDIA CUDA Toolkit**
as well as the driver: `nvidia-smi` alone does not prove the compiler is installed.
If `nvcc` is missing, install a toolkit and compatible C++ compiler using
[NVIDIA's Ubuntu instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
The qualified host used CUDA 13.3.73 and GCC 15.2. Use the toolkit's actual path
if it is not `/usr/local/cuda`.

```bash
export PATH="/usr/local/cuda/bin:$PATH"
nvidia-smi
nvcc --version
g++ --version
```

Stop here if these commands fail. The following build targets **RTX 4090**
(`CMAKE_CUDA_ARCHITECTURES=89`). For another card use its CUDA architecture; this
walkthrough does not qualify other cards. Both GBNF and Responses are disabled
by default, so the build flag and server switch in this guide are required.

```bash
python -c "import setup; setup.get_llama_cpp()"
python tools/prepare_xgrammar.py --out build/xgrammar-json1
cmake -S . -B build-codex -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_ENABLE_GBNF=ON \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" \
  -DSTRATA_XGRAMMAR_DIR="$PWD/build/xgrammar-json1"
cmake --build build-codex --parallel 4 --target strata grammar_native_test
ctest --test-dir build-codex -R '^grammar_native_test$' --output-on-failure
```

The first two commands download the pinned build dependencies. They do not run
the model. The build produces `build-codex/strata`; the grammar test must pass.
It checks the native grammar code, not a full Codex conversation. See the
[native GBNF guide](NATIVE_GBNF.md) for other build/test configurations.

## 3. Start the model server

**Terminal A: on the Ubuntu GPU host.** Keep it open while you use Codex.
Do not run a second model server on the same GPU for this walkthrough.

Create a separate config so the original installation stays available. Set
`STRATA_BASE_CONFIG` to the config from step 1. For an existing installation,
replace the right-hand side with its absolute path. Setup-generated configs
normally use absolute paths; keep the original `cwd` for any relative model paths.

```bash
export STRATA_BASE_CONFIG="$PWD/strata-coder-iq1_m.json"
python - <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path.cwd()
cfg = json.loads(Path(os.environ["STRATA_BASE_CONFIG"]).read_text(encoding="utf-8-sig"))
cfg.update(json.loads((root / "docs/codex/server-coding-settings.json").read_text()))
cfg["exe"] = str(root / "build-codex/strata")
cfg["model_name"] = "qwen3.8-flash-next"
cfg["experimental_responses"] = True
cfg["log"] = str(root / "strata-codex.log")
cfg.pop("vision", None)  # This Codex profile is text-only.
args = cfg["args"]
for flag, value in (("--max-context", "32768"), ("--kv", "int8"),
                    ("--suffix-draft", "0"), ("--conversation-cache-mib", "0")):
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args.extend((flag, value))
cuda = Path(shutil.which("nvcc")).resolve().parent.parent
libs = [str(p) for p in (cuda / "lib64", cuda / "targets/x86_64-linux/lib") if p.is_dir()]
cfg["lib_dirs"] = libs + cfg.get("lib_dirs", [])
target = root / "strata-codex.json"
with target.open("x", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print("Created", target)
PY
```

This chooses the new binary and a 32768-token context matching the Codex catalog.
It keeps the existing model paths and hardware allocation. The coding settings
are temperature **1.0**, top-p **0.95**, top-k **20**, min-p **0**, no presence or
frequency penalty, repetition penalty **1.0** (off), and a **2048-token** local
thinking budget. These are the settings used in the real client tests, not a
guarantee that a model will never loop. The tested setup used ordinary MTP with
`--spec 4`; the script preserves your existing MTP configuration.

Choose a long private API key and enter the **same value on server and client**.
This is your Strata key; you do not need an OpenAI API key or an OpenAI login.
The replay key is a separate secret used only by the Strata server.

```bash
read -r -s -p 'Enter your Strata API key: ' STRATA_API_KEY; printf '\n'
export STRATA_API_KEY
export STRATA_RESPONSES_REPLAY_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
python serve/server.py --engine strata --config strata-codex.json \
  --host 127.0.0.1 --port 8095 --experimental-responses
```

Wait for the server to finish loading and print its listening address. Keep the
same replay key when restarting if you want to resume earlier conversations;
generating a new one invalidates their encrypted replay items. Keep it in your
private deployment environment, not in Git or a shared transcript. For a first
test, the session key above is enough. The server remains on loopback; the SSH
tunnel below carries the client connection.

Use this explicit server command on later runs, with `.venv` activated and both
keys set. The ordinary `run-coder-*.sh` launcher still selects its original config.

## 4. Connect from Windows PowerShell

### Terminal B: open the tunnel

Run this **on your Windows PC**, replacing `YOUR_USER@YOUR_SERVER` with the SSH
login for the GPU host. A configured alias such as `llm-49` also works.

```powershell
ssh -N -L 127.0.0.1:8095:127.0.0.1:8095 -o ExitOnForwardFailure=yes YOUR_USER@YOUR_SERVER
```

After SSH authentication the window normally stays blank. Leave it open. If port
8095 is already in use locally, use another free local port and change every
client URL and the profile's `base_url` to match. The remote port stays 8095.

### Terminal C: install the pinned client and profile

Open a **second Windows PowerShell window**. You need Git and SSH installed.
The small coding task below also uses Python 3 on this PC. Check with
`git --version`, `ssh -V` and `python --version`.

These commands create a separate client folder and Codex home. Use a fresh folder
the first time; on later runs use [the shorter launch commands](#start-codex-again).
Only repository files are needed here; do not install the model on the client.

```powershell
$ErrorActionPreference = 'Stop'
$StrataClient = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'Strata-codex-client'
New-Item -ItemType Directory -Path $StrataClient -Force | Out-Null
$StrataSource = Join-Path $StrataClient 'source'
git clone --branch work/gbnf --single-branch https://github.com/CC-David-CC/Strata-a5500.git $StrataSource
if ($LASTEXITCODE -ne 0) { throw 'Clone failed; check the destination before continuing.' }

$CodexExe = Join-Path $StrataClient 'codex-0.160.0.exe'
Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/openai/codex/releases/download/rust-v0.160.0/codex-x86_64-pc-windows-msvc.exe' -OutFile $CodexExe
$ExpectedHash = 'fdda5fa3cf3fb3d000b876720742857676293e4315e4b045fae6f8bd7e866d1d'
if ((Get-FileHash -LiteralPath $CodexExe -Algorithm SHA256).Hash -ne $ExpectedHash) { throw 'Codex checksum mismatch.' }
& $CodexExe --version

$env:CODEX_HOME = Join-Path $StrataClient 'codex-home'
New-Item -ItemType Directory -Path $env:CODEX_HOME -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $StrataSource 'docs/codex/model-catalog-0.160.0.json') -Destination $env:CODEX_HOME
Copy-Item -LiteralPath (Join-Path $StrataSource 'docs/codex/prompt-examples/windows/codex-instructions-0.txt') -Destination (Join-Path $env:CODEX_HOME 'local-model-instructions.txt')
$ProfileText = [IO.File]::ReadAllText((Join-Path $StrataSource 'docs/codex/strata.config.toml'))
$ProfileText = $ProfileText.Replace('/ABSOLUTE/PATH/TO/model-catalog-0.160.0.json', (Join-Path $env:CODEX_HOME 'model-catalog-0.160.0.json').Replace('\', '/'))
$ProfileText = $ProfileText.Replace('/ABSOLUTE/PATH/TO/local-model-instructions.txt', (Join-Path $env:CODEX_HOME 'local-model-instructions.txt').Replace('\', '/'))
[IO.File]::WriteAllText((Join-Path $env:CODEX_HOME 'strata.config.toml'), $ProfileText, (New-Object Text.UTF8Encoding $false))
$env:STRATA_API_KEY = Read-Host 'Enter the same Strata API key used on the server'
```

The version must be `codex-cli 0.160.0`. The profile selects `strata-local` as the
provider, `wire_api = "responses"`, `qwen3.8-flash-next`, and medium reasoning.
The two absolute file paths are filled in for your PC. The profile belongs in
`CODEX_HOME`, not your project's `.codex` directory. Its settings follow the
[official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

### Check the connection and JSON support

Still in Terminal C:

```powershell
$Headers = @{ Authorization = "Bearer $env:STRATA_API_KEY" }
Invoke-RestMethod -Uri 'http://127.0.0.1:8095/health'
Invoke-RestMethod -Headers $Headers -Uri 'http://127.0.0.1:8095/v1/models'
$Body = @'
{"model":"qwen3.8-flash-next","store":false,"input":"Return an object with ready set to true.","reasoning":{"effort":"none"},"max_output_tokens":128,"text":{"format":{"type":"json_schema","name":"connection_check","strict":true,"schema":{"type":"object","properties":{"ready":{"type":"boolean"}},"required":["ready"],"additionalProperties":false}}}}
'@
$Reply = Invoke-RestMethod -Method Post -Headers $Headers -ContentType 'application/json' -Body $Body -Uri 'http://127.0.0.1:8095/v1/responses' -TimeoutSec 180
$Reply | ConvertTo-Json -Depth 20
if ($Reply.status -ne 'completed') { throw 'The JSON check did not complete. Check the response and server log.' }
```

Health should show `loaded: true`, `max_context: 32768`, and `api_key: true`.
The Responses object should have `status: completed` and an assistant message
whose text is a JSON object such as `{"ready":true}`. A 404, 401 or failed response
needs fixing before you start Codex; see [troubleshooting](#troubleshooting).

### Launch Codex

Start with a disposable practice folder. For your own project, set `$Project`
to its existing absolute path instead. Codex will be able to edit that folder.

```powershell
$Project = Join-Path $StrataClient 'practice'
New-Item -ItemType Directory -Path $Project -Force | Out-Null
git -C $Project init
& $CodexExe --no-daemon --strict-config --profile strata --sandbox workspace-write --ask-for-approval on-request --cd $Project
```

If Codex asks whether to trust this new folder, inspect and confirm the folder you
just selected. The model should be `qwen3.8-flash-next` with medium reasoning, with
no missing-model-metadata warning. Continue to [the first task](#5-try-a-small-coding-task).

### Start Codex again

On later runs, start Terminal A and the tunnel again if they are stopped, then
run this in a fresh PowerShell window. Do not repeat the download/profile setup.

```powershell
$StrataClient = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'Strata-codex-client'
$env:CODEX_HOME = Join-Path $StrataClient 'codex-home'
$env:STRATA_API_KEY = Read-Host 'Enter your Strata API key'
& (Join-Path $StrataClient 'codex-0.160.0.exe') --no-daemon --strict-config --profile strata --sandbox workspace-write --ask-for-approval on-request --cd (Join-Path $StrataClient 'practice')
```

## Ubuntu client alternative

Use this section instead of the Windows client steps. Run it **on the Ubuntu
computer containing the project**. It can be the model server itself; in that
case skip SSH forwarding and use the server's loopback address directly.

For a separate client, keep this command running in its own terminal, substituting
your SSH login:

```bash
ssh -N -L 127.0.0.1:8095:127.0.0.1:8095 -o ExitOnForwardFailure=yes YOUR_USER@YOUR_SERVER
```

In a second terminal, with Git, curl, tar and Python 3 installed:

```bash
set -e
strata_client="$HOME/Strata-codex-client"
mkdir -p "$strata_client/codex-0.160.0" "$strata_client/codex-home"
git clone --branch work/gbnf --single-branch https://github.com/CC-David-CC/Strata-a5500.git "$strata_client/source"
curl --fail --location https://github.com/openai/codex/releases/download/rust-v0.160.0/codex-package-x86_64-unknown-linux-musl.tar.gz \
  --output "$strata_client/codex.tar.gz"
printf '%s  %s\n' '4fcc47ab57f52ff75363951a8761146cd10c8288bd86fed45487dbb204a16b71' "$strata_client/codex.tar.gz" | sha256sum --check
tar -xzf "$strata_client/codex.tar.gz" -C "$strata_client/codex-0.160.0"
"$strata_client/codex-0.160.0/bin/codex" --version
export CODEX_HOME="$strata_client/codex-home"
export STRATA_CLIENT_SOURCE="$strata_client/source"
python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path

source = Path(os.environ["STRATA_CLIENT_SOURCE"]) / "docs/codex"
client = Path(os.environ["CODEX_HOME"])
shutil.copy2(source / "model-catalog-0.160.0.json", client)
shutil.copy2(source / "prompt-examples/ubuntu/codex-instructions-0.txt", client / "local-model-instructions.txt")
profile = (source / "strata.config.toml").read_text()
for name in ("model-catalog-0.160.0.json", "local-model-instructions.txt"):
    profile = profile.replace('"/ABSOLUTE/PATH/TO/' + name + '"', json.dumps(str(client / name)))
(client / "strata.config.toml").write_text(profile, encoding="utf-8")
PY
read -r -s -p 'Enter the server Strata API key: ' STRATA_API_KEY; printf '\n'
export STRATA_API_KEY
curl --fail --silent --show-error http://127.0.0.1:8095/health
curl --fail --silent --show-error -H "Authorization: Bearer $STRATA_API_KEY" http://127.0.0.1:8095/v1/models
curl --fail --silent --show-error --max-time 180 \
  -H "Authorization: Bearer $STRATA_API_KEY" -H 'Content-Type: application/json' \
  --data '{"model":"qwen3.8-flash-next","store":false,"input":"Return an object with ready set to true.","reasoning":{"effort":"none"},"max_output_tokens":128,"text":{"format":{"type":"json_object"}}}' \
  http://127.0.0.1:8095/v1/responses
```

Before starting Codex, check that health shows `loaded: true`, `max_context: 32768`
and `api_key: true`. The JSON request must return `status: completed` and a message
containing an object such as `{"ready":true}`. If it fails, use
[troubleshooting](#troubleshooting) before continuing. Then launch:

```bash
mkdir -p "$strata_client/practice"
git -C "$strata_client/practice" init
"$strata_client/codex-0.160.0/bin/codex" --no-daemon --strict-config --profile strata \
  --sandbox workspace-write --ask-for-approval on-request --cd "$strata_client/practice"
```

Keep the Linux package together; it contains the client's sandbox and
support programs. Do not copy only `bin/codex` to another directory.

For later launches, set `strata_client`, `CODEX_HOME` and `STRATA_API_KEY` again,
then run the final Codex command. Keep the server and, if needed, the tunnel running.

## 5. Try a small coding task

In the new practice folder, give Codex this task:

```text
Create calculator.py with add(a, b). Create test_calculator.py using Python's
unittest, with positive, negative and zero cases. Run the tests and show the
result. Use the shell available on this computer. Stop when the tests pass.
```

Then exercise reading and editing existing files:

```text
Read the files you just created. Add subtract(a, b) and a test for subtract(2, 5).
Run the tests again. Show the diff and the actual test result, then stop.
```

You should see shell calls, their results, edits in the **client's** practice
folder and a passing test command. A prose claim that tests passed is not enough;
inspect the tool result. To try steering, interrupt a turn with Escape and send
a new instruction such as `Stop editing. Inspect the current diff and report it.`
Use `/quit` to leave Codex. Then close the tunnel with Ctrl+C and, when finished
with the model, stop the Strata server with Ctrl+C in Terminal A.

The [qualification harness](PROMPT_EXAMPLES.md#real-codex-cli-on-windows-and-ubuntu)
does a larger automated test, including malformed input, failed commands,
Unicode, newline handling and independent checks of the edited program.

## Baseline and hinted prompts

The setup selects **zero added task examples**. If the model struggles, exit Codex
and replace `CODEX_HOME/local-model-instructions.txt` with the same platform's
`codex-instructions-1.txt`, `-2.txt` or `-3.txt` from
[the committed profiles](codex/prompt-examples). Start a fresh conversation for a
fair comparison. The ordinary native tool-format template remains in all variants.
Examples are explicit client instructions; the server does not add a hidden hint
or silently repair an invalid generated call.

See [readable PowerShell/Bash examples, all schema fixtures and test commands](PROMPT_EXAMPLES.md).
These include the empty-argument `get_goal` case: the recorded baseline invented
an `arguments` field, and one example corrected it. Passing with a hint does not
erase the failed baseline.

## Troubleshooting

| What you see | What to check |
| --- | --- |
| Connection refused / SSH window exits | Terminal A must have finished loading; Terminal B must stay open. Check the SSH destination and both port numbers. |
| 401 Unauthorized | Use the same nonempty `STRATA_API_KEY` in Terminal A and the Codex terminal. Restart Codex after changing its environment. |
| 404 on `/v1/responses` | Check out this branch and start this Python server with `--experimental-responses`. A different service or the base installer command may still own the port. |
| JSON request rejected as unsupported | Select the newly built `build-codex/strata` in `strata-codex.json`, install `requirements-json.txt`, and restart the server. JSON needs native `grammar=gbnf-v4`. |
| Replay-key startup error | Set `STRATA_RESPONSES_REPLAY_KEY` using the server command above. Keep it stable to resume older chats; a different key cannot read their replay items. |
| Missing model metadata | Confirm Codex 0.160.0, `--profile strata`, the chosen `CODEX_HOME`, the catalog's absolute path, and model name `qwen3.8-flash-next`. |
| `System message must be at the beginning` after steering | Confirm the Python server is running current `work/gbnf` source and restart it. This branch normalizes later developer/system instructions for the native template. |
| Failed tool call or repetitive commands | Inspect the actual error, interrupt, and give a focused correction. Try the explicit example profiles; they cannot guarantee model correctness. |
| Context/compaction failure in a long chat | This profile advertises 32768 tokens. Start a new conversation before the context fills; server compaction is not implemented. |
| `strata-codex.json` already exists in step 3 | It is deliberately not overwritten. Reuse it after checking its paths/settings, or choose a new output filename and pass that to `--config`. |

## Supported scope and evidence

The endpoint is stateless: `store:false`, full-history replay and client-owned
function tools. The catalog selects the tested shell tools for edits; it does not
enable Lark `apply_patch`. Hosted tools, image input through Responses, background
jobs, storage, compaction and WebSockets remain unsupported. Strata can replay
its own encrypted reasoning items with the deployment key, not another provider's
encrypted state. Requested reasoning summaries use a separate bounded generation.

JSON answer schemas use native constraints plus validation against the original
schema. An unsatisfiable or unrepresentable schema can be rejected before
generation; validation failures cannot be reported as successful output. Strict
function arguments are validated before a completed call is emitted. Tool results
and reasoning are not constrained by an answer schema. See [JSON output](JSON_OUTPUT.md),
[all captured tool schemas](CODEX_TOOL_SCHEMAS.md), and [Responses capabilities](RESPONSES.md).

The [latest Windows/Ubuntu report](gbnf-evidence/prompt-examples/REPORT.md) links
actual client requests, tool calls/results, tests and limitations. The
[steering checkpoint](responses-evidence/steering/REPORT.md) records an interrupted
turn and a successful next instruction. These tests used Ubuntu CUDA inference;
they do not establish Windows native inference, HIP or multi-GPU qualification.
