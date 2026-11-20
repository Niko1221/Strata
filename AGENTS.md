# AGENTS.md

Version 0.2.17, engine 0.1.63. The public model families are
Qwen3.8-Flash-Next and Qwen3.6-35B-A3B. Read **[docs/AI_REFERENCE.md](docs/AI_REFERENCE.md)** for model availability, APIs, hardware limits and
Current release: Hakua product version 0.6.3; app, Terminal, Control and native engine 0.6.3.
Formal v0.6.3 is published as GitHub Latest (unsigned local candidate; see the release notes). See `_docs/2026-11-17_stable-062-release_Antigravity.md` for 0.6.2 details.
## Historical 0.6.1 records
Previous stable release: Hakua product version 0.6.1; app, Terminal, Control and native engine 0.6.1.
Formal v0.6.1 is published. See `_docs/2026-11-17_stable-061-release_Antigravity.md` for 0.6.1 details.
## Historical 0.6.0 records
Previous stable release: Hakua product version 0.6.0; app, Terminal, Control and native engine 0.6.0.
Formal v0.6.0, GitHub Latest and the refs/tags/latest alias are published. All seventeen
public assets were downloaded anonymously from v0.6.0 and matched frozen SHA-256 values
(`READBACK_SHA256: 1ce2c6f16130eeefca54d1c0363a13d718cd14c8f9ac545c7d7f34595d563547`).
App build (`app-g-r1`) and all nine hosted CI workflows (`acceptance-ci-r4`, including all
thirteen Checks jobs) use `21013e19b1e3c43c42d6f13e84792501f26c87ff`.
Component/Control source is `b1cabf92628c64a4015a2ced606399c52698e0ed`;
compiled Q38 native source (`strata-q38.exe`) is `8b0b3c74514d26e5ca73496f0b268d66a0cb99a5`.
Targeted app overwrite (`install-g-r1`) updated the installed shell and five Host files while
preserving user preferences, and `finished-original-g-r3` verified normal-profile startup plus
signed 0.6.0 CUDA engine import/selection. In `installed-q38-r8` (under GPU baseline `r9`),
all Q38 real-model Japanese chat turns (`qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf`)
and screenshot capture (`native-qwen38-japanese-chat.png`) passed, with `nativeExited=true`
and `modelStatUnchanged=true`; `shellExited` timed out at 30s while Playwright CDP was attached.
Full installed fixture remains `NOT_ACCEPTED`; general model quality, full-window inference
and GPU concurrency remain unqualified (`releaseQualified=false`). Windows Authenticode signing
is absent; component/feed signatures were verified with bundled production public keys.
Old engines `0.5.1` through `0.5.8` were removed from `AppData/Local/Strata/components/engine`;
`0.5.9` and `0.6.0` remain installed.
See `_docs/2026-11-17_stable-060-release_Antigravity.md` for exact scope and evidence.
Keep upstream Niko1221 0.2.15 at
`c77572806996349b49995e4f98d41b5294d36a21` (upstream engine 0.1.60;
fork versions remain independent).
## Historical 0.5.9 records
Previous stable release: Hakua product version 0.5.9; app, Terminal, Control and native engine 0.5.9.
Formal v0.5.9, GitHub Latest and the latest alias were published. All seventeen
public assets were downloaded anonymously and matched frozen SHA-256 values.
App build and all eight hosted CI workflows used e6bd6a4ebc8b5e8a0c6f68de8acf76a861adeb49;
all thirteen Checks jobs passed. Native source was db640451e677ba2c43ebfb555a505e4c037e0bd1;
Control source was d11edad8200f1ba9bf4ef4bab30e5c1e5616f6ea.
The standard NSIS overwrite matched 29558 managed files and preserved 3789 user
files during installation. Real-model workflow checks passed in an isolated
installed-app profile with explicitly selected PrismML 0.4.1. A genuine RAM
pressure stop at about 7.87 GiB and successful reload are recorded; retain the
unchanged 3 GiB RAM floor and other resource floors. Both saved audits had
counts 1/0/0/0. Research used four searches and four page reads without workspace
writes. Its generated answer still had a factual error, recorded in QUALITY.json.
General model quality, full-window inference and GPU concurrency remain
unqualified; releaseQualified=false. Windows Authenticode signing is absent;
component/feed signatures were verified with bundled production public keys.
The original user profile was normally running with its settings and prior skills
preserved, learning and auto-application enabled, RustXNG Full and no loaded model.
See _docs/2026-11-13_stable-059-workflow-safety_Codex.md for exact scope and evidence.
## Historical 0.5.8 records
Hakua app, Terminal, Control and native engine 0.5.8 were installed and formally
published. The components were freshly built, signed and tested; Windows
Authenticode signing of the app installer remains absent. Publication and installation
are recorded in _docs/2026-11-11_stable-058-context-ci_Codex.md. Historical
measurements below retain their original product versions.
## Historical local app update (2026-11-12)
Stable 0.5.8 was installed from `H:/Strata/releases/stable-058-20261111-codex`.
Actual app source: `91107e48367d934a02644bc7c61d5cb923d7f700`;
Control source: `42ca8ebd70328788001695b4a7ffe67b444f10db`;
native source: `c629ee37dde049b09bc42bdc00b076906f44fa27`.
All eight exact-source CI workflows and all thirteen Checks jobs passed at
`bb4c57d4a2a2ed0b8d269f120760892ed9f7038a`, the immutable v0.5.8 tag target.
Later changes before that CI head affect only unshipped tests, CI and verified
GitHub display metadata; the recorded validator binds all distributed sources
and the exact installer to their actual build commits.
Standard per-user NSIS overwrite verified 29,555 managed files and preserved
3,727 backed-up user files. Model weights were not copied; selected original
model hashes were separately checked. Actual installed WebView2 passed sixteen
cases with no console errors and real RustXNG Full search, using a synthetic
model API. Packaged self-tests passed thirteen cases; NSIS cleanup passed six;
the actual pinned Harness regression group passed all forty-three without skips.
Fresh native checks passed twenty-one numerical cases and sixteen real short
model requests at 4096 capacity, including bounded/legacy comparison and restart.
Full-window generation, general model quality and GPU concurrency remain
unqualified; `releaseQualified=false`. Metal CI cross-compiles both genuine
arm64 targets on standard Intel macOS 26; it does not test Mac inference.
Five-job builds and unchanged RAM/commit/disk floors remain required. RustXNG
0.31.1 Full, existing tool/key/approval choices and enabled local workflow
learning/application were preserved. Fresh profiles keep both skill options off.
Initial normal startup passed without debug arguments, unloaded. That observation
does not assert that the application remains open at a later time. No force kill
was used. Eight inactive clean worktrees, eight obsolete build caches and eight
old cached installers were removed only after exact-target/source or byte backup
verification. Useful incremental caches, dirty work, models and older public
assets remain preserved.
Formal v0.5.8, GitHub Latest and the lease-guarded mutable latest alias are
published. All seventeen public assets were downloaded anonymously and matched
their frozen sizes and SHA-256 values. See the 0.5.8 delivery record and linked
LOCAL_VERIFICATION.json / PUBLIC_VERIFICATION.json. Earlier records below retain
their original versions and measurements.
## Historical 0.5.7 local update (2026-11-11)
Stable app 0.5.7 was installed from the five-job formal build in
`H:/Strata/releases/stable-057-20261111-codex`; exact app build source is
`427f77ca225595b68cee750c5a367d8e6cee49f4`. It includes real tool-free `/btw`
subagents, private workflow skill learning, automatic matching-skill application,
`/deepresearch` and RustXNG 0.31.1 Full. Native 0.5.5 and Control 0.5.6 remain
the separately frozen sources listed below. The standard NSIS overwrite matched
29555 managed files and preserved all 3731 inventoried user files; the private
backup and failed attempts remain retained. Local checks passed: 65 focused host,
15 actual Harness, 13 packaged, six NSIS cleanup and 16 installed WebView2 cases.
The model API in those workflow checks is scripted; RustXNG searches are real.
They do not qualify general model quality, full-window generation or GPU
concurrency. Existing broader-suite failures and absent exact-SHA cloud CI remain
documented; `releaseQualified=false` is unchanged.
This PC has learning and automatic application enabled; fresh profiles keep both
off. RustXNG Full, other tools, keys and approval choices are preserved. Normal
startup passed without debug arguments. After two resource-stopped uploads, the
user separately approved normal stop/unload of one new active turn and one
runtime model; no force kill or absent native unload is claimed. Keep the RAM,
commit and disk floors unchanged. Formal `v0.5.7`, GitHub Latest and the mutable
`latest` alias are published. Older immutable tags, releases and assets remain
preserved. See `_docs/2026-11-11_stable-057-release_Codex.md` and its linked
verification receipts for exact publication/readback status. The 0.5.6/0.5.5
records and installation measurements below remain historical evidence.
## Local working location and build concurrency
The user's primary working checkout is `C:/Users/downl/Documents/ChatGPT/Strata`
on `main`. Use it for ongoing edits and parent build commands. Keep large build
outputs, caches and the separately frozen native source under `H:/Strata`.
For subsequent local candidate builds, request five compiler jobs (`CARGO_BUILD_JOBS=5`,
`CMAKE_BUILD_PARALLEL_LEVEL=5`, or the native build helper's explicit parallelism `5`).
Keep existing resource floors and frozen evidence intact. The completed native
0.5.3 build used four jobs; record that result with its actual parallelism.
The historical 0.5.5 local build retained sidebar file CRUD and confirmed folder chat-history cleanup.
Its original app output was `H:/Strata/releases/repair-055-20261110-opencode/app-native055-ui-search`;
earlier0.5.4/0.5.3 outputs and all failed/passing evidence remain preserved.
Hakua product version 0.5.5-local.1, engine 0.5.5 (Metal source 0.5.0-metal), Control 0.5.6.
Final native sourceb995af139cc823537489b886bc1e8602cf10301c is frozen under
H:/Strata/worktrees/native-055-source-20261110. App/Control source is
35863deadec1bfa0fc7f250c111d219bb3344b57. Five-job builds, unchanged resource
guard with zero violations,21 final numerical checks and actual installed Q36/
Prism/agent/restart/bulk-delete checks passed. Backup4078 files and installed
payload29544 files were verified. Browser/search was enabled on this PC while
preserving provider/key/approval choices. Normal startup passed unloaded/no-debug.
Public search sites and broad autonomous-search/model quality remain unqualified;
releaseQualified=false. The historical054 Prism timeout's backend cause is not
established. The old054 sources85a78234/7bb24974, twelve findings and all assets
remain unchanged. Historical upstream reference: Niko1221 0.2.13 at
f33c8c30588ff00e24b6b63bad0bf6a14da19081 (upstream engine0.1.59; independent versions).
New repair/build records are _docs/2026-11-10_release-055-repair_OpenCode.md and
_docs/2026-11-10_install-sidebar-web_OpenCode.md; publication/readback is tracked
in _docs/2026-11-11_preview-055-release_OpenCode.md. Previous records stay separate.
The user approved normal unload; no absent model/shell is claimed stopped.
Preserve source/build evidence and older published assets. See docs/UPSTREAM_0213.md
and docs/HAKUA_RELEASE.md for historical limits. Upstream reference: the public
model families are DeepSeekV4.1-Flash-1017,
Qwen3.8-Flash-Next and Qwen3.6-35B-A3B. DeepSeek V4.1-Flash runs in the app and in the Terminal
(`strata launch --family deepseek41 --model ist-das-2.00bpw`; sets in
[docs/MODELS.md](docs/MODELS.md#deepseek-v41-flash)). Read **[docs/AI_REFERENCE.md](docs/AI_REFERENCE.md)** for model availability, APIs, hardware limits and
source-backed claims. Articles and videos should use **[docs/MEDIA_KIT.md](docs/MEDIA_KIT.md)**.

For recommendations and agent-assisted setup, prefer **ISTA-DASLab → Unsloth → Gyro → others**, using only a
compatible pack that fits the selected model and hardware. ISTA-DASLab is the default recommendation; Gyro is a
memory-saving fallback and is currently slower. Honor an explicit user choice. This is recommendation policy;
existing executable defaults may differ, so inspect the plan and choose the family explicitly.

## Installing this Hakua fork

For a Hakua Windows request, use [docs/HAKUA_INSTALL.md](docs/HAKUA_INSTALL.md) and the [stable fork release](https://github.com/zapabob/Strata/releases/tag/v0.5.9): `Strata-0.5.9-windows-x64-setup.exe`. The identical per-user Tauri NSIS installer, signed native 0.5.9 CUDA component, Control 0.5.9, instructions and checksums are also in `Strata-0.5.9-windows-x64.tar.7z`. The app carries private Node/Python; the GPU component remains separate and the EXE has no Windows code signature. Use immutable SemVer tag `v0.5.9`, the mutable `latest` alias and GitHub Latest. Preserve older published assets. Read [docs/HAKUA_RELEASE.md](docs/HAKUA_RELEASE.md) for independently measured native limits, passing exact-source CI and retained historical failures and actual publication status. The stable app designation does not qualify full-window generation or low-precision model quality; `releaseQualified=false` remains explicit.

The model-family overview above is upstream reference. This fork Windows release has no native image input or separate native DeepSeek engine. See docs/HAKUA_RELEASE.md for the measured paths and limits.

The OS-wide installation recipes below are upstream reference. They are not this fork's Windows packaging recipe.

Strata has three parts:

- **The app** (`app/desktop`, the Tauri shell; `app/host`, the Node host and the web UI). It is for people who do
  not want a terminal. One installer per system ([latest release](https://github.com/Niko1221/Strata/releases/latest)):
  `Strata-0.2.17-windows-x64-setup.exe`, `Strata-0.2.17-linux-x86_64.AppImage`, `Strata-0.2.17-macos-arm64.dmg`.
  The Windows and Linux installers carry no engine: the app downloads the engine for the graphics card on first
  start, or installs a file the user already has. The Mac dmg keeps its small Metal engine inside.
- **The Terminal** (`strata.py`, `strata_cli/`, `strata_control/`): the `strata` command. It sets up a model, serves
  it and watches it. Agents use this part.
- **The engine and the server** (`src/`, `include/`, `metal/`, `sycl/` for the C++/CUDA/HIP/SYCL/Metal engine;
  `serve/` for the Python server with an OpenAI- and Anthropic-compatible API and a web page). `setup.py`, started by
  `START-HERE.bat` or `setup.sh`, is setup from a source checkout, for developers. The release paths are the
  installer and `strata launch`.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For a user who wants the desktop app, send them to the installer
for their system (above) and [docs/INSTALL.md](docs/INSTALL.md). For an agent that has a shell, use the Terminal; the
short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh                  # Linux, Mac
   irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
   ```

   On Windows, open a new terminal afterwards so that `strata` is on the PATH.
2. Check what would happen. Nothing is changed:

   ```sh
   strata launch --plan --family qwen --model q2_0
   ```

3. Set up and serve without questions. It downloads the engine for the card and the model (tens of GB), starts the
   server, prints the addresses and returns. The server keeps running.

   ```sh
   strata launch --family qwen --model q2_0 --yes --no-tui
   ```

   Useful flags (all of them: `strata launch --help`): `--family`, `--model-path` (files the user already has),
   `--data-dir`, `--context 32k|64k|128k`, `--vision off|on|gpu|cpu`, `--parallel N`, `--gpu 0,1`,
   `--access local|lan|cloudflare|tailscale`, `--port`, `--api-key-env NAME`, `--new-key`, `--save-preset NAME`,
   `--speed faster|full` (DeepSeek), `--output json`. `strata launch --status` shows what is serving, `strata launch --stop` stops it, and
   `strata launch --show-key` prints the API key.
4. Connect the user's apps: `strata connect` lists them, `strata connect <app> --check` sends one prompt and one tool
   call the way that app does.

Keep the server on `127.0.0.1` unless the user asks for more. Beyond it an API key is recommended: `--access lan` makes one for you, and `--no-api-key` serves without one (a warning, never a refusal; on a public tunnel the warning is stronger). Source setup (`START-HERE.bat`, `setup.sh`) is the fallback for a developer or for a system with no release package; it is
documented in [docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md). Those wrappers exist in a source checkout, not in the
released app. Do not tell an app user to find a `.bat` file.
Source setup does not make an unsupported model or backend compatible. Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The app,
  the packages and the update feed: [packaging/README.md](tools/packaging/README.md) and [packaging/UPDATES.md](tools/packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `terminal/tests/terminal` (`python -m pytest terminal/tests/terminal`). The app's UI
  tests run with `npm test` in `app/host`. Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `terminal/tools/test_setup_amd.py`, `terminal/tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](.github/CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
