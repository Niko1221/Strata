# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

## Escalating hard reasoning

Reason locally when the answer follows from the code or a test. Hand it to a stronger
model through the user when it does not: an architectural choice between competing
designs, two failed attempts at one problem, a blocker the available evidence cannot
settle, analysis across several subsystems, uncertainty about a consequential decision,
or a destructive change needing review.

Two distinct failed approaches are the budget; a third speculative cycle is not
attempted. Routine edits, searches, and anything a few tool calls settle do not escalate.

When it triggers, stop: no more edits or retries, do not report the task complete,
leave the tree as it is, and say what was already changed. Then produce the
`ESCALATION REQUIRED` block from the `escalate` skill вЂ” objective, current state, exact
problem, real evidence with `file:line`, attempts made, and what is being asked вЂ” with a
copy-ready prompt, and wait for the answer.

Real error text, real paths, real numbers; never an invented result. No secrets or
credentials in the prompt. On the answer, check its APIs, paths and commands against
this repository before applying it, continue from the current state without repeating
finished investigation, and escalate again only for a genuinely new blocker.
## Parking on Windows (local finding, 0.1.42)
The engine on this PC runs out of the system commit limit, not physical RAM: the locked expert
arena and the WDDM device allocations are charged against commit. `conversation_memory.cpp`
admits on `min(ullAvailPhys, ullAvailPageFile)`; that is the number to read. When parking is
refused, `LOCAL-PARKING-PATCH-0.1.42.md` lists what the local tree changes on top of upstream
and why, and the refusal lines now print both resources and the process' own commit.
Build notes for this machine: the old `build_cuda`/`build_tests` caches are configured for a
different path and cannot be re-rooted - configure fresh with
`-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=.../_deps/strata_llamacpp-src` to avoid a re-download.
