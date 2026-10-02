<h1 align="center">Strata-V100</h1>

> **Tesla V100 fork.** This fork keeps Strata compatible with NVIDIA Volta (`sm_70`) and is validated on a
> Tesla V100-PCIE-16GB, Ryzen 5 3600, 48 GB DDR4-3200, and CUDA 12.8. The installed Qwen3.8-Flash-Next Q2_0
> configuration provides the model's full **262,144-token context**, int8 KV cache, and MTP speculative decoding.
> The bundled web chat and live performance monitor listen on port `8088`; open
> `http://<host-or-lan-address>:8088/`. API access is protected by the key stored locally in `.strata-service.env`.
> The single-card benchmark below uses physical GPU1 on its PCIe Gen3 x16 link.

> | Target | Median prompt tokens | Prefill (tok/s) | Decode (tok/s) | Max GPU temp |
> | ---: | ---: | ---: | ---: | ---: |
> | 1K | 1,036 | 398.6 | 52.0 | 56 °C |
> | 4K | 4,111 | 1,214.0 | 50.8 | 61 °C |
> | 8K | 8,185 | 1,442.4 | 48.1 | 63 °C |
> | 16K | 16,367 | 1,490.7 | 49.8 | 67 °C |
> | 32K | 32,765 | 1,495.7 | 47.3 | 72 °C |
> | 64K | 65,533 | 1,484.3 | 43.0 | 81 °C |
> | 128K | 131,063 | 1,033.0 † | 40.9 | 84 °C |
> | 256K | 256,080 | 617.8 † | 36.8 | 84 °C |
>
> Medians of three fresh requests per target, measured 2026-10-02 on physical GPU1 only:
> a single Tesla V100-PCIE-16GB on a PCIe Gen3 x16 link. Both the engine and vision encoder
> used GPU1. GPU0 was not used for inference. Every request was uncached (`reused=0`, seed 20261002) and wrote exactly
> 256 output tokens. The installed Qwen3.8-Flash-Next Q2_0 configuration runs the full
> 262,144-token context with an int8 KV cache, `--prefill auto`, `--spec 8`, draft floor 0.70,
> the paired expert variant, and a 700 MiB vision reserve. Timings come from the engine's
> `/metrics`, not from client wall clocks.
>
> Cooled protocol: before every measured request the card idled at least 120 s, until it was at
> or below 55 °C without software thermal slowdown for 15 s. The gate checked every 3 s;
> separate telemetry ran at 1 Hz. Actual start
> temperatures were 51-55 °C; actual waits were 121.4-588.7 s (about 2-9.8 min). The passively
> cooled card showed no software thermal slowdown through 64K (max 81 °C). The 128K and 256K
> prompts heated it to 84 °C and it throttled during the prompt despite the cooled start; long
> requests may throttle after a cooled start. All requests were kept; none were removed.
>
> † Software thermal slowdown active during the prompt (84 °C).
>
> Engine: the fork's version 0.1.31 (CMake project version), installed build commit `78417ea`
> (SHA-256 `512b1f25d60479a2ddb66fcf1ddca5407963e45378a3c8a463413ad159edae17`). PR #12 merged
> into `main` at `7fbe49a`. Raw rows and methodology:
> [`summary.json`](bench/results/2026-10-02-v100-single-cooled/summary.json),
> [`protocol.json`](bench/results/2026-10-02-v100-single-cooled/protocol.json),
> [`matrix.json`](bench/results/2026-10-02-v100-single-cooled/matrix.json),
> [`completed-requests.json`](bench/results/2026-10-02-v100-single-cooled/completed-requests.json),
> and [`gpu.csv`](bench/results/2026-10-02-v100-single-cooled/gpu.csv) (1 Hz telemetry).
> [Full methodology and timings](docs/DETAILS.md#tesla-v100-fork-benchmark).
>
> This table replaces the September table, which used 64 output tokens, a pre-merge branch, and
> a different protocol. The two runs are not a controlled A/B, so no gains are claimed against
> that table. Use the recorded protocol to repeat the sweep. The benchmark harness does not
> pause automatically; add the documented cooldown gate between measured requests.

<p align="center"><b>Run a 125-billion-parameter AI model on your own gaming PC</b><br>
NVIDIA or AMD graphics card (12 GB or more) · Windows or Linux · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** - a large, smart AI model that
normally needs a server - on a normal PC. It chats, writes code, reads pictures and works with your apps and coding
agents, and nothing leaves your PC.

## How fast is it?

Measured on two ordinary gaming PCs. "Writes answers" is how fast the reply appears in a short chat; "reads your
prompt" is how fast it takes in what you send (a 32K-token document, code or chat history). A token is about ¾ of a
word, so 60 tokens per second is faster than you can read.

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

Measured Strata on your own PC? See [Community benchmark results](docs/COMMUNITY_BENCHMARKS.md)
for a report template and how to share your results in a pull request.

**Two or three NVIDIA cards?** Just run `START-HERE.bat`: it lists your cards, says which ones Strata can use, and
asks whether to share the model across them (recommended when two can). An install made on one card asks once at
its next start. Or choose yourself: `START-HERE.bat --gpus 0,2` (both, remembered), `--gpus all`, or `--gpu 0` (one
card, this start only). Each card keeps the experts of its own layers, and prompts flow through the cards in a
pipeline: on an RTX 5080 + RTX 3090 prompts were read 18-20% faster than on the 5080 alone, decoding on par.
Every card must be an RTX 20 series or newer with 8 GB or more. See [docs/MULTI_GPU.md](docs/MULTI_GPU.md).

A card with more VRAM is faster: an RTX 3090 (24 GB) should write roughly 100-140 tokens per second. Long chats,
other cards: [speed of each model](docs/MODELS.md#how-fast-is-each-size), [community results](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata is free. If it runs well on your PC, a coffee keeps the work on it going.</sub></p>

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series - with **12 GB of VRAM or more** |
| **RAM** | 32 GB or more - how much decides [which model](#which-model-should-i-pick) fits; 64 GB runs every size |
| **Disk** | about 80 GB free, on an SSD if you can (the first start is much faster) |
| **System** | Windows 10 / 11 or Linux, and a current graphics driver from NVIDIA or AMD |

**You need:** an NVIDIA GPU with compute capability 7.0 or newer and 12 GB of VRAM or more, enough RAM for the size
you pick (above), ~80 GB of free disk space (an SSD makes the first start much faster), and Windows 10/11 or Linux.
RTX 30/40/50 cards use the ready-made engine; Volta cards such as the Tesla V100 are compiled locally with CUDA
12.x. The only thing you install yourself is a current **NVIDIA driver**
([nvidia.com/drivers](https://www.nvidia.com/drivers) or the NVIDIA App). Everything else is set up for you.

An **AMD Radeon RX 7900 XT / XTX, RX 9070 / 9070 XT or Radeon AI PRO R9700 on Linux** works too (experimental; the
RX 7800 XT / 7700 XT and RX 9060 XT were validated by their owners):
`./setup.sh --backend hip`, chosen by itself on a PC with no NVIDIA card Strata can use. It installs ROCm without sudo
and compiles the engine (no images yet; several cards with `--gpus`). Details: [AMD HIP](docs/AMD_HIP.md).

Everything else is installed for you. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).
The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

### Let your AI set it up

Use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk, picks the model that fits, installs it, starts it and tells you how to
connect your apps. AI tools can also install, start and stop Strata themselves through its
[MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the Strata folder.

The same steps for NVIDIA and AMD: the installer finds your card and sets up the right engine for it. It asks which
model, which size, how much context (how much text it keeps in mind) and whether it should read pictures - press
Enter each time for the recommended answer. Then it downloads the model (~70 GB; you can stop and it continues where
it left off) and starts it. Your browser opens the Strata app at `http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time): Strata
> loads 35-55 GB into your RAM and locks part of it for the graphics card. That's normal - wait, and don't close the
> window. The window tells you what it is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again: it starts right away, nothing is downloaded twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker,
several cards, where the files go and every option:
[docs/INSTALL.md](docs/INSTALL.md).

## Which model should I pick?

The installer recommends one for your RAM. The same model comes in sizes that are compressed more or less: smaller
is faster, larger is a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best, and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's 4-bit (experimental) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder)** - a coding version with half of the experts removed: 91% of the full model's
  SWE-bench Verified score (by its authors), fits 32 GB of RAM. Weaker outside code, including Chinese and other
  CJK text (#438): for those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15)** - a fine-tune that thinks much shorter before it answers, so you get the
  answer sooner, at about the same quality.
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (experimental) - the closest to the full
  model, but most of it is read from the SSD while it answers: 7-8.5 tokens/s on a 64 GB PC.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)** - a manual setup, not in the
  installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). You can add another model later with
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** `http://127.0.0.1:8080` - **Chat**, a live **Monitor** of the model and your GPU/CPU/RAM, and
  **About** with the settings and addresses.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with base URL **`http://127.0.0.1:8080/v1`**,
  any API key and any model name. Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
  `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
- **Thinking:** choose **off, low, medium or high** in the chat menu or your app's "reasoning effort". Off is
  fastest; high is best for hard questions.
- **Pictures:** say yes to "Images?" in setup, then click **Picture** in the chat, or attach them in your app
  (AMD cards: on Linux through the processor, not on Windows yet).
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>` - always with a key.
- **Good to know:** it answers one request at a time. The first message of a chat is read in full (about 1 minute
  per 30,000 tokens); follow-ups start in seconds.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

### Where things are stored

- **Your chats: only in your browser.** The Chat tab keeps the conversation, its settings and the API key you typed
  in the browser's local storage (`strata.*` keys) - not on the server and not in the Strata folder. Pictures are not
  kept, only their names. Another browser or a private window starts empty; clearing the site's data deletes them.
- **How the model starts:** `strata-<model>.json` in the Strata folder (context, GPUs, host, API key, ...), written
  by setup; next to it `run-<model>.bat` / `.sh`, the log `strata-<model>.log` and, when you use "Use for other
  apps too", `strata-<model>.shared-settings.json`.
- **The model files** (`models/`, `packs/`, `mtp/`, 70-120 GB): in **`Strata-data` next to the Strata folder**, or
  wherever `--data-dir` put them.
- **Where that data folder is:** `%APPDATA%\Strata\settings.json` on Windows, `~/.config/strata/settings.json` on
  Linux ([details](docs/DETAILS.md)).

## Something went wrong?

- **My PC froze the first time Strata started.** Normal while it loads the model: wait, don't close the window.
  Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again: it continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or "the engine stopped unexpectedly".** Not enough free RAM:
  close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** Strata is already running - look for its window.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder.

## How does it work?

Models like this one normally run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes it fit by **sharing the work across your whole PC** - like a kitchen, where the things you use
all the time stay on the counter and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts"),** and each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are asked most often; **your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words and the big model checks them all at once, so
  you get the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time): over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers: [the
details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Benchmarks and regression tests

The V100 table above is reproduced end to end by [`bench/run_v100_bench.py`](bench/run_v100_bench.py):
it builds every row at the exact published prompt size (unique-prefix repeated text, uncached,
256 generated tokens, three fresh requests per size, seed 20261002), targets the running server,
and reads the engine's own timings from `/metrics`. For the 2 October 2026 table, every request
followed the cooldown gate (at least 120 s, at or below 55 °C for 15 s); the per-request waits
are recorded in the raw rows. The API key is read from `.strata-service.env` / `$STRATA_API_KEY`,
so the script contains no credentials and can be committed. Use it for all future benchmark runs:

```sh
.venv/bin/python bench/run_v100_bench.py --model-gguf models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
.venv/bin/python bench/run_v100_bench.py --model-gguf <shard-1.gguf> --only '~8K,128K'   # a subset of rows
```

Contributor-facing summaries and cross-hardware comparisons live in [`benchmarks/`](benchmarks/README.md);
raw rows and methodology notes for each measuring session live in `bench/results/`.

Before shipping any engine change, run the live OpenAI-compatible tool-call regression checks
against the running server (they exercise function calling and a Hermes-style agent loop, and
write full request/response transcripts under the ignored `build/diagnostics/`):

```sh
.venv/bin/python tests/diagnose_openai.py
.venv/bin/python tests/diagnose_hermes_flow.py
```

Both accept `--base-url` / `--api-key` / `--output`; without arguments they use
`http://127.0.0.1:8088/v1` and the key from `.strata-service.env`.


## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team, compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) and Unsloth;
Strata is built with parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE); a few
parts and every model carry their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
