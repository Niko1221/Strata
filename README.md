<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<p align="center"><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a></p>

<h3 align="center">Run a state-of-the-art model on your own gaming PC</h3>

<p align="center">Support for NVIDIA, AMD, Intel · Windows, Linux, Mac · free and open source</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="GitHub stars"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

<p align="center">
  <a href="#install"><b>Install</b></a> ·
  <a href="#models">Models</a> ·
  <a href="#how-fast-is-it">Speed</a> ·
  <a href="#using-it">Using it</a> ·
  <a href="#how-does-it-work">How it works</a> ·
  <a href="#something-went-wrong">Help</a> ·
  <a href="#support-strata">Support</a>
</p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, a 125-billion-parameter model that
normally needs a server, on a normal PC. From 0.2.4 it also runs **DeepSeek V4.1-Flash** in the Terminal. It chats, writes
code, reads pictures and works with your apps and coding agents. Your chats and files stay on your PC.

- **Runs on a gaming PC.** 32 GB of RAM and graphics memory together is enough to start.
- **Works with your tools.** Claude Code, Codex CLI, OpenCode and other agents connect through the OpenAI and Anthropic APIs.
- **Reads pictures**, on by default.
- **Free and private.** MIT license, no account, no cloud, no telemetry ([what it downloads and when](docs/SAFETY.md)).

## What you need

**At least 32 GB of memory: RAM and graphics memory (VRAM) together.** On a Mac or a Ryzen AI Max that is the unified
memory. A 32 GB PC with a 12-16 GB card runs the smaller sizes, and 64 GB runs every size.

| Works on | Guide |
| --- | --- |
| **NVIDIA** GeForce RTX 20, 30, 40 and 50 series, 12 GB or more | [Install](#install) |
| **AMD** Radeon RX 6800 and newer, Radeon AI PRO R9700 | [AMD cards](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md) |
| **Apple Silicon Mac**, M1 or newer, macOS 15 or newer | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max (Strix Halo)** and other unified-memory PCs | [Strix Halo](docs/STRIX_HALO.md) |
| Older cards and processors | [Older GPUs](docs/OLDER_GPUS.md) |

You also need about 80 GB of free disk (an SSD), Windows 10 / 11, Linux or macOS, and a current graphics driver. Two or
three cards of the same make can share a model ([multi-GPU](docs/MULTI_GPU.md)).

## Install

### 1. The Strata app (easiest)

Download the file for your computer, open it, and press install on the model it recommends.

**Windows 10 / 11**

[![Windows, NVIDIA, driver 580 or newer](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda13.exe)
[![Windows, NVIDIA, older driver](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda12.exe)
[![Windows, AMD RX 9000 or R9700](https://img.shields.io/badge/Windows-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip.exe)
[![Windows, AMD RX 7000 or Ryzen AI Max](https://img.shields.io/badge/Windows-AMD%20RX%207000%20%C2%B7%20Ryzen%20AI%20Max-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip-rdna3.exe)

**Linux**

[![Linux, NVIDIA, driver 580 or newer](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda13.AppImage)
[![Linux, NVIDIA, older driver](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda12.AppImage)
[![Linux, AMD RX 9000 or R9700](https://img.shields.io/badge/Linux-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx9070.AppImage)
[![Linux, AMD RX 7000](https://img.shields.io/badge/Linux-AMD%20RX%207000-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx7900.AppImage)
[![Linux, AMD Ryzen AI Max](https://img.shields.io/badge/Linux-Ryzen%20AI%20Max-8A0E15?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-strixhalo.AppImage)

**Mac** (Apple Silicon, macOS 15 or newer)

[![Mac, Apple Silicon](https://img.shields.io/badge/Mac-Apple%20Silicon-333333?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-mac.dmg)

NVIDIA driver older than 580 (`nvidia-smi` shows it)? Take the older-driver file. An AMD RX 6000 card takes the RX 7000
file, which fetches the right engine by itself. All files are on the [releases page](https://github.com/Niko1221/Strata/releases/latest).

| First start | What to do |
| --- | --- |
| **Windows** | Run the `.exe`. No administrator rights needed. If Windows says "Windows protected your PC", choose **More info**, then **Run anyway**. |
| **Linux** | `chmod +x` the file and start it. It needs FUSE 2: `sudo apt install libfuse2t64` (Ubuntu 24.04 and newer) or `libfuse2` (22.04). |
| **Mac** | Open the `.dmg` and drag Strata onto Applications. |

The model download is big (40-70 GB). If it stops, it continues where it left off.

### 2. The Terminal (one line)

The `strata` command, for scripts, servers and AI agents. It is also the way to run DeepSeek.

Linux and Mac:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

Windows (PowerShell):

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Then start it. It asks one question at a time (model, context, pictures, who may connect), downloads what it needs and
starts serving:

```sh
strata launch
```

No admin rights needed. The install checks the SHA-256 of everything it downloads. On Windows, open a new terminal window
after the install. All screens and flags: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md).

### 3. Let your AI set it up

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or another coding assistant:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk, picks the model that fits, installs it, starts it and tells you how to connect
your apps. AI tools can also install, start and stop Strata through its [MCP server](docs/MCP_SERVER.md).

Prefer to build it yourself? See [Build from source](#build-from-source).

## Models

The app and `strata launch` recommend a size for your memory. Smaller sizes are faster, larger sizes are a bit smarter.

### Low on memory? Start with Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) by agentionai is the smallest way to run
Qwen3.8-Flash-Next. It runs on NVIDIA, AMD, Intel Arc and Apple Silicon.

| Size | Memory it needs (RAM + VRAM) | Download | Good for |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27.6 GB | 58.5 GB | PCs that fit nothing else. 9 tokens/s on a 32 GB Mac mini. |
| **Gyro-M** | 35 GB | 92.0 GB | About IQ3_XXS quality with 20% less GPU memory. |

Start it with `strata launch --model gyro-s`. Details: [docs/GYRO.md](docs/GYRO.md).

### Qwen3.8-Flash-Next sizes

| Your memory (RAM + VRAM) | Take | Why |
| --- | --- | --- |
| **32 GB** | **Gyro-S**, or the **[Coder](docs/MODELS.md#coder)** for code | The Coder keeps the experts that code uses |
| **40 GB** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS (39.2 GB) or **Gyro-M** | Swift thinks for a shorter time before it answers |
| **48 GB** | **IQ3_XXS** (47.0 GB) | Slower, better |
| **64 GB** | **IQ3_S** (54.8 GB) | Matches the full model on the published tests |
| **96 GB or more** | Unsloth's **UD-IQ4_XS** to **Q8_0** | Closest to the full model |

Less RAM than a row says can still work, only slower: Strata switches to its
[low-RAM mode](docs/MODELS.md#a-big-graphics-card-and-little-ram) by itself. Also available:
[NVFP4](docs/NVFP4.md) for NVIDIA cards, [Qwen3.6-35B-A3B](docs/QWEN36.md) (smaller, several requests at once) and
[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs). Every size and what fits where:
[docs/MODELS.md](docs/MODELS.md).

### DeepSeek V4.1-Flash (Terminal, from 0.2.4)

A much bigger model that runs on one 12 GB graphics card with 46 to 64 GB of RAM. Strata keeps the busiest parts in memory and
reads the rest from your SSD. [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
made the sets. It runs on NVIDIA RTX 30 to 50 cards and the Radeon AI PRO R9700. Pick it in `strata launch`; the desktop
app does not offer it yet.

| Set | Download | Quality | Writes code at |
| --- | ---: | --- | --- |
| **IQ1_M** (1.75 bits) | 239 GB | The smallest; weaker on plain prose | not measured on a PC yet |
| **IQ2_XXS** (2.00 bits) | 257 GB | Good balance | 10 tokens/s (RTX 3060), 5 (RTX 5070), 48 (4x R9700) |
| **IQ2_XS** (2.25 bits) | 274 GB | Closest to the original | 15 tokens/s (one R9700), 80 (4x R9700) |
| **IQ4_NL** (Engram table) | in every download above | The lookup table all three sets share | |

Start it with `strata launch --model ist-das-2.00bpw` (or `ist-das-1.75bpw`, `ist-das-2.25bpw`). Quality numbers, speeds
per PC and the manual install: [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md).

## How fast is it?

Based on a 12 GB graphics card (RTX 5070, 64 GB RAM). Results from other cards:
[speed of each model](docs/MODELS.md#how-fast-is-each-size) and [community results](docs/COMMUNITY_BENCHMARKS.md).

| Model | Answers appear at | Reads your prompt at |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2,650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2,090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1,750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1,620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2,180 tokens/s |

On a 16 GB AMD card (RX 9070 XT, 47 GB RAM):

| Model | Answers appear at | Reads your prompt at |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1,160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1,110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1,420 tokens/s |

A token is about ¾ of a word, so 60 tokens per second is faster than you can read. "Reads your prompt" is how fast Strata
takes in a long document, code or chat history.

## Using it

### In the app

Chat, a **Monitor** of the model and your GPU, CPU and RAM, a **Models** page and **Settings**. The chat can work in a
folder on your PC. In **Manual** mode it asks before it changes a file or runs a command. It can search the web
(DuckDuckGo or Bing, no account); the search words are the one thing that leaves your PC.

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="The Strata app: a chat where the agent built a small voxel pagoda garden, with the page open in the preview on the right"><br>
<sub>The agent wrote a voxel pagoda garden as one HTML file, and the page opens in the preview next to the chat</sub></p>

### In the Terminal

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="The Strata Terminal monitor overview while the model writes a reply"><br>
<sub><code>strata monitor</code> while the model writes a reply: speed, GPU, memory and the recent requests</sub></p>

```sh
strata launch                              # guided: model, context, pictures, access, then serve
strata launch --model q2_0 --yes --no-tui  # no questions: serve and print the address
strata launch --status                     # what is being served now
strata launch --stop                       # stop it
strata monitor                             # watch speed, GPU, memory and requests
strata update                              # get the newest engine
strata doctor                              # check the installed files
```

To reach Strata from your phone or another PC, use `strata launch --access lan`. It makes an API key for you.
`strata --help` lists everything, and [docs/TERMINAL.md](docs/TERMINAL.md) explains it.

### With AI agents and other apps

An agent installs Strata with the one-line command above and runs `strata launch --model NAME --yes --no-tui`. Steps,
flags and exit codes: [docs/AI_SETUP.md](docs/AI_SETUP.md). Repository rules for agents: [AGENTS.md](AGENTS.md).

| Your app speaks | Base URL |
| --- | --- |
| OpenAI API (most apps and agents) | `http://127.0.0.1:8080/v1` (any API key, any model name) |
| Anthropic API (Claude Code) | `http://127.0.0.1:8080`, without `/v1`. Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| OpenAI Responses API (Codex CLI) | `http://127.0.0.1:8080/v1/responses` ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)) |

`strata connect` prints the exact settings for OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent and OpenClaw,
and `strata connect opencode --write` writes them for you (with a backup).

### Good to know

- **Pictures** are on for every model that can read them. Attach or paste one in the chat, or send it from your app. Details: [docs/IMAGES.md](docs/IMAGES.md).
- **Thinking** can be off, low, medium or high, in the chat menu or your app's "reasoning effort". Off is fastest.
- **The first message** of a chat is read in full, about 1 minute per 30,000 tokens. Follow-ups start in seconds.
- **One request at a time** by default. For more, use `--parallel N` ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card each answer gets slower.

## Build from source

<details>
<summary>Windows, Linux and Mac from the repository</summary>

[Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it, or `git clone` it.
**Windows:** double-click `START-HERE.bat`. **Linux:** run `./setup.sh` in the Strata folder.

**Mac (Apple Silicon, macOS 15 or newer):** install Apple's Command Line Tools once with `xcode-select --install`
(Xcode itself is not needed). Open Terminal in the Strata folder and run `./setup.sh`. It builds the Mac engine on your
Mac (about 7 minutes), asks the same questions, downloads the model and starts it. Next time, `./setup.sh` starts it
right away. Which size fits your Mac: [docs/MAC.md](docs/MAC.md).

The installer finds your card and sets up the right engine. It asks which model and size, how much context (how much text
the model keeps in mind) and whether it should read pictures. Press Enter for the recommended answer. Then it downloads the
model (about 70 GB) and starts it. If the download stops, run it again and it continues. Your browser opens the Strata
page at `http://127.0.0.1:8080`.

> While the model starts, your PC can be slow or stop responding for 1-3 minutes (longest the first time). Strata loads
> 35-55 GB into your RAM and locks part of it for the graphics card. Wait, and don't close the window.

Next time, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close its
window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker, several cards,
where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

To measure your PC, run `BENCH.bat` (`./bench.sh`). It takes 5-10 minutes and writes a report you can share
([how](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

**Optional: more speed on one card when the whole model fits in RAM.** Two opt-in settings by RubenTorney (#1904, #1905).
They help when you run one graphics card and your RAM holds every expert, for example Q2_0 or IQ3_XXS on a 64 GB PC. Open
`strata-<model>.json` in the Strata data folder ([where](docs/INSTALL.md#where-things-are-stored)) and add:

- `"--adapt-async", "1"` to the `"args"` list. The expert cache then updates on a helper thread instead of pausing the
  answer. Measured on an RTX 5070 (Windows, Q2_0): +4-5% decode, +7-10% on code and long documents. Answers can differ
  slightly from the default and stay the same from run to run.
- `"STRATA_PIN_BY_NEED": "1"` in the `"env"` block (Windows). Windows can lock only about half the RAM for the graphics
  card, and this locks the experts the card needs first. Measured on the RTX 5070: +4.4% prompt reading.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Restart Strata after the change. To undo, remove the two entries.

</details>

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by sharing the work across your whole PC. Picture a library.

- The model is a team of 24,576 small specialists ("experts"), like 24,576 reference books. Each word needs only 10 of them.
- **Your graphics card is the desk.** It is small and fast. The roughly 4,000 books used most often lie open on it.
- **Your RAM is the shelves in the reading room.** All 24,576 books are there. When a word needs a book that is not on the
  desk, your processor reads it from the shelf at once, while the card keeps working.
- **Your SSD is the archive.** It holds a 29 GB lookup table. The model reads a few rows of it per word, so it does not
  need to sit in RAM.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

A small helper also guesses the next few words. The big model checks all of them at once and keeps the right ones. You get
the same answer, 1.6-1.8x sooner. Long texts are read in big pieces, up to 8,192 tokens at a time, at over 1,000 tokens
per second.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Something went wrong?

- **My PC froze the first time Strata started.** It loads the model into RAM. Wait, and don't close the window. Still
  frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Start the app or `strata launch` again. It continues where it stopped.
- **It is very slow, or it says "the engine stopped unexpectedly".** Your PC is short on free RAM. Close other programs
  (browsers use a lot), or pick a smaller size such as Q2_0 or IQ2_XS.
- **It says there is no engine for this PC.** In the app, open **Settings, Engine** and press **Update engine**. In the
  Terminal, run `strata update`.
- **It says port 8080 is already in use.** Strata is already running. Look for its window.

More fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach the logs: in the app from **Settings, Engine, Diagnostics,
Save logs**, in the Terminal the output of `strata doctor`. For a security problem, report it privately: [SECURITY.md](SECURITY.md).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
and Unsloth. ISTA-DASLab also made the DeepSeek V4.1-Flash sets. Strata uses parts of
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
Strata is open source under the [MIT License](LICENSE). A few parts and every model have their own licenses
([which ones](docs/HOW_IT_WORKS.md#license)).

## Contributing

Found a bug, or want to send a change or a benchmark? Read [CONTRIBUTING.md](CONTRIBUTING.md) first. It says what to
include so that others can check your work.

## Support Strata

Strata is free and MIT-licensed. I build it and test it on whatever hardware I can get. If it runs well on your PC, a coffee helps
me keep testing new cards and fixing what you report. No pressure. Using it and reporting what you find helps just as much.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Interested in sponsoring Strata? Contact me via [GitHub](https://github.com/Niko1221).
