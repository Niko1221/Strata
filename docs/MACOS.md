# Strata on a Mac (Apple Silicon)

**Experimental: tested on one Mac.** A MacBook Pro M5 Max (40-core GPU, 128 GB) on macOS 26.4, with the Q2_0 model
at a 32K context. Other Apple Silicon Macs and other model files are untested. Read
[Limits and warnings](#limits-and-warnings) before you install.

On a Mac, Strata serves the same web app and APIs (OpenAI, Anthropic, Responses, MCP) as on a PC. The engine under
them is different: `strata-metal` (`metal/`) runs the model on llama.cpp's Metal backend.

## Before you start

- An Apple Silicon Mac (M1 or newer; setup refuses Intel Macs) with **64 GB of memory or more**. 64 GB is Strata's
  floor for a Mac, not a measured minimum: only 128 GB was tried.
- About **80 GB of free disk** on the internal SSD, and an internet connection for the first run (66 GB download).
- Apple's Command Line Tools (the compiler and git). You do not need the full Xcode app.
- Python 3.10 or newer. If it is missing, setup installs it with [Homebrew](https://brew.sh) when Homebrew is there;
  otherwise install it from [python.org](https://www.python.org/downloads/). Everything else (cmake, ninja, Python
  packages) setup installs into the Strata folder.
- Quit virtual machines, big compiles and Docker builds while you use it: they can halve the speed (see below).

## Quick start

In Terminal, one line at a time:

```sh
xcode-select --install       # a dialog opens: click Install, wait until it finishes (skip if already installed)

git clone https://github.com/Niko1221/Strata.git
cd Strata
make check                   # what this Mac can run; installs nothing
make pull MODEL=Q2_0         # builds the engine and downloads the model; asks about images and the context
make run                     # starts it; open http://127.0.0.1:8080 when it says it is ready
```

`make run` keeps the model running in that Terminal window; Ctrl-C stops it. On the test Mac the engine compiled in
3-6 minutes (once) and later starts took about 35 seconds.

**Getting the model.** `make pull` downloads Q2_0, 66.4 GB in two files, from Hugging Face
([ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
pinned to one revision), plus the 0.9 GB image encoder if you answer yes to images. The files go to `Strata-data/`
next to the `Strata` folder; each one's size is checked against the server's when it finishes. If the download stops,
run the same command again: it continues where it stopped. Q2_0 is the only size tested on a Mac. Add
`SETUP_ARGS="--yes"` to skip the questions, and set `HF_ENDPOINT` to use a Hugging Face mirror.

`make run` alone also works the first time: it asks the same questions, downloads, then starts.

| To | Run |
|---|---|
| run in the background | `make start`, then `make status` / `make stop` (log: `strata-run.log`) |
| send one test message | `make chat PROMPT="Write a haiku"` |
| cap the thinking | `make chat PROMPT="..." EFFORT=high MAX_TOKENS=32768 REASONING_BUDGET=4096` (then it must answer) |
| use another port | `make run PORT=8090`, or put `PORT := 8090` in a file named `Makefile.local` |
| use a bigger context window | `make run CONTEXT=131072` (up to 262,144; it stays set, and needs more memory) |
| download without questions | `make pull MODEL=Q2_0 SETUP_ARGS="--yes"` |
| try another size (untested on a Mac) | `make pull MODEL=IQ2_XS`, then `make run` |
| turn on the MTP draft layer | `./setup.sh --setup --mtp on` (see below) |
| see every target | `make` |

If setup stops, it says what is missing and the command that fixes it.

## Limits and warnings

- **Memory.** A Mac's CPU and GPU share one memory. Metal sets a limit on how much the GPU may use at once
  (`recommendedMaxWorkingSetSize`); `make check` prints this Mac's value. On the test Mac it was 107.5 of 128 GB; it
  differs between Macs and macOS versions. Q2_0's weights take about 35 GB of it, plus the context's cache. Everything
  the Mac does shares the same memory, so close other large apps. Below 128 GB only the 64 GB floor applies: nobody
  has measured how close a 64 GB Mac gets.
- **Only Q2_0 was tested.** `make check` marks the other sizes "untested on a Mac". On a PC, the Unsloth sizes
  (UD-Q4_K_XL, UD-IQ4_XS) stream part of their experts from the SSD; the Mac engine cannot, so all of a model's experts
  must fit in Metal's limit. UD-Q4_K_XL (111 GB) is larger than the test Mac's default limit.
- **Speed.** On the test Mac: 13-17 tokens/s for the answer and 225-270 tokens/s to read a prompt, with other programs
  running (see [Measured](#measured)). Other Macs will differ; slower memory means fewer tokens per second.
- **Other programs can halve it.** The same engine and settings gave 24.7 tokens/s with a virtual machine and an Xcode
  build running, and 48.3 without them, while a GPU benchmark run alongside kept its full speed. So the engine needs
  free CPU time, not only a free GPU. Before you judge the speed, quit VMs, compiles and other heavy programs.
- **Power.** Keep a laptop plugged in and out of Low Power Mode. If your Mac has it, *System Settings > Battery >
  Energy Mode > High Power* may help long answers; its effect on Strata was not measured.
- **Disk.** The download is 66.4 GB (67.3 GB with the image encoder); installed with the engine, about 70 GB. `--mtp on`
  adds about 12 GB. Keep the model on the internal SSD: its 28 GB n-gram table is not loaded into memory, its rows are
  read from the model file as they are needed.
- **MTP is opt-in.** It makes answers faster (see Measured), costs about 12 GB of disk and keeps the KV cache 16-bit.
  With it on, a long answer can differ from the answer without it. Two requests at once (`"parallel"`) get no drafts.
- **Not on a Mac yet:** contexts past 262,144 tokens (rope scaling), the PC engine's expert cache and CPU experts.

### If a model does not fit

Pick a smaller size first. If you know what you are doing, you can raise Metal's limit until the Mac restarts:

```sh
sysctl iogpu.wired_limit_mb                  # the current value; 0 means macOS' default
sudo sysctl iogpu.wired_limit_mb=110000      # e.g. ~107 GB on a 128 GB Mac; the value is in MB
sudo sysctl iogpu.wired_limit_mb=0           # back to the default
```

This is an undocumented macOS setting, and Strata never changes it. Leave macOS plenty of memory (Strata's rule of
thumb: 8-16 GB): set too high, the whole Mac can slow down or stop responding until it restarts.

## What works

| | On a Mac |
|---|---|
| Chat, tools, the web app, the APIs | yes, the same server as on a PC |
| Conversation reuse | yes: a follow-up reads only what is new |
| Pictures | yes: `strata-vision` runs on Metal |
| MTP draft layer | opt-in: `--mtp on` |
| Several requests at once (`"parallel"`) | opt-in, the same as on a PC |
| Monitor tab | GPU load, memory and power, the chip's temperature (its die sensors), CPU, RAM; PCIe shows "n/a" (an integrated GPU has no PCIe link) |
| Expert cache, CPU experts, rope scaling, Intel Macs | no |

### The MTP draft layer (`--mtp on`)

The model's own draft head guesses the next 3 tokens, and the model checks them in one pass. Setup builds the head from
the original checkpoint: it downloads the 31 MTP tensors (5 GB, SHA-256 checked) and the embeddings and LM head (2.5 GB),
and llama.cpp's converter makes a 4.1 GB file of them. The converter needs PyTorch, so setup installs it once into
`.venv-mtp` (about 730 MB). llama.cpp pins numpy 2.2.6, which converts this head wrongly (no error, but no draft is ever
accepted), so setup uses numpy 2.4.0 and checks the result against a known SHA-256.

## Measured

On the test Mac: Q2_0, 32K context, through the server's OpenAI API, greedy, thinking off, a fresh prompt each run,
medians of 3, 2026-10-06, with other programs running (the answer speed moved by ±3 tokens/s between runs).

| Prompt | Reads the prompt | Writes the answer | With `--mtp on` |
| ---: | ---: | ---: | ---: |
| 25 tokens (a story) | - | 16.2 tok/s | 17.2 tok/s |
| 3,686 tokens (code) | 241 tok/s | 12.8 tok/s | **21.6 tok/s** |
| 26,051 tokens (code) | 225 tok/s | 17.0 tok/s | 19.4 tok/s |

- The `--mtp on` column was measured on a later engine build (it also reads prompts faster: 272 and 252 tok/s), so
  its ratios to the other column mix two changes. MTP alone, on the same build: 1.23-1.50x on 3 short chat prompts,
  with 64-92% of drafts accepted; prose gained least.
- After the 26K prompt, a follow-up message started answering in 0.3 s (0.7 s with MTP).
- One picture (640×240, a 196-token prompt) was read and answered in 3.4 s in all.
- Two requests at once with MTP: 17.2 tok/s together against 18.7 one after the other (batch slots decode without
  drafts). llama.cpp's `batched-bench` without MTP: 13.2 tok/s for one sequence, 22.5 for two, 29.7 for four.
- The engine adds nothing on top of llama.cpp: `llama-bench` on the same file gives 16.9 tok/s output.

### Compared with MLX (mlx-lm)

Would Apple's MLX run this model faster? Measured on the test Mac, 2026-10-07. The same Q2_0 weights were converted to
MLX: the 2-bit experts copied bit for bit, the other tensors requantized one bit higher, the 28 GB n-gram table at 5
bits. They ran in mlx-lm on MLX 0.32.3 with the community port of this model
([mlx-lm#1788](https://github.com/ml-explore/mlx-lm/pull/1788), not merged as of that date). Each server ran alone, in
turn, on the same prompts: greedy, thinking off, MTP off, through its OpenAI API.

| | Strata (llama.cpp Metal) | mlx-lm (MLX) |
|---|---:|---:|
| Writes the answer, best run | 48.3 tok/s | 41.5 tok/s |
| Writes the answer, worst run (a virtual machine running) | 24.7 tok/s | 17.1 tok/s |
| Reads a 3,290-token prompt | 364-895 tok/s | 286-493 tok/s |
| Memory in use | ~35 GB of weights; the n-gram table stays in the file | ~81 GB, all of it in memory |
| Start | ~35 s | ~160 s |

- The load from other programs was not controlled and moved both engines by up to 2x (an Xcode build pulled MLX
  down to 15-19 tok/s while a GPU benchmark kept its full speed), so read the table as ranges, not as a ranking.
- The answers are not the same: 3 of 12 greedy answers matched; the rest parted after 2 to 58 tokens, because the
  non-expert weights are requantized for MLX.
- mlx-lm's server slowed down on repeated requests (40.7 to about 20 tok/s) until its prompt cache was turned off
  (`--prompt-cache-size 0`).
- Tried on the MLX side: the n-gram lookup on the GPU from one table, and fused hyper-connection steps (same tokens,
  no change beyond the noise); 8-bit hyper-connection weights were slower (26.3 against 32.7 tok/s) and changed the
  answer.
- So MLX was not faster here and needs more than twice the memory: Strata stays on llama.cpp, and the MLX conversion
  is not part of Strata.

## How it fits together

- `metal/strata_metal.cpp`: the engine. It speaks the CUDA engine's line protocol (`GEN`/`GENI`, `T`, `PP`, `RESUME`,
  `DONE`, `STOP`, `SAVE`/`RESTORE`, and `BGEN`/`BT`/`BDONE`/`BADM`/`BSTOP` for batch slots), so every server feature
  works through it. It adapts the protocol instead of porting the CUDA engine, whose kernels and expert cache are CUDA
  code; llama.cpp already runs this model on Metal.
  - Conversation reuse goes back to a checkpoint of the recurrent state, taken before each `<|im_start|>`. It never
    uses a state that does not match: tokens are compared by id, and picture cells by the picture's hash.
  - Pictures get the CUDA engine's M-RoPE positions.
- `metal/setup_mac.py`: setup's Mac steps. `setup.py` runs it by itself on macOS; it changes none of setup's other steps.
- `metal/mtp_gguf.py`: the MTP head for llama.cpp, from the checkpoint, checked by its SHA-256.
- `serve/telemetry.py`: the Monitor's GPU load comes from `ioreg`, its memory limit from Metal, its power from
  IOReport's energy counters and the temperature from the chip's die sensors, all without root. These are private
  macOS interfaces: if a macOS update changes them, the tiles show "–" instead of a wrong number.

## For developers

```sh
make build                                   # build-metal/: the engine and strata-vision
make test                                    # the tests that need no GPU and no model
make test-engine TEST_GGUF=<a small .gguf with <|im_start|>, e.g. LiquidAI LFM2-350M Q8_0>
```

llama.cpp comes at a pinned commit (`-DSTRATA_LLAMA_DIR=<checkout>` builds offline). Strata's own Metal kernels are
patches on that commit in `metal/patches/`; `-DSTRATA_METAL_PATCHES=OFF` builds the plain commit. To A/B the two:

```sh
make build-ab                                # build-metal-a (plain) and build-metal-b (patched)
make ab AB_GGUF=<shard 1> AB_PROMPTS=<json list of token-id lists>
```

`metal/bench/ab.py` runs both builds in turns and exits 1 when their greedy tokens differ. Results so far, on Q2_0 on
the test Mac:

- `0001-metal-fuse-scale-unary`: fuses a scale into the following unary op; the same tokens on the tested prompts, no
  measurable speedup.
- `0002-metal-short-row-mat-vec`: a mat-vec kernel for short rows; 1.8x faster on its own (15.2 vs 27.3 µs), +1.1%
  for the whole model, within the run-to-run noise. `GGML_METAL_MV_SHORT_DISABLE=1` turns it off.

`metal/test_strata_metal.py` speaks the protocol to the binary: a diverging conversation must give a fresh engine's
tokens, an extended prompt must read only its new part, a different picture with the same ids must not be reused, two
batch slots must each give what they give alone, and STOP, BSTOP, SAVE/RESTORE and every refusal must keep both sides
in step. Four of these checks were mutation-tested.
