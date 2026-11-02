# DeepSeek-V4.1-Flash

DeepSeek-V4.1-Flash is a large mixture-of-experts model: 40 layers, 384 experts in each (6 are used for every word), and a 203 GB
n-gram table next to them. Strata runs it with its own engine, `strata-dsv41` (`src/dsv41`, ported from Project Maya's
`maya-ds4`, MIT; the notice is in `third_party/project-maya`). The file is Unsloth's UD-Q8_K_XL, pinned in `setup.py`: nine
shards, 508.4 GB, the experts as MXFP4 and the rest in BF16. The two official 2-bit sets (below) are what a normal PC runs.

The 508 GB set is for a computer with several graphics cards and a lot of RAM. It was run on four Radeon AI PRO R9700 (32 GB each) with
251 GB of RAM. Strata does not refuse a smaller computer; it tells you what to expect first.

## Where it goes

| | size | where |
|---|---|---|
| routed experts (15,360 of 18.8 MB) | 288.8 GB | as many as fit in graphics memory; the rest are computed on the CPU, read in place from the model files |
| attention and other dense weights | 16.8 GB | graphics memory, one block of layers on each card (about 3.9 GB a card on four) |
| n-gram table (Engram) | 202.8 GB | the SSD. A few rows are read for each word it writes. It is not counted in RAM, but a slow disk slows every word, so keep the model files on an NVMe drive |

Measured on the four R9700 cards: 6,197 experts (116.5 GB) in graphics memory, 9,163 (172.3 GB) on the CPU, which needs about that
much free RAM to stay cached. `strata launch --plan` and `setup.py --check` work this out for your cards and RAM
(`strata_control/dsv41_fit.py`; it predicted 6,240 experts on the R9700 box).

With less memory than that, part of the CPU's experts are read from the SSD while the model answers. It runs, slowly. A card
that cannot hold its share of the dense weights (about 4 GB each on four cards, 16 GB on one) can make the engine stop at the start
with an out-of-memory message. Both are said before anything is downloaded.

## The official 2-bit versions (the ones for a normal PC)

Setup offers three official sets (ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF, pinned): **IQ1_M, 1.75 bpw** (`ist-das-1.75bpw`, 121 GB, slightly lower quality; setup recommends it where 2.00 is tight in memory or disk), **IQ2_XXS, 2.00 bpw** (138 GB) and **IQ2_XS, 2.25 bpw** (155 GB). All use the
same 111 GB n-gram table (downloaded once into `models/deepseek-v4.1-flash-shared`) and the 8 GB DSpark drafter: 257 GB of disk for
2.00, 274 GB for 2.25, 239 GB for 1.75, 533 GB for all three. Quality against the original model (4,995 positions, the final files): 2.00 has KL 0.170 and perplexity +15.5% (code +8.6%, agent work +5.3%); 2.25 has KL 0.107 and
perplexity +7.3% (code +3.2%, agent work +2.2%). IQ1_M: perplexity +21% (KL 0.217, top-1 82.5%); code +8.9%, agent work +10.2%, multilingual +14%, docs +24%, plain English prose +53%, so it is weaker on prose. ISTA-DASLab's model card says it keeps 98.7% of the base model's task average (AIME25, GPQA-Diamond, LiveCodeBench v6).

| File | Size | What it is |
| --- | ---: | --- |
| IQ1_M shards (1.75 bpw), 4 files | 121 GB | the model, smallest set |
| IQ2_XXS shards (2.00 bpw), 4 files | 138 GB | the model, middle set |
| IQ2_XS shards (2.25 bpw), 4 files | 155 GB | the model, largest set |
| IQ4_NL: `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | 111 GB | the 4-bit Engram lookup table the model reads from RAM or SSD. Not a model to pick: all three sets use this one file, and it is already in their download sizes |
| DSpark drafter (Q8_0) | 8 GB | the small draft model (Unsloth) |

**Two layouts of the same sets.** Setup downloads the four-shard files (`...-0000N-of-00004.gguf`, the Hugging Face revision `cb02422c`). On 27 October 2026
ISTA-DASLab re-split the three sets: five shards each (`...-0000N-of-00006.gguf`; shard 1 is 5.8 MB of metadata and has no tensors) and the Engram file as the
sixth shard of the split. Strata takes either from files you already have (`--gguf-dir`, `--model-path`, "Use a local GGUF" in the app, `strata models import`),
checks their size and SHA-256, and packs them. The same goes for the four-shard files whose first shard ISTA-DASLab re-published on 26 October with a larger
chat template. What differs in the six-shard files: the two Engram tables are one tensor (the same bytes), the Engram gate is stored already folded (q times k),
the attention output matrix is one 2-D tensor instead of a 3-D one, and there is a routing bias for picture tokens. The packs are the same size and the engine
reads both. The Engram has the same name and size in both layouts but another hash; either one works with either layout, and setup downloads the pinned one
if you have none. The download default is still the four-shard files.

**On a Ryzen AI Max "Strix Halo" (Windows)** the graphics chip and the CPU share one pool of memory, so the engine counts it once: the
128 GB box (a 64 GB BIOS carve-out plus the 64 GB Windows sees) keeps about 109 GB of the 2.00 set's experts in that pool (the dense
weights and 12 GB for Windows and the desktop aside) and reads the other 23 GB from the SSD. Measured there (2.00 bpw, default Faster
mode, DSpark auto): answers 19-21 tokens/s on code (about 18 on agent) and 11 on chat; a 4,000-token prompt 87 tokens/s, 8,000
tokens 125, 32,000 tokens 177 (setup writes `--prefill-chunk 8192` on this PC: 11-15% faster than the engine's 1,024-token default,
decode unchanged). The verify window runs synchronously here (setup writes `DS4_ASYNC_T=4` in the config's env; the engine's
mailbox path is sized for cards that hold most of the experts and costs this box 31% on code - `STRATA_DSV41_ASYNC_T=8` restores
the default). The tested driver is AMD Adrenalin 26.10.1 (driver 32.0.32015.7029); the 2025-03 driver failed at 32K context, so
26.10.1 is the minimum to use. The 32K and 64K contexts work on it; an 8-hour 32K soak passed on this box.

121 + 111 + 8 = 239 GB for 1.75; 138 + 111 + 8 = 257 GB for 2.00; 155 + 111 + 8 = 274 GB for 2.25.

Measured: an RTX 5070 (12 GB) with 64 GB of RAM and a 3.6 GB/s NVMe drive answers at about 4.7 tokens/s and reads prompts at 121
tokens/s (4K) to 231 tokens/s (8K, layer-major path); an RTX 3060 (12 GB) with 46 GB of RAM about 8 to 10 tokens/s (code 9.9, agent work 7.8; 149 tokens/s at 8K; DSpark stays off there, `--dspark on` adds 2-13%); four R9700 about 48
tokens/s on 2.00 and 80 on 2.25 (code, with the shipped hot-expert profile). Under about 40 GB of RAM it is untested and slow; setup says so and still installs.

**RTX 40 series (sm_89).** The engine packages carry machine code for sm_86, sm_89 and sm_120 since 0.2.7 (before, an RTX 40 card ran the sm_86 code through
CUDA's same-generation rule). We have no RTX 40 card: the sm_89 build was compiled here but not run. A user's own build for sm_89 ran the 2.00 set on an
RTX 4090 48 GB with 128 GB of RAM (#3935, their numbers): decode 40.2 tokens/s (median of 5 answers of 64 tokens, `--ssd-cache 108 --dspark on --dspark-conf 0.65`),
prompts 141 tokens/s at 1.2K and 615 at 4.6K. They saw 16 to 22 tokens/s with `--dspark auto`, which decided "off" on their computer. This is one report on one
computer; other cards and RAM sizes can differ, and `--dspark auto` stays the default.

Setup downloads with resume and SHA-256, packs the model with the plain recipe (`pack_gguf.py --model SET --shards ... --engram
engram-IQ4_NL.gguf --engram-meta <UD shard 1> --dspark-gguf dspark-...-Q8_0.gguf`; do not add `--skip-tensors` or `--dense-override`:
that packs without an error and gives garbage text), copies the shipped hot-expert profile (`data/dsv41/profile-2.00.bin` or `-2.25.bin`)
into the pack as `expert-profile.bin`, checks `strata-dsv41 --pack-info` (every file present) and answers a 16-token test question
(`tools/dsv41/smoke.py`, the text must name Paris). The start script passes `--dspark auto` (never `off`: `off` still loads the drafter
and its 7 GB of experts take the graphics memory); the RAM store sizes itself. Long prompts use the engine's fast chunked path on its own (no flag is written: the engine picks the chunk size for your cards).

## The ISTA-DAS Coder

ISTA-DAS's coding version of this model: in every layer 192 of the 384 routed experts are kept and the other 192 removed (the router, the experts and the routing biases are half as wide; every
other tensor is the 2.00 set's, with the same types). `strata launch --model ista-das-coder` (or "ISTA-DAS Coder" in the app's version list). Download 191 GB: 72 GB of model, the 111 GB n-gram
table (the same file the other sets use, downloaded once) and the 8 GB drafter; the picture file is the one every DeepSeek set shares. Where it comes from, the commit, the file names and their
SHA-256 are in one file, `data/dsv41/coder-set.json` (`python tools/dsv41/pin_coder.py OWNER/REPO` writes it from the Hub). Its hot-expert profile is its own (`data/dsv41/profile-coder.bin`, made from the engine's routing over 519,422
tokens of the calibration texts): a profile counted on 384 experts is not read by it (the engine checks the file's size).

Measured on 4 x Radeon AI PRO R9700 (the packaged Terminal, 128K context, 5 experts per word by default there): all 7,680 experts sit in graphics memory, decode about 59 tokens a second, a 93,000-token prompt
read in 80 s; on one R9700: 2,859 experts on the card, decode about 26 tokens a second (with the RAM capped for the whole run: 21 with 40 GB, 18 with 32 GB, 15 with 24 GB, 8 with 16 GB); an RTX PRO 6000 Blackwell
(CUDA 12.8, sm_120, one of two cards on a shared box): about 100 tokens a second. Against the original model (4,995 positions, 6 experts a word, 4 cards, same harness): KL 0.397, top-1 76.4%, perplexity 4.645
(reference 3.140); the same run gives 2.00 KL 0.171, top-1 83.7%, perplexity 3.630 and 2.25 KL 0.109, top-1 87.2%, perplexity 3.405. By part (KL): code 0.230, agent 0.199, documents 0.302, English 0.476, multilingual 0.777
for the Coder; 0.104, 0.108, 0.202, 0.303, 0.138 for 2.00. One run per cell unless the numbers repeat; details and the task suite in `phaseA-tests/maya-port/DS-CODER.md`. Also run (2026-11-02, official files, the 0.2.9
engines): on an Apple M5 Max 19 to 22 tokens a second with all 7,680 experts in memory (a 6,054-token needle read at 300 tokens a second), and on a Ryzen AI Max (Strix Halo, Windows, 64 GB of RAM)
19 to 22 tokens a second with 7,734 of 8,064 experts in graphics memory.

## Install

```
./setup.sh --family deepseek41 --model ISTA-DAS-2.00bpw   # 257 GB: the official 2.00 set + Engram + drafter (resumable, SHA-256 checked)
./setup.sh --family deepseek41 --model ISTA-DAS-2.25bpw   # 274 GB (the Engram is shared with 2.00)
./setup.sh --family deepseek41 --model ISTA-DAS-1.75bpw   # 239 GB, the smallest
./setup.sh --family deepseek41 --model UD-Q8_K_XL        # Unsloth's 508 GB set, for hundreds of GB of RAM or several cards
./setup.sh --family deepseek41 --gguf-dir DIR            # files you already have: the shards (or the folder with 2.00bpw / 2.25bpw / UD-Q8_K_XL inside); the Engram and drafter are looked for beside them
strata launch --model ist-das-2.00bpw --plan             # where it would go on this computer, nothing changed
strata launch --model-path DIR --model ist-das-2.00bpw   # start from files you already have
```

Setup needs `strata-dsv41` next to the engine (or `STRATA_DSV41_EXE`) and says so before it downloads anything. Build it with
`cmake -DSTRATA_BUILD_DSV41=ON`. The default context is 8192 tokens, the size it was checked at; `--context` asks for more.
Loading takes a few minutes (the config sets `"engine_ready_s": 3600`).

**Context up to 1M.** The model reads 1,048,576 positions natively, so `--context` takes 8K to 1M (`--context 512k`, `--context 1m`; the Terminal's context step
lists 8K, 32K, 64K, 128K, 256K, 512K and 1M, and the app lists them up to the limit its pack states in `ds4.txt`, `max_position_embeddings`). A size past what the
pack reads is refused before anything is downloaded. Nothing past 256K has been tested by us: the owner's decision is to offer 512K and 1M and let users try them
(Project Maya v1.1.28 did the same, #442 by @needmorevram, MIT). The cost is real. The context's cache lives in graphics memory, so every position takes memory from
the expert cache and fewer experts stay on the card; and a very long prompt is read slowly, because the indexer's work grows with the square of its length. Start
with 256K or less, and expect the first reply of a 512K or 1M prompt to take many minutes.

## Through the API

The server speaks to the engine as it does for the other models, so `/v1/chat/completions`, `/v1/messages`, streaming, stop and
`/metrics` work as usual (the answer speed is the tokens the model produced over the engine's decode time).

- Thinking is on by default, as the model's template has it. `reasoning_effort: "none"` or `chat_template_kwargs.enable_thinking: false` turns it off.
  Effort words: `low` (50), `medium` (62), `high` (75, the default), `max` (100), or a whole number 1 to 100.
- Prompts are the GGUF's own chat template, the same bytes as DeepSeek's `encoding.py` on the cases checked. Project Maya's own template lost the
  tool associations and later system instructions (their issue #299); Strata does not use it.
- Tool calls: the model writes DSML (`<｜DSML｜ calls>` ... `<｜DSML｜ invoke name="...">` ... `<｜DSML｜ parameter name="..." string="true">`). The
  server turns them into OpenAI `tool_calls` and Anthropic `tool_use`, including `tool_choice`, streamed arguments and calls written inside the thinking.
  A value marked `string="true"` stays a string.
- Defaults when a request names none: temperature 1.0, top_p 1.0.
- Agents and sub-agents that take turns do not read their prompts again. The engine sets a conversation aside when another request takes the model (about 17 MB per 6.5K tokens) and takes it
  back when the next prompt continues it, also after a restart. Slots live in `ds4-slots` next to the run log, are capped at 2 GB
  (`STRATA_DS4_SLOT_GB`) and are removed after a week. The system prompt and tools shared by an agent
  and its sub-agents are kept as a checkpoint too. `STRATA_DS4_SLOT_KEEP=0` and `STRATA_DS4_AUTO_PIN=0` turn this off.

Pictures are on by default: the engine reads them with its own vision tower (no separate picture reader). The tower waits in RAM (about 1 GB of pinned
memory) and takes its 319 MiB of graphics memory only while a picture is being read, out of the room `--reserve-mib` keeps free; the experts get that room
the rest of the time (RTX A6000 as a 12 GB-like card: 370 experts in VRAM against 145, 20.7 against 18.7 tokens/s; Radeon AI PRO R9700 as a 12 GB-like card:
419 against 386 experts, 23.3 tokens/s either way; with a whole 48 GB or 32 GB card: 35-40 experts more, no speed difference). Taking it costs no measurable
time (a picture is encoded in 0.19 s the first time, 0.08 s the next, 0 s when it was sent before). If a picture finds no free graphics memory (another
program took the reserve's room), the experts' last slots on the first GPU are lent to the tower for that picture turn, as they are to a long prompt, and read
back from the model files afterwards (35 experts, 117 ms on an RTX A6000 with 59 MiB free; the picture is read right and the next one is not slower). It stays
on the card on a GPU that shares the PC's memory (Mac, GB10, Strix Halo), and with `--vision-resident`. The server decodes each picture, brings it to a size the tower takes (both sides a multiple of 42, at most 1024 image
tokens) and hands it to the engine; it needs Pillow for that (`python -m pip install pillow`; setup installs it). `--vision no` in `strata launch`
(or `--vision off` for the engine) turns them off; a request with a picture then gets a 400 that says so.

## Several conversations at once (`--parallel`)

Off by default: requests take turns, as before. `--parallel N` (2 to 8) lets the engine answer N conversations at the same time: several chats, an agent with sub-agents,
or a few people on one server. Ask for it where you start the model:

| Where | How |
| --- | --- |
| `strata launch` | `strata launch --family deepseek41 --model ist-das-2.25bpw --parallel 4`; `--save-preset NAME` keeps the number for the next start. The guided screens have the same choice in the Parallel step. |
| `strata serve` | `strata serve --profile NAME --parallel 4`, or `"parallel": 4` in the run config. |
| the app | Expert settings, "Concurrent requests" (the setting the other models have). |
| Docker | `PARALLEL=4`. |

`--plan` with the flag says what it does and what it was measured on. A number above 8 is run as 8. If the graphics cards cannot hold the buffers of every
conversation, the engine runs as many as they can hold and says so in its log (`the graphics memory holds the buffers of N conversations ... running N at once`). A DeepSeek
engine from before the flag answers one request at a time; `strata launch` says so and goes on, and `strata update` fetches the newer engine.

What it does:

- Each conversation has its own attention memory in graphics memory. One extra conversation takes about 0.01 GB at 8K context, 0.05 GB at 32K, 0.17 GB at 128K and 0.68 GB at
  512K (the engine prints the total at start). That memory comes out of the room the experts would use. Measured on 2 RTX A6000 (48 GB each); the engine's line for 8
  conversations at 1M context was 9.47 GB for the 7 extra.
- Several long answers at once. 4 R9700 cards, answers of 150 tokens: one answer alone wrote 41 tokens a second; 2 together 53 tokens a second in all (each at 30), 4 together
  73 (each at 21). 2 RTX A6000: one alone 42, 2 together 52 (each at 31); 4 together 48 to 62 (each at 14 to 19, two runs). The median wait for the first word, with 4
  sessions asking at once, was 0.9 s instead of 8.0 s (4 R9700), because requests no longer queue.
- An agent load: 4 sessions of 5 requests, each reading 2,500 characters of file text and answering 40 tokens. The whole load took 55 to 56 s instead of 64 to 65 s on 4 R9700
  (13% less), 91 s instead of 102 s on 2 RTX A6000 (11% less), and 205 to 216 s instead of 256 to 261 s on a DGX Spark (17 to 20% less).
- One conversation alone: a five-turn chat took 23.3 to 24.9 s with `--parallel 4` and 23.4 to 24.4 s without on 4 R9700, and 34.9 to 35.9 s against 35.2 to 35.6 s on 2 RTX A6000.
- Answers are the same as when each runs alone: 12 combinations of 2 to 4 conversations, with and without the drafter, were identical to the last bit of the model's
  output on a DGX Spark and on 2 RTX A6000, and 6 prompts at once through the server had the same text as alone (greedy), also when a client dropped its connection.

Where it was run: Linux, with 4 Radeon AI PRO R9700 (gfx1201), 1 R9700, 2 RTX A6000 (sm_86) and a DGX Spark (sm_121). Not run: Windows (the engine builds with it, a
Windows run has not been done), a Mac (Metal; the code adds no Metal kernel of its own, nothing has started it there), Ryzen AI Max, cards under 24 GB, other AMD
families. `strata launch` warns for these and still starts. If the engine stops at the start or an answer looks wrong, start without `--parallel`.

## Speed: Faster or Full quality

DeepSeek picks 6 experts for every word it writes. **Faster** uses 5 of them (engine flag `--experts 5`): a small quality change, usually not noticeable, and answers come sooner. It is the default on a PC whose graphics cards total 16 GiB or less, and on an integrated GPU such as Strix Halo: the engine decides that itself (`--experts auto`), and Strata passes `--experts` only when you choose. **Full quality** uses all 6 (`--experts 6`).

The choice needs the DeepSeek engine 0.1.50 or newer (0.1.49.2 does not take `--experts`); with an older engine the step and the plan line are left out and nothing is passed. In `strata launch` the guided screens have a Speed step for DeepSeek; with flags use `--speed faster|full`, or `--full-quality` in scripts. `strata launch --plan` shows which one this PC gets.

## Tuning

`strata launch --tune quick` (or `thorough`, or the Tuning screen) works for DeepSeek with its own short tuner, `tools/dsv41/tune.py`. It starts
the engine a few times on a fixed short question and measures the answer speed: the standard settings first, then the DSpark drafter forced on,
then a shorter draft block (3), then one CPU thread for each physical core (thorough also tries every logical thread). A setting is kept only for a
gain of more than 5%, and the result is written into the launch's settings. Expect about 5 to 10 minutes (each run loads the model, which takes
1 to 3 minutes); close other heavy programs first. If a run fails or the computer is too busy to tell, the standard settings stay and one plain line
says so. The RAM store (`--ssd-cache`) sizes itself when the model starts, so the tuner does not touch it, and the context stays what you chose.

## The engine package

DeepSeek's engine, `strata-dsv41`, is a separate download from the Strata engine: when you choose DeepSeek, setup (or `strata launch`) fetches the
package for your computer from the signed update feed (resumable, signature and every file checked) and installs it in `engine-dsv41/` beside
`engine/`. The Qwen engine's files are never changed. If the feed has no package for your graphics card, setup says so in plain words and stops before
anything big is downloaded. The Windows AMD package (`...-windows-x86_64-hip-rdna3`) carries the ROCm 10.2 runtime and the
rocBLAS / hipBLASLt kernels for gfx1151 in `rocm/bin`, which Strata puts on the engine's library path.

### Getting the engine by hand

If the automatic download does not work for you (no access to the update feed, or an offline computer), download the engine package yourself from
the release page: the file is named `strata-dsv41-<version>-<system>-<backend>-<variant>.tar.gz` (Windows and Linux; pick the one for your system and
graphics card). Check it against `SHA256SUMS.txt` on the release page if you like. Then give it to Strata in either of two ways:

```
strata launch --model ist-das-2.00bpw --dsv41-package PATH-TO-THE-FILE
python setup.py --family deepseek41 --dsv41-package PATH-TO-THE-FILE
```

or put the file in the `engine-dsv41/` folder (setup tells you its exact place when it cannot find the engine) and run the same command again.
Strata checks the package before it uses it: it must be the DeepSeek engine (not the Strata engine) for your system, its manifest must be signed by a key
Strata trusts, and every file must match the hash in the manifest. A damaged or foreign file is refused with a plain line and nothing is installed.
The unpacked folder works as well as the archive. The Strata (Qwen) engine is not touched either way.

### Building the engine for your card

The packages hold machine code for these NVIDIA cards (CUDA 13, Windows and Linux): sm_86 (RTX 30), sm_89 (RTX 40) and sm_120 (RTX 50), plus PTX for
sm_120. If your card is none of these, or you want your own build, compile `strata-dsv41` and put it where Strata looks for it. You need the CUDA toolkit
(the packages are built with 13.0), a C++ compiler, CMake, Ninja and the llama.cpp checkout Strata pins (`third_party/llama.cpp`).

```
cmake -S . -B build-dsv41 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON       "-DCMAKE_CUDA_ARCHITECTURES=89-real" -DSTRATA_GGML_DIR=third_party/llama.cpp
cmake --build build-dsv41 --target strata-dsv41 -j 8
```

Use your card's number in `CMAKE_CUDA_ARCHITECTURES`: 86 for RTX 30, 89 for RTX 40, 120 for RTX 50, or `native` for the cards in this computer. Without the
option a DeepSeek build is made for 86, 89 and 120. On Windows, `packaginguild_dsv41_windows.bat BUILD_DIR` does the same (`DSV41_ARCH` names the cards);
on Linux, `packaging/build_dsv41_cuda.sh` (`ARCH=...`). Then copy the file into the `engine` folder of your Strata install (next to `strata`); setup finds it there and does not download an engine:

```
python setup.py --family deepseek41 --model ISTA-DAS-2.00bpw
```

`strata-dsv41 --version` prints the commit it was built from. A build you made yourself is not signed by Strata, which is fine for your own computer.
Please tell us how it runs on your card (the speed, and the card and RAM): we only list a card as tested after someone has run it.

## Compressed experts (trellis), for testing

`strata-dsv41` reads the routed experts in the trellis types as well as the source MXFP4: TQK6 and TQK7 (agentionai's files), and the
experimental TQK5X / TQK4X (computed codebook) and TQK5T / TQK4T (table codebook), see `tools/trellis/README.md`. A layer may be converted
while the others stay MXFP4. On the GPU they run on the tensor cores of gfx12 (and in plain FMAs elsewhere), on the CPU lane on Strata's
AVX-512 / AVX2 kernels; the build needs `-DSTRATA_NATIVE_EXPERTS=ON` for the CPU lane.

```
# 1. the pack of the source model (also takes the DSpark and mmproj files that sit beside the model folder)
python tools/dsv41/pack_gguf.py --model DIR/UD-Q8_K_XL --out PACK
# 2. re-encode some layers (HIP encoder; one layer at K5 is 2.3 GB, the whole model 93 GB)
python tools/dsv41/tq_convert.py --pack PACK --out CONV --layers 1,7,14 --type K5      # K4 K5 K4T K5T K6 K7
# 3. the pack that reads them
python tools/dsv41/pack_gguf.py --model DIR/UD-Q8_K_XL --out PACK2 --tq-experts CONV
strata-dsv41 --pack PACK2 --score data/quant_quality/v1/set.ids.json --score-out RUN.tsv      # teacher-forced top-64 rows
```

The encode is plain error (no rotation, no activation weights). The tests are `ds4_tq_test` (device and CPU lane against a double
reference, needs one GPU) and `tqw_hip_test`.
