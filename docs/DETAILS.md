# Strata - the details

The technical side of Strata: every measured number, the API, images, all settings and how the engine works.
New here? Start with the [README](../README.md) - it has everything you need to install and use it.

> **On this page:** [Speed](#speed-measured) · [Other GPUs](#other-gpus-estimated) · [Which model?](#which-model) ·
> [Requirements](#before-you-start) · [Windows](#windows) · [Linux](#linux) · [API](#using-it) · [Images](#images-vision) ·
> [Troubleshooting](#troubleshooting) · [How it works](#how-it-works)

---

## Speed (measured)

RTX 5070 **12 GB**, Ryzen 5 7600 (6 cores), 64 GB DDR5-5200, Windows. One code-agent prompt per length, 256 generated
tokens, MTP speculative decoding on. "262K" is the model's full context window (a 259,943-token prompt).

### Prompt processing (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 389 | 539 | 571 | 561 | 543 | 496 |
| **IQ2_XS** | 332 | 463 | 495 | 486 | 472 | 437 |
| **IQ3_XXS** | 285 | 410 | 435 | 427 | 414 | - |

### Output (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 88.7 | 94.6 | 87.5 | 76.4 | 65.1 | 56.3 |
| **IQ2_XS** | 82.0 | 78.0 | 65.3 | 63.7 | 52.0 | 48.0 |
| **IQ3_XXS** | 64.6 | 65.6 | 57.3 | 54.4 | 44.8 | - |

IQ3_XXS at 262K is not measured: with its 43 GB of experts, a 260K-token context brings a 64 GB PC to its memory
limit. Use up to 128K with IQ3_XXS on 64 GB.

Time to first token is prompt length / prompt speed: about 7 s at 4K, 55 s at 32K, 4 minutes at 128K and 9 minutes at
262K. The raw numbers: [`bench/results/`](../bench/results/). The [paper](paper/Strata-Paper.pdf) explains every number.

## Other GPUs (estimated)

Not measured - estimated from the runs above (same CPU and 64 GB RAM): the GPU part scaled by memory bandwidth, the CPU
part by how many more experts the card's VRAM holds. Treat as **±20%**. Numbers are *prompt / output* tokens/s.

| GPU | Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5060 Ti 16GB | Q2_0 | ~341 / ~80 | ~472 / ~87 | ~501 / ~81 | ~492 / ~72 | ~476 / ~62 | ~435 / ~53 |
|  | IQ2_XS | ~291 / ~80 | ~406 / ~77 | ~434 / ~63 | ~426 / ~62 | ~413 / ~51 | ~383 / ~47 |
|  | IQ3_XXS | ~249 / ~66 | ~359 / ~65 | ~381 / ~56 | ~374 / ~54 | ~363 / ~45 | - |
| RTX 3090 24GB | Q2_0 | ~355 / ~128 | ~491 / ~140 | ~521 / ~130 | ~512 / ~115 | ~495 / ~100 | ~453 / ~85 |
|  | IQ2_XS | ~303 / ~131 | ~422 / ~128 | ~451 / ~103 | ~444 / ~102 | ~430 / ~85 | ~398 / ~78 |
|  | IQ3_XXS | ~260 / ~106 | ~374 / ~103 | ~396 / ~89 | ~390 / ~85 | ~378 / ~71 | - |

More VRAM matters more than a faster GPU: every extra GB holds ~700 more experts, and every expert on the GPU is one the
CPU does not have to compute. A 3090's 24 GB takes most of the CPU work away.

## Which model?

All three are [ISTA-DASLab's GSQ-RCO quantizations](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

| Model | Download | RAM it uses | Speed | Quality |
| --- | ---: | ---: | --- | --- |
| **Q2_0** | 66 GB | ~34 GB experts + ~6 GB | fastest | good |
| **IQ2_XS** | 68 GB | ~36 GB experts + ~6 GB | close to Q2_0 | a bit better |
| **IQ3_XXS** | 76 GB | ~43 GB experts + ~6 GB | slower (more CPU work) | best |

With 64 GB of RAM all three fit (close the browser for IQ3_XXS, and keep its context at 128K or less). With 48 GB only Q2_0 / IQ2_XS may fit. 32 GB is not enough.

### Or: Swift 1.5 (a fine-tune that thinks shorter)

The setup's first question also offers **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**,
UkisAI's fine-tune of Qwen3.8-Flash-Next, trained to reach the answer with much less thinking (its authors: 63% fewer
thinking tokens, 1.8x sooner answers, under 1% accuracy loss). Same architecture, the same three sizes, its own
vision encoder; Strata runs it at the same speed (4K, IQ2_XS: 465 prompt / 78.7 output tokens/s, vs 467 / 78.3 for
the original). Its authors recommend **IQ2_XS** (their Q2_0 is marked experimental). Its license is the Swift Open
License 1.0 - read it on the model page.

Our small check (8 reasoning questions, default thinking, IQ2_XS): both models got **8/8**; Swift used **1,234**
output tokens in 28 s, the original **2,682** in 46 s - most of the difference from one question the original
thought about for 1,524 tokens. Not a benchmark, but consistent with the claim.

```
START-HERE.bat --setup --family swift --model IQ2_XS
```

## Before you start

You need **only an NVIDIA driver** (version 580 or newer; update it with the NVIDIA App or from
[nvidia.com/drivers](https://www.nvidia.com/drivers)). Everything else is installed for you the first time.

| | |
| --- | --- |
| GPU | NVIDIA **RTX 30, 40 or 50 series**, **12 GB VRAM or more** (8 GB runs, slowly). Measured on an RTX 5070; RTX 30/40 are untested. |
| RAM | **64 GB** recommended (see the table above). |
| CPU | x86-64 with AVX2 (any Intel/AMD desktop CPU from the last ~8 years). AVX-512 (Ryzen 7000/9000) is a bit faster. |
| Disk | ~70-80 GB free for the model, ~6 GB for the MTP layer (+1 GB with images). **Q2_0 on an AVX-512 CPU** also writes a one-time ~40 GB copy of its experts for the fast CPU kernel. An NVMe SSD is strongly recommended. |
| OS | Windows 10/11, or Linux (Ubuntu 22.04/24.04 get everything installed automatically). |

What the first start installs, all inside this folder (`.venv/`, `engine/`, `third_party/`, `models/`, `packs/`, `mtp/`):
Python 3.12 if you have none (for your user account, no admin), a private Python environment, NVIDIA's CUDA libraries
(from pip, ~0.4 GB), the ready-made Strata engine for RTX 30/40/50, the model and the MTP draft layer. If no
ready-made engine fits your PC, it offers to install the build tools (Visual Studio Build Tools + CUDA Toolkit on
Windows, `build-essential` + CUDA on Ubuntu) and compiles the engine for your GPU (asks first; 20-40 minutes once).

---

## Windows

### Double-click `START-HERE.bat`

**The first time** it asks four questions and does the rest:

1. **Which model?** Qwen3.8-Flash-Next (the original) or Swift 1.5 (the fine-tune that thinks shorter).
2. **Which size?** Q2_0, IQ2_XS or IQ3_XXS (it recommends one for your RAM).
3. **How much context?** 8K to 256K tokens (it recommends one for your VRAM).
4. **Images?** yes / no (see [Images](#images-vision)).

Then it downloads and prepares everything (the model is 66-76 GB, so the first start takes a while; an interrupted
download continues where it stopped) and **starts the model**: your browser opens `http://127.0.0.1:8080`, a small
page that shows it is running and lets you chat. The API is at `http://127.0.0.1:8080/v1` for your apps.

**Every time after that**, `START-HERE.bat` just starts the model (30-90 s to load 34-43 GB into RAM). Nothing is
downloaded again. Closing the window stops the model.

```
START-HERE.bat --setup                          install another model, or change context / images
START-HERE.bat --model IQ2_XS --context 32768 --vision yes --yes     no questions
START-HERE.bat --gguf-dir D:\models\IQ2_XS       use GGUF files you already have
START-HERE.bat --port 8081                      another port
```

With more than one model installed, it asks which one to start. `run-<model>.bat` starts a model directly.

### Chat in the terminal (optional)

```
.venv\Scripts\python chat.py
```

---

## Linux

```bash
./setup.sh
```

The same questions, the same automatic install (it uses `sudo apt` for Python and, only if it has to compile,
for the build tools), and the same start: `http://127.0.0.1:8080`. Later runs of `./setup.sh` (or `./run-<model>.sh`)
start the model directly. Options as on Windows (`./setup.sh --setup`, `--model Q2_0 --yes`, `--gguf-dir /data/Q2_0`).
Terminal chat: `.venv/bin/python chat.py`.

---

## Using it

The server listens on `http://127.0.0.1:8080` (change with `--port` in setup, or edit the run script).

| API | Endpoint |
| --- | --- |
| OpenAI Chat Completions (stream and non-stream, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (stream and non-stream, tools) | `POST /v1/messages` |
| Model list / health | `GET /v1/models`, `GET /health` |

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "strata", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

- **Thinking levels: none, low, medium, high.** The model thinks before it answers (streamed as
  `reasoning_content`, Anthropic: `thinking` blocks). Choose how much per request - in the chat page (the "Thinking"
  menu), in `chat.py` (`/think low`), or over the API:

  | API | how |
  | --- | --- |
  | OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (also `"reasoning": {"effort": ...}`, or `"chat_template_kwargs": {"enable_thinking": false}`) |
  | Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`, `"thinking": {"type": "disabled"}`, or `"thinking": {"type": "enabled", "budget_tokens": N}` (under 2K = low, under 8K = medium, more = high) |

  Without a setting the model uses its own default, **high**. `none` answers at once (fastest); `low` keeps the thinking
  short. The levels are instructions the model was trained with, not a hard token limit: on easy questions all three
  think briefly, on hard ones `high` thinks longest and is most accurate.
- **Chat apps.** Any app with an "OpenAI-compatible" provider works: base URL `http://127.0.0.1:8080/v1`, any API key.
- **Context.** Chosen in setup (8K-262K). Requests longer than that are refused, never silently cut.
- **From other devices / the internet.** The server listens on your PC only (`127.0.0.1`). To reach it from elsewhere,
  put a tunnel in front of it, for example [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/):
  `cloudflared tunnel --url http://127.0.0.1:8080`. **Set a key first**, or anyone with the link can use your PC:
  add `"api_key": "some-long-secret"` to `strata-<model>.json` (or set the `STRATA_API_KEY` environment variable);
  clients then send it as their API key.

**Current limits (v1):** one request at a time; greedy decoding (temperature is ignored); every request processes its
whole prompt again (no conversation cache yet, so long chats have a long time-to-first-token); images only when set up
with them (below); no video.

---

## Images (vision)

The model has a vision encoder: [`mmproj-Qwen3.8-Flash-Next-BF16.gguf`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(0.9 GB, a 27-layer ViT plus the projector into the language model). It is **optional**: say yes when the setup asks
"Images?", or run it again with `--vision gpu` (or `--vision cpu`). The setup downloads the encoder, builds a small
helper (`strata-vision`, from llama.cpp's `mtmd` library) and adds it to your start script. Nothing else changes.

| Encoder on | Time per picture | Cost |
| --- | --- | --- |
| **GPU** (recommended) | **0.1-0.5 s** (up to 1,024 image tokens) | ~1.4 GB of VRAM is kept free for it, so the expert cache is smaller: text output is a few % slower (table below) |
| CPU | 10-30 s (pictures are scaled down to ~300 image tokens) | nothing on the GPU |

A picture becomes up to 1,024 tokens of the context (a 640x480 photo: 300). The same picture sent again, as chat apps
do on every turn, is encoded only once.

### Sending a picture

**Terminal chat:** type `/image <path to a picture>`, press Enter, then type your question.

```
you> /image C:\Users\me\Pictures\receipt.jpg
(picture attached: receipt.jpg - now type your question)
you> What is the total on this receipt?
```

**OpenAI API** (an `image_url` part: a `data:` URL, an `http(s)://` URL or a local file path):

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
img = base64.b64encode(open("photo.jpg", "rb").read()).decode()
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
```

**Anthropic API:** an `image` block with a `base64` (or `url`) source, as usual.

JPEG, PNG, BMP, GIF, WebP, TIFF and AVIF work (the last ones are converted to PNG first; agents such as omp send
WebP). Chat apps with image upload work the same way.

### Speed with images on (4K context, measured)

| Model | Prompt tok/s, images off | Prompt tok/s, images on | Output tok/s, images off | Output tok/s, images on |
| --- | ---: | ---: | ---: | ---: |
| **Q2_0** | 541 | 531 | 90.7 | 86.9 |
| **IQ2_XS** | 468 | 458 | 77.1 | 74.0 |
| **IQ3_XXS** | 411 | 401 | 64.7 | 59.5 |

"Off" is the published setup; "on" runs with the encoder loaded on the GPU and its VRAM kept free, so ~1,000 fewer
experts fit in VRAM and the CPU computes a few more per token: 2-8% slower. For text, turning images on changes
nothing else (the output is bit-identical when the VRAM is the same).

A question about a picture (a 640x480 newspaper page = 300 image tokens, a 328-token prompt; the whole request, encoder
on the GPU, measured through the API):

| Model | Answer ("MEN WALK ON MOON") | Picture encoded | Prompt | Output |
| --- | ---: | ---: | ---: | ---: |
| **Q2_0** | 1.9 s | ~0.1 s | 207 tok/s | 65-73 tok/s |
| **IQ2_XS** | 2.3 s | ~0.1 s | 173 tok/s | 50-60 tok/s |
| **IQ3_XXS** | 2.7 s | ~0.1 s | 144 tok/s | 41-50 tok/s |

(Short prompts run below the 4K prompt speed: a 2,048-token chunk is where the prompt path is efficient. Short answers
run below the long-output speed: the first rounds have no draft yet.)

**How it works inside:** the encoder turns the picture into rows of the same width as the model's word embeddings;
Strata puts them where the prompt has `<|image_pad|>` tokens and gives each one its 2-D position (row and column in the
picture; the model uses interleaved M-RoPE). Answers match llama.cpp's multimodal implementation token for token on our
test images.

---

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `the NVIDIA driver is too old` | Update the driver (NVIDIA App or nvidia.com/drivers), restart, run `START-HERE.bat` again. |
| Python or the build tools could not be installed | Install what it names (links are printed), then run it again. Everything already done is kept. |
| `port 8080 is already in use` | Strata is already running (look for its window), or another program uses the port: `START-HERE.bat --port 8081`. |
| `cudaHostRegister ... out of memory` in the log | Normal on Windows: the engine pins the experts in per-layer slices instead. Only a problem if the load then fails. |
| The first start takes minutes | It is reading 34-43 GB into RAM; the second start is faster while the files are in the OS cache. |
| Slow output, disk light busy | Not enough free RAM: close other programs, or choose Q2_0 / IQ2_XS. |
| `prompt ... exceeds the context` | The request is longer than the context you chose: run setup again with a bigger `--context`. |
| Slower than the tables | The monitor plugged into the GPU and other GPU programs take VRAM from the expert cache; RAM running below its rated speed (enable EXPO/XMP in the BIOS) slows the CPU half. |
| `this server was started without the vision encoder` | The model was set up for text only: run setup again with `--vision gpu`. |
| A picture is refused or `cannot read the image` | The file is not a picture Pillow can open (JPEG, PNG, WebP, GIF, BMP, TIFF, AVIF work). |
| Pictures are slow (10-30 s) | The encoder runs on the CPU: run setup again with `--vision gpu` (needs ~1.4 GB of VRAM). |
| Anything else | The engine log is `strata-<model>.log` in this folder. |

---

## How it works

<p align="center"><img src="paper/tiers.svg" width="760" alt="memory tiers"></p>

- **GPU (VRAM):** attention and DeltaNet mixers, the gated-residual weights, routers, shared experts, output head, the MTP
  draft layer, the KV cache, and an **expert cache** that fills the rest of VRAM with the most-used experts (it adapts to
  the conversation while you chat).
- **RAM:** all 24,576 experts, pinned. The CPU computes the experts that are not on the GPU **in place**, at the same time
  as the GPU works on the cached ones (AVX-512 / AVX2 kernels, ggml's for the i-quants).
- **SSD:** the 28.8 GB n-gram table, read a few rows per token through the OS cache.
- **Speculation:** the model's own MTP layer drafts up to 3 tokens; one pass over all 48 layers checks them. 2.4-3.2
  tokens per pass on average.
- **Prompts** are processed in 2,048-token chunks with the experts streamed to the GPU over PCIe.

The full story, with measurements, bottlenecks and what comes next: **[docs/paper/Strata-Paper.pdf](paper/Strata-Paper.pdf)**.

---

## Credits and licenses

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; quantizations:
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
  Swift 1.5: [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  by UkisAI. Their licenses apply to the weights.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT): the i-quant formats, the GPU dot products and
  dequantizers transcribed in `src/kernels/cuda/iq_kernels.cu`, the CPU backend linked for the i-quant experts, the
  `mtmd` library behind the image encoder (`tools/vision/`), and `gguf-py` used by the tools. See
  `third_party/ggml/LICENSE`.
- Ideas from [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen); references in the paper.
