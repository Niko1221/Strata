# agentionai Signal AP-Q4_K_XL

Setup installs Signal (`--family signal`, sizes AP-Q4_K_XL and AP-IQ4_XS, images included): see
[What setup does](#what-setup-does). The rest of this page is the manual workflow setup automates, and what was checked.

Packed, loaded and run on Linux with an RTX 4090 (24 GB), Ryzen 9 7950X3D and 93 GB RAM on 2026-10-11. The engine
starts on this pack and answers requests; the speeds quoted here were measured with `tools/strata_bench.py` on the
configuration shown, and no output-quality comparison against the base model was made. What was checked is listed
at the end.

## The file

`agentionai/Signal-3.8-Flash-Next-GGUF`, **AP-Q4_K_XL**: one GGUF of 101,142,769,984 bytes (94.19 GiB), 1224
tensors, `general.architecture qwen4exp`, `general.finetune Signal`. Strata's geometry checks pass on it, and it
packs the way any quantization that compresses more than the experts does.

Against the file setup installs on this machine, Unsloth's UD-Q4_K_XL (four shards, 111,434,654,784 bytes):

| | Signal AP-Q4_K_XL | Unsloth UD-Q4_K_XL |
| --- | --- | --- |
| expert gate / up | Q4_K | Q4_K |
| expert down | IQ4_NL | Q5_1 |
| routed expert bytes | 67,947,724,800 (63.28 GiB) | 77,017,907,200 (71.72 GiB) |
| PLE lookup table | IQ4_NL | IQ4_NL |
| attention, shared experts, hyperconnection, PLE key/value, embedding, output head | Q6_K | Q8_0 |

Both expert types already have GPU kernels in the stock build: Q4_K (ggml type 12) is in the gate/up format list and
IQ4_NL (type 20) in the down list (`src/kernels/cuda/iq_kernels.cu:611`, `:615`), and `native_expert_supported()`
(`src/kernels/cuda/iq_kernels.cu:2790`) takes the pair at n_embd 2560, n_ff 640. No CMake option and no rebuild.
The file has to be fully downloaded first: `iq_pack.py` stops if a tensor's bytes are not all there.

## Pack the projections the kernels read as BF16

Build Strata normally; Python needs numpy, regex and gguf-py from the pinned dependency (`STRATA_GGUF_PY` can point
at its `gguf-py` directory).

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf \
  --out packs/signal-ap-q4_k_xl --compat-bf16
```

It converted 460 tensors: 96 F32 to BF16 exactly, and 364 with rounding - 266 Q6_K (worst error 0.00781), 96 Q8_0
(0.01440) and 2 IQ4_NL (0.00922). `dense.bin` is 1,538,035,200 bytes (1.43 GiB) against 1,485,688,320 for the
Unsloth pack. The rounding is relative to the GGUF's dequantized weights; it does not recover the original
checkpoint. Every tensor is listed in `conversions.json` and `compat-bf16.json`.

Use this model's own tokenizer, which the pack exports from the same file, and do not reuse another model's
`dense.bin`.

## The embedding comes from one extra file

Signal stores `token_embd.weight` as Q6_K. The native embedding path takes BF16 and the quantized types `is_iq()`
dequantizes - Q2_0, Q3_K, Q4_K, Q5_K, Q5_1, Q5_0, Q4_1, Q4_0, Q8_0 and the IQ types - and Q6_K is in that set only in
a build configured `-DSTRATA_Q6K_EXPERTS=ON` (`src/kernels/cuda/iq_kernels.cu:1917`, `:2702`, checked at
`src/core/native_head.cpp:124`). The stock engine therefore stops at startup saying
`native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU dequantizer`. The pack
cannot carry the table - `iq_pack.py` leaves it in the GGUF - so `tools/embd_bf16_pack.py --gguf` dequantizes it to
BF16 and `--embd-gguf` hands the engine the file:

```sh
.venv/bin/python tools/embd_bf16_pack.py \
  --gguf /path/to/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf \
  --out /path/to/signal-ap-q4_k_xl-embd-bf16.gguf
```

On this machine that is 248,320 x 2,560 Q6_K turned into 1,271,398,400 bytes of BF16, worst rounding error 0.00036,
measured by the tool as it wrote (file 1,271,398,784 bytes). `--embd-gguf` replaces only the embedding table: the
output head still comes from the `--native` file (`src/program/generate.cpp:2897`, `src/core/native_head.cpp:116`),
and the head itself (`output.weight`, Q6_K) is read natively, because the decode path handles Q6_K with no option at
all (`src/kernels/cuda/native_mmvq.cu:2464`, `:2509`).

## The PLE table needs nothing extra

IQ4_NL is the format the table reader expects by default (`src/kernels/ngram.cpp:167`), and the loader takes
`per_layer_token_embd.weight` from the `--native` file when that file holds it
(`src/program/generate.cpp:2490`), so no `--ple-gguf`. The table is 26.8 GiB and stays on disk, read through mmap.

## Server

Save this as `strata-signal-ap-q4_k_xl.json`, replacing both `/path/to/` entries with the Signal file (the same one
twice - it is a single file, and it carries the PLE table):

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/signal-ap-q4_k_xl",
    "--native", "/path/to/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf",
    "--embd-gguf", "/path/to/signal-ap-q4_k_xl-embd-bf16.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "auto", "--spec", "4",
    "--mtp", "mtp/rt", "--max-context", "262144", "--kv", "int8", "--kv-resident", "32768",
    "--resident-budget-gib", "67", "--pcie-frac", "0.35", "--spec-min-p", "0.70", "--pool-workers", "8"
  ],
  "cwd": ".",
  "tokenizer": "packs/signal-ap-q4_k_xl/tokenizer",
  "model_name": "qwen3.8-flash-next-signal-ap-q4_k_xl",
  "log": "strata-signal-ap-q4_k_xl.log",
  "host": "127.0.0.1",
  "port": 12223
}
```

```sh
.venv/bin/python -m serve.server --engine strata --config strata-signal-ap-q4_k_xl.json --port 12223
```

The web interface is at `http://127.0.0.1:12223`, API clients use `http://127.0.0.1:12223/v1`. The port is the one
`--port` asks for (8095 if you leave it out); a launcher can pass its own, so the `port` entry in the file is only a
note to the reader. Images are not configured in this example (see [Images](#images)), and one graphics card runs one
engine: stop another server before starting this one.

The last four settings belong to one machine (RTX 4090, Ryzen 9 7950X3D, 93 GiB RAM) and are not general defaults.
`--pcie-frac` and `--spec-min-p` are what `setup.py`'s calibration chose for Signal there on 2026-10-11, each the
median of three rounds; `--pool-workers 8` is its pick from one run per value. On other hardware, calibrate again.
The calibration also chose `--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`, from one run each. Measured with
`tools/ab_engine.py` (four rounds, the arms alternating), the engine's default was faster: 44.3 against 35.9 tokens/s
median for 100-token answers (faster in 16 of 16 pairs) and 51.4 against 46.1 tokens/s for answers of up to 512 tokens
(12 of 16). Those three settings are left out.

`--kv int8` follows Signal's authors, who advise against a 4-bit value cache ("a 4-bit value cache makes long traces
degenerate"). The calibration and the comparison above ran with `--kv q4_0`. With `--kv q4_0` the engine had 190 MiB
more VRAM for experts (6216 cache slots instead of 6141) and kept 1.4 GiB less of the KV cache in pinned RAM (1.69
against 3.09 GiB); `tools/strata_bench.py` the same day measured 46.5 tokens/s for short answers, 51.9 in the agent
session and 1978 tokens/s for the 16K prompt, so 8-bit was not slower (table below).

`--resident-budget-gib` caps only the RAM copy of the experts the GPU cache does not hold. With the 6141 cache slots
this card gets (15.81 GiB of VRAM) that is 18,435 experts, 47.47 GiB, and any budget from there up copies all of them
(with `--kv q4_0` and 6216 slots, the engine logged the same 47.28 GiB page-locked copy at 67 and at 69). The 26.8 GiB
PLE table is what gets streamed. Below 47.47 GiB, the least-used experts are read from the SSD instead.

With exactly the configuration above, on 2026-10-11, the engine took 110 and 121 s (two starts) from process start to
the port answering and held 60.3 GiB of RSS after a few short requests. `tools/strata_bench.py` (3 runs per test,
thinking off, temperature 0) measured:

| test | median | range |
|---|---|---|
| short chat, decode (up to 100 tokens) | 49.3 tokens/s | 42.5-60.2 (9 answers) |
| short chat, first token | 0.50 s | 0.41-0.57 |
| 4K prompt | 1223 tokens/s, first token 3.31 s | 698-1233 tokens/s |
| 16K prompt | 2071 tokens/s, first token 7.78 s | 1509-2072 tokens/s |
| agent session, decode | 58.7 tokens/s | 37.7-75.3 (15 turns) |
| agent session, first token | 3.37 s | 2.94-4.63 |
| agent session, prompt reused from the cache | 68.8% | 68.7-68.8% |

The desktop ran on the integrated GPU, with two of the three monitors cabled to the 4090; the Steam client was open and
no game was running.

`--mtp mtp/rt` is the original model's draft head, prepared by `tools/mtp_fetch.py`, `tools/mtp_pack.py` and
`tools/mtp_rt.py` as in the [OrcaRouter setup](ORCA.md); without `--mtp` the server drafts by lookup only. Signal
verifies every draft, and a fine-tune's acceptance is not the base model's. It has not been measured for Signal.

## What setup does

`setup.py --family signal --model AP-Q4_K_XL` (or `AP-IQ4_XS`) does the steps above by itself: it downloads the file
from a pinned revision and checks its SHA-256, packs it with `--compat-bf16`, writes the BF16 embedding beside the
pack (`token-embd-bf16.gguf`) and adds `--embd-gguf` to the config. With images on, it downloads Signal's own encoder
(`mmproj-F16.gguf`, 904 MB), saved as `mmproj-Signal-3.8-Flash-Next-F16.gguf`. On an AMD card setup asks before it
installs AP-Q4_K_XL, as it does for UD-Q4_K_XL: the HIP engine has no prompt kernels for Q4_K experts.

Run on the machine above on 2026-10-11 with the file already downloaded:
`./setup.sh --family signal --model AP-Q4_K_XL --gguf-dir <its folder> --vision gpu --build --yes --no-start`. It took
5 minutes, including the SHA-256 check of the 101 GB file and compiling the image encoder. The pack, the tokenizer and
the embedding file it wrote are byte for byte the ones built by hand above. Its config is not the one above: 32K
context (setup's default on this card), a 63 GiB RAM budget, images on, and none of the calibrated settings; the KV
cache is 8-bit in both. The server was ready 86 s after start, and `tools/strata_bench.py` (3 runs per test,
thinking off, temperature 0) measured:

| test | median | range |
|---|---|---|
| short chat, decode (up to 100 tokens) | 59.9 tokens/s | 48.1-70.8 (9 answers) |
| short chat, first token | 0.38 s | 0.33-0.45 |
| 4K prompt | 1400 tokens/s, first token 2.89 s | 960-1402 tokens/s |
| 16K prompt | 2301 tokens/s, first token 7.00 s | 1834-2317 tokens/s |
| agent session, decode | 71.2 tokens/s | 44.4-89.8 (15 turns) |
| agent session, first token | 2.95 s | 2.44-4.12 |
| agent session, prompt reused from the cache | 68.7% | 68.6-68.8% |

Every speed here is higher than in the table above, but the two configurations differ in five settings at once
(context, RAM budget, the three calibrated settings), plus images, so this does not say which of them makes the
difference.

AP-IQ4_XS has not been downloaded or run. Its header, read from the first 32 MB of the file, differs from
AP-Q4_K_XL's only in the routed experts' gate and up projections: IQ3_S instead of Q4_K, 57,252,249,600 bytes of
experts. IQ3_S gate/up with IQ4_NL downs is the pair Unsloth's UD-IQ4_XS uses, which setup already runs on NVIDIA and
AMD cards.

## Images

Signal ships its own image encoder, `mmproj-F16.gguf`: a `qwen3vl_merger` projector to 2560 wide, the model's width,
in F16. With the configuration setup wrote, `strata-vision` loaded it in 0.7 s, and Signal answered (thinking off,
temperature 0):

- a drawn test picture (the number 7341 at the top, a red circle on the left, a blue square on the right, the word
  LIGHTHOUSE at the bottom), asked for each of them: "7341 red circle blue square lighthouse", 2.8 s; asked how many
  colored shapes: "2", 1.6 s;
- `docs/media/pagoda-preview.webp`, asked for two sentences: "This image shows a pixelated, isometric 3D scene
  featuring a multi-tiered pagoda on a floating island with cherry blossom trees and a small pond. A control panel on
  the left allows users to adjust settings like time, weather, and camera angle for this voxel-style environment.",
  2.7 s.

By hand: build the image encoder (setup does it with `--vision gpu`), add `"--vision", "--vram-reserve-mib", "700"`
to `args`, and give the config a block:

```json
"vision": {
  "exe": "engine/strata-vision",
  "mmproj": "/path/to/mmproj-F16.gguf",
  "model": "/path/to/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf",
  "gpu": true,
  "max_tokens": 1024
}
```

## Scope and validation

Checked on 2026-10-11 on the machine above:

- Header read: 1224 tensors with the keys and geometry the engine expects. `tools/strata_inspect.py` reports the
  routed experts use only quantizations Strata runs (Q4_K gate/up, IQ4_NL down), and names the weights this file
  stores differently from the files setup installs.
- `tools/iq_pack.py --compat-bf16` completed: `index.txt` (1081 rows), `native_experts.txt` (v3, 512 experts per
  layer, 67,947,724,800 bytes), `dense.bin`, `conversions.json`, `compat-bf16.json` and `tokenizer/`.
- `tools/embd_bf16_pack.py --gguf` completed: 1,271,398,784 bytes, worst rounding error 0.00036.
- Packing tests: `.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py` - 24 tests, all pass.
- The engine started on the configuration above. Its log shows each piece landing where this document says it should:
  `--embd-gguf` read as "token embedding BF16 in mapped host memory (1212 MiB, 1.2 s)"; "302 canonical tensors skipped:
  served natively"; 300 native projection matrices, 2294.53 MiB, in 1.1 s; the PLE table read from the same GGUF as
  `--native` at 21487 rows/s; experts mmap'd in place with no `experts.bin`; 47.47 GiB of experts page-locked in RAM
  beside 15.81 GiB in the GPU cache, 63.28 GiB together, the size the header predicts; and the output head read
  natively at 521,472,000 bytes - 2,483,200 blocks of the engine's interleaved Q6_K layout, which is
  `output.weight`'s own block count. The log's wording there, "experimental native Q5_K head", is a fixed string at
  `src/program/generate.cpp:4497`, not a report of the tensor's type.
- The server answered the checks of the [OrcaRouter setup](ORCA.md) on that configuration: arithmetic (391), a Python
  function, a translation, a longer explanation, two requests sharing a 686-token prefix (the second, asked for the
  list's second-largest number, answered 98; it is 99, and with `--kv q4_0` it answered 100), and a cancelled stream
  followed by a correct answer (19), with one engine process throughout.

Not checked:

- Output quality against the base model is not measured, and neither is the draft head's acceptance rate on this
  fine-tune - Signal verifies every draft, but how often it accepts one is unknown here. The speeds above are one
  machine's; on other hardware, calibrate and watch the log for the engine's own complaints.
- Only AP-Q4_K_XL from that repository was downloaded and run. AP-IQ4_XS's header was read, not its weights; Q8_0
  was not examined.
- Setup's download of the 101 GB file (the run above used `--gguf-dir`), setup on Windows, AMD cards, several GPUs,
  and the image encoder on the CPU.
- `tools/embd_bf16_pack.py --gguf` is new with this setup and has no test of its own. For `--model`, the header was
  compared byte for byte with the one the previous code built, and the table is copied by the same loop, moved into a
  function; it was not run end to end, because no safetensors checkpoint was on the machine.
