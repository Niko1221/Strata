# agentionai Signal AP-Q4_K_XL

Setup installs Signal (`--family signal`, sizes AP-Q4_K_XL and AP-IQ4_XS, images included): see
[What setup does](#what-setup-does). The rest of this page is the manual workflow setup automates, and what was checked.

Packed, loaded and run on Linux with an RTX 4090 (24 GB), Ryzen 9 7950X3D and 93 GB RAM. The engine starts on this
pack and answers requests. The speeds quoted here were measured on 2026-10-21 with Strata 0.1.46.3 and
`tools/strata_bench.py`, with the PC's firmware at its defaults (DDR5-3600), on the configuration shown with each
table; no output-quality comparison against the base model was made. What was checked is listed at the end.

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
`--pcie-frac`, `--spec-min-p` and `--pool-workers 8` are what `setup.py`'s calibration chose for Signal on that
machine; they were not calibrated again for the measurements on this page. On other hardware, calibrate again.
The calibration also chose `--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`. Measured with `tools/ab_engine.py`
on 2026-10-21 with the configuration above (four rounds, the arms alternating, median over the three short chats and
the long prompt of each round), the engine's default was faster: 60.8 against 49.8 tokens/s for 100-token answers
(faster in 15 of 16 pairs) and 65.7 against 59.7 tokens/s for answers of up to 512 tokens (16 of 16). With the three
settings the first question's answer came out shorter (419 tokens against 512), so the texts were not the same. Those
three settings are left out.

`--kv int8` follows Signal's authors, who advise against a 4-bit value cache ("a 4-bit value cache makes long traces
degenerate"). With `--kv q4_0` the engine's log for this configuration showed 195 MiB more VRAM free for experts
(16.88 against 16.69 GiB, 6206 cache slots instead of 6132) and 1.4 GiB less of the KV cache in pinned RAM (1.69
against 3.09 GiB). `tools/strata_bench.py`, two runs of each, the two alternating, measured (median of all requests)
60.5 tokens/s with `--kv int8` against 64.5 with `--kv q4_0` for short answers, 74.9 against 72.1 in the agent
session and 2741 against 2693 tokens/s for the 16K prompt. So 8-bit was not slower in the agent session or on the long
prompt; for short answers 4-bit was 4.0 tokens/s faster, against a spread within one run of 52.6-74.7 tokens/s with
`--kv int8`.

`--resident-budget-gib` caps only the RAM copy of the experts the GPU cache does not hold. With the 6132 cache slots
this card gets with the configuration above, Strata 0.1.46.3 page-locks 51.11 GiB of experts (50.80 GiB with
`--kv q4_0` and 6206 slots): every expert outside the GPU cache, so a budget of 67 copies all of them. The 26.8 GiB PLE
table is what gets streamed. Below the size of that copy, the least-used experts are read from the SSD instead.

With exactly the configuration above (the paths of this machine; no images), on 2026-10-21 with Strata 0.1.46.3, the
server was ready 66-84 s after start (16 starts, `tools/ab_engine.py`). `tools/strata_bench.py` (3 runs per test,
thinking off, temperature 0), run four times on this configuration with one start each, measured:

| test | median of the runs | each run's median | slowest to fastest request |
|---|---|---|---|
| short chat, decode (up to 100 tokens) | 64.1 tokens/s | 62.7, 62.3, 66.1, 65.4 | 50.6-77.4 (36 answers) |
| short chat, first token | 0.43 s | 0.46, 0.43, 0.42, 0.38 | 0.30-0.55 |
| 4K prompt | 1710 tokens/s | 1646, 1723, 1696, 1724 | 1314-1731 (12 prompts) |
| 4K prompt, first token | 2.37 s | 2.46, 2.36, 2.38, 2.35 | 2.34-3.08 |
| 16K prompt | 2769 tokens/s | 2743, 2780, 2772, 2765 | 2491-2786 (12 prompts) |
| 16K prompt, first token | 5.82 s | 5.88, 5.80, 5.81, 5.83 | 5.79-6.47 |
| agent session, decode (turns 2-6) | 76.5 tokens/s | 77.4, 73.3, 75.5, 83.2 | 58.8-103.1 (60 turns) |
| agent session, first token (turns 2-6) | 2.39 s | 2.40, 2.39, 2.38, 2.35 | 2.22-3.73 |
| agent session, prompt reused from the cache | 69.3% | 69.3, 69.4, 69.3, 69.3 | 69.3-69.4% |

No run stands out: each run's median for the 4K and 16K prompts is within 5% of the others'. Short answers and agent
turns vary more from one request to the next (50.6-77.4 and 58.8-103.1 tokens/s).

The 4090 held 178 MiB before each run. The PC ran its firmware's defaults: DDR5-3600 on four modules from two
different kits, with the memory, the memory controller and the Infinity Fabric at 1800 MHz (read from the processor's
power table), the processor at stock, and PCIe 4.0 x16 to the card. A desktop session was open during all of the runs:
the Steam client (no game), JetBrains Toolbox and the coding agent that drove the runs.

`--mtp mtp/rt` is the original model's draft head, prepared by `tools/mtp_fetch.py`, `tools/mtp_pack.py` and
`tools/mtp_rt.py` as in the [OrcaRouter setup](ORCA.md); without `--mtp` the server drafts by lookup only. Signal
verifies every draft, and a fine-tune's acceptance is not the base model's. It has not been measured for Signal.

## What setup does

`setup.py --family signal --model AP-Q4_K_XL` (or `AP-IQ4_XS`) does the steps above by itself: it downloads the file
from a pinned revision and checks its SHA-256, packs it with `--compat-bf16`, writes the BF16 embedding beside the
pack (`token-embd-bf16.gguf`) and adds `--embd-gguf` to the config. With images on, it downloads Signal's own encoder
(`mmproj-F16.gguf`, 904 MB), saved as `mmproj-Signal-3.8-Flash-Next-F16.gguf`. Neither size asks on an AMD card: the
HIP engine has kernels for every format the two use - Q4_K experts run on an R9700 as UD-Q4_K_XL's, IQ3_S gate/up and
IQ4_NL downs as UD-IQ4_XS's ([QUANTS.md](QUANTS.md)) - though neither Signal file itself has run on AMD.

Run on the machine above on 2026-10-11 with the file already downloaded:
`./setup.sh --family signal --model AP-Q4_K_XL --gguf-dir <its folder> --vision gpu --build --yes --no-start`. The
pack, the tokenizer and the embedding file it wrote are byte for byte the ones built by hand above.

Run again on 2026-10-21 with Strata 0.1.46.3 (`setup.py --family signal --model AP-Q4_K_XL --data-dir <folder>
--gguf-dir <its folder> --vision gpu --no-start --yes`, into an empty data folder, the engine already built). Its config
is not the hand-built one above: 131072
tokens of context (setup's default on this card; the hand-built one has 262144), every expert kept in RAM instead of a
67 GiB budget (setup does that from about 80 GiB of RAM), the PLE table read from the GGUF into RAM (`--ple-gguf`,
`--ple-io ram`), images on (`--vision`, `--vram-reserve-mib 700`), `--spec-min-p 0.5` instead of 0.70, and neither
`--pcie-frac` nor `--pool-workers`; the KV cache is 8-bit in both. `tools/strata_bench.py` (3 runs per test, thinking
off, temperature 0), run three times on 2026-10-21 with one start each, measured:

| test | median of the runs | each run's median | slowest to fastest request |
|---|---|---|---|
| short chat, decode (up to 100 tokens) | 71.3 tokens/s | 71.3, 71.5, 64.5 | 46.4-78.9 (27 answers) |
| short chat, first token | 0.49 s | 0.49, 0.48, 0.51 | 0.36-0.60 |
| 4K prompt | 1636 tokens/s | 1669, 1636, 1300 | 718-1681 (9 prompts) |
| 4K prompt, first token | 2.48 s | 2.43, 2.48, 3.11 | 2.41-5.62 |
| 16K prompt | 3080 tokens/s | 3108, 3080, 2844 | 2197-3133 (9 prompts) |
| 16K prompt, first token | 5.24 s | 5.19, 5.24, 5.67 | 5.15-7.34 |
| agent session, decode (turns 2-6) | 86.1 tokens/s | 86.1, 89.4, 84.8 | 48.4-117.5 (45 turns) |
| agent session, first token (turns 2-6) | 2.58 s | 2.58, 2.50, 2.62 | 2.36-11.14 |
| agent session, prompt reused from the cache | 69.4% | 69.5, 69.3, 69.4 | 69.3-69.5% |

Two runs have slow requests: run 2's first 4K prompt (984 tokens/s), and in run 3 a 4K and a 16K prompt (718 and 2197
tokens/s) and one agent turn that took 11.1 s to its first token (the other 44 turns: 2.4-6.3 s). In both runs the
kernel logged a page-allocation failure in the engine while it loaded, followed by a message from the NVIDIA driver
("Failed to create a DMA mapping"); the engine carried on and both runs finished without a failed request. With every
expert in RAM, this configuration left 12.8-15.7 GiB of the PC's memory available while it ran, against 22.9-25.4 GiB
during the runs of the other configurations. The cause of the slow requests is not established, and they stay in the
table.

Against the hand-built configuration's table above, setup's is faster for short answers (71.3 against 64.1 tokens/s),
the 16K prompt (3080 against 2769) and the agent session (86.1 against 76.5), and a little slower for the 4K prompt
(1636 against 1710). The two differ in six settings at once (context, RAM budget, PLE table, speculation threshold,
the PCIe fraction and the pool workers), plus images, so this does not say which of them makes the difference.

AP-IQ4_XS (90,447,294,816 bytes) was run through setup on 2026-10-19 and again on 2026-10-21. Its header, read from
the first 32 MB of the file, differs from AP-Q4_K_XL's only in the routed experts' gate and up projections: IQ3_S
instead of Q4_K, 57,252,249,600 bytes of experts. IQ3_S gate/up with IQ4_NL downs is the pair Unsloth's UD-IQ4_XS
uses, which setup already runs on NVIDIA and AMD cards.

`setup.py --family signal --model AP-IQ4_XS --vision gpu` ran on 2026-10-19 with the engine of that day (0.1.45.3):
the 90 GB download, its SHA-256 check, the pack, the BF16 embedding and the MTP head. Its config
has the same settings as AP-Q4_K_XL's setup config above (131072 tokens of context, every expert in RAM, images on).
`tools/strata_bench.py` (3 runs per test, thinking off, temperature 0), run three times on 2026-10-21 with one start
each, measured:

| test | median of the runs | each run's median | slowest to fastest request |
|---|---|---|---|
| short chat, decode (up to 100 tokens) | 91.0 tokens/s | 90.4, 91.0, 92.1 | 70.7-104.6 (27 answers) |
| short chat, first token | 0.33 s | 0.33, 0.33, 0.32 | 0.25-0.42 |
| 4K prompt | 2123 tokens/s | 2123, 2112, 2134 | 1586-2135 (9 prompts) |
| 4K prompt, first token | 1.91 s | 1.91, 1.92, 1.90 | 1.90-2.55 |
| 16K prompt | 3802 tokens/s | 3802, 3796, 3827 | 3358-3840 (9 prompts) |
| 16K prompt, first token | 4.25 s | 4.25, 4.25, 4.22 | 4.20-4.81 |
| agent session, decode (turns 2-6) | 108.0 tokens/s | 107.1, 108.3, 108.0 | 74.6-145.7 (45 turns) |
| agent session, first token (turns 2-6) | 2.00 s | 1.99, 2.00, 2.78 | 1.86-3.46 |
| agent session, prompt reused from the cache | 69.3% | 69.5, 69.3, 69.3 | 69.3-69.5% |

On the same setup config AP-IQ4_XS was faster than AP-Q4_K_XL: short answers 91.0 against 71.3 tokens/s, the 4K prompt
2123 against 1636, the 16K prompt 3802 against 3080 and the agent session 108.0 against 86.1. Its three runs agree
closely (short answers 90.4, 91.0 and 92.1 tokens/s), and the kernel log has no allocation failure or error during
them. Output quality was not compared with AP-Q4_K_XL's, and no picture was sent to it.

## Images

Signal ships its own image encoder, `mmproj-F16.gguf`: a `qwen3vl_merger` projector to 2560 wide, the model's width,
in F16. With the configuration setup wrote, on 2026-10-11, Signal answered (thinking off, temperature 0):

- a drawn test picture (the number 7341 at the top, a red circle on the left, a blue square on the right, the word
  LIGHTHOUSE at the bottom), asked for each of them: "7341 red circle blue square lighthouse"; asked how many colored
  shapes: "2";
- `docs/media/pagoda-preview.webp`, asked for two sentences: "This image shows a pixelated, isometric 3D scene
  featuring a multi-tiered pagoda on a floating island with cherry blossom trees and a small pond. A control panel on
  the left allows users to adjust settings like time, weather, and camera angle for this voxel-style environment."

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
- Packing tests: `.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py` - 35 tests, all pass. Two of them
  are for `tools/embd_bf16_pack.py --gguf`: a small Q6_K table in a model split over two files comes out as the
  BF16 rounding of gguf-py's dequantization, byte for byte, in more than one pass of rows; a missing tensor and a
  1-D tensor are refused.
- The engine started on the configuration above. Its log shows each piece landing where this document says it should:
  `--embd-gguf` read as "token embedding BF16 in mapped host memory (1212 MiB)"; "302 canonical tensors skipped:
  served natively"; 300 native projection matrices, 2294.53 MiB; the PLE table read from the same GGUF as `--native`;
  experts mmap'd in place with no `experts.bin`; 47.47 GiB of experts page-locked in RAM beside 15.81 GiB in the GPU
  cache, 63.28 GiB together, the size the header predicts (for this configuration Strata 0.1.46.3 logs 51.11 GiB
  page-locked in RAM beside 6132 slots in the GPU cache); and the output head read
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
- AP-Q4_K_XL was checked as listed above. AP-IQ4_XS was run through setup and `tools/strata_bench.py` (above) and
  answered an arithmetic question (391), but none of the other checks listed here (the OrcaRouter list, pictures) was
  run on it. Q8_0 was not examined.
- Setup's download of the 101 GB AP-Q4_K_XL file (both runs above used `--gguf-dir`; AP-IQ4_XS's 90 GB download went
  through setup), setup on Windows, AMD cards, several GPUs,
  and the image encoder on the CPU.
- `tools/embd_bf16_pack.py --model` was not run end to end, because no safetensors checkpoint was on the machine. Its
  header was compared byte for byte with the one the previous code built, and the table is copied by the same loop,
  moved into a function. (`--gguf` was already there for Gyro-S's Q6_K table. This setup changed how it reads the
  file - through `iq_pack.py`'s model reader, so a split model works - and added `--tensor` and the printed rounding
  error. The tests above use a small table; the full-size run is the AP-Q4_K_XL one above.)
