# WEX4 expert containers (opt-in, off by default)

This page is for people who build their own native packs with `iq_pack.py` and want the routed experts in one file that
can be checked expert by expert. Nothing changes unless you pass `--experts-wex4`.

WEX4 is a lossless container format from [Wex-WRN2](https://github.com/KairaCode/Wex-WRN2) (Apache-2.0). Each expert is
one record that starts on a 4096-byte boundary and carries its own SHA-256. The index lists every record, and the
SHA-256 of the 4096-byte header page (the **root**) names the whole file. A record holds exactly the bytes `experts.bin`
holds for that expert, which are the GGUF's bytes: nothing is converted or re-quantized, and reading a record gives back
the stored bytes.

## Quick start

From a source checkout:

```sh
# 1. the pack, plus experts.wex4 beside it (the experts are copied from the GGUF, byte for byte)
python terminal/tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --out <pack dir> --experts-wex4

# 2. optional, once: check every expert (reads the whole file; prints the root)
python terminal/tools/wex4_check.py <pack dir>/experts.wex4

# 3. run with it: the file tier reads the experts from the container
strata generate --pack <pack dir> --native <model>-00001-of-00002.gguf --mmap-experts \
  --experts-wex4 <pack dir>/experts.wex4 --experts-wex4-root <the root printed above> ...
```

`--experts-wex4` needs the file tier: `--mmap-experts` or `--resident-experts`. Without one of them the engine stops
and says so. `--experts-wex4-root` is optional.

The flags were tried with `strata generate` and with `serve/server.py --engine strata --config <run config>`, the
flags written in the config's `args` (with `--resident-experts`, MTP drafting and a 512K context: the same greedy reply
as the same config on `experts.bin`). The app and `strata launch` pass extra engine flags through `user_args` in the
run config, but that path was not tried with these flags.

## What is checked, and when

| Check | Engine, every start | `wex4_check.py` |
| --- | --- | --- |
| The header page's SHA-256 | yes | yes |
| The index's SHA-256 | yes | yes |
| File size equals the size the header records (truncated or extended file) | yes | yes |
| Every expert of `native_experts.txt` has exactly one record, with the right names, types and sizes (rules below) | yes | - |
| The root equals the one you pinned | with `--experts-wex4-root` | with `--root` |
| Every expert's own SHA-256, and the record padding is zero | **no** | yes, and it names the layer and expert |

At start the engine reads only the header page and the index (a few MB). It does not read the 25 GB of experts to hash
them, so **a byte that went bad inside an expert after packing is not caught by the engine**. Run `wex4_check.py` for
that, for example after copying the file to another drive. The file tier then behaves as it does with `experts.bin`.

What exists today without WEX4: setup checks each model shard's size and SHA-256 after it downloads it
(`terminal/setup.py`). Nothing checks the files again later, and an `experts.bin` that `iq_pack.py --experts-bin`
writes has no hash of its expert bytes (`experts.bin.src.json` records the shards' names and sizes and a hash of
`native_experts.txt`).

Every refusal starts with `--experts-wex4:`, names the file, and names the layer, the expert and the rule where one
applies, for example:

```text
strata generate: --experts-wex4: <file>: layer 1 expert 2: stored with codec WNR3 (5); this engine reads RAW records only
```

## What it costs

Measured on one PC: RTX 5090 32 GB, Ryzen 9 7950X3D, 32 GB RAM (about 20 GB free during the runs), the files on one
1 TB SK hynix NVMe drive, Windows 11. Model: Qwen3.8-Flash-Next Coder IQ1_M (the two ISTA shards), packed with
`iq_pack.py`. Every engine run used the file tier (`--mmap-experts`).

| What | experts.bin | experts.wex4 |
| --- | --- | --- |
| File size | 25,146,163,200 B | 25,179,885,568 B (+33,722,368 B, +0.13 %: the 5.7 MB index and the record padding) |
| Engine start: header and index checks | - | 28 to 39 ms (14 starts) |
| `wex4_check.py`, every expert hashed | - | 20.0 s and 20.2 s for 25.18 GB (1.25 GB/s, Python `hashlib`, one thread) |

Generate, 999-token prompt, 128 new tokens, greedy, `--spec 2 --suffix-draft 0 --prefill 512 --expert-cache 1000`,
`STRATA_ROUTE_TAIL_SKIP=0`. Two series, each arm three times; the second series runs each arm twice in a row and counts
the second run, so each counted run starts with its own file in the page cache:

| Series | experts.bin decode tok/s | WEX4 decode tok/s | experts.bin prompt tok/s | WEX4 prompt tok/s |
| --- | --- | --- | --- | --- |
| 1 (B, C, base B) x 3 | 30.28, 28.53, 37.88 | 31.55, 31.34, 37.22 | 70.47, 76.75, 76.28 | 65.38, 74.37, 70.88 |
| 2 (B, B, C, C) x 3 | 21.66, 25.83, 31.86 | 25.99, 24.09, 31.66 | 73.47, 71.36, 61.27 | 75.44, 74.88, 63.07 |

The spread inside one arm (21.66 to 37.88 tok/s) is far larger than any difference between the arms, so these runs show
no speed difference either way. The engine built without this change, on experts.bin, ran 27.21, 35.68 and 15.99 tok/s
in series 1. In series 1 the experts.bin run always followed a run that had just read
`experts.bin`, which explains why WEX4's prompt rate is lower there and not in series 2. All 21 runs above and 6 more
gave the same 128 tokens.

The output does not change. With the yardstick's fixed settings (`quant_quality.py run`, 5 segments x 255 positions,
every expert computed on the CPU from the file tier), the per-position log-probabilities of the text token and of the
64 most likely tokens were byte-identical across six runs: experts.bin twice, WEX4 twice, the GGUF in place once, and
experts.bin with the engine built without this change once. The WEX4 records were also compared byte for byte with the
`experts.bin` that `iq_pack.py --experts-bin` wrote from the same GGUF: all 12,288 are equal.

## The record convention

The engine accepts a container when, for every layer L and expert E of the pack's `native_experts.txt`:

1. A tensor named `blk.L.ffn_gate_exps.weight#E` exists, and the file has no expert `E = n_expert` in any layer.
2. Its record holds exactly three tensors, in this order: `blk.L.ffn_gate_exps.weight#E`, `blk.L.ffn_up_exps.weight#E`,
   `blk.L.ffn_down_exps.weight#E`. So the record is the `experts.bin` blob, `[gate | up | down]`.
3. Their dtype codes are 128 plus the ggml type of gate/up, of gate/up, and of down.
4. Their sizes are the layout's: gate `up_off`, up `down_off - up_off`, down `blob - down_off`.
5. The record is stored RAW (codec 0): its size is the blob size, stored rounded up to 4096 with zero padding.
6. A layer's records are consecutive: expert E starts at the layer's first record plus E times the record size.

`iq_pack.py --experts-wex4` writes exactly this. Other tensors in the file (for example the dense weights of a
container made from a whole GGUF) are ignored.

## Limits

- **File tier only.** The resident expert arena (the default without `--mmap-experts`) does not read WEX4; the engine
  refuses the flag there.
- **RAW records only.** A record stored with WNR3 or WBP1 is refused by name. WNR3, the format's compressed codec, codes
  MXFP4 experts only, and Strata has no MXFP4 expert kernel, so no Strata pack could use it.
- **Not for NVFP4 or Gyro packs.** An NVFP4 expert's scale tail is not a GGUF tensor, and Gyro's trellis types
  (ggml types 144-146) do not fit WEX4's one-byte dtype code. Both `iq_pack.py` and the engine refuse them.
- **No disk saving.** The container is slightly larger than `experts.bin` (the padding and the index).
- **Not in the Metal or SYCL engines.** Metal refuses the flag; the SYCL engine has its own expert source and does not
  have it.
- **No check at run time.** An expert is not hashed when it is read; see [What is checked](#what-is-checked-and-when).

## Licence

The engine reads WEX4 with its own small reader (`engine/src/core/wex4_experts.cpp`, MIT like the rest of Strata).
`iq_pack.py` and `wex4_check.py` use Wex-WRN2's reference reader and writer, `wex4.py`, vendored unchanged in
`engine/third_party/wex` with its Apache-2.0 `LICENSE` and its `NOTICE` (see `UPSTREAM.md` there).
