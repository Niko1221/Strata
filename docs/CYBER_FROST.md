# CYBER-FROST-3.8 (Blackfrost)

A fine-tune of Qwen3.8-Flash-Next that **declares its own nextn (MTP) prediction block**. This
page covers packing it, the engine changes it needs, and a working configuration. Back to
[Which model?](MODELS.md).

- Model: [`Blackfrost-AI/CYBER-FROST-3.8-BF16`](https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-BF16)
- Requant used here: [`peasantsmith/CYBER-FROST-3.8-PS-GUFF`](https://huggingface.co/peasantsmith/CYBER-FROST-3.8-PS-GUFF) (Q5_K_M)
- Its own license applies (see its page).

## Why it needs engine changes

Its GGUF declares:

```
qwen4exp.block_count            = 49      # 48 trunk layers + 1 prediction block
qwen4exp.nextn_predict_layers   = 1
```

**Nothing in the engine reads `nextn_predict_layers`**, so the prediction block is counted as a
trunk layer and each affected path fails differently:

| path | failure |
|---|---|
| architecture check (`gguf_reader.hpp`) | `block_count` 49 is compared against the family's 48, so the model is rejected |
| dense loading (`native_dense.cpp`) | the three 2-D nextn projections are not dense-eligible, so they hit the packed fallback as a shape-only row and the loader aborts |
| packing (`iq_pack.py`) | `n_layers` counts the prediction block and emits an expert row the engine rejects (`a malformed line`) |

A patch for all three is in `0001-nextn-mtp-blocks.patch` (3 files, 41 insertions, 6 deletions).

## Packing it

The pack step is the standard native-pack flow; the only change is that the expert table must
cover the **trunk** only (48 layers here, not 49).

```bash
python tools/iq_pack.py \
  --gguf  <path>/CYBER-FROST-3.8-PS-GUFF.gguf \
  --out   <path>/packs/planb2-guff-eb
```

`iq_pack.py` writes the pack, including `native_experts.txt`. **Check that table has 48 rows and
no reference to `blk.48`** — if the prediction block is in it, the pack will not load.

```
$ wc -l native_experts.txt
48 native_experts.txt
$ grep -c blk.48 native_experts.txt
0
```

**Free space:** the expert table is ~78 GB and is written in one pass. Check `df` first.

## Working configuration

Nothing here is tuned to one machine: `--expert-cache auto` and `--layer-split auto` size the
cache and place the layer boundary from the GPUs actually present.

```json
{
  "exe": "<path>/build/strata",
  "args": [
    "--pack", "<path>/packs/planb2-guff-eb",
    "--native", "<path>/CYBER-FROST-3.8-PS-GUFF.gguf",
    "--ple-gguf", "<path>/CYBER-FROST-3.8-PS-GUFF.gguf",
    "--expert-profile", "<path>/data/expert-profile.bin",
    "--expert-cache", "auto",
    "--expert-cache-per-layer",
    "--kv", "int8",
    "--pcie-frac", "0.75",
    "--max-context", "131072",
    "--spec", "4",
    "--spec-min-p", "0.5",
    "--mtp", "<path>/mtp/rt",
    "--prefill", "auto",
    "--mmap-experts"
  ],
  "gpu": [0, 1],
  "layer_split": "auto",
  "host": "127.0.0.1",
  "port": 8099
}
```

**`--mmap-experts` is required when the machine's RAM is smaller than the expert arena** (~78 GB
for this size). Without it the engine reserves the whole arena as one anonymous mapping and
exits 1 with no `READY`. If your RAM is larger, drop the flag and the engine uses the resident
arena, which is faster.

**`--vram-reserve-mib` is deliberately absent.** The engine sizes its own reserve. A
`cudaMalloc(...) for the weight arena failed` message with very little free VRAM is almost always
another process holding the card — check what is resident on the GPU before changing flags.

## The MTP draft layer

This fine-tune ships no MTP weights of its own, so the draft layer comes from the base model —
**the same source `tools/mtp_fetch.py` already reads** for the family's other versions.

```
python tools/mtp_fetch.py inventory --out <path>/mtp/raw
python tools/mtp_fetch.py fetch     --out <path>/mtp/raw
python tools/mtp_fetch.py verify    --out <path>/mtp/raw
python tools/mtp_pack.py  --src <path>/mtp/raw --out <path>/mtp/rt
```

Speculative decoding is output-equivalent by construction, so the generated text does not change.
**A draft head trained against this fine-tune's distribution would accept more tokens than one
from the base does** — it works, it is simply not optimal.

## Measured

Draft acceptance: the **MTP drafter** accepted 9,258 of 10,809 drafts (**85.7%**) over 274 drafting
windows; 42% of windows accepted every draft. This is the drafter the engine binds from the model's
own nextn block. **The `--spec` verify window is a separate mechanism** and reports its own,
different rate — do not compare the two.

**Treat acceptance as the portable number; re-measure throughput on your own hardware with
`--calibrate`.**

**Throughput here is bounded by the expert-miss path, not by flags.** On a machine whose RAM is
smaller than the expert table, expert rows that miss the cache are read from the SSD while the
model answers, and that I/O is the ceiling. Two consequences worth knowing before tuning:

- **Longer prompts prefill better than short ones** — fixed per-request overhead dominates the
  short case, so a short prompt can prefill slower than a long one.
- **`--pcie-frac` is not a tuning lever.** A sweep across its whole range on an identical cold
  prompt moved prefill within run-to-run noise. The engine probes PCIe bandwidth at startup and
  derives it. Leave the default.

**The real lever is RAM:** if the machine's RAM holds the expert table, the SSD is out of the
answer path entirely.
