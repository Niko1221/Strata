# Community benchmark on RTX 5060 Ti 16 GB + RTX 3060 12 GB (layer split)

Measured on 2026-11-11/12 by AndreyOsipkov. Flash-Next IQ3_S at a 262,144-token context on two mismatched cards in a
layer split: three opt-in switches that raise prompt speed, MTP against DFlash2, and the layer boundary.
Main limitation: two prompt runs per size, not three.

## Hardware and software

- RTX 5060 Ti 16 GB (CPU slot, PCIe 5.0 x8) and RTX 3060 12 GB (chipset slot, PCIe 4.0 x4); no P2P. Default power
  limits. `gpu: [1, 0]`: the 3060 is the first stage, the 5060 Ti the last (head and drafter).
- The 3060 runs with [gpu-unlock](https://github.com/Zagitta/gpu-unlock) 1.0.0 (full-rate BF16/FP16 with FP32
  accumulation). All results are with it.
- Core i5-14400F, 62.6 GB DDR5, Samsung 990 PRO NVMe. Linux Mint 22.3, kernel 6.17, driver 595.84, CUDA 13.1.
- Source builds, Release, `CMAKE_CUDA_ARCHITECTURES=86;120`: **0.1.59.1** (tag `v0.2.13.1`) and, for the rows marked so,
  0.1.45.3 (`3fa1b524`). Both carry three opt-in conversation-cache commits that are not in main; they do not touch the
  prompt or decode path.
- A GitLab instance and its runner were on the same host, not stopped (night, no jobs scheduled).

## Model and configuration

- [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  `IQ3_S`, experts read from the GGUF (`--resident-experts`).
- MTP: `unsloth/Qwen3.8-Flash-Next-GGUF` `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, `--spec 4`. DFlash2:
  `qwen3.8-flash-next-dflash2.q4_0.pack`, `--spec 6 --draft-conf 0.35`, without `--pipeline-windows` (not supported
  with DFlash2).
- `--kv int8 --kv-resident 32768 --prefill auto --expert-cache auto --pcie-frac 0.17 --trim-stage-weights
  --pipeline-windows 2`, `layer_split: "17"`. Full configs in [`config/`](config/) (`<root>` is the install folder).

## Method

Every variant is a fresh engine start, then:

- **Prompt:** [`scripts/strata-prefill.py`](scripts/strata-prefill.py): C++ source text (15 llama.cpp files, sha256
  `282e2504…8c1e17`) cut to 7.9K, 31.3K and 94.7K tokens, a unique tag in front (`cache_n` 0 in every row), 8 output
  tokens, two runs per size. The engine's `prompt_per_second`.
- **Decode, short:** three fixed questions, thinking off, temperature 0, 400-token cap, each twice. **Decode,
  reasoning:** three reasoning tasks, thinking on, 16,000-token cap. The engine's `predicted_per_second`.
- Order of variants: [`scripts/strata-night-0213.sh`](scripts/strata-night-0213.sh) and
  [`scripts/strata-night-1111.sh`](scripts/strata-night-1111.sh). Per-run rows: `data/<run>/<variant>/*.jsonl`.

## Results

Medians, range in parentheses, tok/s.

### Three switches: `STRATA_SPLIT_MTP_BATCH=1 STRATA_PROMPT_ATTN_IMMA=1 STRATA_PF_FUSED=1`

| 0.1.59.1, MTP | 8K prompt | 32K | 100K | Decode, short |
| --- | --- | --- | --- | --- |
| without | 1060 (1021-1099) | 1684 (1670-1697) | 1933 (1924-1943) | 76.3 |
| **with** | **1127** (1082-1172) | **1792** (1778-1807) | **1997** (1986-2008) | 75.2 |
| | +6% | +6% | +3% | unchanged |

On 0.1.45.3 the same three gave **+8% / +9% / +6%** (1060 → 1149, 1648 → 1793, 1894 → 2001; the baseline measured
again at the end of that run moved 0.2%). One at a time on 0.1.45.3 at 32K: `SPLIT_MTP_BATCH` +5%,
`PROMPT_ATTN_IMMA` +2%, `PF_FUSED` +2%; `STRATA_QSA_SPLIT_ATTN` and `STRATA_PREFILL_HELP` 0%.

### MTP against DFlash2 (0.1.59.1, three switches on)

| Drafter | 8K prompt | 32K | 100K | Decode, short | Decode, reasoning |
| --- | --- | --- | --- | --- | --- |
| **MTP** | 1127 (1082-1172) | 1792 (1778-1807) | 1997 (1986-2008) | 75.2 | **87.9** (75.8-88.1) |
| DFlash2 | 1107 (1031-1183) | 1816 (1801-1830) | 2018 (2009-2027) | 73.8 | 77.6 (67.3-79.3) |

DFlash2 decodes **12% slower** on reasoning; prompts within 1-2%. The engine log confirms DFlash2 ran
(`data/night-0213b/v0213-df2-c3/drafter.txt`).

### Layer boundary (0.1.45.3)

17 layers on the 3060 is the best: 18 is equal (1666 / 1894 at 32K / 100K against 1644 / 1890), 22 is 11% / 15%
slower and 24 is 17% / 23% slower. Decode unchanged within noise.

## Limitations

- Two prompt runs per size; spreads about ±1% at 32K and 100K, ±5% at 8K. Short-answer decode varies about ±10%
  between questions.
- One machine, one quantization, Linux only; memory, power and clocks not recorded. With and without the switches on
  0.1.59.1 were separate engine starts a few hours apart the same night.
- Answer correctness with the switches was not checked: answers at temperature 0 differ between two runs of the same
  setting here, so a comparison would not show a numerical change.
