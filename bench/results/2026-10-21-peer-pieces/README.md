# STRATA_PF_PEER_PIECES on 2x Tesla V100-PCIE-32GB (host route), 2026-10-21

The prompt path with `--peer-device 1` and no P2P (`STRATA_PEER_NO_P2P=1`), without and with
`STRATA_PF_PEER_PIECES=1`. With the variable, each MoE layer's two transfers through host memory go in 8 pieces:

- the FP16 activations start towards the peer before the router, and the peer converts each piece as it lands;
- the peer's FP16 per-token sums come back the same way, and the primary copies each piece into `bo` as it lands.

## Machine and build

- 2x Tesla V100-PCIE-32GB (sm_70, PCIe 3.0 x16 each, `nvidia-smi topo`: SYS between them), driver 550.120,
  CUDA 12.8 (nvcc 12.8.93). A third card (Quadro RTX 4000) was idle and not visible to the engine.
- Intel Core i9-7960X (no AVX-512), 125 GiB RAM, Ubuntu, Linux 6.8.0-51.
- Release build, `-DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON
  -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON`,
  this branch on `011c9e60`.
- Two arms, each a separate server start of the same binary: `pieces_off` (the variable not set) and
  `pieces_on` (`STRATA_PF_PEER_PIECES=1`). For the output check, `main` is an unchanged build of `011c9e60`.

## Model and settings

Qwen3.8-Flash-Next GSQ-RCO IQ3_S (native pack), MTP draft head. Engine arguments, the same for every arm:

    strata --serve --pack <data>/packs/gsq-iq3_s --native <models>/.../Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
      --expert-profile <strata>/data/expert-profile.bin --expert-cache auto --prefill auto:16384 --spec 2 --spec-min-p 0.5
      --mtp <data>/mtp/rt --max-context 262144 --kv int8 --ple-io ram --peer-device 1

Environment for every arm: `STRATA_PEER_NO_P2P=1`, `STRATA_GR_DOWN_MAX4=1`, `STRATA_PEER_HOT_AT=11283`,
`STRATA_PEER_HOT=0.45`, `STRATA_PREFILL_RING=96`, `STRATA_NO_LARGEPAGES=1`. The engine chose 16,384-token prompt
chunks and a 96-slot ring.

## Prompt throughput

`scripts/abpr.py`: per server start, one warm-up request, then two new prompts made from the repository's docs, shuffled
with a seed per round (32,000 and 128,000 characters, about 10K and 38K tokens). Temperature 0, thinking off, 8 output
tokens. Prompt throughput = prompt tokens / time to the first streamed token. Four rounds; the arms' order changes
from round to round. Every run is in `runs.jsonl`.

| prompt | round | tokens | pieces_off | pieces_on | change |
|---|---:|---:|---:|---:|---:|
| ~10K | 1 | 9,891 | 1,604 | 1,711 | +6.7 % |
| | 2 | 7,170 | 1,545 | 1,637 | +6.0 % |
| | 3 | 9,542 | 1,594 | 1,711 | +7.3 % |
| | 4 | 11,680 | 1,631 | 1,731 | +6.1 % |
| | **median** | | **1,599** | **1,711** | **+7.0 %** |
| ~38K | 1 | 37,815 | 1,674 | 1,799 | +7.5 % |
| | 2 | 36,429 | 1,661 | 1,790 | +7.8 % |
| | 3 | 33,685 | 1,663 | 1,768 | +6.3 % |
| | 4 | 39,885 | 1,678 | 1,796 | +7.0 % |
| | **median** | | **1,669** | **1,793** | **+7.5 %** |

All values are tok/s. Earlier the same day, with the same prompts, the unchanged `main` build measured 1,596 / 1,672
tok/s (medians of 4 rounds), the same as `pieces_off`.

`STRATA_PREFILL_TIMING=1`, one extra server start each (`engine-logs/fin_off-1-t.txt`, `engine-logs/fin_on-1-t.txt`):

| prompt | arm | GPU timeline | combine (includes the wait for the peer) |
|---|---|---:|---:|
| 9,884 tokens | pieces_off | 5,666 ms | 806 ms |
| | pieces_on | 5,282 ms | 283 ms |
| 37,808 tokens | pieces_off | 21,949 ms | 3,131 ms |
| | pieces_on | 20,548 ms | 1,196 ms |

## Output

`scripts/lp.py` with `STRATA_LOGPOS` (teacher-forced: five prompts of ~34K, 24K, 16K, 8K and 3K tokens, the log
probabilities of the top 20 tokens at 2,493 positions), compared by `scripts/logpos_compare.py`. In the result files
`main011` = `main`, `fin_off` = `pieces_off`, `fin_on` = `pieces_on`.

- `logpos-vs-main-noP2P.txt`: with `STRATA_PEER_NO_P2P=1`, `pieces_off` and `pieces_on` are **bit for bit
  identical** to `main` at all 2,493 positions.
- `logpos-vs-main-P2P.txt`: with P2P (the variable has no effect there), `pieces_on` is bit for bit identical to `main`.

## Not measured

- Decode: the change is in the prompt path only. No decode runs were made.
- Other GPUs, PCIe 4.0 or 5.0 links, HIP and SYCL.
- Prompts below 7K tokens, and chunks other than 16,384 tokens.
- The two halves separately in this final form. On an earlier revision with a switch for each, the early activations
  gave +6.0 % and the sums in pieces +1.6 % alone (+1.0 % on top of the first) at ~38K tokens.
