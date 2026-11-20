# STRATA_PF_PEER_PIECES on 2x Tesla V100-PCIE-32GB (host route), 2026-11-21

The prompt path with `--peer-device 1` and no P2P (`STRATA_PEER_NO_P2P=1`), without and with
`STRATA_PF_PEER_PIECES=1`. With the variable, each MoE layer's two transfers through host memory go in 8 pieces:

- the FP16 activations start towards the peer before the router, and the peer converts each piece as it lands;
- the peer's FP16 per-token sums come back the same way, and the primary copies each piece into `bo` as it lands.

This replaces the 2026-11-10 run on `7726f80c` (+7.6 % at ~9K tokens, +7.7 % at ~38K). The branch is now on
`3142b66b` (after the 0.2.16 layout move). Since 0.2.16 there are no llama.cpp MMQ prompt kernels in the tree, and
this card runs every prompt chunk on the int8 expert kernels (`prompt kernels: every chunk runs 48 of 48 layers on the
int8 kernels` in the engine log). The early activations were tied to MMQ layers, so after the plain rebase only the
sums in pieces were left (`runs-without-fix.jsonl`: +2.8 % at ~9K, +1.2 % at ~38K). The branch now sends the
activations early for every MoE layer on the host route; the numbers below are for that version.

The baseline got faster since 2026-11-10 (~38K without the variable: 1,672 -> 1,712 tok/s); with the variable the
throughput is about where it was (1,800 -> 1,785 tok/s), so the relative gain is smaller.

## Machine and build

- 2x Tesla V100-PCIE-32GB (sm_70, PCIe 3.0 x16 each, `nvidia-smi topo`: SYS between them), driver 550.120,
  CUDA 12.8 (nvcc 12.8.93). A third card (Quadro RTX 4000) was idle and not visible to the engine.
- Intel Core i9-7960X (no AVX-512), 125 GiB RAM, Ubuntu, Linux 6.8.0-51.
- Release build, `cmake -S engine -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON`
  (the remaining options at their defaults; `STRATA_NATIVE_EXPERTS` and `STRATA_Q6K_EXPERTS` are on).
- `pieces_off` / `pieces_on`: one binary built from this branch, a separate server start per run. `main`: an unchanged
  build of `3142b66b` with the same options, for the output check.

## Model and settings

Qwen3.8-Flash-Next GSQ-RCO IQ3_S (native pack), MTP draft head. Engine arguments, the same for every arm:

    strata --serve --pack <data>/packs/gsq-iq3_s --native <models>/.../Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
      --expert-profile <strata-main>/data/expert-profile.bin --expert-cache auto --prefill auto:16384 --spec 2 --spec-min-p 0.5
      --mtp <data>/mtp/rt --max-context 262144 --kv int8 --ple-io ram --peer-device 1 --conversation-cache-mib 0

Environment for every arm: `STRATA_PEER_NO_P2P=1`, `STRATA_GR_DOWN_MAX4=1`, `STRATA_PEER_HOT_AT=11283`,
`STRATA_PEER_HOT=0.45`, `STRATA_PREFILL_RING=96`, `STRATA_NO_LARGEPAGES=1`. The conversation cache is off, because
parking the previous conversation counts into the next prompt's time. Engine logs of every run are in `engine-logs/`.

## Prompt throughput

`scripts/abpr.py` (driven by `scripts/run.sh`; `scripts/run-main.sh` is the `main` output run): per server start, one warm-up request, then two new prompts made from
the repository's docs, shuffled with a seed per round (32,000 and 128,000 characters). Temperature 0, thinking off,
8 output tokens. Prompt throughput = prompt tokens / time to the first streamed token. Four rounds; the arms' order
changes from round to round. Every run is in `runs.jsonl`.

| prompt | round | tokens | pieces_off | pieces_on | change |
|---|---:|---:|---:|---:|---:|
| ~9K | 1 | 7,959 | 1,616 | 1,685 | +4.3 % |
| | 2 | 8,589 | 1,634 | 1,673 | +2.4 % |
| | 3 | 9,079 | 1,637 | 1,700 | +3.8 % |
| | 4 | 9,912 | 1,653 | 1,717 | +3.9 % |
| | **median** | | **1,636** | **1,693** | **+3.5 %** |
| ~38K | 1 | 35,486 | 1,703 | 1,780 | +4.5 % |
| | 2 | 36,832 | 1,712 | 1,781 | +4.0 % |
| | 3 | 38,727 | 1,712 | 1,788 | +4.4 % |
| | 4 | 38,974 | 1,714 | 1,788 | +4.3 % |
| | **median** | | **1,712** | **1,785** | **+4.2 %** |

All values are tok/s. Every one of the 8 pairs is faster with the variable.

## Output

- **Log probabilities** (`scripts/lp.py` with `STRATA_LOGPOS` and `STRATA_LOGPOS_TOPK=20`): five teacher-forced
  requests of about 34K, 24K, 16K, 8K and 3K tokens, then the five greedy requests below on the same server; 60
  positions in all. Files in `logpos/`, compared line by line (`scripts/logpos_identical.py`, `logpos-host-route.txt`):
  `pieces_off` and `pieces_on` are **bit for bit identical** to `main` at all 60 positions, top 20 included.
- **Greedy answers** (`scripts/gen.py`, `greedy.txt`): five prompts of 4K to 39K tokens, 200 tokens each,
  temperature 0. `main`, `pieces_off` and `pieces_on` give the same text for all five (SHA-256 of the answer equal).

The prompts of `lp.py` and `gen.py` are made from the docs and the `engine/src` sources of the checkout; the `main`
run read a copy of the branch's files, so every arm got the same prompts.

## Not measured

- The P2P route (the variable has no effect there; on 2026-11-10 it was bit for bit identical to `main`).
- Decode: the change is in the prompt path only. No decode runs were made.
- Other GPUs and PCIe 4.0 or 5.0 links.
- HIP, SYCL and Metal: `engine/src/prefill/prefill.cpp` and `engine/src/prefill/kernels.cu` are built by those backends
  too; only the CUDA engine was built here.
- Prompts below 7K tokens, and chunks other than 16,384 tokens.
- Together with `STRATA_ADAPT_OVERLAP=1` or `STRATA_PF_PEER_CPU=1`.
- The two halves separately in this form.
