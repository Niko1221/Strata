# STRATA_PF_PEER_PIECES on 2x Tesla V100-PCIE-32GB (host route), 2026-11-10

The prompt path with `--peer-device 1` and no P2P (`STRATA_PEER_NO_P2P=1`), without and with
`STRATA_PF_PEER_PIECES=1`. With the variable, each MoE layer's two transfers through host memory go in 8 pieces:

- the FP16 activations start towards the peer before the router, and the peer converts each piece as it lands;
- the peer's FP16 per-token sums come back the same way, and the primary copies each piece into `bo` as it lands.

This replaces the 2026-10-21 run, which was made on `011c9e60` (+7.0 % at ~10K tokens, +7.5 % at ~38K). The branch is
now on `7726f80c` (0.2.13.1 plus later main commits), which has the new peer options `STRATA_ADAPT_OVERLAP` and
`STRATA_PF_PEER_CPU` (both off here, as by default).

## Machine and build

- 2x Tesla V100-PCIE-32GB (sm_70, PCIe 3.0 x16 each, `nvidia-smi topo`: SYS between them), driver 550.120,
  CUDA 12.8 (nvcc 12.8.93). A third card (Quadro RTX 4000) was idle and not visible to the engine.
- Intel Core i9-7960X (no AVX-512), 125 GiB RAM, Ubuntu, Linux 6.8.0-51.
- Release build, `-DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON
  -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON`.
- `pieces_off` / `pieces_on`: one binary built from this branch, a separate server start per run. `main`: an unchanged
  build of `7726f80c` with the same options, for the output check.

## Model and settings

Qwen3.8-Flash-Next GSQ-RCO IQ3_S (native pack), MTP draft head. Engine arguments, the same for every arm:

    strata --serve --pack <data>/packs/gsq-iq3_s --native <models>/.../Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
      --expert-profile <strata>/data/expert-profile.bin --expert-cache auto --prefill auto:16384 --spec 2 --spec-min-p 0.5
      --mtp <data>/mtp/rt --max-context 262144 --kv int8 --ple-io ram --peer-device 1 --conversation-cache-mib 0

Environment for every arm: `STRATA_PEER_NO_P2P=1`, `STRATA_GR_DOWN_MAX4=1`, `STRATA_PEER_HOT_AT=11283`,
`STRATA_PEER_HOT=0.45`, `STRATA_PREFILL_RING=96`, `STRATA_NO_LARGEPAGES=1`. The conversation cache is off, because
parking the previous conversation counts into the next prompt's time. Engine logs of every run are in `engine-logs/`.

## Prompt throughput

`scripts/abpr.py` (driven by `scripts/ab.sh`): per server start, one warm-up request, then two new prompts made from
the repository's docs, shuffled with a seed per round (32,000 and 128,000 characters). Temperature 0, thinking off,
8 output tokens. Prompt throughput = prompt tokens / time to the first streamed token. Four rounds; the arms' order
changes from round to round. Every run is in `runs.jsonl`.

| prompt | round | tokens | pieces_off | pieces_on | change |
|---|---:|---:|---:|---:|---:|
| ~9K | 1 | 8,031 | 1,546 | 1,674 | +8.3 % |
| | 2 | 8,609 | 1,558 | 1,686 | +8.2 % |
| | 3 | 9,080 | 1,585 | 1,695 | +6.9 % |
| | 4 | 9,912 | 1,605 | 1,723 | +7.4 % |
| | **median** | | **1,572** | **1,691** | **+7.6 %** |
| ~38K | 1 | 35,891 | 1,663 | 1,793 | +7.8 % |
| | 2 | 37,088 | 1,669 | 1,802 | +8.0 % |
| | 3 | 38,600 | 1,675 | 1,798 | +7.3 % |
| | 4 | 38,997 | 1,678 | 1,805 | +7.6 % |
| | **median** | | **1,672** | **1,800** | **+7.7 %** |

All values are tok/s. Every one of the 8 pairs is faster with the variable.

## Output

- **Log probabilities** (`scripts/lp.py` with `STRATA_LOGPOS` and `STRATA_LOGPOS_TOPK=20`, `scripts/logpos_arms.sh`):
  five teacher-forced requests of about 34K, 24K, 16K, 8K and 3K tokens. On this revision the prompt path reads all but
  the last 6 tokens of each request, so `STRATA_LOGPOS` writes 30 positions (on `011c9e60` it wrote 2,493); each
  depends on the whole prompt's cache. Files in `logpos/`, compared line by line (`scripts/logpos_identical.py`):
  - host route (`logpos-host-route.txt`): `pieces_off` and `pieces_on` are **bit for bit identical** to `main` at
    all 30 positions, top 20 included.
  - with P2P (`logpos-p2p.txt`; the variable has no effect there): `pieces_on` is bit for bit identical to `main`.
- **Greedy answers** (`scripts/gen.py`, `greedy.txt`): five prompts of 5K to 39K tokens, 200 tokens each, temperature 0.
  `main`, `pieces_off` and `pieces_on` give the same text for all five (SHA-256 of the answer equal).

## Not measured

- Decode: the change is in the prompt path only. No decode runs were made.
- Other GPUs and PCIe 4.0 or 5.0 links.
- HIP and SYCL: `src/prefill/prefill.cpp` and `src/prefill/kernels.cu` are built by those backends too; only the CUDA
  engine was built here.
- Prompts below 7K tokens, and chunks other than 16,384 tokens.
- Together with `STRATA_ADAPT_OVERLAP=1` or `STRATA_PF_PEER_CPU=1`.
- The two halves separately in this form. On an earlier revision with a switch for each, the early activations gave
  +6.0 % and the sums in pieces +1.6 % alone (+1.0 % on top of the first) at ~38K tokens.
