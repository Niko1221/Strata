# STRATA_HC_REQ8 - measured

RTX 2060 SUPER 8 GB (sm_75), Ryzen 9 3900X, 60 GB RAM, Q2_0, `--max-context 8192 --kv int8`, 400 MiB reserve
(the distribution check's engines below):

| | expert cache slots | dense weights loaded |
| --- | ---: | ---: |
| BF16 (default) | 2,044 | 1,416 MiB |
| `STRATA_HC_REQ8=1` | 2,441 (+397) | 216 MiB + 675 MiB int8 |

### Distribution check (teacher-forced)

The method of `docs/UNSLOTH_Q4.md`: a serve engine with `STRATA_LOGPOS` + `STRATA_LOGPOS_TOPK=256`, `--short-read 1600`
so every prompt token is read through the verify windows and scored, `--adapt-every 100000 --pcie-frac 0` (fixed
experts), greedy. Three ~1,000-token texts of this repository (code: `native_rope.cu`; docs: `DETAILS.md`; prose:
`BATCHING.md`). KL is bf16 || other over the bf16 run's top 256 plus one bucket for the rest. Control: a second BF16
run.

| comparison | text | positions | argmax same | top-10 overlap | KL mean | KL median | KL p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 vs bf16 (a second run) | all three | 3,327 | 100% | 100% | 0 | 0 | 0 |
| bf16 vs REQ8 | code | 1,084 | 93.3% | 91.1% | 0.069 | 0.0077 | 1.03 |
| bf16 vs REQ8 | docs | 1,145 | 99.7% | 92.2% | 0.0063 | 0.00017 | 0.066 |
| bf16 vs REQ8 | prose | 1,098 | 97.7% | 95.4% | 0.0066 | 0.0012 | 0.088 |
| bf16 vs REQ8 | all | 3,327 | | | 0.027 | 0.0011 | 0.43 |

The BF16 reruns are bit-identical, so the REQ8 rows are the int8 projections alone. For scale: the prompt-attention
kernel's check (`bench/results/2026-10-03-v100-prompt-attn`) moved the median KL by 0.0007-0.022 with controls that
only change the rounding order, and the experimental speed projection (`bench/results/2026-09-27-esp`) measured
mean 0.063. Code is the most sensitive of the three texts here (its heavy tail: p99 1.03, max 6.9).

Reproduce: `bench/results/2026-10-06-hc-req8/req8_kl.py run`, then `compare` (paths at the top of the script).
