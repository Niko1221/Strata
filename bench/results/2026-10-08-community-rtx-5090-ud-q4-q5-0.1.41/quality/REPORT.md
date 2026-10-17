# Quant quality against Q8_0 on Strata

Generated 2026-10-08 18:04 by strata_quality.py. Teacher-forced: every arm read the same 30 requests (~560 scored tokens each) through the verify windows; `STRATA_LOGPOS_TOPK=256`. KL = KL(Q8_0 || arm) over Q8_0's top 256 + one bucket (nats). Decisive = positions where Q8_0's top-2 gap is >= 0.5 nats.

Sources: **prose** https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip (wikitext-2-raw/wiki.test.raw, then wikitext-2-raw/wiki.train.raw; sha256 ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11); **agent** https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/resolve/main/func-calling.json (sha256 769478035a886678d525d057ea66e3fd1e247f43e35c8d6d7cc866e080b6742b), conversations rendered as '### ROLE' blocks; **code** llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (src, ggml/src, common, gguf-py; shuffled with seed 1234)

## all

| arm | requests | positions | mean KL [95% CI] | median KL | p99 KL | top-1 same | top-1 same (decisive) | top-10 overlap | PPL arm / Q8_0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ud-q4_k_xl | 30 | 17,339 | 0.0558 [0.0412, 0.0714] | 0.0040 | 0.827 | 94.0% | 96.9% (15,728) | 89.8% | 52.807 / 51.866 |
| ud-q5_k_xl | 30 | 17,339 | 0.0346 [0.0249, 0.0450] | 0.0021 | 0.501 | 95.4% | 97.9% (15,728) | 92.2% | 54.344 / 51.866 |

## prose

| arm | requests | positions | mean KL [95% CI] | median KL | p99 KL | top-1 same | top-1 same (decisive) | top-10 overlap | PPL arm / Q8_0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ud-q4_k_xl | 10 | 5,781 | 0.0449 [0.0262, 0.0675] | 0.0044 | 0.637 | 94.2% | 97.2% (5,204) | 90.9% | 213.493 / 214.828 |
| ud-q5_k_xl | 10 | 5,781 | 0.0251 [0.0144, 0.0369] | 0.0018 | 0.383 | 95.7% | 98.6% (5,204) | 93.2% | 246.003 / 214.828 |

## code

| arm | requests | positions | mean KL [95% CI] | median KL | p99 KL | top-1 same | top-1 same (decisive) | top-10 overlap | PPL arm / Q8_0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ud-q4_k_xl | 10 | 5,776 | 0.0660 [0.0356, 0.1040] | 0.0050 | 0.971 | 93.6% | 96.6% (5,225) | 89.0% | 27.082 / 24.014 |
| ud-q5_k_xl | 10 | 5,776 | 0.0363 [0.0207, 0.0579] | 0.0031 | 0.497 | 95.4% | 97.8% (5,225) | 91.6% | 24.446 / 24.014 |

## agent

| arm | requests | positions | mean KL [95% CI] | median KL | p99 KL | top-1 same | top-1 same (decisive) | top-10 overlap | PPL arm / Q8_0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ud-q4_k_xl | 10 | 5,782 | 0.0564 [0.0337, 0.0819] | 0.0028 | 0.841 | 94.3% | 96.9% (5,299) | 89.6% | 25.457 / 27.030 |
| ud-q5_k_xl | 10 | 5,782 | 0.0424 [0.0232, 0.0645] | 0.0017 | 0.608 | 95.1% | 97.4% (5,299) | 91.8% | 26.672 / 27.030 |

## ctx short

| arm | requests | positions | mean KL [95% CI] | median KL | p99 KL | top-1 same | top-1 same (decisive) | top-10 overlap | PPL arm / Q8_0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ud-q4_k_xl | 12 | 6,847 | 0.0401 [0.0194, 0.0663] | 0.0006 | 0.676 | 96.3% | 98.1% (6,501) | 88.8% | 810.417 / 709.144 |
| ud-q5_k_xl | 12 | 6,847 | 0.0217 [0.0107, 0.0380] | 0.0003 | 0.400 | 97.1% | 98.6% (6,501) | 91.6% | 751.920 / 709.144 |

## ctx 8192

| arm | requests | positions | mean KL [95% CI] | median KL | p99 KL | top-1 same | top-1 same (decisive) | top-10 overlap | PPL arm / Q8_0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ud-q4_k_xl | 9 | 5,245 | 0.0523 [0.0342, 0.0768] | 0.0061 | 0.806 | 93.5% | 96.6% (4,642) | 91.0% | 9.487 / 10.377 |
| ud-q5_k_xl | 9 | 5,245 | 0.0352 [0.0265, 0.0454] | 0.0039 | 0.499 | 95.1% | 98.0% (4,642) | 92.9% | 10.764 / 10.377 |

## ctx 32768

| arm | requests | positions | mean KL [95% CI] | median KL | p99 KL | top-1 same | top-1 same (decisive) | top-10 overlap | PPL arm / Q8_0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ud-q4_k_xl | 9 | 5,247 | 0.0796 [0.0531, 0.1147] | 0.0118 | 1.006 | 91.6% | 95.5% (4,585) | 90.0% | 8.323 / 8.535 |
| ud-q5_k_xl | 9 | 5,247 | 0.0509 [0.0307, 0.0734] | 0.0058 | 0.741 | 93.5% | 96.9% (4,585) | 92.3% | 8.894 / 8.535 |

## Paired differences (row arm minus column arm, same positions)

KL diff > 0: the first arm is further from Q8_0. Interval: 95% bootstrap over requests. Requests worse: share of requests where the first arm's mean KL is higher. Decisive top-1: difference in agreement with Q8_0 at decisive positions, percentage points (> 0: the first arm agrees more often).

### all

| first - second | requests | mean KL diff [95% CI] | requests worse | decisive top-1 diff (pp) |
|---|---:|---:|---:|---:|
| ud-q4_k_xl - ud-q5_k_xl | 30 | +0.0212 [+0.0122, +0.0307] | 83% | -1.03 |

### prose

| first - second | requests | mean KL diff [95% CI] | requests worse | decisive top-1 diff (pp) |
|---|---:|---:|---:|---:|
| ud-q4_k_xl - ud-q5_k_xl | 10 | +0.0198 [+0.0062, +0.0356] | 80% | -1.40 |

### code

| first - second | requests | mean KL diff [95% CI] | requests worse | decisive top-1 diff (pp) |
|---|---:|---:|---:|---:|
| ud-q4_k_xl - ud-q5_k_xl | 10 | +0.0297 [+0.0120, +0.0488] | 90% | -1.19 |

### agent

| first - second | requests | mean KL diff [95% CI] | requests worse | decisive top-1 diff (pp) |
|---|---:|---:|---:|---:|
| ud-q4_k_xl - ud-q5_k_xl | 10 | +0.0140 [+0.0019, +0.0274] | 80% | -0.51 |

### ctx short

| first - second | requests | mean KL diff [95% CI] | requests worse | decisive top-1 diff (pp) |
|---|---:|---:|---:|---:|
| ud-q4_k_xl - ud-q5_k_xl | 12 | +0.0185 [+0.0075, +0.0319] | 92% | -0.52 |

### ctx 8192

| first - second | requests | mean KL diff [95% CI] | requests worse | decisive top-1 diff (pp) |
|---|---:|---:|---:|---:|
| ud-q4_k_xl - ud-q5_k_xl | 9 | +0.0171 [+0.0031, +0.0353] | 78% | -1.38 |

### ctx 32768

| first - second | requests | mean KL diff [95% CI] | requests worse | decisive top-1 diff (pp) |
|---|---:|---:|---:|---:|
| ud-q4_k_xl - ud-q5_k_xl | 9 | +0.0287 [+0.0101, +0.0495] | 78% | -1.40 |

## Notes

- Every arm runs on the same engine build; the Unsloth packs (Q8_0 too) have their Q8_0 hyper-connection projections rounded to BF16 (`--compat-bf16`), so the reference is Q8_0 as Strata runs it.
- Experts held in VRAM run GPU kernels and the others CPU kernels; the split differs per arm (expert cache size), which is part of what is measured. `--adapt-every 100000` keeps it fixed within a run.
- A token Q8_0 ranks in its top 256 but the arm does not gets a share of the arm's leftover probability (capped at its 256th), so KL can only be underestimated where the distributions differ a lot.
- `x@2` is a second run of `x` (same everything): its row is the run-to-run floor, expected 0. `x@ecN` is `x` with `--expert-cache N`: the same weights with another GPU/CPU split of the experts, i.e. the floor from engine numerics alone. A quant's KL means something only above that floor.
- q8_0: engine 0.1.41, expert slots 4766, load 36s, arm total 31 min.
- ud-q4_k_xl: engine 0.1.41, expert slots 7940, load 33s, arm total 5 min.
- ud-q5_k_xl: engine 0.1.41, expert slots 6217, load 45s, arm total 8 min.
