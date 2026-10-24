# DFlash2 quantization versus original BF16 and MTP — 2026-10-19

Metadata correction (2026-10-20): the shared target head is Q8_0 (GGML type 8, 2560 × 248320, 675430400 bytes). The older engine hard-coded Q5_K in its log. This corrects the format label; the original timings and raw logs are unchanged. See the [follow-up](../2026-10-20-dflash2-long-tuned/README.md).

## Scope and artifacts

Hardware: RTX 4070 Ti SUPER 16 GiB, Ryzen 9 5900X, 125.7 GiB RAM; Linux,
CUDA 13.4, Release build for SM89. The target is the local Qwen3.8-Flash-Next
NVFP4 pack, BF16 token embeddings, FP8 PLE, INT8 KV and the shared Q8_0
output head. `*-command.json` records the exact executable SHA-256, command
and environment. No engine code or user configuration is changed for this study.

The official [incoai checkpoint](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
is pinned to revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6`.
Reading its safetensors header confirms **81 tensors, all BF16**, with
1,206,131,200 bytes of payload (603,065,600 parameters). There is no `lm_head`
or embedding tensor in that checkpoint. Its SHA-256 is
`d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f`.

The quantized artifacts are derived directly from the bit-exact BF16 GGUF,
not from one another. All 36 backbone matrices are quantized. The other 45
tensors (norms, dynamic convolution and selector) keep their BF16 bits:
`extension-parity.json` verifies this against the original for all three formats.
Those remaining BF16 tensors total 288,627,200 bytes. This is partial drafter
quantization; it does not quantize the shared target output head again.

| Drafter | GGUF MiB | Backbone format | Other tensors |
| --- | ---: | --- | --- |
| Original | 1150.26 | BF16 | BF16 |
| Q8_0 | 740.11 | Q8_0 | BF16 |
| Q5_0 | 576.04 | Q5_0 | BF16 |
| Q4_0 | 521.36 | Q4_0 | BF16 |

Files and SHA-256 hashes are recorded in `artifacts.json`. Models stay outside
the checkout in `../Strata-data/dflash2-official/`.

## Method

**Use all available VRAM**, as requested: `--expert-cache auto`. Different
drafter footprints therefore give different numbers of GPU-resident target
experts. This measures the practical benefit of smaller weights on this card,
including the resulting expert-cache gain; it does not isolate matrix kernels.

Every arm starts with the same 512 MiB allocation reserve. Verifier arenas and
graph captures consume most of that reserve after the expert cache is sized.
`vram` in each result records actual GPU and process memory at READY, after
warm-up and after measurements. A 256 MiB reserve did not fit the verifier
arena (`reserve-256-oom/`); that failed startup is excluded. A preliminary
fixed-2000-expert run was interrupted when the user requested filling VRAM;
its partial results in `fixed-residency-aborted/` are also excluded.

Context capacity 8192, prefill chunk 512, frozen routing profile, 11 pool workers
plus the host, CPU prompt sharing and PCIe expert sharing off, adaptive expert
swaps off, prompt/conversation snapshots off, suffix and lookup drafting off.
The latest DFlash2 kernel optimizations are enabled; experimental graph buckets
and interleaved draft head kernels remain off.

Four fixed tokenized prompts: code, math, prose, and a 2425-token notes prompt.
Greedy decoding, seed 1234, 96 output tokens per request. Each arm warms up on
all four prompts before measurements. Each speculative arm has three measured
repeats per prompt. Target-only has one measured repeat. MTP is run again at
the end to expose drift. DFlash2 uses three proposals plus the anchor
(`--spec 8 --dflash-block 3`); MTP uses up to three proposals (`--spec 4`).

Engines run sequentially; there is no compilation or other benchmark during
measured decoding. Throughput is based on native decode time, excluding
prefill and engine startup. Wall request times, acceptance counts and raw token
IDs are retained. Changed GPU expert placement can change floating-point
rounding, so equality across these arms is an observation to check, rather
than an assumed property of quantization. After graph warm-up, the Q8_0,
Q5_0 and Q4_0 arms reported 120, 133 and 128 MiB free; the runtime printed
its low-VRAM warning. These measurements reflect nearly filling the card.

Reproduce with `.venv/bin/python bench/results/2026-10-19-dflash2-quant-mtp/suite.py`;
then run `analyze.py` and `quality.py` in the same directory. Existing local
model artifacts are required. The scripts do not download models.

## Results

Median engine decode tokens/s across three repeats. Warm-up and prefill are not
included. MTP's reference pools its first and last runs (six samples per task);
the geometric mean shifted by 1.04% between these two MTP series.

| Arm | Code | Math | Prose | Long | Geometric mean | Drafter VRAM MiB | GPU experts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Target only | 32.22 | 30.99 | 37.27 | 38.62 | 34.62 | — | 3554 |
| MTP, pooled | 38.40 | 43.41 | 43.07 | 47.43 | 42.96 | 801 | 3146–3149 |
| DFlash2 BF16 | 43.09 | 43.07 | 36.40 | 40.93 | 40.78 | 1354.6 | 3040 |
| DFlash2 Q8_0 | 45.16 | 44.83 | 39.65 | 43.88 | 43.32 | 945.0 | 3205 |
| DFlash2 Q5_0 | 45.45 | 43.37 | 33.49 | 43.68 | 41.21 | 780.9 | 3263 |
| DFlash2 Q4_0 | 49.27 | 45.42 | 34.89 | 45.15 | 43.33 | 726.2 | 3284 |

The Q4_0 and Q8_0 geometric means are 0.87% and 0.84% above pooled MTP,
while MTP itself moved 1.04% between its beginning and end series. That is a
tie at this test's resolution, not a demonstrated overall win. DFlash2 Q4_0
is 28% faster on this code prompt, and is 19% slower on prose and 5% slower
on the long prompt. Q8_0 is more even: 18% faster on code, 8% slower on prose,
and 7% slower on the long prompt. Q5_0 did not beat Q8_0 despite its smaller
file; on this backend it is slower on prose and its overall median is below MTP.

Quantization reduces the BF16 GGUF from 1150.3 MiB to 740.1 MiB (Q8_0),
576.0 MiB (Q5_0) or 521.4 MiB (Q4_0). Runtime savings are partly used by the
larger target cache: Q4_0 fits 244 more GPU experts than BF16 and 138 more than
MTP, with about 128 MiB free after the run. The target output head is shared
with all DFlash2 arms and already Q8_0; these size reductions come from the
drafter backbone, not from quantizing that head.

The greedy output check passed all 16 DFlash2 quantization/workload pairs.
Code and math matched the maximum-cache target-only reference. The BF16 long
answer and prose answers can differ from that reference when GPU expert
placement changes: second target-only requests at the corresponding arm's
own placement reproduced all 96 speculative tokens exactly. `quality-summary.json`
and the emitted target-logit fixtures keep those checks. These text differences
therefore do not show that quantizing the drafter changed the verified target
result.

`extension-parity.json` records bit-exact preservation for all 45 BF16
extension and norm tensors in each quantized GGUF. `exit-codes.json` has a zero
exit code for each benchmark engine. The `reserve-256-oom/` attempt is retained
as a failed startup record; its 256 MiB reserve did not leave room for the
verifier arena. A fixed-residency run was stopped when the test was changed to
use automatic maximum VRAM placement and is excluded from these results.
