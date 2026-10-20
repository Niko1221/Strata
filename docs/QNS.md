# Mirai Qwen3.8-Flash-Next-S (QNS trellis)

Validated on Linux (Ubuntu 22.04, CUDA 12.8) with RTX 3090s at 300 W, an EPYC 7642 (AVX2) and 125 GB RAM on
2026-10-20.

Mirai's Flash-Next-S is a trellis build of Qwen3.8-Flash-Next: the routed experts take 1 to 5 bits each (2.4 on
average), the backbone and head 5 bits. Mirai packed it for Strata as two files: `Qwen3.8-Flash-Next-S-qns.gguf`
(the experts and a Q8_0 backbone) and `Qwen3.8-Flash-Next-S-qns-dense.gguf` (the backbone and head as trellis, which
takes less VRAM). There is also a 32 GB build, S-small. The weights are on Hugging Face from Mirai
(`trymirai/Qwen3.8-Flash-Next-S-strata-experimental`).

Every expert is its own blob of its own size, so the RAM arena is compact and each VRAM slot is sized to its expert.
The format, the CUDA and AVX2 kernels and the packer support come from Mirai's `strata-qns.patch` for v0.1.43.

## Pack

```sh
.venv/bin/python tools/iq_pack.py --gguf /models/Qwen3.8-Flash-Next-S-qns.gguf --out packs/qns
```

It takes seconds and writes `native_experts.txt` v6 with `qns_offsets.bin` and `qns_rotation.bin`. A pack made by the
v0.1.43 patch (v5) is refused with a note to repack: v5 means Hadamard-folded experts in this engine.

## Run

The n-gram table is the PLE shard of any ISTA-DASLab size (shard 2, 28.8 GB); the MTP layer and the expert profile
are the original model's.

```sh
build/strata --serve --pack packs/qns \
  --native /models/Qwen3.8-Flash-Next-S-qns.gguf --native-dense-gguf /models/Qwen3.8-Flash-Next-S-qns.gguf \
  --qns-dense-gguf /models/Qwen3.8-Flash-Next-S-qns-dense.gguf \
  --ple-gguf /models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp <mtp/rt> --kv int8
```

`--layer-split auto` with `--gpu 0,1` runs it on two cards.

## Measured

Greedy, thinking off, 32K context, MTP on, the server's own tok/s per request (six short requests and a 6,210-token
document); one run each, so the ranges are indicative, not a benchmark:

| Model | Cards | Experts in RAM | Decode | 6,210-token prompt |
|---|---|---:|---:|---:|
| S | 1 | 33.8 GiB | 35-81 tok/s | 4.1 s |
| S | 2 (layer split) | 33.8 GiB | 60-125 tok/s | 4.2 s |
| S-small | 1 | 26.5 GiB | 47-106 tok/s | 3.9 s |
| S-small | 2 (layer split) | 26.5 GiB | 83-166 tok/s | 4.0 s |

On S-small the greedy answers of this engine and of v0.1.43 with Mirai's patch are the same on five of six probes
(the sixth parts on a near-tie after 10-55 words), on one card and on two.

## Limits

- NVIDIA (CUDA) and the AVX2 CPU pool. Not built or run on AMD (HIP), Intel (SYCL) or Mac.
- Setup and the Terminal do not offer these models yet; start them with a config like the command above.
