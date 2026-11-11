# DeepSeek-V4.1-Flash (dsv41) on a Tesla P40 from the CUDA 12.9.2 Docker image (2026-11-10)

`strata bench run` (standard-v1, protocol `strata-qwen38-thinking-v1-draft.1`) against
`Dockerfile.dsv41` built with `nvidia/cuda:12.9.2-devel-ubuntu24.04` and
`CUDA_ARCHITECTURES=60;61;70;75;80;86;89` - the first dsv41 run on a Pascal card in a container. CUDA 13
dropped sm_60/61, so this is the only CUDA that serves DeepSeek on a P40. Terminal and engine source at
`87dfca0e7c3ee24ad36cc3f69a31b859587245f8`, engine version 0.1.59.1, dsv41 engine `ds4-0.1`.

Medians of 3, from [results.json](results.json); the runner's lines are in [bench-run.txt](bench-run.txt)
and the engine's in [engine-run.txt](engine-run.txt).

| | median |
| --- | ---: |
| short prompt: first token / decode tok/s | 6.1 s / 3.40 |
| 4k prompt: prompt / decode tok/s, TTFT | 93.0 / 3.60, 30.3 s |
| 16k prompt: prompt / decode tok/s, TTFT | 145.1 / 3.60, 74.1 s |
| sanity checks | 12 of 12 |
| required suite after loading | 36 min 0 s |

7 of the 9 performance requests stop at the protocol's 512-token cap while still reasoning, so "first answer"
and "total" are times to the cap; the `short` prompt is ~80 tokens, so its prompt rate is a fixed request cost,
not a speed (both as noted in `bench/results/2026-10-20-0.2.1-bench/README.md`).

## Rig

- Tesla P40 24 GB (sm_61), driver 580.178.04, passed to the container alone (`--gpus device=1`); the machine's
  other card (a GT 1030) is not visible to it. AMD Ryzen 9 5950X (16 cores, AVX2 only), 125.7 GiB RAM,
  Ubuntu 24.04.5, kernel 7.0.0-42-generic. Nothing else used the card or the CPU during the run.
- Image: `strata:dsv41-cuda12` from `Dockerfile.dsv41`, `--build-arg CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
  CUDA_ARCHITECTURES='60;61;70;75;80;86;89' BUILD_VISION=0`. The engine package's `engine-component.json` says
  `build_id ...-cuda12.9`, `variant cuda12`; `cuobjdump` over the packaged `strata-dsv41` lists cubins for
  sm_60, sm_61, sm_70, sm_75, sm_80, sm_86 and sm_89.
- Model: DeepSeek-V4.1-Flash `IST-DAS-2.25bpw` (the pack and shared model files already in `/data`), served by
  `/opt/strata/components/engine-dsv41/installed/strata-dsv41` (the image's own dsv41 package, pinned with
  `STRATA_DSV41_EXE`; the engine line in `engine-run.txt` names it). Engine flags as started: `--max-context 262000
  --eos 1 --dspark on --dspark-conf 0.65 --reserve-mib 900 --tp 2 --layer-split auto --vision on` - tp 2 becomes
  1 on one card, and the image has no vision files (`BUILD_VISION=0`), so no image requests were sent.
- From the engine's own lines: layers 0-42 plus the head on the card (weights 6.58 GB, 329 of 15,744 experts in
  VRAM), 15,415 routed experts (150.5 GB) read in place from the model's files over mmap, a 111.6 GB RAM store
  with 89.7 GB of the hottest experts read in, card at 22,083 MiB used after load, container resident 100 GiB of
  the 125.7 GiB. The draft path (`--dspark`) kept 60-90% of drafts per position.

## The run's status and what that means

`results.json` records `status: nonconforming`. The run measured everything (9 of 9 performance requests, 12 of
12 sanity, no failures) and fails conformance only on metadata the `--url` runner cannot read from a running
server: the quantization repository's and source revisions are not pinned, model files cannot be hashed (the
server reports no model path), engine options are not exposed over the API, the model drive is not identified,
and `software.runtime` shows the driver's newest CUDA (13.0) because `--url` mode cannot read the engine's
`engine-component.json` (known runner behaviour, item 3 of `bench/results/2026-10-20-0.2.1-bench/README.md`);
the engine that ran is the `cuda12.9` build above. A conforming run would need `--profile`, whose name rule
rejects the dots in `deepseek-v4.1-flash-ist-das-2.25bpw`.

## Not tested

No image (vision) requests, no multi-GPU or `--tp 2`, no concurrency stage (not selected), nothing past the
512-token cap, no CUDA 13 image (it cannot serve this card), and the sm_60 path was compiled but not run (this
card is sm_61).
