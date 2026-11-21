# The split engine across hosts

`STRATA_SPLIT_DISTRIBUTED=ON` builds the existing Flash-Next split engine for
one CUDA device per MPI process. MPI carries engine commands and host results;
NCCL carries GPU tensors. It uses the split engine's prefill, sampler, conversation
cache, images, lanes, MTP and DFlash2. The normal engine and the default split
build keep their existing transport.

This is an experimental, opt-in Linux CUDA build. It does not start an RPC service
or add another HTTP backend. The ordinary Strata server can drive its existing
line protocol through an MPI launcher.

## Build

Install the CUDA toolkit, an MPI C++ development environment, and NCCL headers
and library. Both hosts need compatible runtimes and the same model files.

```sh
cmake -S engine -B build-cluster \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_SPLIT=ON \
  -DSTRATA_SPLIT_DISTRIBUTED=ON \
  '-DSTRATA_SPLIT_CARDS=1;2' \
  -DSTRATA_SPLIT_ARCH=121 \
  -DCMAKE_CUDA_ARCHITECTURES=121-real \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DSTRATA_NCCL_INCLUDE_DIR=/opt/nccl/include \
  -DSTRATA_NCCL_LIBRARY=/opt/nccl/lib/libnccl.so.2
cmake --build build-cluster --target strata-split-1 strata-split-2 -j8
```

Replace the NCCL paths with the local installation. Architecture 121 is for
DGX Spark. The resulting executables are `strata-split-mpi-1` and
`strata-split-mpi-2` under `build-cluster/src/split`. The one-process variant
provides a reference with the same engine. Card counts 3, 4, 6 and 8 can also be
built; a successful build does not establish correctness on that many hosts.

## Run

The distributed engine requires one process per host. Each process must see exactly one CUDA device. Keep the
executable, GGUF shards and drafter at the same paths on every host. Compare
their SHA256 checksums before a run. Passwordless SSH between the participating
hosts must already work.

The following uses SSH aliases `spark-a` and `spark-b`; they are examples.
Set NCCL's HCA and socket interface for the physical link actually connected on
the machines. A separate management network can remain connected.

```sh
export CUDA_VISIBLE_DEVICES=0
export FNX_ARENA=0
export FNX_EXPERT_RAM_GIB=off
export FNX_RING_F32=1
export FNX_KV8=1
export NCCL_IB_DISABLE=0

mpirun -np 2 --host spark-a:1,spark-b:1 --map-by ppr:1:node --bind-to none \
  --mca pml ob1 --mca btl self,tcp \
  -x CUDA_VISIBLE_DEVICES -x NCCL_IB_DISABLE \
  /opt/strata/strata-split-mpi-2 --serve \
  --native /models/model-00001-of-00004.gguf \
  --draft dflash2 --draft-model /models/dflash2.pack \
  --draft-vocab /models/draft_vocab.bin \
  --max-context 65536 --spec 6 --lookup 0 --batch 2
```

MPI's control traffic can be bound to the same network with the launcher's TCP
interface settings. NCCL's network settings must reach every rank; use explicit
`-x` arguments for any additional NCCL variables. The coordinator synchronizes
the `FNX_` compute settings before loading the model.

For MTP replace the DFlash2 arguments with `--draft mtp --mtp /models/mtp/rt`.
All routed expert shards remain resident. Expert RAM caching and the arena
are currently unavailable in this mode. Incompatible explicit settings are
refused at startup. Decode and prompt sums use FP32.

Budget memory per host. Each process also keeps the model's RAM embedding tables;
the hosts do not form one allocation pool. Unused memory is headroom for context
and lanes, rather than an instruction to allocate more copies of the weights.

The model determines the context limit. `--rope-scale 4` explicitly requests
YaRN; it changes every position's rotary embedding. Capacity alone does not
prove that a model can recover information from a full context. The existing
frontend reserves eight tokens beyond the requested reply: account for them
when preparing a request close to the context limit.

## HTTP API

Use the ordinary `--engine strata` server and a run config whose executable
is a launcher named `strata-split-cluster-2`. Its name lets the server apply the
existing split protocol's capabilities. The launcher should `exec mpirun ...
/opt/strata/strata-split-mpi-2 "$@"`; the config supplies the model arguments.
Keep `--batch` consistent with the server's parallel request count. DFlash2 runs
on the coordinator and verifies its proposals through the distributed engine.

Existing split-engine limits still apply, including its unsupported logprobs
and per-request engine tuning. This build does not add support for model
families the split engine cannot load.

## Validation

CPU transport and command-frame checks can run without a GPU:

```sh
cmake -S engine/src/tp -B build-runtime
cmake --build build-runtime -j4
ctest --test-dir build-runtime --output-on-failure
```

`strata-tp-nccl-runtime-test` additionally checks real GPU sums, gathers and
strided peer exchanges, including CUDA graph replay. Build the standalone runtime
with `STRATA_TP_CUDA=ON` and the NCCL paths, then run it with one and two ranks.

`strata-split-mpi-tile-test` is a separate, explicitly built GPU check for the
prompt kernel's weight decoding. `terminal/tests/split/tile_check.py` compares its dumps
with CPU dequantization of the original GGUF, including Q5_0 routed weights.
`engine/tests/split/distributed_engine_test.cpp` exercises the complete engine's logits,
snapshot restore, lane isolation, batched rows, penalties and image payloads
larger than a command frame. It needs real model files.

```sh
cmake --build build-cluster --target \
  strata-split-mpi-engine-test-1 strata-split-mpi-engine-test-2 \
  strata-split-mpi-tile-test -j8
```

Run the engine check through the same MPI launcher as the frontend, passing
`MODEL CASES OUTPUT_DIRECTORY`. Each line of `CASES` contains the prompt token
count, the continuation token count, the prompt IDs, then the teacher-forced
continuation IDs. Supply at least two prompts. The check uses a 1024-token
context and two lanes, including startup warmup, to catch pool-sizing failures
that a large context can hide. Its image rows are synthetic transport fixtures;
they do not measure image understanding. Keep output directories distinct for
the one- and two-rank runs and compare the saved full-vocabulary logits.

Preserve exact source revisions, launch commands, individual runs and failures.
Compare one and two hosts with identical prompts, quantization, context and
draft settings. Separate compilation, functional checks, numerical comparison,
free generation speed, concurrent throughput and long-context retrieval.
Floating-point reductions can change free generation across card counts;
compare teacher-forced distributions as well as visible answers.

This MPI job is a single serving unit: losing one rank ends the job. A process
supervisor must restart the whole group. Repeated functional checks do not
establish unattended 24/7 operation. HIP, SYCL and additional physical host
counts need their own validation before requesting review.
