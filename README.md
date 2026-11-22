# THE PLATINUM BUILD — Strata on 4× MI50 (gfx906)

**What this is:** the fastest Strata build we have ever measured for
Qwen3.8-Flash-Next on gfx906 hardware. Branch `rp-graft-0214` = upstream Strata
**v0.2.14.1** + 13 gfx906 graft commits (MMQ dense/expert repacks + the "rp"
prompt-path stack, vendored MIT code from mx-llama.cpp), compiled with the
**clang 24** bundled in **ROCm 10.1** on the community
[`mixa3607/rocm-gfx906:10.1-complete`](https://hub.docker.com/r/mixa3607/rocm-gfx906)
image.

**Verified on GOLEM (4× MI50 32 GB, Threadripper 2990WX, 250 GB RAM):**

| probe | result |
|---|---|
| decode_128 / decode_256 (bench probe) | **148–149 tok/s** |
| prefill_8192 (bench probe) | **1290–1305 tok/s** |
| long cold prefill (60–70K-token prompts, live traffic) | **~2150–2190 tok/s** |
| generation, real-world (17.8K prod generations ≥100 tok) | **typical 58, worst 10% ≤44, best 10% ≥81 tok/s** |
| context | 262 144 tokens (int8 KV) |
| VRAM | 93.6 GiB across 4 GPUs, layer-split auto |

The bench probes are short, cache-warm best cases. Day-to-day numbers with
50–70K-token conversation contexts are honest lows of ~45–50 and an average
near 60 tok/s — every token pays real attention/KV cost at those depths,
and that is what the production-traffic row measures. Long cold
prefills (~2.2K tok/s on 60–70K-token reads) run faster per token than the
8K bench probe — we have not isolated why, so treat that as an observation,
not a guarantee.

For comparison, stock upstream engines *regress* on gfx906 (0.1.60+ measures
~40 % slower decode on wave64 hardware). This branch exists because nothing
upstream targets gfx906 prompt-path performance.

## Hardware envelope (read before anything else)

- **4× gfx906** datacenter cards (MI50 32 GB; MI60/VII should work, untested).
  The config assumes ≥ 96 GiB aggregate VRAM and the full model HBM-resident.
- **~250 GB system RAM** in our box; the working set is the expert arena +
  ~3–4 GiB engine runtime + ~3 GiB pinned KV at 256 K ctx. Budget accordingly
  before copying our flags.
- **PPTable + memory-clock cap applied first.** Our numbers assume the cards
  run a PPTable with the memory clock hard-capped at 1080 MHz (stability
  ceiling on used HBM2). Without proper PPTable/clocks you will see lower
  numbers and intermittent hangs — that is a hardware layer, not this build.
- **NUMA matters.** On the 2990WX we pin to the two RAM-bearing nodes
  (`--membind=0,2`, CPUs 0-15,32-47). Wrong-NUMA placement costs real tok/s.
- Docker, `/dev/kfd` + `/dev/dri` exposed, `memlock` unlimited.

## Path A — use the prebuilt binary

1. Download `strata` from the [`platinum-r101` release](releases/tag/platinum-r101)
   and verify: `sha256sum strata` must equal
   `accc4ac0031123a39b9f681fd3b9820df1d55f03d60e43437ec0d2d6db5f2451`.
2. Pull the runtime base image (we do **not** host a container — this is the
   community image everything is built on):
   `docker pull mixa3607/rocm-gfx906:10.1-complete`
   and build the serve image with `Dockerfile.runtime` from the release
   (adds the 4 Python packages the serve wrapper needs).
3. Get the model data (below), edit `strata.json.example` paths, launch with
   `strata_serve.sh` from the release.

## Path B — build from source (~20 min)

```
git clone -b rp-graft-0214 https://github.com/rpcribari/Strata rp-graft
cd rp-graft
bash build_r101.sh          # dockerized; log: build-r101.log
sha256sum build-r101/strata # must equal the release sha above
```

Do not drift the flags in `build_r101.sh` — `STRATA_HIP_GFX906=ON`,
`STRATA_MMQ=ON`, `STRATA_MMQ_KQUANTS=ON`, `STRATA_PREFILL_MMQ=OFF`, gfx906
arch, clang 24 at `/opt/rocm/llvm/bin`. The +8 % over the previous champion
build came specifically from clang 24 / ROCm 10.1; clang 23 on the older
7.2.4 image reproduces the old 137–139 numbers.

Optional vision encoder (HIP): re-run cmake with `-DSTRATA_VISION_HIP=ON` and
build target `strata-vision-hip` (see `tools/vision/CMakeLists.txt`).

## Model data

All from Hugging Face [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):

- `UD-IQ4_XS/` — the 3-shard GGUF (`...-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf`)
- `MTP/` — the MTP draft layer (this is the 1.6–1.8× decode speedup; keep it on)
- `mmproj-F16.gguf` — vision projector (only if you built the vision encoder)

Then build the expert pack from shard 1 (needs numpy):

```
python3 tools/iq_pack.py --gguf <path>/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --out ~/Strata-data/packs/ud-iq4_xs --experts-bin
```

`--experts-bin` is required with `STRATA_ARENA_MMAP=1` (which we run).

**Expert profile:** the release ships `expert-profile-starter.bin` (ranked
routing pairs learned from our production traffic — better than nothing on day
one). Regenerate from your own traffic once you have some:
`tools/collect_profile_traces.py --mode routing` + `tools/make_profile_mix.py`.

## Serve

`strata.json.example` is our exact production config with paths replaced by
`/home/YOURUSER` — `sed` them to your layout. The serve wrapper is the `serve/`
Python package in this checkout:

```
python3 -m serve.server --engine /home/YOURUSER/rp-graft/build-r101/strata \
  --config /home/YOURUSER/strata.json --port 8081 --host 0.0.0.0
```

`strata_serve.sh` wraps that in the container with the flags we need:
`--device=/dev/kfd --device=/dev/dri --group-add video --group-add <render-gid>
--ipc=host --ulimit memlock=-1:-1`.

**Required env** (baked into the example config):

- `HSA_OVERRIDE_GFX_VERSION=9.0.6`
- `HSA_USERPTR_FOR_PAGED_MEM=0` and `HSA_SVM_FOR_PAGED_MEM=0` — REQUIRED on the
  ROCm 10.1 stack, the engine misbehaves without them
- `STRATA_ARENA_MMAP=1`, `STRATA_ENGINE_READY_S=2400` (long load on big RAM)
- attention knobs `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1`,
  `STRATA_GFX906_ATTN_REDUCE12=1`, `STRATA_ATTN_MERGE_V2=1`

## Verify

```
BENCH_BASE=http://127.0.0.1:8081/v1/chat/completions python3 bench_api.py
```

Expect decode ≈ 148, prefill_8192 ≈ 1290 on GOLEM-class hardware.
**If decode reads 58–86, your probe is missing `"reasoning_effort":"none"`** —
you are counting reasoning tokens, not a regression.

## Pitfalls (each cost us real time)

1. `--group-add render` **by name** fails intermittently in dockerd — resolve
   the GID with `getent group render | cut -d: -f3`.
2. Never run a second engine while one serves traffic — VRAM overcommit stalls
   the GPUs and kills the engine.
3. Bring-up scripts must not send docker errors to `/dev/null`.
4. Bench probes must replicate the real request body (stream, include_usage,
   `reasoning_effort:none`) or you will chase ghosts.
5. The engine reads `strata.json` only at launch — config edits are inert
   until restart.
6. Upstream engines newer than 0.1.52-era regress gfx906 ~40 %; that is the
   whole reason this graft exists. Don't "just upgrade" back onto it.

## Rollback reference

Previous champion ("crown build"): 0.1.52-era fork, clang 23, older 7.2.4
image — 137–139 tok/s decode. See branch `int6` (legacy).
