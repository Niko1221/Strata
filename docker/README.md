# Strata in Docker, on an AMD RX 7700 XT (gfx1101)

Two images, both named `strata`:

| Image | Built by | Contains |
| --- | --- | --- |
| `strata-hip-builder:<arch>` | `./build.sh` | ROCm 7.2 toolchain; compiles the engine into `./build-hip` |
| `strata-hip:<arch>` (+ `-<gitsha>`, `-latest`) | `./build.sh` | the engine, the OpenAI-compatible server, the model bootstrap |

```sh
./build.sh                  # ~7.4 GB of base image, then a 10-25 minute compile (ccache: ~2 min after)
./run.sh --detach           # server on http://127.0.0.1:8080
docker logs -f strata-gfx1101
./run.sh --check            # asks /v1/models
```

Measured here (RX 7700 XT, gfx1101, ROCm 7.2.1 image), so "documented" and "observed" stay apart:

| | |
| --- | --- |
| engine build | `[233/233] Linking HIP executable strata`, 16.6 MB, ~2 min with a warm ccache |
| engine on the card | `strata-device` → `device 0: AMD Radeon RX 7700 XT`, 27 CUs, wave32, 11.984 GiB |
| GPU test suite | 42 of 44 pass; the 2 failures are CPU-side pack-format tests that also fail with no GPU attached |
| VRAM under generation | peak **9 898 MiB** against the 10 240 MiB budget |
| speed, IQ3_XXS | prefill 24.3 tok/s, decode **12.3 tok/s**, speculative draft accepting 33 of 37 tokens |
| first start | 76 GB download ~12 min, pack (39.97 GiB `experts.bin`) ~7 min, MTP draft layer ~1 min |

`./run.sh` refuses to start anything silly: the wrong card, `HSA_OVERRIDE_GFX_VERSION` set (breaks
gfx1101 - ROCm 7.x handles it natively), or a filesystem too small for the quant it is about to
download. Read its `--dry-run` output before the first real run.

## Models come from `~/Development/models` (the HF cache, on the big disk)

The container reads **and, if needed, downloads into** a Hugging Face cache in
**`~/Development/models`** (`--hf-cache` to choose elsewhere). Deliberately *not*
`~/.cache/huggingface/hub`: that lives on `/`, which here has ~90 GB free - less than one quant plus
its pack. `~/Development` is on the 4 TB disk (3.7 TB free). Packs, the MTP draft layer and logs go
to `~/Development/strata-work` (`--work`). Nothing is baked into an image, and a model fetched by the
container is immediately visible to the host's own `hf` client and to every other container.

The cache mounts **read-write** (a first run may download), and you must pass the *cache root* - the
directory that holds `models--<org>--<repo>/` and `blobs/`, i.e. what `hub/` is for the `hf` client.
HF stores every file as a relative symlink into `blobs/`, so mounting a `snapshots/` subfolder instead
of the root gives the engine nothing but dangling links.

The quant defaults to **IQ3_XXS**. First start, if it is not cached:

1. `hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include 'IQ3_XXS/*'` (~76 GB, resumable)
2. `tools/strata_tokenizer.py` → `<pack>/tokenizer/`
3. `tools/iq_pack.py --experts-bin` → `native_experts.txt` + `experts.bin` (~43 GB; the HIP mmap path
   needs that file, [docs/AMD_HIP.md:61-67](../docs/AMD_HIP.md))
4. the MTP draft layer (~5 GB from the original Qwen checkpoint) - optional; without it the engine
   starts with `--spec 0` and decodes slower

`--offline` turns step 1 off and fails with the command to run instead. `./run.sh --check-only`
reports what is missing without writing anything.

Space is checked before anything is downloaded, and counted honestly: when the cache and the work
directory share a filesystem, the pack's ~43 GB is added to the download's ~76 GB and both must fit.
If it does not fit, `./run.sh` stops and says so - it never starts a download that will die at 90 %.

```sh
~/Development/models        # default cache: models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/
~/Development/strata-work   # default: packs/<quant>/, mtp/rt/, logs/

./run.sh --hf-cache /mnt/storage/models --work /mnt/storage/strata-work   # keep models somewhere else
export HF_HUB_CACHE="$HOME/Development/models"   # make the hf CLI share this cache too
mv ~/.cache/huggingface/hub/* "$HOME/Development/models/"   # a cache is content-addressed: moving it
                                                             # is enough, no re-download
```

## The 10 GiB VRAM budget

The HIP engine admits allocations against a **10,240 MiB / 10 GiB ceiling**.
It rounds tracked allocations to 64 KiB and preserves at least **1,024 MiB for
runtime/library resources** and **256 MiB of slack**, so explicit allocations
can consume at most **8,960 MiB**. Allocation admission also checks physical
free memory. Managed and asynchronous pool allocations fail before allocation
because their residency cannot be tracked by this ledger.

`hipMemGetInfo` reports free memory within that quota. The launcher therefore
reserves only later explicit allocations when sizing the expert cache:
`max(700, STRATA_VRAM_LATER_MIB)`, default 768 MiB. Runtime reserve and slack
are already excluded; subtracting them twice can leave no expert cache.
The measured default uses a 2,048-token prefill chunk and an 800-slot cache
budget (971 native slots on this pack). Prefill scratch borrows cache storage;
the exact planner reduces the chunk if it cannot fit. The later allowance
covers subsequent explicit buffers; the runtime reserve remains separate.

The configured context remains **131,072 tokens**. The launcher rejects other
values and budgets above 10,240 MiB. Worker count defaults to **0**, allowing
the engine to select physical cores within its CPU affinity. Explicit settings
reach the container, including `STRATA_PREFILL`, `STRATA_POOL_WORKERS`, staging
and prefill ring sizes, and `STRATA_HIPBLASLT_TUNING`.

Verify actual memory while requests run:

```sh
python3 docker/vram-guard.py --budget-mib 10240 --interval 0.05 \
  --for 240 --output /work/logs/vram-audit.json
```

Inside the container, `--pid $(pgrep -x strata)` audits allocation totals from
AMD DRM fdinfo, deduplicating clients and filtering the selected PCI device.
It fails if attribution is unavailable.

The guard compares exact bytes and permits no over-budget tolerance. Raw card
usage includes the desktop and is a conservative check. `--others-mib` uses a
fixed subtraction and is only an estimate of process usage. Sampling does not
prove that every transient driver allocation fits inside the runtime reserve.
The allocator covers Strata's synchronous device allocations and the deployed
BLAS libraries' `hipMalloc` calls; reassess coverage and reserve when changing
runtime versions or allocation APIs. Build the updated image before using these
settings; `run.sh` rejects older inference images without the allocation-guard
label. Run `./build.sh` before launching. See [discovery results](../bench/results/2026-10-01-discover-128k-10g/README.md)
for measurements and experiment status.

## What the container is configured with

Matched to the client profile this engine is driven from (`amanda/profiles/yolo-auto__yolo.json`),
with one deliberate deviation:

| | default | why |
| --- | --- | --- |
| `--max-context` | **131072** | fixed at 128K for this deployment; other values are rejected |
| `max_tokens` | **32768** | the profile's `maxOutputTokens`; a request that names none gets it, and `fit_max_tokens` clamps it to the room left rather than answering 400 |
| reasoning | **high** | the profile's `"reasoning": "high"`. Omitting the level already behaves like high (this model's template defaults to xhigh); the server now states it, and a request naming `off` / `low` / `medium` still wins |

Measured behaviour of those three, on the live server: `n_ctx = 131072`; a request with no effort
reasons (107 chars) while `"reasoning_effort":"off"` answers in 4 tokens with none; a request with no
`max_tokens` stops at its own EOS rather than running to 32768.

```sh
./run.sh --max-context 131072                # retain 128K
./run.sh --reasoning-effort off              # serve as a non-thinking model
./run.sh --max-tokens 4096
```

## Useful environment

Pass with `./run.sh -e KEY=VALUE`. Inside the image these already have defaults.

| Variable | Default | Meaning |
| --- | --- | --- |
| `STRATA_MODEL` | `IQ3_XXS` | IQ3_S, IQ2_XS, Q2_0, IQ1_M (the Coder) |
| `STRATA_VRAM_BUDGET_MIB` | `10240` | Strata allocation ceiling |
| `STRATA_VRAM_RUNTIME_RESERVE_MIB` | `1024` | runtime/library overhead reserve; cannot be lowered |
| `STRATA_VRAM_SLACK_MIB` | `256` | headroom within the ceiling |
| `STRATA_VRAM_LATER_MIB` | `768` | VRAM bound after the cache is sized: draft head, spec window, prompt workspace |
| `STRATA_HF_CACHE` | `/hf-cache` | in-container mount point of the model cache (`~/Development/models`) |
| `STRATA_WORK` | `/work` | in-container mount point of the work dir (`~/Development/strata-work`) |
| `STRATA_DOWNLOAD_MODEL` | `1` | `0` = never download (`./run.sh --offline`) |
| `STRATA_AUTO_PREPARE` | `1` | `0` = check only, fail with instructions |
| `STRATA_PACK_DIR`, `STRATA_MTP` | `/work/...` | pack and draft-layer locations |
| `STRATA_MAX_CONTEXT` | `131072` | fixed 128K context |
| `STRATA_PREFILL` | `2048` | requested chunk, reduced by the exact cache-borrow planner if necessary |
| `STRATA_EXPERT_CACHE` | `800` | slot budget; native cache uses per-pair sizes within the allocation cap |
| `STRATA_POOL_WORKERS` | `0` | engine chooses physical cores within CPU affinity, reserving the host core |
| `STRATA_SPEC` | `4` | `0` without an MTP runtime |
| `HF_TOKEN` | - | only for releases that need license acceptance |
| `HIP_VISIBLE_DEVICES` | `0` | the discrete card; the iGPU (gfx1036) must stay invisible |
| `HSA_OVERRIDE_GFX_VERSION` | **unset, deliberately** | setting it breaks gfx1101 |

## Troubleshooting

* `HIP backend requires validated gfx1100 wave32 hardware` — the engine treats gfx1101 as
  unsupported. The patch set in [docs/DOCKER_GFX1101_PLAN.md Part C](../docs/DOCKER_GFX1101_PLAN.md)
  has not been applied; `./build.sh` says so before it even pulls an image.
* `no HIP device is present` — the container needs `--device /dev/kfd --device /dev/dri/renderD<N>`
  and the `video`/`render` groups; `./run.sh` works out `<N>` from the KFD topology.
* Configure fails with *Strata HIP currently supports only gfx1100 wave32* — same as the first item.
* Answers are correct but slow: check `docker logs` for
  `expert cache auto: ... -> 0 slots` (inspect the cache reserve and allocations; keep `--max-context 131072`) and for
  missing MTP (`--spec 0`).
* A test you expected to run was skipped: `ple_parity` needs an external model fixture and is
  excluded **by name** in `./build.sh --tests` - never counted as a pass.
