# The all-cards split engine on V100 / sm_70 (opt-in)

The automatic rule keeps sm_70 on the layer split because the split has only been run on sm_86/sm_120 and,
for the automatic path, requires all experts to fit in the cards. The engine itself is not sm_80-only:
`src/split/src/moe_mma.cuh` has an `FNX_ARCH < 80` path (`mma.sync.aligned.m8n8k4`, Volta's HMMA.884), and its
prompt-tile comment names the V100. It also has a host expert cache (`FNX_ARENA=1`, `FNX_EXPERT_RAM_GIB`).
`--split-engine on` now admits an sm_70 pair, marks the architecture and a longer context as untested, and uses
that cache for the experts the two cards cannot hold. Auto still starts the layer split.

## Build for sm_70

```sh
cmake -S . -B build-split -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_SPLIT=ON \
  -DSTRATA_SPLIT_CARDS=2 -DSTRATA_SPLIT_ARCH=70 \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12 \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_EXPERIMENTAL_SM60=ON
ninja -C build-split strata-split-2 split_moe_tile_test
```

The kernels compile for sm_70. With CUDA 12.9 + gcc-12 the split targets also need `m` and `stdc++` on their
link line (`src/split/CMakeLists.txt`); without them the link stops on `exp` / `std::runtime_error` even
though the newer targets also link `cublasLt`.

## Run it

`--split-engine on` (or `"split_engine": "on"` in the run config) applies the rule in
`strata_control/split_engine.py`. For an sm_70 pair where the routed experts do not fit, it keeps
`FNX_ARENA=1` and sizes `FNX_EXPERT_RAM_GIB` from the VRAM overflow (26 GiB for the UD-Q4_K_XL set on 2x32 GB;
20, 24 and 28 GiB also worked, 40 GiB left too few experts on the cards). `split_config` also passes the base
config's `--rope-scale`, so 393216 works with `--rope-scale 1.5`.

v0.2.11 added a faster prompt path for the big cards (the hyper-connection split and 8192-row passes). On this
pair it is **slower**: the extra pass buffers push 1,548-1,716 more experts out of VRAM. The sm_70 config
therefore sets `FNX_PF_HCSPLIT=0` and `FNX_PF_ROWS=4096`; that is the path measured below. The engine also
auto-moves experts to RAM when its first placement is short (`VRAM short by ...`), so a host cache that starts
at 26 GiB can become 31 GiB with the defaults.

The engine locks the 29.5 GB embedding table, so the host needs `LimitMEMLOCK=infinity` (systemd) or a
privileged container (`--ulimit memlock=-1:-1`). A host with the default 8 MiB limit stops with
`mlock of token_embd.weight failed: Cannot allocate memory`. A local `FNX_ALLOW_UNLOCKED=1` keeps the table in
the page cache for a test host whose container path is unavailable; it is not the intended deployment.

## Kernel test on the V100

`split_moe_tile_test` on the real UD-Q4_K_XL v3.1 file ran on a V100 and `tests/split/tile_check.py` compared
the tiles with gguf-py:

- Q5_1 and Q8_0 tiles: exact (max step 0).
- Q4_K and Q5_K gate/up tiles: `max|d|` 1e-4, exact ~31% under the checker's f16-step rule. This is the f16
  scale/min rounding the tile code documents (`moe_mma.cuh`: "the sub-block scale d * sc and min dmin * m are
  f16 here, rounded where llama.cpp keeps them f32, at most an ulp of f16 per weight"), not an sm70 crash.

## Measured: 2x V100-PCIE-32GB, UD-Q4_K_XL Dynamic v3.1, 393216 context

The 26 GiB host-cache configuration was tuned against the v0.2.11 prompt switches (one seed each unless
noted; 247K prompt, 200 max tokens):

| prompt path | 247K intake | host cache | short reply |
| --- | ---: | ---: | ---: |
| v0.2.11 defaults (HC split, 8192 rows) | 1383 tok/s | 31.0 GiB | 94.6 tok/s |
| HC split off, 8192 rows | 1523 tok/s | 30.5 GiB | 95.5 tok/s |
| HC split on, 4096 rows | 1655 tok/s | 26.0 GiB | 103.1 tok/s |
| HC split off, 2048 rows | 1499 tok/s | 26.0 GiB | 102.8 tok/s |
| **HC split off, 4096 rows** | **1856 / 2135 tok/s** | 26.0 GiB | **102.8 / 132.3 tok/s** |

The best two-seed run (the last row) gives generation 70.4 / 72.3 tok/s during the 247K intakes and a warm
repeat of **1.43-1.48 s** with 247,210 tokens reused, 81.7-85.4 tok/s. The pre-v0.2.11 engine measured
1871 / 1886 tok/s intake and 1.45-1.49 s / 81.7-85.5 warm; the tuned v0.2.11 path is the same speed within the
seed spread. The new defaults are the regression here, not the version.

The normal layer split on the same model measured 987-996 tok/s intake and 4.75-4.82 s on the warm repeat.
The first short prompt pays a 3.7-3.8 s warm-up, then the warm replies run at 132 tok/s.

## Limits

- Quality was not re-run: the split engine's arithmetic differs from the layer split's (upstream reports 29 of
  61 suite answers identical). The numbers above are speed only.
- Two cards and 393216 context were run. Three or four V100 cards, a host without the memlock change, and a
  shared GPU were not.
- The split engine has no clock-clamp hook: it idled at 1380 MHz / 41 W and 40 W per card, against 135 MHz /
  ~24 W with the normal engine's optional clock clamp.
