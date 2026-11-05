# Installing the training tree

`training/` is not in the engine's install path: `setup.py` / `START-HERE.bat` bring the
engine up with `requirements.txt`, which is standard-library-plus-numpy and nothing else.
Training adds a second, separate set of packages. This file says which, per platform,
and what to do when one of them has no wheel for your OS.

## Which platform can do what

| platform | `--method mezo` (zero-order) | `--method backward` (autograd) | what it costs |
|---|---|---|---|
| Linux (x86-64, NVIDIA) | ✅ reference platform — everything below was measured here | ✅ | — |
| Windows 10/11 (NVIDIA) | ✅ `preadv_shim` gives the same `preadv` semantics | ✅ since `triton-windows` (see §3) | `mmap.madvise` is missing ⇒ mapped pages are not returned; ~15 GiB stays resident |
| macOS | ❌ blocked at the guard | ❌ | no engine binary for this model, and pack IO (`preadv`) was never measured here |
| Cygwin / MSYS2 | ❌ | ❌ | `sys.platform` is `cygwin`/`msys`, no `preadv`; the shim is not exercised there |

The guard is `platform_check.py`, and `train.py` calls it before it loads anything — a
blocked platform exits with code 2 and a reason, instead of `AttributeError` from
`os.preadv` 76 GB later. `python3 platform_check.py` prints the matrix above plus what it
found on your machine.

## 1. What each layer needs

| layer | third-party packages | where they come from |
|---|---|---|
| core (`mezo`, the default method) | numpy, jinja2, regex, psutil | already in `requirements.txt` — **nothing to install** |
| backward | torch, transformers, tokenizers, accelerate, safetensors, gguf, flash-linear-attention, triton | `requirements-train.txt` (§2) |
| bench only | psutil, numpy (+ the backward set for the torch benches) | as above |

So the zero-order line — the one that runs on the engine's own inference and is the
default — has no separate install. Only backprop needs this file.

## 2. Install

```bash
# 1) torch, from the index that matches your driver (pick one):
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132   # CUDA 13.2
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu     # no GPU

# 2) the rest (linux and windows take the same file; the triton line is platform-marked)
python -m pip install -r requirements.txt -r requirements-train.txt
```

Python 3.10+ (the wheel you can get is the real constraint: for 3.13 the CUDA 13.2 index
above has wheels; for 3.14+ check the index before pinning).

## 3. The one package without a Windows wheel: triton

`backward/nf4_fused.py` decodes the NF4 expert weights inside a Triton kernel, and
`loader_qwen4.py` imports it unconditionally, so backprop needs `import triton` to work.

- **Linux** — nothing to do. The torch wheel depends on triton and pip installs it.
- **Windows** — upstream triton publishes no win32 wheel. `triton-windows` is the same
  `triton` module compiled for Windows; it is what `requirements-train.txt` pins, and it
  installs from PyPI. If PyPI does not have the build for your torch (new torch, old
  PyPI), the wheels are on the project's Releases page —
  `github.com/woct0rdho/triton-windows/releases` — and its issue tracker carries the
  build for each torch release:

  ```powershell
  # if pip cannot find a matching version, download the .whl from Releases, then:
  python -m pip install C:\path\to\triton_windows-3.9.0.post30-cp313-cp313-win_amd64.whl
  ```

  The version before `.postN` is the triton release being patched. It must match what
  torch was built against — bumping torch ⇒ bump this line in the same commit, or
  `torch._inductor` will reject the version at the first compile.

Check it in one line: `python -c "import triton, torch; print(triton.__version__, torch.__version__)"`.

## 4. Verify

```bash
python3 platform_check.py --deps     # module by module: present/missing, which layer, install command
python3 platform_check.py            # platform + torch/CUDA, exit code 2 when blocked
python3 train.py --list              # algorithms, methods this branch can run, datasets
python3 train.py --dry-run           # the command it would run, nothing loaded
```

`--deps` only reads `sys.path` and the installed-metadata (`find_spec` + `version`), so it
never imports torch — it costs nothing and it works in CI's no-torch job.

## 5. The combination the numbers were measured on

Windows 11, RTX 4090 48 GB, Python 3.13.12, torch 2.13.0+cu132, triton-windows
3.9.0.post30, transformers 5.17.0, tokenizers 0.23.3, accelerate 1.14.0, safetensors
0.8.0, gguf 0.19.0, flash-linear-attention / fla-core 0.6.0. Engine: the CUDA 12 build
from the signed feed (see `docs/INSTALL.md` for the engine side, which has its own,
unrelated requirements).

Two facts that bite on Windows and are not installation problems:

- `set PYTHONUTF8=1`. Without it, on a Chinese-locale Windows, torch's inductor reads
  Triton/CuteDSL templates with the GBK codec ⇒ `UnicodeDecodeError` in the `fla` import
  chain ⇒ transformers silently falls back to the torch GDN reference ⇒ 60 GiB
  allocated and OOM within 20 steps. `platform_check.py:ensure_utf8` warns, and
  `warn_if_gdn_fell_back` checks the result instead of the cause.
- No `madvise`: mapped GGUF pages are not returned to the OS. It is a degrade (≈15 GiB),
  listed by `platform_check.py:degrades()`, not a blocker.
