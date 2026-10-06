# Windows HIP vision on RX 6900 XT

On 2026-10-07 (Asia/Tokyo), the installed Strata 0.1.40 image encoder was
changed from CPU execution to HIP execution on a physical RX 6900 XT. The
server was restarted at `http://127.0.0.1:8081`. Two distinct images passed
recognition through Chat Completions and Responses. The official language
engine and existing Huihui Swift IQ3_XXS model were retained.

## Source and build

- `tools/vision/CMakeLists.txt` now accepts `STRATA_VISION_HIP=ON` and rejects
  selecting CUDA and HIP together.
- `tools/vision/build_windows_hip.bat` builds a separate executable using
  ROCm Clang and Visual Studio's Windows SDK/linker. It does not deploy it.
- The encoder passes an explicit GPU device to mtmd, logs its backend and
  hardware name, and fails before `READY` if GPU selection or warm-up fails.
  CPU mode masks both CUDA and HIP devices before backend initialization.

The build used the unchanged release-pinned llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, ROCm `10.2.0a20260930`, Clang 24,
Visual Studio 2022, Ninja and CMake. The private `.rocm-win` environment
matches the official engine's ROCm runtime. Compile and link rules both
contained `--offload-arch=gfx1030`. HIP VMM was disabled; the CPU kernels
use the portable AVX2 baseline.

```bat
set STRATA_HIP_ARCHS=gfx1030
set LLAMA_DIR=C:\path\to\llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d
tools\vision\build_windows_hip.bat
```

The installed encoder SHA-256 is
`0abe56ac988a727849602e652ed6501a4b023ba837bdfbcb3d8e705f692c29ed`.
The main engine still matches the official 0.1.40 executable SHA-256:
`e7605caf89b939e262ff9c0ca763aa99493277c9fa61d3cc9ee630f902b9ff96`.

## Deployment and execution evidence

Hardware: Windows 11 Pro build 26300, Threadripper 3990X, RX 6900 XT
(gfx1030, 16 GiB), AMD driver `32.0.21045.5002`.

The config, BUILD metadata and CPU executable were backed up in
`logs/windows-hip-vision-20261007/`. The previous server was idle when
stopped. The only model-config change was `vision.gpu: false -> true`;
the mmproj, 300-image-token cap, 64 CPU helper threads, 131072 context,
loopback binding, port and main-engine arguments were preserved.
`engine/BUILD.json` records the local HIP encoder separately from the
official engine.

The restarted encoder logged:

```text
strata-vision: GPU backend ROCm, device ROCm0 (AMD Radeon RX 6900 XT)
strata-vision: model files loaded in 2.8 s
strata-vision: warmed up at 289 image tokens in 1.2 s
```

Windows reported 1,306,030,080 bytes of dedicated GPU memory for the live
encoder process at the captured startup snapshot. Its loaded module paths
pointed to `engine/amdhip64_7.dll`, `engine/amd_comgr.dll`, and the bundled
`engine/rocm/bin` hipBLAS/rocBLAS libraries. The main engine then loaded,
adjusted its automatic expert cache to the available VRAM (2142 slots),
and reported `session is up (engine 0.1.40)`.

## Checks

| Check | Result |
| --- | --- |
| HIP build, followed by an incremental rebuild | Passed |
| GPU requested with `HIP_VISIBLE_DEVICES=-1` | Exit 1 with explicit error; no `READY` |
| Same HIP executable in CPU mode, external GPU visibility set to 0 | HIP reported no visible device; CPU returned valid, finite embeddings |
| Standalone GPU encoder | RX 6900 XT, successful warm-up; valid finite 240-token, 2560-wide embeddings |
| `/health`, `/props`, `/v1/models`, web root | Loaded, images enabled, Strata 0.1.40, expected model, HTTP 200 |
| Text chat | Exact `HIP vision ready.`; normal stop |
| Image Chat Completions | Correct red circle, blue square and `HIP2026`; normal stop |
| Image Responses | Correct red circle, blue square and `HIP2027`; completed |
| Final health and source whitespace check | Passed |

The [Chat image](vision-chat.png) and [Responses image](vision-responses.png)
are distinct 640 x 384 fixtures. The prompt asked for shapes and printed
text without supplying the expected answer. Different image contents
ensure both API checks exercised encoding instead of sharing cached image
embeddings. Both responses reported zero cached prompt tokens.

Requests used temperature 0, reasoning effort `none`, and an output limit
of 128 tokens for images. The 300-image-token cap was intentionally retained.
The upstream encoder warns that grounding tasks need at least 1024 image
tokens; fine-detail OCR and grounding accuracy were not validated. Single
standalone encode timings in the evidence are functional observations,
not a throughput benchmark. Other GPUs, long-context images and sustained
load have not been tested.

This is a local source-built encoder. The stock installer and official
release packaging were not changed. Raw build/test scripts and logs remain
in `logs/windows-hip-vision-20261007/`; [verification.json](verification.json)
contains the runtime responses, backend logs, DLL paths and memory snapshot.
No push was performed.
