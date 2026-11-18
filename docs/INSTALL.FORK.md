# Installing this fork (Strata-WebSearch) via the Terminal

Strata's ready-made app downloads are the vanilla upstream build ([Niko1221/Strata](https://github.com/Niko1221/Strata)) and
do **not** contain this fork's web search. To run the search-enabled built-in Chat and the serve frontend, install from
this fork's source: clone branch `web-search`, then `./setup.sh` (Linux/Mac) or `START-HERE.bat` (Windows). Everything
stays in the checkout, so you can update and rebuild it yourself.

For the app install route, the full hardware table and every other option, use the official [INSTALL.md](INSTALL.md).
This page is the fork's CLI and source install only.

## What you need

The requirements are the same as Strata itself (the fork adds no new dependencies: the search backend is stdlib-only).
In short:

- **GPU:** NVIDIA RTX 20/30/40/50 or AMD RX 7000/9000 (and Radeon AI PRO), 12 GB VRAM or more.
- **RAM:** 16-32 GB for the small sizes, 48-64 GB for the large 125B quantizations you actually want.
- **CPU:** x86-64 with AVX2 (a Ryzen 7000/9000 with AVX-512 is a bit faster).
- **Disk:** ~70-80 GB free for the model, ~6 GB for the MTP layer, an NVMe SSD strongly recommended.
- **OS:** Windows 10/11, Linux, or a Mac with Apple Silicon and macOS 15 (the fork also works from source on Mac, same as Strata).

The complete hardware table is in the official [INSTALL.md](INSTALL.md#what-you-need).

## Install via the Terminal (CLI)

### Linux and macOS

```sh
git clone -b web-search https://github.com/c-OOL-io/Strata-WebSearch.git && cd Strata-WebSearch
./setup.sh       # first run: .venv, the engine build (build/, or build-hip/ on a Radeon), the model, then it serves
```

### Windows

In git-bash or a PowerShell with `git` on the PATH:

```sh
git clone -b web-search https://github.com/c-OOL-io/Strata-WebSearch.git
cd Strata-WebSearch
START-HERE.bat   # first run: .venv, the engine build, the model, then it serves
```

Double-clicking `START-HERE.bat` in Explorer works too.

`./setup.sh` / `START-HERE.bat` asks which model, what size, how much context and whether it should read pictures
(Enter = the recommended answer), downloads the model and the MTP draft layer, and starts serving on
`http://127.0.0.1:8080`. **Next time**, run `./setup.sh` (or `./run-<model>.sh`) / `START-HERE.bat` again: it starts
right away and downloads nothing twice. Close its window to stop the model.

## Building from source

Most NVIDIA cards get a ready-made engine, so nothing compiles. When setup has to build (Radeon, or a card without a
ready-made engine), it needs what the official guide needs:

- **What it needs:** `git`, a C++ compiler and `cmake` (setup installs these through `sudo apt` on Debian/Ubuntu; on
  Windows it offers Visual Studio Build Tools + the CUDA Toolkit), and the CUDA Toolkit for NVIDIA or ROCm with the HIP
  compiler for a Radeon ([AMD_HIP.md](AMD_HIP.md)). Python goes into the checkout's `.venv`.
- **After you change the code:** run `./setup.sh` again. It compiles only what changed and starts the model.
- **To update without starting:** `git pull`, then `./update.sh`.
- **Rebuild by hand:** the build folder is an ordinary CMake build: `cmake --build build --target strata -j8`
  (`build-hip` for a Radeon; the options are in [AMD_HIP.md](AMD_HIP.md)).
- **To try a pull request:** `git fetch origin pull/<number>/head:pr-<number> && git checkout pr-<number>`, then
  `./setup.sh`.

## The search

Web search is **off by default**. Turn it on per chat via the globe button, or under **Chat Settings, Web search**
(enable, engine picker, SearXNG URL, API key field, max results, page reading). Keys stay in the browser
(localStorage); nothing search-related is stored on the server. Engines: DuckDuckGo (no key), SearXNG (self-hosted),
Brave, Tavily, Serper, Exa and Perplexity. The search runs server-side in `serve/websearch.py` (stdlib-only, no new
dependencies), so it works on a headless server too.

## Check that it works

From the checkout, run the fork's test suite:

```sh
python -m unittest serve.test_websearch -v
```

All tests green means the `/search` route, the providers and the page reader are wired up correctly.
