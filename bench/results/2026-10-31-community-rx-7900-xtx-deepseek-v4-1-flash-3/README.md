# Community benchmark on AMD Radeon RX 7900 XTX

Measured on 2026-10-31 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GiB VRAM
- CPU: Intel(R) Core(TM) i7-14700KF (20 cores, 28 threads); installed RAM: 62.6 GiB; storage: sda     ST2000DM008-2FR102           sata; zram0; nvme1n1 Samsung SSD 970 EVO Plus 2TB nvme; nvme0n1 Samsung SSD 990 EVO Plus 1TB nvme
- OS: Linux 7.3.0-rc6-2-cachyos-rc; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.54, commit 870aedd9; engine ds4-0.1
- Background workloads: not recorded (GPU power limit/draw read above)

Added by hand, because the tool could not read it:

- GPU driver: `amdgpu`, the module that ships with the running kernel (7.3.0-rc6-2-cachyos-rc, `amdgpu.ko.zst`, srcversion `807FFCD9D65F5CD6B23E554`, no `version` field). Neither `amd-smi version` nor the `DRIVER:` block of `amd-smi static` names a version for it (`amdgpu version: N/A`, `BOOTUP_DEFAULT`), and the card ships no EEPROM version (`EEPROM_VERSION: N/A`); card firmware `022.001.002.034.000001`. The `rocm-smi` on the PATH belongs to a second, separate ROCm install at `/opt/rocm` (release `7.2.4`, HIP `7.2.53211`); its `--showdriverversion` answers `Driver version: 7.3.0-rc6-2-cachyos-rc` - the kernel release, not an AMD package version. The `not read` line above is how the tool records it: on the AMD path each card is stored with `driver: None` (`tools/strata_bench.py:511`), and the machine record prefers that per-card value over the `rocm-smi` answer it also collects (`:716`).
- ROCm: the install the engine was built with is a TheRock install at `/home/cayaf/therock-tarball/install`, release `10.2.0`: its `.info/version` file says `10.2.0`, and its `amd-smi version` prints `AMDSMI Tool: 27.1.0+105dd4ff | AMDSMI Library version: 27.1.0 | ROCm version: 10.2.0`. The components inside it carry their own numbers - `include/hip/hip_version.h` and `hipcc --version` say HIP `7.17.26402-0000000`, the compiler is AMD clang `24.0.0git (ROCm/llvm-project 7667f52dd687)`. The build cache names that root (`CMAKE_HIP_COMPILER_ROCM_ROOT`, `--rocm-path=` on the same folder), and `ldd` on `engine/strata-dsv41` resolves `libhipblaslt.so.1` and `librocm_kpack.so.0` from its `lib`. It is not on the PATH; the `/opt/rocm` install above supplies the `rocm-smi` that is. There is no CUDA on this machine and no `nvidia-smi`.
- Card, from `amd-smi static`: `gfx1100`, `PCIE_INTERFACE_VERSION: Gen 4`, `SOCKET_POWER_LIMIT: 399 W` (the slider allows 294-402 W and was left alone), 24560 MiB of VRAM.
- Engine: built by hand on this machine, unsigned, not from the update feed - `strata-dsv41 0.1.54 build 870aedd9e`, gfx1100 only, from a checkout of `main` at 870aedd9e with one local change in `CMakeLists.txt`: the `-include` of `include/strata/dsv41/hip_extra.h` written as one joined option instead of `-include <path>`, which a checkout under a path with a space needs (without it the build stopped with `cannot specify -o when generating multiple output files`). `$ROCM` is the ROCm root:

  ```text
  cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON \
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=third_party/llama.cpp \
    -DCMAKE_HIP_COMPILER=$ROCM/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM \
    "-DCMAKE_HIP_FLAGS=--rocm-path=$ROCM --rocm-device-lib-path=$ROCM/lib/llvm/amdgcn/bitcode" \
    -DCMAKE_PREFIX_PATH=$ROCM
  cmake --build build-hip --target strata-dsv41 -j 6
  ```

  The tree already held `STRATA_PREFILL_MMQ=ON`, `STRATA_PORTABLE=OFF` and `GGML_BMI2=OFF` from the Qwen build made in it the day before, so only the DeepSeek objects were new. The binary was checked by running it (`--version` prints the version and build id, `ldd` finds every library, and `grep -o gfx1100` on it finds 21 code-object markers); nothing about its correctness was tested by that.
- Background: nothing else used the GPU while the suite ran. This is read off times, not recorded live: the suite started its own server at 10:50:05 and answered on port 8080, and the install's other server (the Qwen engine, same port 8080) is a process started at 11:11:47 - after the suite's last request, which fits its 1244 s from 10:50:05. Had that other server been serving, the suite's server could not have taken the port.

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-dsv41 --pack deepseek-v4.1-flash-ist-das-1.75bpw --max-context 65536 --eos 1 --vision on --dspark auto --dspark-conf 0.65
```

Added by hand:

- Set and files. The 1.75 bpw set is IQ1_M, called `ISTA-DAS-1.75bpw` by the setup: four shards from `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` (39358830464 + 39176113248 + 39997334720 + 2188841824 B, 120.7 GB together) and the shared `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (110595290816 B) with its meta file. `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (7976914240 B) is in the same folder and was never loaded, see below. Hub revision: not recorded; every file has a `.done` sidecar holding its SHA-256. The pack is `deepseek-v4.1-flash-ist-das-1.75bpw` (`ds4.txt`, `tensors.txt`, `engram.bin`, `tokenizer/`, `expert-profile.bin`), made by the setup at 10:21 on the morning of the run.
- Storage. Model, pack and conversation folder are all on `/dev/nvme1n1p1` (btrfs), a Samsung 970 EVO Plus 2 TB, which is also the drive the install sits on. No second drive and no network storage.
- Where the experts went, from the engine's own start-up lines in `engine-session-log.txt`: 1896 of the 15360 routed experts sit in VRAM (14.48 GB) next to the dense weights (5.86 GB) and the caches (0.06 GB), with 568 MiB free; the other 13464 (100.8 GB) are read in place from the model's files over the PCIe link on 20 threads; the RAM store is 43.7 GB, 5015 slots of 8.7 MB, and it opens full with the 5015 hottest experts (34.8 GB, read in 10.8 s). KV is fp4 with all 65536 positions in VRAM (`kv_resident 65536`), and pictures are on with the engine's own reader on the GPU (318 MiB). `ds4: tensor parallel 2 -> 1` is the engine picking the split one card allows, not a fault.
- Drafting off, three ways: the start-up line `DSpark not loaded: the routed experts (115 GB) are far more than the GPU holds (25 GB) ...`, the server's own report `drafting: off`, and the session log, which has no `drafts ... kept` line for any of its 40 requests (the 36 measured ones and the 4 warm-ups), and `drafts_offered: 0` for each in `results.json`. Every number here is therefore the 1.75 bpw set with no draft decoding, on the `--dspark auto` decision.
- Experts per token: `--experts` was not passed and the engine prints no line for it. `docs/DEEPSEEK41.md:121` puts the Faster default (5 experts) at 16 GiB of total graphics memory or less, so this 24 GiB card should have run the full 6. Read from the docs, not measured.
- The hot-expert profile. For this set the setup copies the shipped `data/dsv41/profile-2.00.bin` into the pack as `expert-profile.bin` (`setup.py:783`, `setup.py:8466-8467`), and the start-up line says the RAM store filled from it (`RAM store: 5015 of the profile's hottest experts read in`). The engine then rewrites that file with its own counts after each request and once more at the end (`save_profile` in `src/dsv41/ds4_model.cu`, called from `src/dsv41/ds4_main.cpp`; `--profile-readonly` was not used), so the pack now holds a 61440 B file - 40 rows of 384 experts, md5 `148bc84cc62688d86edddee18dcee06f` - where the shipped file is 66048 B, 43 rows, md5 `4aef7ba0400bd3d81143a7e14820aea7`; the loader sizes the file as (model layers + DSpark layers) x experts (`src/dsv41/ds4_model.cu:2881-2890`), so the three rows this run dropped are the DSpark rows it had no use for. A re-run on this box now seeds the RAM store from what this run learned, not from the shipped profile.

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

Added by hand:

- Conversation cache when the measured session started: `<strata>/ds4-slots` already held 8 conversations from earlier runs (0.11 GB), and none of them helped the fresh tests - each one logged `0 reused`, which is what the unique-tag prompts are for. The session saved its own and left 21 conversations in that folder, so a later run here starts with a warmer cache than this one did.
- The engine starts that are not in this table. The log for this pack has ten starts that morning. Four exited at once with `unknown option --vram-reserve-mib`: `strata-dsv41` has no such option, only `--reserve-mib` (`src/dsv41/ds4_main.cpp:104`), while the server suggests `--vram-reserve-mib 3072` to an AMD card that also drives a Linux desktop (`serve/server.py:8037-8054`, a note that changes nothing by itself). Three came up to its `maya ds4 serve: GEN ...` line, served nothing, and were replaced by the next start a few minutes later. Two ran the suite at 8192-token context and are the folders beside this one, `2026-10-31-community-rx-7900-xtx-deepseek-v4-1-flash` and the same name with `-2`: 199 s and 235 s, chat and 4K only, because 16K, 32K and the agent session do not fit 8192. They are not part of this report. The measured session was the tenth, at 65536 and with no reserve flag.
- Concurrency was not run (`clients: 0` in `results.json`), so nothing here speaks about two or more requests at once.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 10.0 (8.0-12.3) | 2.14 (1.52-3.86) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 164.6 (153.8-166.1) | not measured | 25.07 (24.84-26.83) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 3 | 282.3 (243.9-332.8) | not measured | 58.69 (49.79-67.91) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 3 | 252.7 (203.6-261.5) | not measured | 131.12 (126.71-162.72) |
| agent session, follow-up turns | 12661 (12654-12667) at the last turn | see below | up to 64 | 3 | see below | 14.3 (8.3-19.7) | 19.23 (15.41-23.21) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.6-69.6); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 11796 (11796-11796) new tokens of the files are read once); first turn (nothing cached) first token: 15.45 s (15.29-17.14).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 39.4 | 15.295 | 14.1 |
| 1 | 3340 | 640 | 2700 | 131.9 | 20.516 | 8.3 |
| 2 | 4851 | 3397 | 1454 | 74.4 | 19.559 | 15.5 |
| 3 | 7957 | 4909 | 3048 | 141.7 | 21.528 | 18.2 |
| 4 | 10038 | 8017 | 2021 | 108.2 | 18.69 | 13.0 |
| 5 | 12667 | 10089 | 2578 | 135.6 | 19.036 | 14.1 |

The engine also prints, on each of its request lines, what share of that token's routed experts came over the PCIe link from the CPU lane (`engine-session-log.txt`): short chat 4.2-19.0 % (median 13.6), 4K 5.0-9.6 % (6.5), 16K 1.7-3.0 % (2.1), 32K 1.0-1.3 % (1.1), agent follow-up turns 1.6-1.8 % (their first turn 1.7-1.8 %). The short prompts lean on the CPU lane and on the SSD the most; from 16K up, nearly every expert a token needs is already in VRAM or in the RAM store. The one line above 20 % (47.2 %) is the 27-token warm-up request, which the suite discards. The decode column prints `not measured` on the prompt rows because the report writes that word there for them (`tools/strata_bench.py:882`); the engine's own numbers for those 24-token tails are in `results.json` (`decode_tps`, source `engine`: 4K 7.4-18.1, 16K 16.3-20.5, 32K 5.9-12.3 tok/s) and are as noisy as 24 tokens allows.

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server). All 66 lines that engine session wrote - start-up choices, warm-ups, the 40 request lines with their CPU-lane shares, and the conversations it saved - in `engine-session-log.txt`, added by hand from the install's own run log for this pack.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 1244 s.

Added by hand:

- No correctness work of any kind: no needle test, no coding test, no answer compared against another quantization or another card. A fast wrong answer gives exactly this table.
- Not run on this box: the 2.00 and 2.25 bpw sets; `--dspark on` forced (`auto` chose off here); `--experts` chosen by hand; any context but 8192 (the two earlier reports) and 65536 (this one); 128K and above; two or more requests at once; a real picture, since the reader was loaded and on the GPU but the suite only sends text; a second conversation slot or `--parallel`; `--ssd-cache`; a desktop taking VRAM while the model runs, which is the case `--vram-reserve-mib` is about.
- One box, one card, one hand-made binary. Strata's HIP packages build `strata-dsv41` for the rx9070 variant and leave it out of the others, the rx7900 line included (`packaging/build_hip_linux.sh:14-15, 41`), so this is a hand-built DeepSeek engine on a card the DeepSeek package does not cover. It says nothing about the shipped binaries on gfx1100, and nothing about other cards.
- Nothing was thrown away: `skipped` is empty, `failures` is empty, `results.json` keeps every measured request with its prompt tokens, reused tokens, both timings and its draft counters, and `engine-session-log.txt` has the same requests in order.
