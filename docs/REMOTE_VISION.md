# The image encoder on another PC

Strata starts the image encoder as a separate process and speaks a small line protocol to it. That
process needs no GPU of its own and is not tied to the model server's machine, so it can run on a
second PC — which takes the encoder's VRAM off the card that runs the language model.

This is for the case the [troubleshooting table](DETAILS.md) describes as a choice between two
unsatisfying options:

| today | what it costs |
| --- | --- |
| `--vision gpu` | ~1.4 GB of VRAM, which comes out of the expert cache |
| `--vision cpu` | 3–30 s a picture, and all the CPU cores while it runs |

Remote vision is the third option: **the speed of a GPU encoder at no local VRAM.** Measured below.

## What it does

The wrapper (`tools/remote-vision/strata_vision_remote.py`) is a man in the middle. Strata starts it
where it would start `strata-vision`, and it forwards the protocol to an encoder on the other machine
over one ssh connection:

```
server --stdio--> wrapper ==(one multiplexed ssh)==> remote encoder
                    |                                     ^
                    +-- the picture, and the embeddings, cross as base64 over that same connection
```

The protocol passes **paths**, not data (`ENC <image> <out>`), and the server's temporary directory
only exists on this machine — so the wrapper sends the picture over and brings the embeddings back.
Two small things cross the network per picture: the image, and the embeddings (**10.0 MB** for 1024
image tokens at `n_embd` 2560, F32 — measured, the file is `tokens × 2560 × 4` bytes). The remote
encoder stays resident: the mmproj is loaded once, not per picture.

The remote side needs no Strata install:

```
<remote_root>/build/bin/strata-vision
<remote_root>/mmproj-*.gguf
<remote_root>/models/text-vocab-stub.gguf      metadata+vocabulary only; see make_vocab_stub.py
```

The stub is why the remote host does not need the 34–55 GB of text model: `strata-vision` opens the
text model with `mp.vocab_only = true` (`tools/vision/strata_vision.cpp`) and llama.cpp returns right
after the vocabulary, so no tensor data is ever read. `make_vocab_stub.py` builds that stub — 11 MB
instead of copying a 51 GB shard.

## Measured

Reference machine: Ryzen 7 9700X (8C/16T), 62.6 GB RAM, RTX 5080 16 GB. Vision host: i7-11800H,
RTX 3060 Laptop 12 GB, over a 10 GbE LAN. Model IQ3_S at a 524288-token context, `--kv int8`,
`--vram-reserve-mib 700`; the three configs differ **only** in the image encoder, and every run used a
**distinct** picture (the server caches embeddings by the image's hash).

Engine 0.1.40.2, the release current on 2026-10-07/08 when these were taken: a table like this one wants
the three modes measured the same day on the same build, which is what the deltas below are. The encoder
binary and its mtmd path have not changed since (`tools/vision` and llama.cpp's revision are the same on
0.1.42), and this config gives `--vram-reserve-mib 700` explicitly, so 0.1.42's Windows auto-reserve floor
(#1376) does not apply to these numbers either.

One picture request, 1024×1024 (1088 prompt tokens), as the client sees it:

| encoder | median | first run | VRAM used | expert cache |
| --- | ---: | ---: | ---: | --- |
| this GPU | **1.63 s** | 2.46 s | 1383 MiB | 3112 experts, 5.96 GiB |
| **another PC** | **2.83 s** | 4.75 s | 0 | 3641 experts, 6.96 GiB |
| this CPU (8 threads) | **15.27 s** | 15.80 s | 0 | 3641 experts, 6.96 GiB |

So: **5.4× faster pictures than the CPU encoder, and 1.00 GiB more expert cache than the GPU one**
(+17% expert slots, straight from the engine's own startup line). It is 1.2 s slower per picture than
a local GPU encoder, which is the price of not spending the VRAM.

Isolated encoder timings agree: 0.182 s (RTX 5080), 1.53 s (remote, RTX 3060), 13.36 s (8 threads).
The whole-stack differences (+1.20 s, +13.64 s) match the isolated ones within 11%.

### What it does not do: make text faster

Moving the encoder off the card **does** grow the expert cache, and that is visible in the engine log
(3112 → 3641 experts). It does **not** show up as text speed. Long-context text, same config, three
modes:

| prompt | this GPU | another PC | this CPU |
| ---: | ---: | ---: | ---: |
| 97 | 167.5 / 89.7 | 172.5 / 92.3 | 167.2 / 82.3 |
| 257,116 | 3268.5 / 79.3 | 3298.8 / 83.0 | 3306.5 / 87.1 |
| 462,743 | 3029.2 / 81.1 | 3051.6 / 81.0 | 3037.1 / 81.0 |

(prompt tok/s / decode tok/s.) The spread is within run-to-run noise. The reason is arithmetic: the
expert set is ~47 GiB, so 1 GiB more of it cached moves the coverage by ~2% and the hit rate barely.

**This is a picture-latency and VRAM trade, not a text-speed optimisation.** If the card has room
for the encoder, a local GPU encoder is faster than this one.

## Configuration

The config's `vision` section points `exe` at the wrapper and turns the FIFO exception on:

```json
"env": { "STRATA_VISION_SSH": "vision-host" },
"vision": {
  "exe": "tools/remote-vision/strata-vision-remote.cmd",
  "mmproj": ".../mmproj-Qwen3.8-Flash-Next-BF16.gguf",
  "model": ".../Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
  "gpu": true,
  "remote": true,
  "max_tokens": 1024
}
```

`STRATA_VISION_SSH` has **no default**: the encoder's host is the operator's, not this project's. It
is an ssh_config alias or `user@host`; `STRATA_VISION_ROOT` gives the directory on it (default
`~/Developer/strata-vision`). Both are read from the environment, and the server passes its own
environment to the encoder, so the config's `env` block is a convenient place for them.

| variable | meaning |
| --- | --- |
| `STRATA_VISION_SSH` | **required** — the ssh target |
| `STRATA_VISION_ROOT` | the directory on it |
| `STRATA_VISION_SSH_OPTS` | extra ssh options |
| `STRATA_VISION_LIVENESS_PORT` | the port the remote watchdog probes back on this machine (default 1918) |
| `STRATA_VISION_PROBE_FAILS` | failures in a row before the encoder is reaped (default 3) |
| `STRATA_VISION_TIMEOUT` | one ENC reply (default: a fifth under the server's `STRATA_VISION_ENCODE_S`) |
| `STRATA_VISION_START_TIMEOUT` | the READY line (default: a fifth under `STRATA_VISION_READY_S`) |
| `STRATA_VISION_SSH_TIMEOUT` | one helper ssh |
| `STRATA_VISION_MUX` | ssh connection multiplexing. **Do not set this on Windows** — see below |

`strata-vision-remote.cmd` is the `exe` because the server starts it with
`subprocess.Popen(exe, stdin=PIPE, stdout=PIPE)`: a `.py` cannot be that process. The shim runs the
script with `.venv\Scripts\python.exe`, or whatever `STRATA_VISION_PYTHON` names. It was verified as
the server starts it — `Popen` can run a `.cmd`, and cmd.exe forwards the pipes untouched. Packing
the script into an `.exe` with PyInstaller also works and is not needed.

## Design notes

**The encoder must not hold the request FIFO.** A local encoder shares the GPU with the engine, so
upstream serialises encodes with the requests — an encode during a running request left that request
stuck at "reading the prompt" for good. A remote encoder shares nothing, and holding the FIFO across
the upload to a starved host froze the whole API, text requests included: one stalled encode left
every later request queued behind it. So with `"remote": true` the encode happens outside the queue.
The wrapper serialises its own encodes, and `encode_all` (#1072) is used either way.

**The server starts without the encoder.** A remote host can be off, asleep, or unreachable when the
server comes up. The server then serves text, and the first image asks for the encoder again
(`Service.ensure_vision`). Without this, a boot-order race would cost the whole server.

That retry has to fail as an **answerable request error**, not as an exception: anything that escapes
`prepare()` reaches socketserver, which logs a traceback and drops the connection, so the client is
told nothing at all. Measured before it was fixed: the client saw `status 0: An error occurred while
sending the request`. It now answers

```
400 {"error": {"type": "invalid_request_error",
               "message": "the image encoder is not available: the vision encoder did not start: ..."}}
```

in about 4 s, and the next image after the host returns succeeds on the same running server.

**The encoder is reaped when the server goes away.** The remote side runs under a watchdog that
TCP-probes the server's own port back over `$SSH_CONNECTION`: port open = the service is running;
closed for a few probes = reap the encoder. That covers a hard-killed server, a sleeping PC and a
dropped link, where no signal reaches the remote side at all. An earlier design — the wrapper
touching a heartbeat file over ssh every 20 s — died in production when one heartbeat ssh failed: the
watchdog reaped a healthy encoder while the server kept running.

**Windows: ssh multiplexing is not available.** `ControlMaster`/`ControlPath` fails with
`getsockname failed: Not a socket`, so it is off by default and `STRATA_VISION_MUX=1` is for hosts
where it works. That matters, because **one helper ssh costs 240–275 ms end to end** on Windows
(process, handshake, auth, close) and it is the whole of this wrapper's overhead:

| per picture | cost |
| --- | ---: |
| one helper ssh (`put`), one for the embeddings (`get`) | ~520 ms |
| the transfer itself (10 MB down, 200–380 KB up) | ~150 ms |
| base64 rather than raw bytes | 36 ms |

The per-image `ensure_scratch` that an earlier revision did was a third ssh and was removed: the
scratch directory is made in `preflight()` and re-made by `recover()`, and an encoder dying does not
remove it. That alone took a 1024-token picture from 1.81 s to 1.53 s. Folding the two remaining
helper ssh calls into one persistent file channel would save another ~500 ms; it is deliberately not
done here, because it adds a protocol and a state machine to the remote side for half a second.

**Residency on Windows.** The model server runs this machine close to its memory ceiling (a 47 GiB
model in RAM), and under that pressure Windows trims small background processes first. Measured in
production: a 5 MB `ssh.exe` helper authenticated, opened its session and then sat for 10 minutes
without delivering its exec request, and one encode took 1131 s while the vision host was idle and
healthy. So the wrapper asks for a working-set floor for itself and for every ssh it spawns
(`STRATA_VISION_RESIDENT_MIB`, `STRATA_VISION_HELPER_MIB`; `0` switches it off). It does not create
memory — it keeps ~100 MiB of already-resident pages from being trimmed, which was the difference
between a 2 s encode and a 19 minute one.

**Timeouts are derived from the server's.** The server waits `STRATA_VISION_ENCODE_S` /
`STRATA_VISION_READY_S` for these same two lines and kills the wrapper when its wait runs out, so the
wrapper's defaults are a fifth shorter: a slow host is then reported as an `ERR` line the server can
read, instead of looking like an encoder that died.

## Trying it

```sh
# on the vision host, once
cmake -S tools/vision -B build -DLLAMA_DIR=<llama.cpp> -DSTRATA_VISION_CUDA=ON && cmake --build build -j
# and place mmproj-*.gguf plus a text-vocab stub beside it (make_vocab_stub.py)
```

```powershell
# on the server machine
$env:STRATA_VISION_SSH = 'vision-host'
pwsh -File tools/remote-vision/test-wrapper.ps1
```

The server's side of it — start with the host down, text keeps working, the first image answers 400,
and the next image after the host comes back succeeds — was verified against a running server with a
`ProxyCommand` whose reachability is a flag file, so one server run sees the host go away and return.
`Service.ensure_vision`'s half of it is pinned by
`serve/test_server.py::RemoteVision`.

`tools/remote-vision/` also has `remote-build.sh` (builds the encoder on a Linux host) and
`remote-vision-watchdog.sh` (the reaper described above).
