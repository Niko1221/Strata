# Community test: one hour of back-to-back requests on a resident RTX 5090 in a container

This answers the item `docs/TEST_REQUESTS.md` lists as *"A long soak (an hour or more of back-to-back requests) on a
split or resident setup"* (#848). It is **not** a `strata bench run` standard benchmark and does not claim to be one:
it is one hour of continuous chat traffic against a server that was already serving, with the card and the server
sampled next to the requests, so drift, heat and failures over an hour can be read. The standard benchmark from the
same machine, card and model is in
[`../2026-10-24-community-rtx-5090-container-gen3`](../2026-10-24-community-rtx-5090-container-gen3/).

## The setup

| | |
| --- | --- |
| Card | NVIDIA GeForce RTX 5090, 32,607 MiB, driver 580.142, **PCIe Gen 3 ×16** (the engine's probe: 13.9 GB/s host-to-device, `pcie_frac` 0.38) |
| Mode | **resident**: the engine's expert cache held 18,240 slots (23.48 GiB) and its own line for the hour reads `decode expert cache hit rate: 100.0%`; no expert came from the SSD |
| Context | `--max-context 65536 --kv int8 --kv-resident 32768`: the K/V of 32,768 of the 65,536 cells per QSA layer in VRAM, the rest in 0.77 GiB of pinned host RAM. Its own line for the hour: `KV streaming: 100.00% of … block reads hit VRAM` |
| Engine | `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp` (draft layer on), `--vision`, `--vram-reserve-mib 700`, one slot (no `--batch`) |
| Container | official image `ghcr.io/niko1221/strata:0.2.2-cuda13` (mirror `ghcr.nju.edu.cn`), engine `0.1.47-linux-x86_64-cuda-cuda13`, started 2026-10-24 08:09:04 and not restarted since |
| Docker | 19.03.15, default seccomp profile plus `--security-opt seccomp=unconfined` (issue #4309), `-p 0.0.0.0:8899:8080` |
| Host | Ubuntu 20.04.3, 2 × Xeon Gold 5218 (no AVX-512, so the expert kernels run on AVX-2), 125.6 GiB RAM |
| Model | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, Q2_0 |
| Server | `/health` answered `{"status":"ok","max_context":65536,"loaded":true,"images":true}` at all 119 samples |

The model files were hashed before the run: shard 1
`69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720` (37,623,740,192 bytes), shard 2
`316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` (28,800,138,432 bytes) — the values `setup.py`
pins for this model.

## The traffic

`soak.py` in this folder is the client, unchanged. One request at a time, no pause between them (the next starts when
the previous one's stream ends), `max_tokens` 192, `temperature` 0.7, one fixed prompt of 99 tokens, streaming over
loopback so the first visible delta can be timed. Every 30 s it also samples `nvidia-smi` for the card the server runs
on and the server's `/health`. `analyze.py` turns its JSONL into `summary.json`, `requests.csv`, `gpu.csv` and
`health.csv`.

Two numbers are stated apart, because they mean different things:

- **end-to-end tokens/s** = completion tokens ÷ the whole request's wall time. It includes prefill and any thinking
  the model did, so it cannot be inflated by counting hidden tokens against a short visible window.
- **TTFT** = the client's clock to the first *visible content* delta. 254 requests spent their whole 192-token budget
  in thinking and produced no visible content; their TTFT is not counted.

The client also records a per-request `decode_tps` and this report does **not** use it: it divides all completion
tokens (thinking included) by the visible-content window, which is not a decode rate. The engine's own number is
reported instead, and it comes from the engine's log.

## Results, measured by the client

| | |
| --- | --- |
| Duration | 3,599 s of traffic (60.0 min), 2026-10-27 22:18:49 – 23:18:49 local |
| Requests | **4,578 sent, 4,578 answered, 0 failed** (no HTTP errors, no timeouts, no retries) |
| Rate | 76.3 requests/min sustained, one at a time |
| Generated | 844,832 completion tokens, 14,083 tokens/min |
| End-to-end | **median 237.2 tok/s** (p10 216.8, p90 253.5) |
| Latency | median 0.792 s, p95 0.898 s, max 1.446 s |
| TTFT (visible content) | median 0.529 s, p95 0.780 s |
| Capped at `max_tokens` 192 | 2,822 of 4,578 requests (61.6%) |

## Results, measured by the engine itself

The engine's log carries a per-request summary line. **3,131 of the 4,578 requests left one**; the other 1,447 were
answered (the client saw all 4,578 replies with usage) but left no summary line, and I did not establish why. Over the
3,131 lines:

| | |
| --- | --- |
| Decode | **median 249.3 tok/s** (p10 227.4, p90 267.7, min 192.3, max 299.2) |
| Prompt reuse | 94.9% of prompt tokens came from the prefix cache (median 94 of 99 tokens) |
| Speculative drafts | 316,654 of 455,054 accepted (69.6%) |
| Warnings or errors in the lines the engine added during the hour | **0** |

The engine's decode rate and the client's end-to-end rate agree within about 5% (249.3 against 237.2), which is what
it should be: the end-to-end figure also pays the prefill and the thinking.

## Drift over the hour

| | first 10 min | last 10 min |
| --- | --- | --- |
| Requests | 761 | 765 |
| End-to-end | 236.9 tok/s | 236.5 tok/s |
| Latency median | 0.79 s | 0.79 s |
| TTFT median | 0.531 s | 0.535 s |

No drift: the last ten minutes are within 0.2% of the first ten on every figure.

## Heat and clocks

119 samples of the card, one every 30 s:

| | min | median | max |
| --- | --- | --- | --- |
| GPU utilization | 77% | **98%** | 99% |
| Temperature | 34 °C | 54 °C | **55 °C** |
| Board power | 269 W | 365 W | 373 W |
| SM clock | 2,827 MHz | 2,835 MHz | 2,865 MHz |
| VRAM used | 31,850 MiB | 31,850 MiB | 31,850 MiB |

The card sat at a steady 54–55 °C for an hour at 365 W with no clock drop, and its VRAM figure did not move at all
(31,850 MiB in every sample), which is what "resident" means here.

## Files

| File | What it is |
| --- | --- |
| `summary.json` | the aggregates above, as written by `analyze.py` |
| `requests.csv` | every request: index, wall-clock time, elapsed seconds, status, TTFT, total time, prompt and completion tokens, visible characters, end-to-end tokens/s |
| `gpu.csv` | every 30 s: utilization, memory used and total, temperature, power, SM clock |
| `health.csv` | every 30 s: the server's `/health` answer |
| `soak.py` | the client that produced the traffic, unchanged |
| `analyze.py` | the script that turned the client's JSONL into the files above and the numbers here |

## Limits

- One machine, one card, one model, one context size, one hour, concurrency 1. Nothing here says how a second slot, a
  second card or a split setup behaves.
- The client and the server run on the same host: the timings include loopback, not a network.
- Temperature and power are the card's own sensors, read once every 30 s; this report cannot describe the room.
- The engine-log figures cover 3,131 of the 4,578 requests (see above). The client-side figures cover all of them.
- `nvidia-smi` on this host lists the RTX 4090 as card 0 and the RTX 5090 as card 1. Every sample here is card 1, the
  one the container uses. The 4090 was serving another program throughout and is not part of this test.
- No answer was checked for quality. 61.6% of the replies were cut at `max_tokens` 192, and 254 replies were spent
  entirely on thinking: this measures the machine under load, not what the model said.
