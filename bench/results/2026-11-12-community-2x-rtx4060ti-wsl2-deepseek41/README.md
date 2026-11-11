# 2026-11-12: 2x RTX 4060 Ti 16 GB (WSL2), Xeon W-2145, 110 GB RAM — DeepSeek V4.1 Flash IST-DAS 1.75 bpw

Measured on 2026-11-12 by [@onkanat](https://github.com/onkanat).

**This is not a standard-protocol result.** `bench/protocols` defines one protocol
(`strata-qwen38-thinking-v1-draft.1`) and it covers Qwen3.8-Flash-Next only; there is no DeepSeek
protocol to run. So this report re-uses that protocol's performance fixtures — the same 4,093-token and
16,387-token prompts, the same 512-token cap, the same sampling from the model card — against
DeepSeek-V4.1-Flash, and measures what it finds. Numbers here are comparable with a Qwen run on the same
fixtures (both are in this folder's sibling report), not with the published standard results.

The interesting part is that this works at all: a 239 GB model on two 16 GB cards, because 95 GB of the
CPU-side experts fit in RAM and stay there.

## Hardware and software

- 2x NVIDIA GeForce RTX 4060 Ti 16 GB (sm_89, no NVLink), Xeon W-2145 @ 3.70 GHz, 110 GiB visible inside
  WSL2 (host 128 GB, `.wslconfig` ceiling 112 GiB).
- Ubuntu 26.04.1 LTS, kernel 6.18.58.1-microsoft-standard-WSL2, driver 580.97, CUDA 13.0. Windows 11 host.
- Model files on the WSL2 ext4 virtual disk (about 1.4 GB/s measured read); `dd` from the Windows drive
  through `/mnt/d` gave **124 MB/s**, which is why the model was copied instead of used in place.
- Strata app 0.2.13.1 (`628f749`), DeepSeek engine package `dsv41` fetched by the launcher (the engine
  calls itself `maya ds4 serve` in its log). No source build.
- Background workloads: none. The Qwen3.8 server that normally serves port 8080 was stopped for the run.

## Model and configuration

- DeepSeek-V4.1-Flash, ISTA-DAS **1.75 bpw**: `…-GSQ-RCO-IQ1_M` in four shards (113 GB), the shared
  Engram n-gram table `DeepSeek-V4.1-Flash-Engram-IQ4_NL` (110.6 GB), DSpark drafter `dspark-…-Q8_0`
  (8 GB), vision `mmproj-BF16` (972 MB). 239 GB in total, already on the disk since 2026-11-07.
- 131,072-token context, `kv q4_0`, vision on, `--parallel 1`, `--tune skip`. Layer/matrix sharing across
  the two cards (`--tp 2`) as chosen by the launcher.
- The launcher's own estimate for this machine: 2,592 of 15,360 experts (20 GB) on the cards, the other
  12,768 (95 GB) on the CPU, and *"that fits this computer: 110 GB of RAM keeps all 95 GB of the CPU's
  experts cached"*. After loading, `free` showed 99 of 110 GB used — the estimate held.
- Model load took **6.5 minutes**; the whole model is read from disk once at startup.

```text
strata launch --family deepseek41 --model ist-das-1.75bpw \
  --model-path /home/onkanat/ds41/models/deepseek-v4.1-flash-IST-DAS-1.75bpw \
  --data-dir /home/onkanat/ds41/data --context 128k --kv q4_0 --vision on \
  --parallel 1 --tune skip --access local --port 8081 --api-key-env STRATA_KEY --no-tui --yes
```

## Method

Three passes, all through the HTTP API with streaming, one request at a time, thinking on and preserved:

1. `std` — the protocol's performance fixtures (short, prompt-4k, prompt-16k), 3 repetitions each,
   512-token cap. This is the arm that lines up with the Qwen standard run.
2. `long` — the short fixture twice with a 1,024-token cap, to see how much reasoning arrives before the
   answer when the cap is not the binding limit.
3. `nothink` — the short fixture twice with `chat_template_kwargs: {"enable_thinking": false}`, for a
   decode baseline without reasoning.

`reasoning_content` and `content` stream deltas are counted separately, so "first answer" is the first
answer token, not the first token of any kind. Decode rate is `completion_tokens / (total − first
streamed token)`; prompt rate is the engine's own `timings` from its log. One engine session, no restart
between passes; the server had answered the launch's own smoke requests before the first fixture.

The client is in this folder (`bench-client.py`), the raw per-request records in `results.jsonl`, and the
engine's own log lines in `engine-log.txt`.

## Results

### `std` (512-token cap, thinking on, 3 runs)

| Case | Input tokens | Output tokens | Reasoning streamed | **Decode tok/s** | **First answer s** | Total s |
| --- | ---: | ---: | ---: | --- | --- | --- |
| short | 54 | 512 / 512 / 355 | 1,237 / 1,193 / 365 characters | **13.2** (12.9–13.3) | **21.0** (8.9–24.3) | 31–43 |
| prompt-4k | 3,014 | 512 / 500 / 507 | 1,126–1,225 characters | **10.0** (9.8–10.5) | **42.7** (42.6–43.1) | 66–68 |
| prompt-16k | 12,072 | 501 / 512 / 507 | 1,195–1,251 characters | **8.8** (8.2–9.1) | **64.3** (62.8–66.1) | 90–95 |

All nine requests produced an answer. Prompt ingestion, from the engine log: **4K 169.6 / 188.8 / 192.5
tok/s**, **16K 350.2 / 359.6 / 360.4 tok/s** across the three repetitions.

### `long` (1,024-token cap) and `nothink`

| Pass | Reasoning streamed | **Decode tok/s** | **First answer s** | Answered |
| --- | ---: | --- | --- | --- |
| `long` short | 1,758 / 1,302 characters | 7.5 (7.4–7.6) | **48.9** (37.9–59.8) | 2/2 |
| `nothink` short | 0 | 8.0 (7.6–8.4) | **1.5** (0.1–2.8) | 2/2 |

So the reasoning budget is visible: with thinking on, roughly 300–450 tokens of reasoning arrive before
the first answer token, which at 7–13 tok/s costs 20–60 s of silence. With thinking off, the first
answer token follows the first token in about a second.

### Rate drift during the session

Decode fell from **13.2 tok/s** in the first pass to **7.4–8.4 tok/s** in the last two, over about
fifteen minutes; the engine log agrees (last requests: 697 tokens in 94.3 s = 7.39 tok/s, 512 in 65.6 s =
7.80 tok/s). The cause was not measured. Candidates: the Engram table's page cache competing with the
expert cache in RAM (99 of 110 GB used), or heat. The ranges above are reported instead of a single
number because of this drift.

### Same machine, same fixtures: Qwen3.8-Flash-Next Q2_0 (standard protocol run)

| | Qwen3.8 Q2_0, 256K, 2 slots | DeepSeek V4.1 1.75 bpw, 128K, 1 slot |
| --- | --- | --- |
| Decode, short / 4K / 16K | **44.1 / 44.7 / 47.8** tok/s | 13.2 / 10.0 / 8.8 tok/s |
| First answer, short / 4K / 16K | 3.3 / 12.0 / 21.0 s | 21.0 / 42.7 / 64.3 s |
| Prompt, 4K / 16K | 647.6 / 1,126.5 tok/s | 170–192 / 350–360 tok/s |
| Requests that answered inside the cap | 2 of 9 (7 hit the 512 cap while thinking) | 9 of 9 |
| Footprint | 30.6 of 32.7 GiB VRAM, 66 of 110 GB RAM | ~20 GB VRAM, ~95 GB RAM experts |

Read that as: DeepSeek runs here, answers every time, and is 3.3–5.4x slower per token with roughly 6x
longer first-answer latency, using the machine's RAM instead of its cards. For interactive agent work it
is not a replacement for the Qwen pack; for batch work where latency does not matter, it is usable.

## What was not measured

- Quality: the twelve sanity checks were not run against DeepSeek (the protocol that defines them does
  not cover this model), so nothing here says how good the 1.75 bpw weights are beyond the launcher's
  label ("about 21% above the original model's perplexity … noticeably weaker on plain prose").
- Temperature, clocks, power draw, PCIe throughput, per-card memory.
- Concurrency (`--parallel 1` fixed).
- The 2.00 bpw set (a 257 GB download) against this 1.75 bpw set.
- Why the decode rate drifted; whether a fresh engine session per pass behaves like the first pass.
- Native Windows, outside WSL2.
