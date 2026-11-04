# Community benchmark on Radeon 8060S (Strix Halo laptop, Windows, 96 GB carve-out): Qwen3.6-35B-A3B against llama.cpp, Qwen3.8-Flash-Next Q2_0 and DeepSeek-V4.1-Flash

Measured on 2026-11-03 and 04 by @pdmaclennan-art, with an AI assistant (Claude Code) running the commands; every number below comes from the
files in this folder and in the two standard-v1 runs beside it. Strata 0.2.10 (engine 0.1.56) on Qwen3.6-35B-A3B UD-Q4_K_XL with setup's
own settings, against llama.cpp (a gfx1151 ROCm build) on the same file, same machine, same day; then Qwen3.8-Flash-Next Q2_0 with setup's
own settings through the same tests ([below](#qwen38-flash-next-q2_0)), and DeepSeek-V4.1-Flash 2.00 bpw the next morning
([below](#deepseek-v41-flash-ist-das-200bpw)). Main limitation: one run per configuration in the comparison (the
standard-v1 runs have three repetitions per case).

## Hardware and software

- ASUS ProArt PX13 (HN7306EAC) laptop: Ryzen AI Max+ 395 (16 cores, 32 threads), Radeon 8060S (gfx1151), 128 GB LPDDR5X with a
  **96 GB BIOS carve-out** for the GPU, so Windows sees 31.6 GiB (34 GB). Storage: Sandisk PC SN5100S 1 TB NVMe.
- Plugged in; ASUS Armoury Crate Performance mode; Windows power plan "Performance".
- Windows 11 Pro build 26300; AMD driver 32.0.22032.6002 (2025-12-15), older than the 32.0.32015.7029 that
  `docs/STRIX_HALO_WINDOWS.md` was measured on.
- Strata 0.2.10 release: Terminal and engine 0.1.56 (`windows-x86_64-hip-rdna3`, commit `c5ab5d40`) from the signed feed through
  `install.ps1`. Release binaries, no source build.
- llama.cpp for the comparison: [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) `93e6d0285` (build 11957),
  built for gfx1151 on ROCm 10.2.
- Background: another local model server stopped for the whole run; a desktop session and a text-to-speech server stayed up
  (about 25 GB of committed memory, mostly paged out).

## Model and configuration

- [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) @ `5bc3e238`, `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
  (setup's pinned file, SHA-256 checked by setup). No vision.
- Setup: `strata install --family qwen36 --model UD-Q4_K_XL --context 131072 --vision no --parallel 1 --yes --no-start`.
  Engine arguments it wrote: `--native <gguf> --max-context 131072 --prefill auto --mtp 2`. Start log: every routed expert in VRAM
  (18.75 GiB), prompt chunk 4096, prompt experts on MMQ (int8), hipBLASLt tuning table (32 rows, gfx1151, version 100500).
- "No MTP" row: the same config with `--mtp 0`.
- llama.cpp: `llama-server -m <same gguf> -ngl 99 -fa on -b 2048 -ub 2048 -c 90000 -np 1 --jinja --no-warmup`. The "Q4_K_M" row uses
  Unsloth's `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (the non-MTP repository) on the same server.

## Method

- **Standard-v1:** `strata bench run --profile <setup's config> --track default`. Its report is in
  [`../2026-11-03-standard-radeon-8060s-20261103T093522Z-224ddae3/`](../2026-11-03-standard-radeon-8060s-20261103T093522Z-224ddae3/README.md).
- **Comparison harness** (my own script, stdlib Python): one fresh server per configuration, model loaded before the clock starts (for Strata,
  after `/health` reports `loaded`). Streaming `/v1/chat/completions`, temperature 0, **`presence_penalty: 0` on every request** (llama.cpp's
  default; without it Strata applies the model card's 1.5), thinking on unless stated. Time to first token and decode are measured by
  the client: decode = (completion tokens - 1) / (last chunk - first chunk), completion tokens from `usage`. In order:
  1. an 18,023-token prompt, 128 new tokens;
  2. a second turn: the same messages plus the answer and a new question, 128 new tokens;
  3. needle-in-a-haystack at 24K and 80K tokens (thinking off);
  4. fresh prompts of 4K, 16K, 32K and 64K tokens, 256 new tokens each, every one starting with a different line so none reuses another.

  The prompts are private Markdown notes, so they are not attached; their token counts are in `results.json`. One run each.
- **Prompt-reuse probe** (Strata only, setup's config): six requests in a row on one server, each one's `usage.prompt_tokens_details.cached_tokens`.
- **Memory:** Windows' per-process GPU counters for the engine process, and available RAM and system commit after the second turn.

## Results

**Standard-v1** (temperature 1.0, top_p 0.95, top_k 20, presence 0, thinking on; three repetitions; sanity **12 of 12**):

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: |
| short | n/a | 53.3 (52.8-53.9) | 0.20 |
| prompt-4k | 730.6 (728.7-732.4) | 46.1 (46.1-46.2) | 5.58 |
| prompt-16k | 589.9 (588.4-590.8) | 29.9 (29.9-29.9) | 27.73 |

**Comparison, 18K prompt** (one run each):

| Configuration | Prompt tokens | TTFT s | Prompt tok/s (client) | Decode tok/s | Second turn s (its TTFT) | Engine GPU GB | Free RAM GB | Needles 24K / 80K |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| llama.cpp, UD-Q4_K_M | 18,023 | 9.3 | 1,938 | 44.1 | 3.3 (0.5) | 25.9 | 7.7 | pass / pass |
| llama.cpp, UD-Q4_K_XL (same file as Strata) | 18,023 | 8.9 | 2,025 | 43.4 | 3.4 (0.4) | 26.1 | 11.5 | pass / pass |
| llama.cpp, UD-Q4_K_XL, `--spec-type draft-mtp --spec-draft-n-max 2` | 18,023 | the server exited after reading the prompt (llama.cpp's problem, not Strata's) | | | | | | |
| Strata, setup's settings (MTP 2) | 18,023 | 31.9 | 565 | 30.1 | 35.5 (32.0) | 35.0 | 0.5 | pass / pass |
| Strata, `--mtp 0` | 18,023 | 30.2 | 597 | 28.7 | 35.0 (30.6) | 35.0 | 0.4 | pass / pass |

**Comparison, fresh prompts**, TTFT s / decode tok/s, 256 new tokens:

| Prompt tokens | llama.cpp UD-Q4_K_M | llama.cpp UD-Q4_K_XL | Strata MTP 2 | Strata `--mtp 0` |
| ---: | --- | --- | --- | --- |
| 4,033 | 2.2 / 47.7 | 2.1 / 51.6 | 5.5 / 67.2 | 5.3 / 49.0 |
| 16,034 | 7.8 / 43.2 | 7.7 / 50.0 | 27.2 / 35.7 | 26.1 / 31.3 |
| 32,036 | 18.3 / 46.5 | 18.1 / 42.2 | 69.7 / 12.8 | 66.1 / 14.3 |
| 64,036 | 48.5 / 38.4 | 48.6 / 39.9 | 212.3 / 6.4 | 198.0 / 7.6 |

Strata's own timings for the MTP run ([engine-log.txt](engine-log.txt)): prompt 733 tok/s at 4K, 591 at 16K, 460 at 32K, 302 at 64K, 265 at 80K.

**Prompt-reuse probe** (Strata, setup's settings, 18K-token system prompt, up to 200 new tokens, `reasoning_budget_tokens: 100`):

| Request, in order | Prompt tokens | Cached tokens | Seconds |
| --- | ---: | ---: | ---: |
| 1. fresh, thinking off | 18,022 | 0 | 33.3 |
| 2. the same request again | 18,022 | 0 | 33.2 |
| 3. follow-up turn (request 1, its answer, a new question), thinking off | 18,082 | 0 | 33.6 |
| 4. same system prompt, a different question | 18,024 | 0 | 34.3 |
| 5. same system prompt, thinking on | 18,022 | 0 | 36.2 |
| 6. follow-up turn of request 5 (answer without its reasoning, as agent clients send it), thinking on | 18,066 | 0 | 67.9 |

The engine log agrees. Of its 54 requests that day, one reused anything: the server's own continuation after the thinking budget of
request 5 (`prompt 18140 tokens = 18122 reused + 18 read`), which sends back exactly the tokens the engine generated. The same
continuation for request 6 read all 18,184 tokens again (`0 reused`), which is why request 6 took 67.9 s. On llama.cpp the second
turn of the comparison started in 0.4-0.5 s.

## What I see

1. **Prompt reading** on this path is 565-733 tok/s up to 18K and falls to 265 at 80K; llama.cpp reads the same file at about 2,000.
2. **Decode falls with context**, as in #2464: 67 -> 36 -> 13 -> 6.4 tok/s at 4K / 16K / 32K / 64K with MTP (greedy), while llama.cpp
   stays at 38-52.
3. **No prompt reuse on the Qwen3.6 path.** In `src/qwen35/q36_engine.cu`, `match_len` reuses a session only when everything it holds
   (prompt and generated tokens) is a proper start of the new prompt ("the recurrent state cannot step back"). A repeated request,
   a new question on the same system prompt and an agent client's next turn (the template drops the earlier reasoning) all read the whole
   prompt again. llama.cpp keeps context checkpoints for this model and reused the 18K prefix.
4. **MTP drafts only for greedy requests** (`mtp_d`, "greedy requests only"). At temperature 0 with no penalty: 72-98% of drafts accepted
   and 81-85 tok/s on short prompts. Standard-v1 at temperature 1.0: "drafts accepted 0 of 0" in all 22 requests. A request without
   `presence_penalty` gets the card's 1.5 and no drafts either (the tool-call checks show "0 of 0", 53-54 tok/s).
5. **Memory with a 96 GB carve-out:** every expert sat in VRAM, yet available RAM fell to 0.4-0.5 GB of the 34 GB Windows sees while Strata served
   (7.7-11.5 GB with llama.cpp), and setup warned that the model "needs about 32 GB of RAM". System commit: 110 GB of a 171 GB limit.
6. `strata bench run --profile` passes the environment to the engine: with `HIP_VISIBLE_DEVICES=-1` set user-wide (here, to keep another
   app off ROCm) the engine stops with "no ROCm-capable device is detected" (first start at 11:27 in the log). Clearing it fixed it.

## Qwen3.8-Flash-Next Q2_0

Same afternoon, same machine, same harness and prompts (the other local model server stopped again).

- Setup: `strata install --family qwen --model Q2_0 --context 131072 --vision no --yes --no-start` (the two `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0`
  files; hashes in the standard run's README). Engine arguments it wrote: `--pack <q2_0 pack> --native <file 1> --ple-gguf <file 2>
  --expert-profile <expert-profile.bin> --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <draft layer> --max-context 131072
  --kv int8 --resident-experts`, env `STRATA_RESIDENT_PIN=0`. Setup left KV streaming off ("it needs ~46 GB of RAM for Q2_0 ... this PC has 32").
  Start log: every expert in the GPU cache (24,576 of 24,576 slots, 31.64 GiB), 0 GiB of experts in RAM, conversation cache 2,304 MiB.
- No llama.cpp row on these files (not tried). For scale only: on 16 Oct, the same llama.cpp fork (as built then) ran Unsloth's
  Flash-Next UD-Q2_K_XL with its MTP head (another 2-bit file) on this machine: 18K prompt 583 tok/s, TTFT 30.9 s, decode 19.6 tok/s,
  second turn 6.0 s, both needles pass, 61.7 GB of GPU memory.

**Standard-v1**: [`../2026-11-03-standard-radeon-8060s-20261103T120237Z-8ca6387a/`](../2026-11-03-standard-radeon-8060s-20261103T120237Z-8ca6387a/README.md)
(temperature 1.0, top_p 0.95, top_k 20, presence 0, thinking on; three repetitions; sanity **11 of 12**):

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: |
| short | n/a | 48.2 (47.3-49.2) | 0.84 |
| prompt-4k | 1,021.4 (1,014.9-1,034.6) | 43.1 (39.8-47.0) | 4.08 |
| prompt-16k | 1,061.2 (1,056.7-1,062.4) | 42.2 (39.0-45.1) | 15.52 |

T2 failed: the model answered "I used `get_weather` for Zagreb with `unit: celsius`." with no tool call (finish `stop`). 7 of the 9
performance requests reached the 512-token cap and are listed as truncated, as on the 35B.

**Comparison harness** (temperature 0, `presence_penalty: 0`, one run each):

| Prompt tokens | TTFT s | Prompt tok/s (client) | Decode tok/s | Second turn s (its TTFT) | Engine GPU GB | Free RAM GB | Needles 24K / 80K |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 18,061 | 21.2 | 852 | 41.9 | 3.8 (1.2) | 47.1 | 8.5 | pass / pass |

| Fresh prompt tokens | TTFT s / decode tok/s |
| ---: | --- |
| 4,071 | 5.3 / 49.3 |
| 16,072 | 18.6 / 37.3 |
| 32,074 | 36.9 / 39.5 |
| 64,074 | 74.2 / 44.2 |

The engine's own prompt rates: 783 tok/s at 4K, 854-871 from 16K to 80K. System commit peaked at 126 GB of 171.

**Prompt-reuse probe** (same six requests as above):

| Request, in order | Prompt tokens | Cached tokens | Seconds |
| --- | ---: | ---: | ---: |
| 1. fresh, thinking off | 18,022 | 0 | 18.7 |
| 2. the same request again | 18,022 | 18,015 | 0.5 |
| 3. follow-up turn, thinking off | 18,048 | 18,031 | 1.1 |
| 4. same system prompt, a different question | 18,024 | 18,005 | 0.8 |
| 5. same system prompt, thinking on | 18,060 | 0 | 19.6 |
| 6. follow-up turn of request 5, thinking on | 18,091 | 18,055 | 1.4 |

What I see on Flash-Next:

1. **No slowdown with context**: prompt reading stays at 850-870 tok/s up to 80K and decode at 37-49 tok/s up to 64K, unlike the 35B path
   (6.4 tok/s at 64K).
2. **Prompt reuse works**: repeats, follow-up turns and a new question on the same system prompt start in about a second (conversation cache
   and checkpoints). Request 5 read everything again after thinking-off requests; I have not traced why (the template differs between the two).
3. **Drafts run at temperature 1.0**: 50-88% accepted per request in standard-v1 (66% overall), where the 35B path drafts only for greedy requests.
4. Tool calls (write, then read after the result) pass; `reasoning_budget_tokens: 128` holds; with no limit the server ended the thinking
   itself after 2,253 tokens and continued the answer.
5. With every expert in the 96 GB carve-out, 8.5 GB of the 34 GB Windows sees stayed free.

## DeepSeek-V4.1-Flash IST-DAS-2.00bpw

Next morning (4 Nov), same machine, same harness and prompts, the other local model server stopped.

- Setup: `strata install --family deepseek41 --model IST-DAS-2.00bpw --yes --no-start`. Download and pack fine; setup's test question
  failed. With setup's settings the DeepSeek engine (`strata-dsv41.exe` 0.1.56, build `06698c674`) crashes while loading:
  `cudaMemcpy(x.vram_tab, ...): unspecified launch failure`. It runs only with `--vram-experts 0.35` or less (30 GiB of experts on the
  GPU); 0.4 and above crash. Details and every setting tried: [#5181](https://github.com/Niko1221/Strata/issues/5181); the per-setting lines are in [engine-log.txt](engine-log.txt).
- Served config: setup's (`--max-context 131072 --eos 1 --vision on --dspark auto --dspark-conf 0.65 --prefill-chunk 16384`, env
  `DS4_ASYNC_T=4`) plus `--vram-experts 0.35`: 24.33 GB of experts (2,330 of 15,744) and 6.58 GB of dense weights on the GPU, 13,414
  experts (115.7 GB) on the CPU lane, read in place from the SSD. Setup was re-run with `STRATA_DSV41_NO_SMOKE=1` to write it.
- No standard-v1 run and no role checks: at 3-4 tok/s each would take about an hour.

| Prompt tokens | TTFT s | Prompt tok/s (engine) | Decode tok/s (client) |
| ---: | ---: | ---: | ---: |
| 44 | 11.7 | | 4.3 |
| 3,937 | 27.6 | 137-143 | 3.4 |
| 15,723 | 52.2 | 303 | 2.8 |
| 17,358 (the 18K prompt) | 51.0 | 341 | 2.6 |
| 31,130 | 69.5 | 451 | 3.6 |
| 62,427 | 124.1 | 506 | 3.7 |

Second turn 65.0 s (10.1 s to the first token: 17,357 tokens reused). Needles at 24K and 80K: pass. Engine decode 3.1-4.5 tok/s, against
the 10-16 the DeepSeek docs give for this chip with a 64 GB carve-out; with most experts on the SSD, decode is where it loses. Free RAM
0.2-0.3 GB while it served, system commit 114 GB of 171.

## Correctness and limitations

- Needles at 24K and 80K: pass on every configuration. Tool calls (a `write`, then a `read` after the tool result): pass on llama.cpp and
  Strata. `reasoning_budget_tokens: 128` closed the thinking at 128 tokens on Strata; `reasoning_budget` (llama.cpp's name) was ignored.
  Standard-v1 sanity: 12 of 12.
- One run per configuration in the comparison; no interleaving. Medians and ranges exist only for standard-v1.
- A 96 GB carve-out with 31.6 GiB seen by Windows, not the 64 / 64 split `docs/STRIX_HALO_WINDOWS.md` measured; an older driver than the
  tested one. Results may differ on a small carve-out.
- Flash-Next has no llama.cpp row on the same files, and Q2_0 is only one of setup's four sizes. DeepSeek ran only with a third of the
  GPU for its experts, so its numbers are not what this chip can do once the crash is fixed.

Files: [results.json](results.json) (every row of the comparison and both probes, with the role checks), [engine-log.txt](engine-log.txt)
(the engine's start lines and one line per request for both models, home folder replaced by `<user>`).
