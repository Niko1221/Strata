# Community benchmark: Strata 0.2.8 on an RX 7900 XTX over OCuLink x4 (NAS host)

Measured on 2026-10-31 by faf-net. This is the same machine as my earlier report
([PR #2904](https://github.com/Niko1221/Strata/pull/2904), closed with the request to send new numbers for a new
version). It adds the measurements around the two `standard-v1` runs made in the same session:

- [`2026-10-31-standard-rx-7900-xtx-669d34e7`](../2026-10-31-standard-rx-7900-xtx-669d34e7/): release packages 0.2.8
  with my everyday run config (`--track custom`), complete, 12/12
- [`2026-10-31-standard-rx-7900-xtx-93cf5301`](../2026-10-31-standard-rx-7900-xtx-93cf5301/): the unchanged
  `0.2.8-rocm-gfx110x` image with `MODEL=IQ3_XXS` and `API_KEY` only (`--track default`), complete, 12/12

Main limitation: one machine, one session, one quantization (IQ3_XXS). The comparison with 0.2.3.1 below is the only
same-session, same-config comparison; the other 0.2.3.1 numbers are from 23 Oct with more background load.

## Hardware and software

- GPU: AMD Radeon RX 7900 XTX 24 GB (gfx1100) in a UGREEN eGPU dock with its own PSU, **OCuLink PCIe 4.0 x4**
- Host: UGREEN iDX6011 Pro NAS, Core Ultra 7 255H (6 P + 8 E + 2 LP-E, no AVX-512), 64 GB LPDDR5X, model files on NVMe
  RAID 1 (btrfs)
- OS: UGOS Pro (Debian 12 base), kernel 6.18.15; the engine runs in a container with the ROCm runtime of the release
  package
- Strata: release 0.2.8, engine `strata-engine-0.1.54-linux-x86_64-hip-rx7900`, control 0.1.54 (prebuilt, checked
  against `SHA256SUMS.txt`); for the comparison release 0.2.3.1 (engine 0.1.48, control 0.1.48.1)
- Background: every NAS service that was not needed was stopped for the session (photo library, media server, an agent,
  a search engine, a mail bridge, file sync). Running idle: a reverse tunnel, three small MCP servers, a container agent,
  and the proxy in front of the engine. No other model was loaded.

## Model and configuration

- Original Flash-Next IQ3_XXS (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf`), native pack `iq3_xxs`,
  MTP head `rt`. `b028` and `p0231` in the data are only the labels of the two run configs.
- Both versions use the identical run config, only engine and control are swapped:

```text
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8
--vram-reserve-mib 832 --pcie-frac 0 --conversation-cache-slots 4 --conversation-cache-min-tokens 8192
--conversation-cache-mib 9216 --kv-resident 32768
env STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100200.txt (shipped); reasoning_effort high; default pool workers (10)
```

## Results

### 0.2.8 against 0.2.3.1, same session and config

3 interleaved pairs (0.2.8, 0.2.3.1, 0.2.8, ...), each from a freshly started engine: one warm-up answer, then two
measured answers to a 63-token prompt with 900 new tokens. Raw data: [`version-pairs.jsonl`](version-pairs.jsonl).

| median (min-max), n=6 | 0.2.8 | 0.2.3.1 |
| --- | ---: | ---: |
| decode tok/s | 61.4 (58.9-62.9) | 61.6 (58.4-63.9) |
| prompt tok/s, 63-token prompt | 111.2 (90.7-119.2) | 121.2 (113.3-126.2) |
| MTP draft acceptance | 45.4 % | 44.8 % |

Decode is equal within the noise. Reading the **short 63-token prompt is about 8 % slower on 0.2.8**; the ranges overlap
(two of the six 0.2.8 runs reach the 0.2.3.1 range), and a nightly run on 30 Oct (with more background load) pointed the same way (110.3 against
113.0 tok/s). At 4K and longer I see no difference (below). This is a small effect on a prompt where a request's fixed
cost dominates; I report it, I do not claim a cause.

### Prompt reading, context and recall on 0.2.8

| test | result |
| --- | --- |
| `standard-v1`, package, custom track | prompt **791.3 / 1,201.4** tok/s at 4K / 16K, short decode 68.3 tok/s, 12/12 |
| `standard-v1`, unchanged image, default track | prompt **740.2 / 1,222.2** tok/s at 4K / 16K, short decode 67.8 tok/s, 12/12 |
| `tools/strata_bench.py --long` ([folder](strata-bench/)) | chat decode 68.3; prompt 784.8 / 1,211.5 / 1,199.3 tok/s at 4K / 16K / 32K; agent cache hit 68.6 %, agent decode 75.6 |
| `tools/needle_bench.py` 32K and 128K, depths 10/50/90 ([json](needles-32k-128k.json)) | **6/6**, 28 s at 32K, 98-111 s at 125K tokens |
| `tools/needle_bench.py` 250K, depths 10/50/90 ([json](needles-250k.json)) | **3/3**, 223-237 s at 251,626 tokens |
| 8 fresh engines, two ~3.8K prompts each ([jsonl](fresh-engines.jsonl)) | 694-749 tok/s, none slow (my 0.2.3.1 run: 706-749) |

For reference, my 0.2.3.1 numbers from 23 Oct (more background load, same config): `standard-v1` 799.9 / 1,203.0 tok/s,
`strata_bench.py --long` 791 / 1,218 / 1,244 tok/s at 4K / 16K / 32K; the 0.2.3 image 748.6 / 1,220.4 tok/s. 0.2.8 is in
the same range. The needles got faster than in my nightly run on 30 Oct (128K: 129-149 s, 250K: 274 s), which I put
down to the stopped background services, not to the engine.

**The unchanged image now leaves more VRAM free on this card:** 854 MiB after loading, and no "only N MiB of VRAM free"
warning during the run. With the 0.2.3 image it was 594 MiB, with one warning (my note in #3263).

### Tool calls

0.2.8 lists recovery of slightly malformed tool calls as a change. A small check, not a statistic: 10 agent-style tasks
with four declared tools (`read_file`, `run_command`, `search_web`, `write_file`), thinking on, temperature 0, each a
fresh conversation, on both versions. Raw data: [`tool-calls.jsonl`](tool-calls.jsonl).

| | 0.2.8 | 0.2.3.1 |
| --- | ---: | ---: |
| expected tool called first, arguments valid JSON | 10/10 | 9/10 |
| tool calls returned at all, all arguments valid JSON | 10/10 | 10/10 |

The one difference is a choice, not a parse failure: asked to write a file, 0.2.3.1 first called
`run_command("pwd && ls -la")`. All 20 answers on both versions came back as well-formed tool calls, so this sample
**does not exercise the new recovery path** and says nothing about it either way. The four tool checks of `standard-v1`
pass on both tracks.

## Data files

Field names in the `.jsonl` files are German:

- `version-pairs.jsonl`: `variante` (b028 = 0.2.8, p0231 = 0.2.3.1), `paar` (pair), `messung` (measurement),
  `prompt_token`, `neu_token` (new tokens), `prompt_tps`, `decode_tps` (engine timings), `draft_n` / `draft_ok`
  (MTP drafts proposed / accepted), `sekunden` (client wall time)
- `fresh-engines.jsonl`: `engine` (1-8, each newly started), `prompt` (1-2), same fields as above
- `tool-calls.jsonl`: `variante`, `nr`, `soll` (expected tool), `aufrufe` (tools called), `argumente` (arguments as
  returned), `argumente_json` (each parses as JSON), `richtig` (expected tool first and valid JSON), `finish`, `usage`,
  `sekunden`
- `strata-bench/`: unedited output of `tools/strata_bench.py --long`. It was run from my older 0.1.44.1 checkout of the
  tool against the 0.2.8 server, which is why its README names that project version; the engine line says 0.1.54.

## Correctness and limitations

- One machine, one session, IQ3_XXS only; no IQ2_XS or vision measurement this time.
- Card power, clocks and PCIe link state were not recorded in this session.
- The pairs use a short prompt; the 8 % prompt-reading difference is limited to that case in my data.
- The tool-call check is 10 tasks per version and did not produce a malformed call to recover.
- Host names, addresses and keys are removed. No model files or packs.
