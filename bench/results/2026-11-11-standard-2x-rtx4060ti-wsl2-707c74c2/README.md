# Community benchmark on 2x NVIDIA GeForce RTX 4060 Ti 16 GB (WSL2)

Measured on 2026-11-11 by [@onkanat](https://github.com/onkanat). One model, one quantization, one
configuration: `qwen3.8-flash-next-q2_0` at a 262,144-token context with `kv q4_0`, served by engine 0.1.59
and measured with `strata bench run` (standard protocol `strata-qwen38-thinking-v1-draft.1`, draft, run
`20261111T194337Z-707c74c2`, status `complete`, 4 min 10 s of measured work).

The main limitation is the machine: the server runs inside WSL2 on Windows 11, the runner records **one**
GPU, and the engine splits the model across **two** cards. The per-card rates below are therefore the
rates of the pair, and the split itself is not separable from this data.

## Hardware and software

- 2x NVIDIA GeForce RTX 4060 Ti 16 GB (sm_89, no NVLink), layer split across both cards by the engine
  (`--layer-split auto`). The runner lists device 0 only (16.0 GiB), because WSL2 exposes a single device
  to its probe. Power limit 165 W per card as reported; actual draw not measured.
- Intel Xeon W-2145 @ 3.70 GHz; 128 GB RAM on the host, 110 GiB visible inside WSL2 (`memory=112 GiB` in
  `.wslconfig`).
- Storage: the model files sit on the WSL2 virtual disk, which the runner reports as `hdd Virtual Disk
  (unknown)`. PCIe link: not measured.
- Ubuntu 26.04.1 LTS, kernel 6.18.58.1-microsoft-standard-WSL2; GPU driver 580.97; CUDA 13.0.
- Strata commit `628f74955a56fcba092495b800dd02af75373a0f` (app 0.2.13.1); engine **0.1.59**,
  `linux-x86_64-cuda-cuda13` release binary from the signed update feed (not a source build).
- Background workloads: none during the run. A document-summarization job on another host, which uses
  this server, finished at 22:40 (+03), three minutes before the run started. Not measured otherwise.

## Model and configuration

- Qwen3.8-Flash-Next, `q2_0` pack, Q2_0 quantization, two GGUF files (`…Q2_0-00001-of-00002.gguf`,
  `…00002-of-00002.gguf`); file SHA-256 in `results.json`. The quantization repository revision and the
  original weights revision are **not measured** (`null` in `results.json`).
- Vision on. No custom packs or profiles. DFlash2 drafting is on (`--spec 6`).
- Context 262,144; `kv q4_0`; expert cache 14,758 MiB (11,195 expert slots, 5,785 on the first card);
  `--prefill auto`; no low-RAM mode. These values come from the running process, not from a launch
  profile: the runner could not read them, so `configuration.kv_cache`, `expert_cache`, `prefill` and
  `mtp` are `null` in `results.json`.
- Server capacity 2 slots (`--parallel 2`), one request measured at a time. Sampling is the model card's:
  temperature 1.0, top_p 0.95, top_k 20, `reasoning_effort=xhigh`, thinking on and preserved.

```text
strata bench run --url http://127.0.0.1:8080 --credential-env STRATA_KEY --track default \
  --track-notes "0.2.13.1 app, engine 0.1.59 cuda13, serve.server --parallel 2, 2x RTX 4060 Ti 16 GB, WSL2 112 GiB" \
  --out ~/strata-bench/run-20261111T2240

# the server (systemd unit, started before the run):
python3.12 -m serve.server --engine strata --config /root/.local/state/strata/launch/q2_0-256k-vision.json \
  --host 0.0.0.0 --port 8080 --parallel 2 --shutdown-timeout 30

# the engine process, from ps:
strata --serve --pack /opt/modeller/strata/packs/q2_0 \
  --native …/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf …/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --expert-profile /opt/strata/terminal/0.1.57/data/expert-profile.bin \
  --expert-cache auto --prefill auto --max-context 262144 --kv q4_0 --vision \
  --mtp-draft-vocab /opt/modeller/strata/dflash2/draft_vocab.bin --spec 6 \
  --layer-split auto --pipeline-windows 2 --batch 2
```

## Method

The frozen fixtures of `bench/prompts/standard-v1` (corpus
`e75eff8b9feea04f745520b392c4c17d97aefc1ba185e222aa6740efa8f68407`) were sent over loopback HTTP,
streaming, one request at a time: one warm-up (excluded), three repetitions each of the short, 4K and 16K
prompts, then the twelve sanity checks. Output cap 512 tokens. No request was retried. The engine was
**not** restarted for the run: `started_engine=false` and `server_requests_before_run=307`, so the
expert cache was warm and the report describes a server that had already answered traffic. The optional
concurrency stage was not run. Prompt and decode rates are the engine's own `timings`; time to first
token is the client's monotonic clock from dispatch to the first non-empty delta. Memory measurement: not
measured per card.

Raw evidence: `outputs.jsonl` (every request, usage, timings and client event offsets), `results.json`
(validated result, `schema_version` 1), `runner-README.md` (the runner's own report), `config.json`,
`corpus-build.json`, `corpus-manifest.sha256`, `chat_template.jinja`, and `engine-log.txt` (the server's
journal for the run window). `ARTIFACTS.sha256` hashes the files above.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short, single request, 512 cap | 77 | 0 | 375 (325–512) | 3 | not shown (request overhead dominates) | 44.1 (43.5–45.5) | 1.34 (1.29–1.44) |
| prompt-4k, single request, 512 cap | 4,093 | 0 | 512 | 3 | 647.6 (500.4–650.2) | 44.7 (44.0–47.8) | 6.35 (6.33–8.21) |
| prompt-16k, single request, 512 cap | 16,387 | 0 | 512 | 3 | 1,126.5 (1,117.4–1,127.7) | 47.8 (47.0–48.0) | 14.58 (14.57–14.71) |

Total request times, median (range): short 9.9 s (8.8–12.6), prompt-4k 17.8 s (17.0–19.8), prompt-16k
25.4 s (25.2–25.5). First answer (thinking finished) median: 3.32 s, 11.96 s, 20.96 s (the 16K figure has
one sample; the other two were still reasoning at the cap).

Request counts: warm-up 1 attempted / 1 truncated; performance 9 attempted / 2 completed / 7 truncated at
the 512-token cap / 0 failed / 0 skipped; sanity 12 attempted / 12 completed; concurrency 0 (not run).

## Correctness and limitations

- **Sanity: 12/12 passed** — instruction 4/4, retrieval 4/4, tool call 4/4.
- Seven of the nine performance requests reached the 512-token cap — thinking at `reasoning_effort=xhigh`
  fills the cap before the answer starts. That is the protocol's own cap, not a failure; the decode rate
  is measured over the tokens the engine produced, and the answer-time medians have the sample sizes
  noted above.
- The runner records **one** GPU. Read "2x" from this report, not from `hardware.gpus`; a single-card run
  on this machine is not available for comparison.
- Storage is a WSL2 virtual disk; the engine's `kv_cache`, `expert_cache`, `prefill` and `mtp` settings are
  `null` in `results.json` because the server runs from a hand-written systemd unit, not a launch
  profile.
- Not measured: GPU temperature, clocks, power draw, PCIe throughput, per-card memory use, concurrency,
  and behaviour on native Windows on the same box.
- Separate from this run, six hours of the same server's journal (1,390 requests, 15:08–21:08 +03) show
  about 50 tok/s decode for a lone request of 150+ tokens and about 35 tok/s per request when both slots
  are busy. That is continuous-traffic behaviour, not part of the protocol result.
