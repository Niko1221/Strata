# Community benchmark on RTX 5080 Laptop GPU 16 GB

Measured on 2026-10-11 by a community member. Strata 0.1.42 (official Windows engine), Flash-Next IQ3_XXS,
131,072-token context; three fresh runs each at about 3.5K, 28K and 109K prompt tokens with a 256-token output cap,
plus six needle checks at 32k and 128k. Main limitation: a laptop with normal desktop applications open in the
background, and the prompt lengths came out below the 4,096 / 32,768 / 128,000 targets (actual counts below).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB (16,303 MiB); power limit 175 W as reported by the engine's NVML
  telemetry; PCIe Gen 5 x16 (maximum link as reported by nvidia-smi)
- CPU: AMD Ryzen 9 9955HX3D, 16 cores / 32 threads; RAM: 64 GB (63.3 GiB visible); storage: 1 TB NVMe SSD (SK hynix)
- Machine: Lenovo Legion laptop (model 83RU), on AC power
- OS: Windows 11 Pro 10.0.26300; NVIDIA driver 582.05; CUDA libraries from Strata's setup (CUDA 13.0)
- Strata commit `61b3fb5` (main after `UPDATE.bat` on 2026-10-11), engine 0.1.42 ready-made Windows release
  (sm_75/86/89/120 build)
- Background: browser and IDE open; no other GPU workload

## Model and configuration

- Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` + `-00002-of-00002.gguf`),
  pack from `setup.py` (iq3_xxs), default expert profile, MTP draft (`mtp/rt`)
- No vision encoder; no custom packs
- Context 131,072; `--kv int8`, `--kv-resident 32768`; `--expert-cache auto`; `--prefill auto`; no low-RAM mode
- `--spec 4 --spec-min-p 0.5`; reasoning effort not set in the requests; `temperature 0`; no calibration, no
  experimental speed projection

```text
strata.exe --pack <packs>/iq3_xxs --native <models>/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
  --ple-gguf <models>/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile data/expert-profile.bin
  --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <data>/mtp/rt --max-context 131072 --kv int8
  --kv-resident 32768
(started through serve/server.py on port 18438, as written by setup.py; local paths shortened)
```

## Method

- Script: [`bench_sweep.mjs`](bench_sweep.mjs) (Node 26). Each prompt starts with a unique random seed and filler
  text, so no prefix is reused (`reused` = 0 in every run); the task is "Summarize the text above in one short
  paragraph", `max_tokens` 256, `stream: true`.
- Prompt and decode throughput come from the engine (`/metrics`, `requests[0]` right after each call:
  `prompt_read / prompt_ms` and `decode_tok_s`). TTFT is measured in the client on the stream: first non-empty delta,
  reasoning or answer text. Total latency is client-side for the whole request.
- One engine session, model already loaded; the server had served a few short requests before (expert cache warm,
  no prompt cache hit). Runs one after another, no pause.
- Memory: snapshot from `/metrics` `hardware` right after each run (GPU memory used, system RAM used including the
  other applications). No peak measurement.
- Recall: `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90` ([needles.json](needles.json)).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| IQ3_XXS, ~3.5K prompt | 3,512 / 3,536 / 3,527 | 0 | 195 / 256 / 227 | 3 | 2,205 (1,952–2,228) | 116.3 (92.5–117.8) | 1.62 (1.60–1.83) |
| IQ3_XXS, ~28K prompt | 27,887 / 27,895 / 27,864 | 0 | 256 / 256 / 256 | 3 | 2,668 (2,656–2,689) | 107.1 (103.2–115.3) | 10.49 (10.41–10.53) |
| IQ3_XXS, ~109K prompt | 108,858 / 108,865 / 108,880 | 0 | 227 / 198 / 256 | 3 | 2,604 (2,603–2,605) | 103.4 (99.6–106.9) | 41.89 (41.87–41.90) |

Per-run data: [sweep.json](sweep.json). Total latency (client): 3.5–3.9 s, 12.7–12.9 s and 43.7–44.4 s.
Memory after each run: 15.68 GiB GPU memory used; 59.5–60.1 GiB system RAM used (whole machine). GPU power at the
snapshot: 125–171 W. No failures, no cancelled requests. The first ~3.5K run is slower than the other two
(1,952 vs about 2,200 prompt tok/s, 92.5 vs about 117 decode tok/s).

## Practical workload (structured JSON)

In a private application the model writes a long structured business record as structured output (`response_format` json_schema,
about 4,300 prompt tokens including a 6 KB example record, reasoning on). Ten fresh requests on engine 0.1.41 took 36–48 s
for 4,000–5,000 completion tokens; five requests on 0.1.42 took 30–37 s for 4,050–4,600 completion tokens (three of
them reused a 4,336-token prefix). All 15 answers were valid JSON. Rare outliers looped in their reasoning
(one answer cut at 16,384 tokens without an answer); a `reasoning_budget_tokens` of 3,072 with `max_tokens` 12,288
keeps such cases short. Prompts and data of this workload are internal and not included.

## Correctness and limitations

- Needles 6/6 found: 32k (32,804–32,805 prompt tokens, 13 s each) and 128k (125,056–125,058 prompt tokens,
  44–50 s) at depths 10/50/90 %.
- Prompt lengths are below the suggested 4,096 / 32,768 / 128,000 because the filler text tokenized shorter than
  estimated; the actual counts are in the table.
- Laptop with other applications open; no clock or temperature capture; single engine session; no comparison run
  of 0.1.41 with this sweep.
