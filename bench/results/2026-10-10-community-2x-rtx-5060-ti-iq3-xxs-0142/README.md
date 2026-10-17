# Two RTX 5060 Ti 16 GB, i7-12700, IQ3_XXS — Strata 0.1.42

Measured on **10–11 October 2026** by **inlanger**, through the API in Docker:
**150–162 tok/s** on full high-reasoning responses at 2.3–2.5K input,
**123–139 tok/s** at 122K input. **Peak single-response decode: 178.2 tok/s.**

## Hardware and model

- **2× RTX 5060 Ti 16 GB**, PCIe **Gen4 ×8 / Gen4 ×4** under load, no P2P;
  180 W power limit per card.
- **i7-12700**, 8 P + 4 E cores, AVX2; **64 GB DDR4-3200**.
- Crucial P310 500 GB Gen4 NVMe for models and Docker; Ubuntu 26.04,
  kernel 7.0.0-38-generic, NVIDIA 595.91.07.
- Source build of **0.1.42**, `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`,
  clean tree, Release, CUDA 13.0, architectures `120;86`.
- [ISTA-DASLab Flash-Next GSQ-RCO IQ3_XXS](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  native GGUF experts, a previously calibrated routing profile, **Huihui refusal-direction
  projection** (layers 4–44, scale 1), and the **Cyrillic MTP vocabulary** (58,963 tokens).
  Artifact hashes and GGUF filenames are in [environment.json](environment.json).

## Configuration

Layer split **28**, pipeline windows **2**, MTP **3** with `spec-min-p=0.7`,
**131,072** context, int8 KV with **20,480** resident cells, prefill **16,384**,
`pcie-frac=0`. Selected: `STRATA_ROUTE_TAIL_SKIP=7`, `STRATA_SPLIT_MTP_BATCH=1`,
auxiliary CPU affinity off. Full commands and environment: [configs.json](configs.json).
Loaded cache: **14,230 experts / 23,802 MiB**; RAM arena: **40,925 MiB**;
11 CPU workers plus the host thread.

## Results

### Tail skip with batched split MTP

Same binary and inputs, fresh engine starts in **A/B/B/A** order, two full answers
per start. A has tail skip 0; B has tail skip 7. Both have split MTP batching on.

| Setting | First start, tok/s | Second start, tok/s | Pooled tok/s | Responses |
| --- | ---: | ---: | ---: | ---: |
| Tail skip 0 | 148.3 | 151.0 | 149.5 | 4 |
| Tail skip 7 | 161.5 | 162.4 | **161.9** | 4 |

**+8.3%** from this setting on the same 0.1.42 binary. Tail skip can change answers.

The initial screen used four full answers per setting:

| Tail skip | Split MTP batch | Aux CPUs | Pooled decode, tok/s | 32K prefill, seconds (one request) |
| ---: | ---: | --- | ---: | ---: |
| 0 | off | off | 145.1 | 16.006 |
| 7 | off | off | 149.0 | 15.996 |
| 7 | off | auto | 152.4 | 16.041 |
| 7 | on | off | **160.8** | **15.487** |
| 7 | on | auto | 145.7 | 15.535 |

The **178.2 tok/s peak** was one full response with tail skip 7, aux CPUs auto,
and split MTP batching off: 2,509 input tokens, 2,225 output tokens in 12.485 s of decode.

### P-cores

Selected configuration, fresh starts in **all / P / P / all** order, four full
answers per start. All: 11 workers; P-only: 7 workers.

| Comparison | All cores, tok/s | P-cores, tok/s | P-core change |
| --- | ---: | ---: | ---: |
| First pair | 148.00 | 155.17 | +4.84% |
| Second pair | 152.82 | 151.63 | −0.78% |
| Pooled, 8 responses per setting | 150.26 | 153.41 | +2.10% |

The original 11-worker pool was retained.

### Long inputs, selected 11-worker configuration

| Date | Input tokens | Output tokens | Prefill, seconds | Decode, tok/s | Requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oct 10 | 122,159 | 2,120 | 46.665 | 123.2 | 1 |
| Oct 11 | 122,160 | 2,028 | 46.899 | 138.8 | 1 |
| Oct 11 | 32,269 | 2,548 | 15.503 | 132.5 | 1 |

P-only measured 131.3 tok/s at 122K and 129.2 at 32K; prefill was unchanged.

## Method and data

**81 requests**, all natural `stop`, all `reused=0`, high reasoning,
temperature 0, no requested output cap or thinking budget. Main tasks: synthetic
Russian/English record extraction, **1,802–3,528 output tokens**, including reasoning.
Long inputs add filler and recall records; small JSON checks are separate.

Pooled decode = **sum(output tokens) / sum(decode seconds)**; prefill and loading
excluded. [summary.json](summary.json) includes medians and ranges. Paired requests
match exactly. Fresh engine per arm, retained state within it, no discarded warm-up.
Natural output length and draft acceptance vary. Other Docker services remained running;
background CPU/disk load was not controlled. TTFT was not recorded.

Final snapshots: **15.1 GiB VRAM per card**, 7.6–8.4 GiB available RAM,
5.7–6.5 GiB swap occupied. No new Xid/AER/OOM messages during the test periods;
the old root-port correctable-error counter stayed at 1. Peak memory and paging rate
were not measured.

- [results.json](results.json): every request's timings, validation flags and request parameters.
- [fixtures.json](fixtures.json): exact synthetic messages and expected JSON.
- [outputs for Oct 10](outputs-2026-10-10.jsonl) / [Oct 11](outputs-2026-10-11.jsonl): answers and reasoning, keyed by request ID.
- [engine-log-excerpts.txt](engine-log-excerpts.txt): startup and request timing evidence.

Recalculate the tables with `python summarize.py`. To replay an arm, configure and
restart a 0.1.42 server using its entry in `configs.json`, replace the path placeholders,
then run:

```bash
python replay.py --url http://127.0.0.1:8080 --arm 2026-10-10/confirm-tail-mtp-a
```

`replay.py` only sends the archived requests; it does not change server settings.
Not tested: concurrent clients, vision, 262K context, general model quality.
