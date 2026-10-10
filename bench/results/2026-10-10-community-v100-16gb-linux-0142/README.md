# Community benchmark on a Tesla V100-SXM2 16 GB (sm_70, Linux, one card): engine 0.1.41 against 0.1.42

Measured on 2026-10-10 by **@taxah92**. Both versions were built from their tags and run on the same machine,
the same day, with the same configuration: 0.1.41 (`fb58e0d`) first, 0.1.42 (`61b3fb5`) after the upgrade. The
question was what 0.1.42's two new NVIDIA defaults (`STRATA_ROUTE_TAIL_SKIP=7` and the measured PCIe share) do
on a 16 GB Volta card whose expert working set does not fit in VRAM and whose CPU pool is much faster than its
PCIe link.

On this machine: decode **+34% to +62%** on the same prompts, prompt throughput unchanged, recall unchanged
(`tools/needle_bench.py` 6/6 on both versions, and the same answers on an own 128K needle). The larger part is
the PCIe share — the new rule timed the CPU pool at 45–52 GB/s against the 13.1 GB/s link and moved `pcie_frac`
from the start-up rule's **0.36 to 0.10, the floor of the 0.10–0.60 search range**.

Main limitation: one machine, one model, and the two new defaults changed together — this report does not
separate the tail skip from the PCIe share. The long-context arms are single runs; the short arms are three.

## Hardware and software

- **GPU:** Tesla V100-SXM2-16GB (sm_70), 16,384 MiB, driver 580.178.04, PCIe Gen3 x16 (`pcie.link.gen.max 3`,
  `pcie.link.width.max 16`). The card is passed through to a Proxmox VE guest (one GPU, nothing else on it).
  The engine's own start-up probe measures the host→device link at **13.1 GB/s**.
- **CPU:** AMD EPYC 7532 (32 vCPU visible in the guest); **RAM:** 94 GiB visible (96 GiB allocated);
  **storage:** SSD/NVMe on the hypervisor; **OS:** Ubuntu 26.04 LTS, kernel 7.0.0-34-generic.
- **CUDA:** 12.4.131. Source builds for Volta, identical options in both arms:

  ```text
  cmake -S . -B build-sm70 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON
  ```

  No local patches in either tree. The `#if CUDART_VERSION >= 12050` gate in `src/core/vmm.cpp` is
  byte-identical between the two tags.
- **Background workloads:** none during the runs. The server serves one request at a time; the engine is the
  machine's production LLM, so every request below ran alone.

## Model and configuration

- Qwen3.8-Flash-Next **GSQ-RCO IQ3_S**, two GGUF shards (native shard 1 and a PLE shard 2), 46.84 GiB of
  experts. Native-experts pack (`tools/iq_pack.py`), MTP runtime (`tools/mtp_rt.py`), cyrillic draft vocabulary,
  learned expert profile (`--expert-profile-save-every 10`), vision off.
- Context **1,048,576** with YaRN scale 4; **`--kv k8v4`**, `--kv-resident 32768` (9.56 GiB of pinned RAM);
  `--expert-cache auto` → 3,010 slots planned, **3,903 experts resident (7.46 GiB of VRAM)**, 16,058 of
  16,384 MiB used; `--prefill auto:16384`; `--spec 3 --spec-min-p 0.5 --suffix-draft 3`; `--prompt-cache 16
  --prompt-cache-every 8192`; conversation cache 24,576 MiB / 3 slots / 1,280 MiB minimum free;
  `--vram-reserve-mib 600`. Sampling: `temperature 0`, thinking on (the model's default).
- `env` in both arms: `STRATA_SM70_TABLE=1`, `STRATA_SELECT_SIMT=1`. In the 0.1.42 arm only:
  `STRATA_ROUTE_TAIL_SKIP=7` (the release default, pinned explicitly so the line stays reproducible).
  **`--pcie-frac` is not set in either arm** — the 0.1.41 arm therefore uses the start-up link probe's share
  and the 0.1.42 arm the new measured one.
- The launch line (paths shortened, no credentials):

  ```text
  strata --serve --pack <packs>/iq3_s \
    --native <models>/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    --ple-gguf <models>/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
    --expert-profile <data>/expert-profile.learned.bin \
    --expert-profile-save <data>/expert-profile.learned.bin --expert-profile-save-every 10 \
    --expert-cache auto --prefill auto:16384 --spec 3 --spec-min-p 0.5 --mtp <mtp>/rt \
    --max-context 1048576 --rope-scaling yarn --rope-scale 4 --kv k8v4 --kv-resident 32768 \
    --vram-reserve-mib 600 --prompt-cache 16 --prompt-cache-every 8192 --suffix-draft 3 \
    --conversation-cache-mib 24576 --conversation-cache-slots 3 --conversation-cache-min-free-mib 1280 \
    --control-vector-scaled <data>/experimental-speed-projection/...gguf:1.0 \
    --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
  ```

  The control vector is inactive unless a request asks for `experimental_speed_projection`; every request in
  this report ran without it.

## Method

- **Same-day A/B.** 0.1.41 was measured first, then the production unit was switched to 0.1.42 and the same
  workload was repeated. One engine fits the 16 GB card, so the arms are separate engine runs, each with a
  fresh load (46.84 GiB of experts at 3.68 GiB/s, about 50 s to READY).
- **Long-context recall:** the repository's own `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`
  (6 tests) on both versions, plus an own 128K Russian-haystack needle at depths 10/50/90 (greedy,
  `temperature 0`, 256-token cap) — `ab.py`, attached.
- **Speed arms, three runs each per version:** a Russian story prompt (450-token cap) and four short prompts
  (ru-chat, code, math, tool-json; 400-token cap). Prompt and decode throughput are the engine's own timing
  fields; client wall seconds are in the JSON. Requests are non-streaming, so **TTFT was not measured**.
- **Cache state:** `cache_n` is recorded per run. The needle prompts are fresh (0 reused tokens) except the
  90 % depth, where 65,280 tokens came from the prompt cache in both arms — the same value on both sides.
- **Warm-up:** the first request after READY is the 20,000-character calibration prompt; the measured runs
  follow it.

## Results

Decode throughput, medians of three runs (range in parentheses), engine timings:

| Leg | 0.1.41, tok/s | 0.1.42, tok/s | Δ median |
| --- | ---: | ---: | ---: |
| Russian story, 450-token cap | 49.9 (49.2–53.0) | 68.3 (66.9–71.5) | **+36.9%** |
| short ru-chat | 45.2 (45.1–45.6) | 60.8 (58.8–70.7) | **+34.5%** |
| short code | 49.5 (49.1–49.7) | 72.0 (70.3–72.3) | **+45.5%** |
| short math | 56.7 (56.5–57.5) | 82.3 (82.1–83.7) | **+45.1%** |
| short tool-json | 48.1 (47.1–48.4) | 78.0 (77.1–80.7) | **+62.2%** |

Draft acceptance on the story leg was 75–85% (0.1.41) and 76–89% (0.1.42) — the gain is not from better
drafting; the range does not overlap on any leg.

Own 128K needle (single run per depth per version; the same haystack and question on both sides):

| Depth | Prompt tokens | 0.1.41 prompt / decode tok/s | 0.1.42 prompt / decode tok/s | Answer |
| --- | ---: | ---: | ---: | --- |
| 10% | 131,100 | 1,320 / 51.5 | 1,306 / 81.9 | same code word |
| 50% | 131,101 | 1,308 / 60.2 | 1,285 / 83.7 | same code word |
| 90% | 131,101 | 1,250 / 59.1 | 1,221 / 88.4 | same code word |

(`cache_n` 0 / 0 / 65,280 on both versions, in that order; prompt throughput is within noise, as the release
says the prompt path is untouched.)

`tools/needle_bench.py`, wall seconds including the answer, both versions 6 of 6 found:

| Test | 0.1.41 | 0.1.42 |
| --- | ---: | ---: |
| 32k, depths 10 / 50 / 90% | 28.5 / 25.9 / 16.8 s | 28.2 / 25.7 / 16.7 s |
| 128k, depths 10 / 50 / 90% | 95.6 / 78.2 / 58.8 s | 95.2 / 77.8 / 58.2 s |

What the 0.1.42 log says about the two defaults (verbatim lines):

```text
strata serve: STRATA_ROUTE_TAIL_SKIP=7 is ON: a missed expert that every token of a verify window routes at
              rank 7 or lower is skipped (+10..20% decode on 12-16 GB cards that miss experts, answers differ
              slightly from 0.1.41); STRATA_ROUTE_TAIL_SKIP=0 turns it off
strata serve: PCIe share: the CPU pool takes 45 us per missed expert (45.4 GB/s of expert bytes), the link
              13.1 GB/s -> pcie_frac 0.15 (was 0.36; --pcie-frac N fixes it, STRATA_PCIE_FRAC_DEFAULT=old
              keeps the start-up rule)
strata serve: PCIe share: the CPU pool takes 40 us per missed expert (51.8 GB/s of expert bytes), the link
              13.1 GB/s -> pcie_frac 0.10 (was 0.15; ...)
strata serve: route tail skip: 151,008 missed experts skipped (158,991 entries) since the start
```

The expert cache numbers are identical in both versions (`expert cache auto: 8.17 GiB free, 600 MiB reserved
(+123 MiB for the draft head) -> 3010 slots`, `expert cache 3903 slots, 7.46 GiB of VRAM`), so the decode
difference is not a different number of resident experts.

## Correctness and limitations

- **Recall unchanged.** `needle_bench.py` 6/6 on both versions, and the own Russian 128K needle returned the
  same code word at all three depths on both versions (greedy).
- **Answers drift slightly**, as the release documents. On the short `tool-json` prompt the values are
  identical but 0.1.42 prints the object pretty-printed; one ru-chat answer differs in wording. The attached
  JSONs keep the full texts and reasoning for both arms.
- **Not separated:** tail skip against PCIe share. No `STRATA_ROUTE_TAIL_SKIP=0` or
  `STRATA_PCIE_FRAC_DEFAULT=old` arm was run — both defaults changed at once, so the +34…+62% is their sum.
  The release's own tables (+15.6% RTX 3060, +14.6% Tesla P100 for the tail skip; +30% on a single card for
  the share) suggest the share is the larger half here, and the log lines above show it moved a long way.
- **Not measured:** TTFT (non-streaming requests), KL or perplexity, concurrency (one request at a time),
  other quantizations, other context lengths (128K needles and short prompts only), multi-GPU.
- **One machine, one model.** Volta on Linux in a VM is not a configuration the release notes cover, so the
  size of the gain should not be generalized beyond "a card that misses experts, with a CPU pool several
  times faster than its PCIe link".
- **Question for the maintainers:** the share search stopped at its **0.10 floor** on this host (CPU pool
  3.5–4× the link). Is that floor right, or would a lower one pay off on machines shaped like this one?

## Files

| File | Contents |
| --- | --- |
| `ab.py` | the measurement stand: `chars/token` calibration, the 128K Russian needle, the story leg, the four short prompts; writes one JSON per phase |
| `baseline-0141*.json` | the three 0.1.41 runs (the first one also holds the needle and calibration) |
| `post-0142*.json` | the three 0.1.42 runs, same order |
| `needles-0141.json`, `needles-0142.json` | raw output of `tools/needle_bench.py` for both versions |

Fields in the stand's JSON: `usage.prompt_tokens`, `prompt_tokens_details.cached_tokens` and the engine's
`timings` (`prompt_per_second`, `predicted_per_second`, `draft_n`, `draft_n_accepted`) per request, plus the
client's wall seconds and the answer text. The only edit to the raw data: the private LAN address used in the
`tool-json` prompt was replaced with `127.0.0.1` (in the prompt, the answers and the endpoint field).
