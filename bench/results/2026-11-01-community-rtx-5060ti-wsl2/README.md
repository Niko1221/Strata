# Community benchmark on RTX 5060 Ti 16 GB — plus a helper-cache A/B with an RTX 3060 Ti 8 GB card (WSL2)

Measured on 2026-11-01 by `@xss1992`. What was tested: the everyday config of Qwen3.8-Flash-Next
**UD-Q4_K_XL** at 256 K context on a 16 GB Blackwell card under **WSL2**, and the second-card
**helper expert cache** (`--expert-cache-device1` + `--remote-expert-opt`) on the same box.
Main results: the everyday single-card config measured from the engine's own timing lines, a
**net-loss helper-cache A/B** with a specific cause (all experts are RAM-resident already), and
a **standard-protocol run** (`strata bench run`, `strata-qwen38-thinking-v1-draft.1`) in the
sibling folder `../2026-11-01-standard-rtx5060ti/` (this PR). Main limitation: the machine was
not idle (an interactive session ran on the same engine).

## Hardware and software

- GPU 0: RTX 5060 Ti 16 GB (sm_120), GPU 1: RTX 3060 Ti 8 GB (sm_86, idle in all final runs);
  CPU: Intel i9-14900KF; RAM: 128 GB DDR5 (WSL limit 110 GiB); storage: NVMe.
  PCIe link: not readable from WSL sysfs; the engine's startup measurement: `the link
  24.4 GB/s`, CPU pool `34.0 GB/s` of expert bytes.
- OS: Windows 11 host, Ubuntu 24.04 on WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2);
  driver 610.88; engine build `linux-x86_64-cuda-cuda13`.
- Strata 0.2.9 / Control 0.1.55 / engine 0.1.55 (release binaries).

## Model and configuration (the one actually serving)

- Model: Unsloth `Qwen3.8-Flash-Next-UD-Q4_K_XL` (4-shard GGUF, ~111 GB), expert pack
  `unsloth-ud-q4_k_xl`, expert-profile (24,576 ranked pairs).
- Engine command line as launched by the server (this is the live process, verbatim except
  the home path and with `~` for the Strata data folders):

```text
strata --serve --pack ~/Strata-data/packs/unsloth-ud-q4_k_xl \
  --native ~/strata-models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --expert-profile ~/terminal/0.1.49.2/data/expert-profile.bin \
  --expert-cache 3400 --prefill auto --spec 4 --max-context 262144 \
  --kv int8 --kv-grow --expert-cache-per-layer --vram-reserve-mib 128 \
  --vision --conversation-cache-mib 16384 --adapt-min-gain 3.0 \
  --spec-min-p 0.85 --mtp ~/Strata-data/mtp/rt
```

- Env: `STRATA_HC_Q8=1 STRATA_ROUTE_TAIL_SKIP=0 STRATA_ROUTE_PRIOR=1.0
  STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_CPU_SHARE_MAX=1024`; vision encoder on CPU
  (16 threads); card 1 stays `off` at role level (`"gpu": 0`, driver holds ~0.92 GB).
- Startup keeps **3,120 resident experts** (engine line `token graph hit path: 3120 resident
  experts`; the R4 on-device hit path is on); during a long live conversation the adaptive
  policy used 2,832 at one point. Resident expert set in host RAM ~71.7 GiB of ~88 GiB.
- Full sanitized launch config: `config.final.json` (identical to the running profile).

## Results — engine-measured lines from the live server (current config)

Direct transcriptions of `strata serve:` timing lines (prompt tok/s is over freshly-read
tokens only; `reused` tokens are conversation-cache hits; decode includes reasoning):

| Fresh prompt tokens | Reused | Prompt tok/s | Generated | Decode tok/s | Drafts accepted | Context |
| ---: | ---: | --- | ---: | --- | --- | --- |
| 32,111 | 0 | **1,205.9** | 22,047 | **58.2** | 9,390 / 10,863 | 32K |
| 90,005 | 0 | **1,203.3** | 3,899 | **58.4** | 2,296 / 2,679 | 90K |
| 65,417 | 16,384 | 1,379.2 | 122 | 38.5 | short output | 82K |
| 94,194 | 30,950 | 1,304.6 | 1,724 | 44.2 | 647 / 719 | 125K |
| 47,960 | 30,950 | 1,212.7 | 2,280 | 48.6 | 889 / 1,012 | 79K |
| 22,104 | 32,106 | 1,305.4 | 256 | 46.9 | 132 / 140 | 54K |

Live-session decode at 100K+ cached context (incremental turns) runs **~48–63 tok/s** with
draft acceptance 80–90 %.

## Helper expert cache A/B (RTX 3060 Ti 8 GB) — a measured loss

Same prompt, same settings, both sides without the MTP adapter (the A/B predated it):

| | single card | helper on 3060 Ti | delta |
| --- | ---: | ---: | ---: |
| prefill ~100K | 1,515 tok/s (66.6 s) | 616 tok/s (163.6 s) | **−59 %** |
| decode | 39.8 tok/s | 33.6 tok/s | **−16 %** |
| main expert cache | 3,120 slots | 2,016 slots | −35 % |
| helper cache | — | 2,185 slots / 6.37 GiB | |
| VRAM | ~15.5 GB | ~15.2 + 7.8 GB | |

Why: the documented wins (dual 4090 +63 %, 5090+4090 +28–63 %) assume expert misses take a
slow path. Here every expert byte is already in RAM — the engine measured the host path at
34 GB/s of expert bytes against a 24.4 GB/s link — so moving hot experts to the second card
swaps a faster path for a slower one, and additionally costs 35 % of the primary cache,
per-layer cross-card staging (~220 ms + 50 ms per request), and a prefill penalty.
Conditions where the helper should win: hit share below ~60 %, low memory bandwidth
(e.g. dual-channel DDR4), or experts spilling to SSD — none apply here.

## WSL2 notes

- The helper card is only visible to the engine when the launch config `env` overrides
  `CUDA_VISIBLE_DEVICES` (the server derives it from `"gpu"`, which must stay `[0]` so no
  `--layer-split` is added); `tools/calibrate.py` and issue #447 document the same pattern.
- With a helper active, only 8 GiB of the host expert arena is registered with CUDA under
  the WDDM host (`STRATA_ARENA_PIN_GIB` overrides); explicit helper slot counts are safer
  than `auto` there.
- Mixed-arch visibility (sm_120 + sm_86): per-arch default tables are not applied to the
  other arch's card (see #3313); no attempt was made to force them.

## Correctness and limitations

- Standard-protocol results (12 sanity checks, fixed corpus) are in the sibling
  `2026-11-01-standard-rtx5060ti/` folder of this PR; its `results.json` states the
  conformance status, exact counts and scores.
- The engine-line table above comes from a busy machine (an interactive session ran on the
  same engine; it is also the largest prompt source), single run per row, no TTFT recorded;
  it is labelled as live-traffic data, not idle-box data.
- The helper A/B is one paired run at one prompt length, measured without the MTP adapter.
- Earlier findings on engine 0.1.48–0.1.49 (layer-split OOM on the 8 GB card with 0.71 GiB
  free, peer tier needing a non-empty primary cache) are unchanged in spirit.
