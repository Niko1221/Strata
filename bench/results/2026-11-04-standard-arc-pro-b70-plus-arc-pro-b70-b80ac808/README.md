# Strata standard benchmark: Intel Arc Pro B70, Intel Arc Pro B70

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261104T092532Z-b80ac808`, started 2026-11-04T09:25:32Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Intel Arc Pro B70; Intel Arc Pro B70 (32 GB VRAM each; submitted through vfio into a QEMU/KVM guest, so the runner could not read VRAM or the physical PCIe link: see the submitter addendum)
- CPU: AMD EPYC-Milan Processor (QEMU CPU model; the real host CPU is an AMD Ryzen 7 5700G, 8c/16t, of which the VM uses 5 cores / 10 threads); RAM: 94.3 GiB
- Storage of the model files: hdd QEMU HARDDISK (sata)
- OS: <host> 26.04.1 LTS 7.0.0-42-generic; drivers: Intel `xe` (Battlemage) kernel driver; runtime: SYCL/AOT (oneAPI) engine build - the runner's `nvidia-smi`/`rocm-smi` probes do not see Intel cards
- Strata commit: `ffa80d73133ea5c548aefe57889a16e8085a938f`; engine 0.1.56; source build
- Background work: Hermes agent idle during the run; fresh engine reload immediately before start

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs; quantization UD-IQ4_XS; revision 38bb39ee97821de2c9009abb7e93950eec396e66; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` `5ce89370720f8bf9...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` `577a38a2392b40ca...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` `d4634e6d84f0ebb0...`; `expert-profile-sycl.bin` `ddddf9b361dd1cb1...`
- Track: **custom** (engine 0.1.56 (repo v0.2.10+ffa80d73, AOT build-sycl-aot, container strata-sycl-dev), quant UD-IQ4_XS (from model filename, no override), ctx 262144, kv int8 + kv-resident 32768, prefill 8192, stream-experts + expert-cache auto (~47.8 GiB, primary GPU 24.5 GiB), spec 2 + spec-min-p 0.5 + lookup, MTP draft layers loaded (drafting=MTP, DFlash2 not downloaded), NO_HOST=0 CPU-experts enabled, STRATA_PCIE_FRAC_DEFAULT=auto (engine pcie_frac 0.20 with 2-card layer split), STRATA_ROUTE_PRIOR=1.5, idle_unload 1800s, 2x Intel Arc Pro B70 32GB (xe driver, --layer-split auto, Gen3 link - host CPU limit, physical link invisible to guest via vfio), VM 10 vCPU (Ryzen 7 5700G host, 5C/10T) / 94 GiB RAM, local server + local disk)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 43 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 71.7 (71.1-74.7) n=3 | 1.00 (1.00-1.03) n=3 | 2.45 (2.10-2.46) n=3 | 4.6 (4.4-7.9) n=3 | 1/3 |
| prompt-4k | 880.2 (842.7-884.0) n=3 | 73.1 (68.7-73.3) n=3 | 4.68 (4.66-4.89) n=3 | 8.00 (7.94-9.23) n=3 | 11.6 (11.2-12.3) n=3 | 2/3 |
| prompt-16k | 1502.3 (1500.6-1512.6) n=3 | 63.3 (61.8-69.5) n=3 | 10.94 (10.87-10.96) n=3 | 15.71 (15.71-15.71) n=1 | 18.9 (18.3-19.2) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **12 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | pass | the expected JSON value |
| I2 | pass | the expected JSON value |
| I3 | pass | the expected JSON value |
| I4 | pass | the exact expected text |
| R1 | pass | the exact expected text |
| R2 | pass | the exact expected text |
| R3 | pass | the exact expected text |
| R4 | pass | the expected JSON value |
| T1 | pass | the expected tool call |
| T2 | pass | the expected tool call |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
- The engine was restarted deliberately seconds before the run; round 1 of each case carries load-after-start warm-up, so its maxima (TTFT, first answer, total) are not steady-state numbers - read the medians and rounds 2-3 for that.
- The engine log starts with `Drafting: off (no drafter is downloaded yet)`: that line names the separately downloadable DFlash2 drafter. The `--mtp` draft layers passed on the command line are loaded and request-level `drafts_*` counters are non-zero across the run (the companion quick-suite run measured 71.9 % acceptance), and the engine's startup line `Drafting: off` refers to the optional DFlash2 drafter, not to this MTP path.
- `engine.log` is a sanitized `journalctl --user` slice of the service: a `--no-start` run uses the already-running server, so the runner itself captures no engine stdout.

## Submitter addendum

- Tool gaps on Intel inside a VM (all inline in the hardware section): VRAM is 32 GB per card; the driver is the Intel `xe` kernel driver; the CPU line shows the QEMU model; the model files live on a SATA-passed-through HDD, model load took about 80 s.
- PCIe: the host is an ASUS ProArt B550-CREATOR with a Ryzen 7 5700G, so the links run at PCIe Gen3 rate (the CPU is the ceiling; board and cards are newer). The first card sits electrically x16 in slot 1; the second card was added on 2026-11-03 and its per-card link width was not re-measured. A vfio-assigned device never shows its physical link to the guest (sysfs reports the QEMU stub), so no link width is claimed from inside the VM. The engine's own PcieAuto probe measured `pcie_frac 0.20` for this two-card layer split.
- Provenance: the three GGUF SHA-256 equal the hashes recorded at download time (the `.done` sidecars), the setup script's pinned SHAs, and the current Hugging Face LFS objects of revision `38bb39ee97821de2c9009abb7e93950eec396e66` (`unsloth/Qwen3.8-Flash-Next-GGUF`, `UD-IQ4_XS/`). The declared custom pack `data/expert-profile-sycl.bin` (`ddddf9b3...`) is the xe expert ranking shipped in this repository (selected by `sycl/setup_intel.py`), not a submitter artifact.
- Configuration is not the setup default (track **custom**): two cards with `--layer-split auto`, `spec 2` + `spec-min-p 0.5`, MTP draft layers, CPU-resident experts (`STRATA_VERIFY_NO_HOST=0`), `STRATA_ROUTE_PRIOR=1.5`, `STRATA_PCIE_FRAC_DEFAULT=auto`, `idle_unload 1800s`. Every value is spelled out in the track notes above.
