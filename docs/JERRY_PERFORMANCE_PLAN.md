# Jerry Performance Plan for Harry / Hermes

**Status:** Hardware verified; candidate configuration prepared; model download and inference benchmarks pending.  
**Date:** October 11, 2026  
**Source baseline:** [`7695859aa13afb248bdb4d36f9b2e95707db079d`](https://github.com/kcmrshll9-ux/StrataQwen3.8-Flash-Next-/commit/7695859aa13afb248bdb4d36f9b2e95707db079d)

The owner confirmed the deployment: **Harry, the Hermes agent, runs on the current PC; Jerry runs the model and serves its API over the wired LAN.** The priority is **balanced coding/tool quality and speed**.

The objective is the fastest reliable completion of Harry's coding and tool tasks at an acceptable quality level. Token throughput is one measurement, not the sole selection criterion. No performance gain or optimal configuration is claimed before benchmarking.

## Verified Hardware and Preparation

Read-only SSH inspection and `python3 setup.py --check --gpus 0,1` completed successfully. This fork is checked out on Jerry at the source baseline above.

| Component | Verified observation |
|---|---|
| OS | Ubuntu 26.04.1 LTS, x86_64; kernel 7.0.0-38-generic |
| CPU | AMD FX-8350; 8 reported cores/threads; AVX and SSE4.2 present; **AVX2 absent** |
| GPUs | Two NVIDIA GeForce RTX 3060; compute capability 8.6 |
| VRAM | **2 × 12 GiB**, separately allocated per GPU |
| Driver | NVIDIA 595.99.02 |
| CUDA compiler | CUDA 13.1.115 at `/usr/local/cuda/bin/nvcc`; absent from the default command PATH |
| Host compilers | GCC 15.2 default; GCC 14 is also installed |
| Build utilities | Git, CMake 4.2.3 and Ninja available |
| Python | Python 3.14.4; project virtual environment/dependency compatibility remains to be validated |
| System memory | 30 GiB usable; approximately 28 GiB available during inspection |
| Swap | 8 GiB; unused during inspection |
| Model storage candidate | Main filesystem on Samsung SSD 860 EVO 500GB, SATA; 195 GiB free during inspection |
| Other disk | A separate rotational SATA disk is present; not selected for active model storage |
| GPU links | Board-limited PCIe Gen2 x16 capability on both cards; PHB path between GPUs |
| Ethernet | Interface reports 1,000 Mb/s |
| Existing runtime | Ollama executable present but its service was not running; no running Strata server observed |

The PCIe generation was lower at idle; the setup check identifies the board's Gen2 limit. Actual sustained transfer rates and clock behavior need measurements under load. Neither the link labels nor theoretical bandwidth are benchmark results.

### CPU Compatibility Is Mandatory

The repository's [older-CPU guide](INSTALL.md#older-cpus-experimental) and Jerry's setup check agree: the standard ready-made engine requires AVX2, while Jerry requires the **experimental source build with `STRATA_ISA_FLOOR=avx`**. Setup selected that path automatically.

We should build for Jerry's actual CPU and the GPUs' compute capability rather than reuse an AVX2-only prebuilt binary. We should validate the installed CUDA/host-compiler pair with a build; the setup code supports explicit `CC`, `CXX`, `CUDAHOSTCXX` and `STRATA_NVCC` choices. Any compiler override should be recorded in the build metadata. This inspection did not build the engine.

## First Candidate

The repository's hardware check reports **Coder IQ1_M fits in low-RAM mode** and that its estimates reject Q2_0, IQ2_XS, IQ3_XXS, IQ3_S and the larger Unsloth variants on this configuration.

Coder is therefore the first candidate for the owner's coding/tool workload. It is a compressed, expert-pruned coding variant; it should not be assumed to match the full model on general or multilingual tasks. The [publisher's model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) identifies the artifact and its tradeoffs. This is a candidate selection, not a quality result on Harry's tasks.

| Setting | Initial candidate | Reason and validation |
|---|---|---|
| Family / quantization | `coder / IQ1_M` | The setup check identifies a fitting low-RAM option |
| GPUs | `--gpus 0,1` | Use both cards for one model, then compare against one-card control runs |
| Distribution | `--layer-split auto` | Per-card VRAM-aware baseline; manual splits require measurements |
| Expert memory | `--low-ram auto` | Record the actual effective mapped/resident mode; do not force the entire expert arena into RAM |
| Context | 32,768 for standalone baseline; evaluate 65,536 for Hermes integration | Measure context's memory/latency cost and match Hermes to the actual server context |
| KV precision | `int8` first; compare `k8v4` only if worthwhile | Keep a quality-preserving baseline before testing the hybrid memory tradeoff |
| KV streaming | Automatic initially | Record effective mode and host/GPU memory effects before overriding |
| Vision | Off initially | Reserve per-card memory and establish text/tool performance first |
| Parallel requests | One active Harry request initially | Establish single-agent latency before measuring concurrent workloads |
| Experimental switches | Defaults initially | Change one variable at a time and keep only reproducible wins |
| Network | Loopback with authenticated SSH tunnel during validation, or authenticated LAN endpoint | Verify authentication and ingress controls before LAN deployment |

**Context compatibility:** Current [Hermes documentation](https://hermes-agent.nousresearch.com/docs/reference/faq#can-i-use-it-offline--with-local-models) describes a 64,000-token minimum in its custom-endpoint setup and requires matching the server's actual context. A 32K standalone benchmark is not a claim that Harry can be configured through that wizard at 32K. Validate the installed Hermes version and a 65,536-token server profile before integration. Do not advertise a larger context than the server actually provides.

A prospective standalone setup command, after download confirmation and build/dependency checks, is:

```bash
PATH=/usr/local/cuda/bin:$PATH ./setup.sh --setup \
  --family coder --model IQ1_M \
  --gpus 0,1 --layer-split auto \
  --context 32768 --kv int8 --low-ram auto \
  --vision no --no-browser --no-start
```

This command is a prepared candidate, **not an executed installation**. It may compile the engine and download large artifacts. Before using it, establish authentication in the local run configuration and confirm the download/storage budget. Keys belong outside Git; do not put a literal key into this document or a shared command.

The [setup guide](AI_SETUP.md#0-ground-rules) requires confirmation before the large model download. The model definitions estimate a 58.4 GB Coder download, with roughly another 6 GB for the draft layer; source/build products, dependencies and any prepared packs need additional disk space. Install one candidate initially and verify the complete projected disk footprint, including temporary and duplicate artifacts, against available space.

## Benchmark Sequence

1. **Build and smoke test.** Verify CPU-compatible CUDA build metadata, successful startup, both GPUs selected, correct model identity and finite outputs. Keep the server private during tests.
2. **Warm baseline.** Measure short chat, approximately 4K and 16K prompts, and multi-turn tool-style histories with fixed prompts and sampling. Separate cold model loading from warmed requests.
3. **One GPU versus both.** Use identical model, context and inputs. Record per-card VRAM, effective split, expert cache hit/miss behavior, CPU load, storage activity and total task latency.
4. **CPU workers.** Use the repository's calibration workflow or controlled comparisons such as 4/6/8 workers. Do not assume all eight reported cores produce the best floating-point throughput on this older CPU.
5. **Context and KV.** Compare standalone 32K against integration-sized 65,536 context; compare int8 and k8v4 only with the same tool/quality checks. Keep sufficient memory headroom and avoid sustained swap activity.
6. **Split and transfer options.** Test manual split positions or documented opt-ins only when baseline logs identify a relevant bottleneck. PCIe Gen2 makes transfer costs material; measure before accepting any claimed improvement from another machine.
7. **Real Harry tasks.** Measure successful tool-call parsing, correct arguments, task completion, retries, time to first token and complete task time. Include representative coding edits and multi-turn histories.
8. **Stability.** Run repeated tasks long enough to expose memory growth, thermal throttling, clock changes, failed tool loops or engine resets. Keep the previous working configuration available.

The repository's [`tools/strata_bench.py`](../tools/strata_bench.py) provides a repeatable HTTP suite with medians, first-token measurements and an agent-style conversation test. It turns thinking off by default, so also run representative Harry tasks with the intended reasoning settings before judging useful agent performance.

Record:

- Prompt-processing and generation tokens/s.
- Median and tail time to first token, and complete task latency.
- Tool-call correctness and task success rate.
- Per-GPU peak VRAM and utilization; CPU load, available RAM and swap activity.
- Storage activity/page faults, cache reuse and expert movement where logs expose them.
- Temperature, clocks, power behavior and engine errors.
- Exact commit, model artifact revision, build, configuration and prompt corpus.

No throughput target is invented here. A candidate wins only if its improvement repeats and the task-quality/stability checks remain acceptable. Retain one variable's comparison at a time; do not combine several unmeasured switches.

## Harry / Hermes Connection

The intended data path is:

```text
Harry / Hermes on this PC
    -> authenticated OpenAI-compatible API over the LAN
    -> Strata on Jerry
    -> model split across GPU 0 and GPU 1, with host RAM/SSD support
```

Use Hermes' custom endpoint with the actual reachable `/v1` base URL, the server's real model ID, its verified context limit and a secret supplied locally. [Hermes supports this endpoint configuration](https://hermes-agent.nousresearch.com/docs/reference/faq#can-i-use-it-offline--with-local-models). A secret should not be committed or copied into public benchmark reports.

Harry's agent/tool execution should remain on this PC unless separately requested. Jerry can initially serve inference with Strata's built-in server-side MCP disabled; Harry supplies tool definitions and executes returned calls. Validate that separation with a real harmless tool-call test.

Keep conversation prefixes and tool definitions stable where the agent permits it, and record cache reuse in multi-turn runs. Do not remove necessary tools or shorten required task context merely to inflate token-throughput numbers.

## Deployment Gates and Remaining Work

- [x] Confirm Harry/Hermes host and Jerry inference host.
- [x] Inspect Jerry read-only and verify per-GPU memory.
- [x] Prepare the repository checkout and pass its hardware-only check.
- [ ] Confirm the large download and validate full disk budget.
- [ ] Create the isolated project environment and compile the AVX-compatible engine.
- [ ] Download/verify the selected artifacts, then smoke test.
- [ ] Run baseline and tuning comparisons; publish sanitized results.
- [ ] Resolve or apply verified mitigations for the [security review](security/2026-10-11-security-review.md).
- [ ] Verify API authentication, connection/read limits and private/LAN reachability.
- [ ] Configure Harry's actual endpoint and context, then validate tool use.
- [ ] Record the winning measured profile and restart behavior.

Read-only inspection and source checkout preparation are complete. No model download, driver modification, engine build, inference benchmark, firewall change or Harry configuration change has been performed.
