# Experimental Hermes image-to-video resource handoff

One bounded CLI workflow completed: a Hermes supervisor delegated sequential image and video workers, Strata yielded resources for each GPU tool, the tools completed and shut down, and Hermes resumed inference to inspect the outputs and produce its final report. No operator supplied mid-run instructions or repaired the running task.

This is useful functional evidence for [adaptive resource leases, Strata #1493](https://github.com/Niko1221/Strata/pull/1493). It is **not a general agent reliability benchmark, a matched speedup result, or a standalone media beta release**. The experiment also uses local integration and checkpoint-bank components that are not all in that PR. The public experimental components remain open for review while the portable integration and Desktop path are developed.

## What ran

- Windows; Ryzen 9 7940HS; RTX 4070 Laptop 8 GiB; 64 GiB DDR5-5600.
- ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_S; 65,536 context capacity; high reasoning; CPU vision; experimental projection off.
- One model-serving engine, one supervisor and two sequential direct workers. Context capacity was 64K; the largest observed request was 43,163 tokens, not a fully occupied 64K test.
- The Hermes supervisor selected Qwen Image Q8_0 using a resource forecast; the host validated and prepared the assets. Three verified cached assets were reused; no model download occurred in this run. The image worker generated the reference; the video worker used it with the local MiniMax H3/ComfyUI workflow.
- Protected tool admission required model-process exit and fresh resource checks. The broker maintained the handoff through tool execution and cleanup; subsequent inference reloaded through admission. No simultaneous second heavy job was launched.
- Hermes inspected the generated image, a contact sheet and selected video frames. The final report was written and the supervisor ended normally.

## Result and audit correction

Hermes exited normally. The API, request/dispatch correspondence, SQLite lineage, sequential worker returns, pixel delivery, resource-lease lifecycle and owned-process cleanup checks passed. All 31 recorded model requests completed, with no pending or failed request and no output/reasoning truncation in the audited records.

The original inner and outer test verdicts are still **FAILED**. One preparation-link auditor compared argument dictionaries literally: the native tool call used forward slashes in its Windows request path, while the trusted host recorded the same file with backslashes. The host's own completion verification and the native occurrence identity were valid.

A separately reviewed correction canonicalizes only the exact one-key request argument using the existing approved-root path validator. It requires that exact argument shape, rejects duplicate keys, and preserves root/traversal checks, call identity, chronology and receipt verification. Twelve focused positive/negative cases passed. A read-only replay against the unchanged actual records changes only that formerly false link check to true. Original source, outputs, receipts and failed verdicts were preserved; no model/media rerun was used to repair the evidence.

The supported conclusion is **a completed functional CLI workflow with a separately corrected audit false negative**, not an original all-green test run. This correction is in our test auditor, not an upstream Strata or Hermes runtime fix.

## Measurements

| Measurement | Observed value | Interpretation |
|---|---:|---|
| Agent interval | 3,448.60 s (57m 28.60s) | Whole agent work, including tools and model calls |
| Outer run interval | 3,505.64 s (58m 25.64s) | Includes startup and final validation/cleanup |
| Model requests | 31 | 15 supervisor, 8 image worker, 8 video worker; no overlap |
| Model request wall-time union | 2,577.00 s | Nested inside the agent interval |
| Recorded prefill / decode / load | 942.95 / 1,482.06 / 115.66 s | Components inside request time; do not add these to whole-run time |
| Image tool wrapper / Comfy execution | 72.99 / 42.66 s | Same image job; Comfy time is nested |
| Video tool wrapper / Comfy execution | 737.94 / 680.49 s | Same video job; Comfy time is nested |
| Actual NVML free-VRAM minimum | 881.29 MiB | 33,067 samples, 100 ms target, maximum observed gap 313 ms |
| Native allocator-reported free minimum | 312 MiB | Different measurement from NVML; not interchangeable |
| Agent guard available-RAM minimum | 7.29 GiB | Separate video-tool collector observed 7.25 GiB |
| Owned cleanup | No survivors or errors | Both inner and outer cleanup confirmed |

Aggregate request accounting is 633,552 input, 538,276 cached and 29,404 completion tokens across repeated requests. Those are cumulative counts, not unique context size. Sampling cannot exclude short unobserved resource peaks or establish an OS memory reservation.

Four actual disk checkpoint restores returned prefixes of 11,555, 19,814, 11,132 and 24,974 tokens. The first supervisor requests after the worker returns reused 19,814 and 24,974 tokens and took 49.59 and 72.82 seconds including their new input and generation. This demonstrates reuse across the tested handoffs; there is no matched cache-off media run from which to claim a whole-task speedup.

## Quality and remaining costs

The output video is 960 x 544, 141 frames at 24 fps (5.875 seconds), with non-silent 32 kHz stereo audio. The image and sampled frames show a usable racing scene with distinct cars and coherent track structure. They retain a glossy, CGI-like appearance and fine-detail limitations. Selected-frame vision checks do not prove every frame, continuous motion or audio semantics. The D3-derived temporal diagnostic is not a calibrated probability of realism.

One avoidable cost remains: a worker recursively printed timestamp fields from a receipt, producing a 29,269-character terminal response. Its next model call processed 21,634 new tokens despite good prefix reuse; prefill took 257.85 seconds. A bounded receipt-summary proposal has passed focused checks and immutable receipt replay, but has not yet demonstrated end-to-end time savings. This run is not evidence that the delegation guidance eliminates all redundant inspection or oversized tool output.

Automatic history compression was **not exercised**. Separate qualification is still needed for normal compression and continuation, foreground CPU/GPU contention, and the actual Desktop pipeline. General process-loss recovery is also outside this run. No guarantee against OOM, unlimited resource elasticity, arbitrary detached tools, or perfect autonomy is made.

## Exact implementation boundaries and attribution

- The serving composition extends the runtime in public Strata draft `999e1279cae89ab1b0055309575672c7ea515143` (v0.1.42 base). Local Python differences include opt-in request-correlation headers and a bounded persistent-checkpoint shutdown wait. Those differences are not attributed to the unchanged public head.
- The tested native binary SHA-256 is `1d33e452f2c0fc7a963f7cc8a5aac673bc5255104845904c3090a59c8b7ad603`. Its checkpoint bank incorporates xrip's [PR #1937](https://github.com/Niko1221/Strata/pull/1937) at `1d52b02c18973db3d468df88d5120c4eea9b7813`, plus separately reviewed local bounded-read checks and adaptive integration. Later author revisions were not tested in this media run. The bank was enabled; the separate current-session SAVE feature was off.
- The tested Hermes paired source is `873419f6bfc7c19cd18406d4da91027b4cbcf65e`. The terminal middleware seams match [Hermes #135418](https://github.com/NousResearch/hermes-agent/pull/135418); the completed-worker evidence changes match [#136266](https://github.com/NousResearch/hermes-agent/pull/136266). The newer private-child-environment changes in public head `c14b6711b9243ab1381b02ec2345936498e0cde1` were not part of this media run; their separate tests remain separately reported.
- Existing [implementation provenance](../../../docs/ADAPTIVE_PROVENANCE.md) applies: medking82's [#726](https://github.com/Niko1221/Strata/pull/726) supplies the allocator foundation, and the narrow vision cleanup adapts weituotian's [#1862](https://github.com/Niko1221/Strata/pull/1862).
- The media stack uses [ComfyUI](https://github.com/Comfy-Org/ComfyUI) and [City96's ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), with local CPU mmap-detach and BF16 scratch-allocation corrections. They belong to that loader integration, not Strata. Existing licensing and notices remain. They were applied together here, so this trial does not isolate their individual memory or speed gains.

The path-audit correction and reporting proposal are local integration work, not imported algorithms. No TensorFold implementation was used. Source provenance is separated from related-work discussion in the linked provenance document. Public evidence excludes private prompts, local paths, credentials, session identifiers and raw histories.

## Useful community review

The main review targets are admission/release ownership, cancellation and uncertain tool completion, checkpoint compatibility across unload/reload, and portable host integration. Reproductions on other machines would help distinguish hardware-specific behavior from protocol defects. Please keep the sampled-memory and draft implementation limits in reports, and include the exact component revisions.

This report records one functional experiment, not a ready-to-run portable release. The existing PRs expose the reusable engine and Hermes extension pieces; packaging the complete host integration and qualifying Desktop operation remain work in progress.
