# On-device autotuning for the RX 7900 XTX (gfx1100)

Strata's AMD backend runs CUDA-shaped kernels whose launch shapes were chosen on an RTX 5070. This adds
two measured layers of tuning for the 7900 XTX, in the spirit of engines that tune kernels on the user's
device instead of shipping one shape for a whole class of hardware. Both layers keep answers exactly the
same.

| Layer | What is measured | Changes answers? | Cost |
|---|---|---|---|
| Kernels | Rows per thread block of the decode pass's GPU expert kernels (gate/up, down) and `iq_mmvq` | No: bitwise-equal variants only, checked on the machine and again end to end | ~1–3 min + 4 engine starts |
| Engine settings | Draft floor (`--spec-min-p`), CPU expert workers (`--pool-workers`) | No (greedy verification keeps the tokens) | 5–10 min |
| Drafting (opt-in) | MTP window (`--spec`), prompt-lookup match (`--suffix-draft`) | No | +6 engine starts |

## Run it

Stop the server first: the timings need the GPU idle and about 1 GB of free VRAM. Then:

```
python tools/hip/autotune.py strata-iq3_xxs.json                 # kernels only
python tools/hip/autotune.py strata-iq3_xxs.json --settings      # + draft floor and CPU workers
python tools/hip/autotune.py strata-iq3_xxs.json --draft --prompts my-agent-prompts.json
python tools/hip/autotune.py strata-iq3_xxs.json --off           # back to the default kernel shapes
```

`--prompts` takes a JSON list of strings, or plain text with prompts separated by a line `---`. Use requests your
agent really sends. Prompt lookup only pays where the answer repeats the context (edits, refactors, quoting),
so the built-in prompts are a weak proxy for an agent workload.

The script needs the `tune_decode` tool. It builds it in `build-hip` when that build directory exists, or you
can build it yourself:

```
cmake --build build-hip --target tune_decode -j2
```

When it finishes, the config's `env` carries `STRATA_DECODE_TUNING` (next to `STRATA_HIPBLASLT_TUNING`), and the
engine log says `decode tuning: N shapes from ...` at start. A report goes to `strata-<model>.autotune.json`, and the
original config is kept once as `strata-<model>.json.bak-autotune`.

The settings (`--settings`, `--draft`) also work through setup: `./setup.sh --calibrate`, `--calibrate-draft`,
`--calibrate-prompts FILE`. On AMD the PCIe share is left exactly as the config has it, because the HIP expert
paths (mmap, resident CPU experts, the staging buffer) were measured with that share. `tools/calibrate.py
--sweep-pcie` sweeps it anyway.

## How the kernel layer stays exact

`native_expert_grouped` (the GPU experts in every verify window) and `iq_mmvq` give one warp to each weight row.
How many rows share a block changes occupancy and scheduling but never the arithmetic: each row is still reduced
by one warp, over the same lanes, in the same order. `tune_decode` checks this anyway:

1. Every candidate's output is compared **byte for byte** with the default's before it may be timed. A
   difference excludes the candidate, fails the run (exit 4), and nothing is written.
2. The workload is the verify pass's: windows of 1, 3 and 5 tokens (weighted 0.2/0.3/0.5), `--hit-frac` of each
   token's 10 experts on the GPU, and experts drawn from a 768 MB pool. The pool is far larger than the 96 MB
   Infinity Cache, because decode streams around a gigabyte of experts per token and a cache-warm benchmark would lie.
3. Candidates are timed interleaved over several rounds, in a rotating order. A winner must beat the default by more
   than 3% on the weighted median, then again in a second interleaved confirmation.
4. `autotune.py` then starts the engine without and with the table, twice each, interleaved. It keeps the table
   only if the generated tokens are identical and decode is not slower. If two runs *without* the table already
   differ (an adaptive expert tier, for example), the token check is reported as not checkable and speed decides.

The table is refused at engine start, with a message, unless its header matches the running engine: the GPU
architecture, the HIP runtime version and a hash of the compiler that built the kernels. A rebuild with another
ROCm needs a new run. CUDA builds compile only the default shapes and ignore the variable.

## What to expect

Honestly: unknown until it runs on your card. The tunable kernels are a slice of each token's time. In the
hybrid setup, CPU-side expert work and the GPU-reach wait are large too, and kernel shapes cannot touch those.
Measure where your time goes before and after:

```
"env": {"STRATA_DECODE_TIMING": "1", ...}
```

With this set, each request logs one line: windows, average window size, tokens per window, and ms per window
split into verify (GPU-reach wait, per-layer host and CPU expert time), commit and draft. It also logs VRAM hits,
CPU expert entries and PCIe experts per layer-window, followed by the GPU stage table. If "verify" is dominated by
the CPU expert part, the drafting and worker settings matter more than kernel shapes.

## Drafting notes for Qwen3.8-Flash-Next

- **MTP** is the model's own draft layer. Its window grows while the draft's confidence is at least
  `--spec-min-p`. Every extra row in a window brings more experts the VRAM cache misses onto the CPU. On a
  CPU-bound machine a higher floor or a shorter `--spec` often wins, so it is measured, not assumed.
- **Prompt lookup** (`--suffix-draft`) drafts the continuation of an earlier repeat of the context. A built-in
  policy learns the measured round cost per window size and lookup acceptance per match length, and takes a lookup
  window only when its expected tokens per millisecond beat the MTP's. It fires only when its first token agrees
  with the MTP's first guess.
- **DFlash2** (block-diffusion drafters) exists for Qwen3.8-27B, not for Flash-Next. A drafter for Flash-Next would
  have to be trained, and it would take VRAM from the expert cache (about 700 experts per GB). This is not part of
  this tuning.
