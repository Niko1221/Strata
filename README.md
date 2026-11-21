<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="exstrata-logo-dark.svg">
    <img src="exstrata-logo.svg" alt="exstrata" width="340">
  </picture>
</p>

<h3 align="center">Strata, reworked to use less memory on one consumer PC.</h3>
<p align="center">An experimental fork of <a href="https://github.com/Niko1221/Strata">Strata</a> for large MoE models whose experts live in system RAM.</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-experimental-orange" alt="Status: experimental">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/commit/3142b66b9"><img src="https://img.shields.io/badge/based%20on-Strata%203142b66-555" alt="Based on Strata 3142b66"></a>
</p>

**exstrata is a fork of Strata.** It keeps Strata's engine and changes how much memory a model needs. Every change is opt-in: with nothing switched on, exstrata runs exactly as Strata does. exstrata is not affiliated with or endorsed by the Strata project.

## Current design

These are the directions exstrata is working on now. They are plans and work in progress, not finished features.

### Use less memory

The target is Qwen3.8-Flash-Next: about 10% less VRAM plus RAM than Strata needs for the same model, with the same task accuracy.

- **A smaller n-gram table.** The per-layer n-gram (PLE) table is the second-largest thing in RAM. exstrata stores it in a 3-bit format, which takes 62 bytes a row instead of 90.
- **Smaller fixed VRAM.** The dense weights go into one allocation instead of many. The draft head reads its rows from the main head instead of keeping a copy.
- **Reduce, do not offload.** exstrata makes the stored data smaller. It does not move data to disk, because that hurts machines with slow SSDs.

### Draft for an engine whose experts are on the CPU

Every token in a verify window reads experts from RAM, whether or not the token is accepted, so a wrong draft costs memory bandwidth. exstrata tunes the drafting with that cost in mind:

- choosing the MTP drafter's settings;
- comparing the MTP drafter with the DFlash2 drafter;
- ending a verify window when the expected accepted tokens no longer pay for the reads.

### Keep the model as capable as before

A change may alter the output slightly. It is accepted only if task accuracy does not drop.

1. A fast teacher-forcing check against Strata screens each change.
2. The final check is a paired comparison on verifiable tasks: subsets of GSM8K, MATH-500, HumanEval, MMLU-Pro and JCommonsenseQA.

## Relationship to Strata

exstrata is its own repository, not a GitHub fork. Its history starts at Strata's `main` commit 3142b66, and later Strata changes are merged into the `exstrata` branch from time to time. The "based on" badge names the Strata commit the code is currently based on.

## Building and running

Building and running work as they do in Strata. See [Build from source](docs/SOURCE_INSTALL.md) and the [install guide](docs/INSTALL.md). The opt-in switches that exstrata adds will be listed here once they land.

## License and credits

exstrata is released under the MIT license, as Strata is. [LICENSE](LICENSE) keeps Strata's copyright notice: "Copyright (c) 2026 Niko1221 and the Strata contributors". The engine, tools and documentation come from [Strata](https://github.com/Niko1221/Strata). Please report problems that also happen in Strata to Strata, and problems specific to exstrata here.
