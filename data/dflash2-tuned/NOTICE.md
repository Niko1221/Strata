# Strata's tuned DFlash2 drafter (Apache-2.0, modified by Strata)

`qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack` is a **modified copy** of the DFlash2 draft model
[incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), revision
`a22a1744e85c17364dcc54a6a420af7135d4c7f6`, Copyright Inco AI, licensed under the Apache License, Version 2.0 (the text is in `LICENSE`
beside this file). The upstream card names Inco AI as the author; the upstream repository carries no separate NOTICE file.

**What Strata changed:** the weights. Every parameter of the published draft model (603 M) was fine-tuned by Strata on the text that
Qwen3.8-Flash-Next quantizations (Unsloth UD-Q4_K_XL and UD-IQ4_XS-v31) wrote for Strata-made prompts (repository files and documents,
synthetic records; no user data), then packed into the engine's runtime format (Q4_0 matrices, Q4_0 selector tables; the layout of
`tools/dflash2_pack.py --q4_0-selector`). The pack is the same size as the unmodified one (339,694,592 bytes). The model code and the
architecture are unchanged. The pack's SHA-256 is `4df2e5ab2b47c064afbe662e16e4f06569877125c2d0e2123a4ac755266c664c`.

On Macs Strata runs it with `--draft-conf 0.7`; on CUDA (every expert resident) and on Strix Halo it runs at the confidence those platforms already use. It changes how many words are guessed right, never the words
the model writes: the text is the target model's, checked word for word against the unmodified drafter on the measured sets.

Training on the outputs of the Qwen3.8-Flash-Next quantizations: the owner of Strata confirmed on 2026-11-06 that the Qwen model terms allow
training on its outputs.

The idea of the Q4_0 selector tables is nazeshinjite's (Apache-2.0): see `docs/DETAILS.md`, "Credits and licenses".

This file and `LICENSE` are copied next to the drafter file (`<data folder>/dflash2/`) when Strata fetches it, and are part of Strata's own
tree. Strata's code stays MIT ([`LICENSE`](../../LICENSE)); this drafter's weights are Apache-2.0.
