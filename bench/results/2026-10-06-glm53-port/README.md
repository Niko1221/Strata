# GLM-5.3 port measurements, 2026-10-06

Ryzen 9 8940HX (16 cores), 63 GB RAM, RTX 5070 Ti Laptop 12 GB, Kingston NV3 1 TB (D:), model
`D:\models\GLM-5.3-colibri-int4-g64`, `--expert-ram-gb 28 --threads 16`. The engine was the `build\strata-glm.exe`
of that afternoon: before the prompt-attention and multi-row kernels in `src/glm/kernels.cpp` (docs/GLM53.md).

| Files | What | Result |
| --- | --- | --- |
| `baseline-tf.*`, `cpu-f32-tf.*`, `gpu-f32-tf.*` | `--tf` over colibri's 19 reference tokens | argmax = colibri at 12 of 12 positions |
| `cpu-bf16-tf.*`, `cpu-fp8-tf.*` | the same with `--kv bf16` / `--kv fp8` | 11 of 12 (position 13: `1947`, a near-tie in f32: 20.18 vs 20.47) |
| `baseline-chunk{256,1024,2048}.*` | 1,536-token prompt (`prompt1536.txt`), chunked, CPU | 604.5 / 508.5 / 484.5 s; 1,756 / 662 / 360 GB read |
| `cpu-layer1536.*` | the same prompt, whole prompt per layer, CPU | 417.6 s (attention 186.5 s, experts 222.8 s), 359 GB read |
| `gpu-layer1536.*` | the same, `--gpu 0` | 146.5 s (attention 25.0 s, experts 119.5 s of which 44.3 s waiting for reads) |
| `gpu-chunk1536.*` | the same, `--gpu 0 --prefill chunk --prefill-chunk 256` | 1,496 s, 1.75 TB read, 1,138 s waiting for reads |
| `cpu-decode.*` | 100 tokens after a 7-token prompt | stopped at 35 tokens: the drive had fallen to 0.1-0.36 GB/s (see docs/GLM53.md, "The drive") |

`baseline.json` and `validation.json` are the two driver scripts' records (`build-work/baseline.py`,
`tools/glm_validate.py`). The raw logits (`*.f32`, `--logits-out`) are not kept in git.
