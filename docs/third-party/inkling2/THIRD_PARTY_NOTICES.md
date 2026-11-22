# Third-party notices

YASAKANI itself is licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`). This file lists what else is in the release, what is
only required, and which published work it builds on.

## Redistributed with YASAKANI

### Qwen3.8-27B (model, tokenizer and chat template)

* By the Qwen team, Alibaba Cloud. License: Apache License 2.0 - the `LICENSE` file of the model repository is the Apache License 2.0 text with "Copyright 2026 Alibaba Cloud" (it is shipped with the model files; on 2026-11-12 the model page listed this license and carried the same `LICENSE` file).
* Original: <https://huggingface.co/Qwen/Qwen3.8-27B>
* The files `qwen38_27b_quality.yskn` (quality model), `qwen38_27b_speed.yskn` (fastest model) and `qwen38_27b_max_quality.yskn`
  (max-quality model) are **derived** from it: converted to the
  YASAKANI file format and compressed (quantized). They are not the original weights. `tokenizer.json`, `tokenizer_config.json`,
  `chat_template.jinja`, `config.json` and `generation_config.json` are the original files, unchanged.

### DFlash 2 draft model for Qwen3.8-27B (helper model)

* By z-lab (a mirror of `incoai/Qwen3.8-27B-DFlash2`, credited to Inco AI). The Hugging Face page lists the Apache License 2.0.
* Original: <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2> - background: <https://inco.ai/blog/dflash2/> and <https://github.com/z-lab/dflash>
* The files `bdraft2_k8.yskn` and `bdraft2_k8.shead` are **derived** from it: converted to the YASAKANI file format, compressed (quantized) and
  re-trained by the YASAKANI authors against a compressed Qwen3.8-27B model (an earlier YASAKANI conversion). They are not the original
  weights.

### nlohmann/json 3.11.3 (source, `third_party/nlohmann/json.hpp`)

* Copyright (c) 2013-2023 Niels Lohmann <https://nlohmann.me> - MIT License - <https://github.com/nlohmann/json>
* The header is included unmodified (`third_party/nlohmann/PROVENANCE.txt` records its SHA-256).

```
MIT License

Copyright (c) 2013-2023 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### NVIDIA CUDA Runtime (inside `yskn.dll` and `yskn_inkling2.dll`)

* The prebuilt `yskn.dll` and, since 0.3.1, `yskn_inkling2.dll` (the engine for Inkling-2 Mini) each contain the NVIDIA CUDA Runtime library
  (`cudart`), linked statically (`cudart_static.lib`; the release is built
  with CUDA Toolkit 13.2: nvcc 13.2.78, CUDA Runtime 13.2.75). NVIDIA's CUDA Toolkit End User License Agreement lets an application distribute the files that its
  Attachment A lists; for the CUDA Runtime on Windows the list is `cudart.dll`, `cudart_static.lib` and `cudadevrt.lib`. The runtime inside
  each library is used only by that engine itself. See <https://docs.nvidia.com/cuda/eula/> (sections 1.1.1, 1.1.2, 2.2 and 2.6).
* The Apache License 2.0 of YASAKANI does **not** apply to this component. It stays under NVIDIA's agreement; please do not take it out of
  `yskn.dll` or `yskn_inkling2.dll` for other use.
* The agreement text was read on 2026-10-31 in its CUDA 13.3 and 13.4 versions ("last updated January 26, 2026"); both say the same about the
  runtime. The `EULA.txt` installed with the CUDA 13.2 toolkit that built the release DLL was compared on 2026-11-12: it carries the
  same date, has the sections named above, and lists the same three files for the CUDA Runtime on Windows.

### CPython `Lib/textwrap.py` (an input of the speed measurement)

* The code-edit task of the speed tables holds this file of CPython as its input and, with one local variable renamed (`margin` to
  `common_margin` inside `dedent`), as its expected answer: in `bench/20261103/prompts/cases.json`, in the answers recorded under `bench/`,
  and since 0.1.6 in the package file `python/yasakani/edit_tasks.json` of `yasakani bench --edit` (also in the engine zip). Changes: the
  e-mail address in its header is replaced by `<email>` (except in `bench/20261108/demo/`, kept as recorded); nothing else.
* Its header: Copyright (C) 1999-2001 Gregory P. Ward; Copyright (C) 2002, 2003 Python Software Foundation.
  PSF's notice: Copyright (c) 2001 Python Software Foundation; All Rights Reserved.
  License: the PSF License Agreement (below, as in the `LICENSE.txt` of Python 3.14); <https://docs.python.org/3/license.html>.

```
PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
--------------------------------------------

1. This LICENSE AGREEMENT is between the Python Software Foundation
("PSF"), and the Individual or Organization ("Licensee") accessing and
otherwise using this software ("Python") in source or binary form and
its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
analyze, test, perform and/or display publicly, prepare derivative works,
distribute, and otherwise use Python alone or in any derivative version,
provided, however, that PSF's License Agreement and PSF's notice of copyright,
i.e., "Copyright (c) 2001 Python Software Foundation; All Rights Reserved"
are retained in Python alone or in any derivative version prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on
or incorporates Python or any part thereof, and wants to make
the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to Python.

4. PSF is making Python available to Licensee on an "AS IS"
basis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON,
OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between PSF and
Licensee.  This License Agreement does not grant permission to use PSF
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using Python, Licensee
agrees to be bound by the terms and conditions of this License
Agreement.
```

## Required, but not redistributed (you install them)

| What | Why | License | Tested with |
|---|---|---|---|
| NVIDIA display driver | runs the GPU | NVIDIA driver license | 616.92 |
| Microsoft Visual C++ 2015-2022 x64 runtime | `yskn.dll` and `yskn_inkling2.dll` are built with Visual C++ (`yskn_inkling2.dll` also uses its OpenMP library, `vcomp140.dll`) | Microsoft software license terms | - |
| Python 3.10 or newer | the command-line tool and server | PSF License | 3.14.4 on Windows 11 (the unit tests also ran on 3.10) |
| numpy | array handling | BSD-3-Clause | 2.4.4 and 2.5.4 |
| tokenizers (Hugging Face) | tokenizer | Apache-2.0 | 0.22.2 and 0.23.3 |
| jinja2 | chat template | BSD-3-Clause | 3.1.6 |

`pip` also installs what these packages themselves depend on (for example `huggingface-hub` with `tokenizers`, and `MarkupSafe` with `jinja2`),
each under its own license; `pip list` and `pip show <package>` give the exact versions and licenses on your machine. YASAKANI loads the
tokenizer from the local model folder only and never downloads anything.

The first version in a row is that of the PC where the release candidate was tested (2026-10-31). Where there is a second one, it is
what `pip` picked in a fresh virtual environment on 2026-11-12 (jinja2 was 3.1.6 again); the self-check passed there for both models.

### Optional: image input (since 0.1.2)

Needed only if you start the server with `--mmproj`, `--llama-dir` and `--vision-model` (see `python/README.md`). None of these files
is in the release; you download them yourself.

| What | Why | License | Tested with |
|---|---|---|---|
| llama.cpp, CPU build for Windows x64 (`llama-b10488-bin-win-cpu-x64.zip`, <https://github.com/ggml-org/llama.cpp/releases/tag/b10488>) | its multimodal library (`mtmd.dll`, with `llama.dll` and the `ggml` libraries) turns an image into features | MIT | b10488 |
| llama.cpp b10488 built from source (Linux only; <https://github.com/ggml-org/llama.cpp/tree/b10488>) | the same libraries as shared libraries (`libmtmd.so`, `libllama.so`, the `ggml` libraries), built by you | MIT | b10488 (by the contributor of #8) |
| `mmproj-Qwen3.8-27B-BF16.gguf` (<https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF>) | the image encoder of Qwen3.8-27B in the GGUF format | made from Qwen3.8-27B (Apache-2.0); see the model page | the file with SHA-256 `de2a49866988ea272c43dc2a43ee2662c25725a717ca5b9059ea420a97f53fe8` |
| a Qwen3.8-27B GGUF file | settings and vocabulary for llama.cpp; its weights are not read | made from Qwen3.8-27B (Apache-2.0); see the page you download it from | `Qwen3.8-27B-Q4_K_M.gguf` of <https://huggingface.co/bartowski/Qwen3.8-27B-GGUF> (17,442,399,968 bytes); other files not tried |
| Pillow | reads the image (`pip install ".[vision]"`) | MIT-CMU | 13.0.0 |

`python/yasakani/_mtmd.py` declares, with Python's ctypes, the structures and functions of the llama.cpp b10488 headers (`llama.h`,
`mtmd.h`) that it calls; no llama.cpp source file or binary is part of the release.

## Published work this builds on (no code taken from it unless listed above)

* Jian Chen, Yesheng Liang, Zhijian Liu. *DFlash: Block Diffusion for Flash Speculative Decoding.* arXiv:2602.06036, February 2026.
  The helper model above is a drafter of this kind.
* Qwen team. The Qwen3.8 model family (see above).
* Thinking Machines Lab. Inkling-2 Mini (<https://huggingface.co/thinkingmachines/Inkling-2-Mini>, Apache License 2.0; its model card
  also names a Model Acceptable Use Policy, <https://thinkingmachines.ai/model-acceptable-use-policy/>). The engine in `src/inkling2/` and
  the reader `python/yasakani/_ink2_ckpt.py` compute what the published model code computes (the Inkling-2 code proposed for Hugging Face
  transformers, <https://github.com/huggingface/transformers/pull/51050>, Apache License 2.0); they were written for YASAKANI. The model
  files are not part of the release: the converter `tools/convert/inkling2_convert.py` makes a model folder from the files you download
  and writes the model's notice into it.

## Not included

The measurement kit under `bench/` refers to other engines (llama.cpp, SGLang, vLLM, TensorRT-LLM) and to test sets published by others
(NInfer, AIME 2025) by name and checksum only. Their code and data are not redistributed; each has its own license.
