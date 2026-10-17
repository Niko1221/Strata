# Huihui abliterated UD-IQ4_XS

Validated on Linux with 2x RTX 4090 (24 GB each), a Xeon Silver 4309Y and 188 GB of RAM on 2026-10-10 and 2026-10-11.

The target is the `unsloth-UD-IQ4_XS` folder of
[`huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF)
at revision `06a318de4011598a142ee54bcfec47c6921c958e`: an abliterated (refusal-removed) Qwen3.8-Flash-Next quantized
in the layout of Unsloth's `UD-IQ4_XS`. Shard 1 is byte for byte Unsloth's; shards 2 and 3 have Unsloth's sizes and
other bytes (the abliterated weights). The pack has the same expert formats as Unsloth's UD-IQ4_XS (43 layers IQ3_S
gate/up with IQ4_NL down, 4 with Q8_0 down, 1 IQ4_XS gate/up with Q8_0 down; 59.5 GB of experts), so it runs with
the same settings and at the same speed.

The model's safety alignment is reduced; its own license applies (see its page). Refusal behaviour was not measured
here.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | 10,946,624 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` | 49,835,229,856 | `c67b468b3b83bc12a987e455ec4a0c2c3937250da1c234ae1638d9505180c161` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` | 43,836,407,744 | `130560b6a447e0febf61de9b03476f52143160bc4bcbacc9bfa07d9538718e2c` |

## Preparation

Put the three shards in one folder, for example `Strata-data/models/huihui-UD-IQ4_XS`, and check the hashes above
(`sha256sum`). Then pack it as setup packs Unsloth's UD-IQ4_XS:

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf ../Strata-data/models/huihui-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --out ../Strata-data/packs/huihui-ud-iq4_xs --compat-bf16
```

The model card suggests making `verify_sha256` in `setup.py` return early; that is not needed for this path, which
does not go through setup's download.

The draft layer is the base model's (`mtp/rt` in `Strata-data`), as for the other packs; it accepted as many drafts
here as with Unsloth's file (49-56% of proposed tokens per run, against 50-58% for Unsloth's, in the runs below).

## Local server

Copy the config setup wrote for Unsloth's UD-IQ4_XS (`strata-unsloth-ud-iq4_xs.json`) to
`strata-huihui-ud-iq4_xs.json` and change five values: `--pack` (the new pack), `--native` (the new shard 1),
`"tokenizer"` (`<pack>/tokenizer`), `"model_name"` and `"log"`. Then:

```sh
.venv/bin/python -m serve.server --engine strata --config strata-huihui-ud-iq4_xs.json
```

`./update.sh` keeps this config up to date like setup's own (the 0.1.42 update reported it "up to date").

## Measured

All runs on the machine above, engine 0.1.41 built from source with CUDA 13.0, the same config for every model:
layer split at 24 with `--trim-stage-weights`, `"parallel": 4` (pipelined, 2 groups of 2), 262,144 context,
`--kv int8 --kv-resident 32768`. Each model: a fresh start, one warm-up round, then 5 rounds; medians. Nothing else
used the server during the runs. The two OrcaRouter files ([ORCA.md](ORCA.md)'s source repository, packed the same
way with `--compat-bf16`) are included because they are the other uncensored option people ask about.

| | Unsloth UD-IQ4_XS | Huihui UD-IQ4_XS | OrcaRouter IQ4_XS | OrcaRouter Q5_K_M |
| --- | ---: | ---: | ---: | ---: |
| Expert bytes | 59.5 GB | 59.5 GB | 65.4 GB | 91.9 GB |
| Decode, one request (400 tokens), tok/s | 112.8 | 110.5 | 107.8 | 87.4 |
| Prompt, 30K tokens, tok/s | 4,032 | 4,036 | 3,727 | 2,284 |
| 4 concurrent requests (300 tokens each), total tok/s | 189.2 | 189.4 | 185.9 | 83.4 |
| 13 short tasks (tool calls, JSON, code run against tests, 20K-token needle, ...), temperature 0, `max_tokens` 6000 | 12/13 | 12/13 | 12/13 | - |
| base64 set: 4 strings x encode/decode x 3 runs, no `max_tokens` | 21/24 | 17/24 | 12/24 | 16/24 |
| - of which encode / decode | 11/12, 10/12 | 8/12, 9/12 | 5/12, 7/12 | 8/12, 8/12 |

The base64 set (`foz hau jiko`, `Hello, Strata!`, `kiwi mango`, `Привет`; one run at temperature 0 and two at 1.0 for
each, `reasoning_effort` medium, no tools) is a character-level task the model has to work out in its thinking, so it
shows small precision differences. Its run-to-run spread is large: Unsloth's file encoded `foz hau jiko` correctly in
1 of 6 runs in an earlier session and in 3 of 3 here. In the 13-task set each model failed one base64 task, cut by the
6,000-token limit (Unsloth and Huihui the encode, OrcaRouter IQ4_XS the decode). These are smoke tests, not a quality benchmark.

What the table shows for the uncensored choices on this machine: Huihui's file is as fast as Unsloth's and solved
more of the base64 set than both OrcaRouter files; OrcaRouter's IQ4_XS reads prompts 8% slower and solved the fewest;
OrcaRouter's Q5_K_M, the larger file, did not make up for it and is 23% slower in decode, 43% in prompts and 56% with
four clients (its experts are 54% larger than Unsloth's, so fewer of them fit in VRAM).

## Scope

- Only the `unsloth-UD-IQ4_XS` folder at the revision above was checked; the repository's other folders (UD-Q4_K_XL,
  Q8_0, BF16, Swift) were not.
- One machine, NVIDIA, a 2-GPU layer split. Single-GPU, AMD and Intel runs are not measured; given the identical
  layout they should behave like Unsloth's UD-IQ4_XS there.
