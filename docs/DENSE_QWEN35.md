# Dense `qwen35` (Qwen3.5 / Qwen3.8 4B) — port plan

Status: plan, no code yet. Fork of `niko1221/strata` (upstream stays `origin`; this fork is `fork`).

## Why this is small (and why it is not free)

Qwen3.5-4B and Qwen3.8-4B distills carry `general.architecture = qwen35` (dense). Metadata read from
`unsloth/Qwen3.5-4B-GGUF` and `empero-ai/Qwen3.8-4B-Distill-GGUF` shows the **same hybrid skeleton** the
engine already runs for `qwen35moe`: identical GDN keys (`ssm.conv_kernel 4`, `ssm.state_size 128`,
`ssm.group_count 16`, `ssm.time_step_rank 32`, `ssm.inner_size 4096`), `full_attention_interval 4`,
`embedding_length 2560`, `attention.head_count 16 / kv 4`, `head_dim 256`.

The difference is only the FFN:

| | `qwen35moe` (runs today) | `qwen35` (this port) |
|---|---|---|
| FFN | 256 experts, top-8 + shared expert | one dense MLP (`ffn_gate/up/down.weight`, 9216) |
| router | `ffn_gate_inp.weight` | none |
| MTP block | `blk.<n>.nextn.*` | same keys, present in the 3.8-4B distill (`nextn_predict_layers = 1`) |
| head | `output.weight` | absent (tied) in the 3.8-4B distill → `token_embd.weight` |

So the GPU work already exists: the MoE path's **shared expert** is a dense MLP and is computed both per token
(`q36_engine.cu:1701`) and in prefill (`q36_engine.cu:2061-2072`). The port wires the dense FFN into that path
and removes routing; it does not write new kernels.

## Milestones

**M0 — recognition.** `ModelFamily::Qwen35` (`model_family.hpp`), `check_architecture_qwen35` in
`gguf_reader.hpp` (no `expert_*` keys required, `nextn_predict_layers` optional), geometry branch in
`family_layout.hpp:31` (`n_ff` from `feed_forward_length`, `n_expert = 0`, `n_ff_shexp = n_ff`,
`hc = idx = 0`), dense layout table in `layer_tensors` (`ffn_gate/up/down.weight` instead of
`ffn_*_inp/_shexp/_exps`), `layout_table_complete` for the dense family, family switch at
`src/program/generate.cpp:2981`. Exit test: `strata generate` loads the file and refuses templates/args it
cannot honour instead of "architecture is 'qwen35'".
Unit tests: `tests/core/model_family_test.cpp`, layout shapes.

**M1 — loader.** Dense branch in `src/qwen35/q36_engine.cu` load: `sh_gate/up/down ← ffn_*`,
`n_ff_shexp ← n_ff`, skip router/expert blobs, head from `output.weight` else `token_embd.weight`.

**M2 — one token.** Greedy continuation for a fixed prompt vs `llama.cpp` (`llama-cli -p … -n 32 --temp 0`,
build `b10387` here already knows `LLM_ARCH_QWEN35`). Top-1 agreement token by token; that is the real gate.

**M3 — prefill + MTP.** Batch path parity, then `--mtp` on/off (a community distill's `nextn` block may be a
donor head: keep `--mtp` optional per model, never default on).

**M4 — serve + CLI.** `FAMILY_MODEL_NAMES`, model card sampling, chat template **read from the GGUF**
(`tokenizer.chat_template`, present: 7 756 chars) instead of a hardcoded jinja, `setup.py` family entry so
`strata launch --family dense35 --gguf-dir ~/models/qwen38-4b` writes a run config; then `pi` through
`~/.pi/agent/models.json`.

**M5 — shrink the diff.** Re-read upstream `CONTRIBUTING.md`, keep the patch upstream-shaped (one family, no
kernel changes), open a PR if it lands cleanly; otherwise this branch is the only owner.

## Risks

1. **Tied head.** `token_embd.weight` is Q6_K in the distill, `output.weight` is absent; the head matmul must
   accept the embedding tensor (check `src/core/native_head.cpp` and the `--native` variant).
2. **Compiled geometry pins.** `compiled_geometry_mismatch` is only consulted on the Flash-Next path
   (`generate.cpp:4335`), so the q36 path's runtime-shaped kernels should carry 2560/9216/128/16/32/4/4096 —
   confirm before M1, this is the one assumption that could still sink the plan.
3. **MTP quality** from a distilled model — see M3, never enabled by default.
4. **Model quality** (a 4B distill of a 1T-class teacher) is a separate question from architecture; the port
   is judged by parity with llama.cpp, not by how good the model is.
5. **Maintenance**: `strata update` must keep using the locally built engine (`engine/BUILD.json`
   `"source": "local"`); after any upstream pull the engine is rebuilt, not replaced.

## Verification rule

No claim of "works" without: `./setup.sh --check` output, a build with `-DCMAKE_CUDA_ARCHITECTURES=75`,
`cuobjdump --list-elf` showing `sm_75`, and a llama.cpp parity run quoted verbatim.
