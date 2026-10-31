# Dense `qwen35` (Qwen3.5 / Qwen3.8 4B) — port plan

Status: **M0 + M1 done, greedy parity with llama.cpp** (guard, geometry, layout tables, family switch; 37/37 unit checks, and the real
`Qwen3.8-4B-Q4_K_M` header reads as 32+1 MTP layers, 0 experts, FF 9216). Fork of `niko1221/strata` (upstream stays
`origin`; this fork is `fork`).

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

**M2 — one token.** Greedy continuation for a fixed prompt, **through `pi`** (the harness we actually use), against the
same request sent to `llama.cpp`'s server (build `b10387` here already knows `LLM_ARCH_QWEN35`). Two comparisons: the
prompt's **token count** must match exactly (that proves template and tokenizer), then the greedy text. `llama.cpp` is
the reference engine, not the harness. Top-1 agreement token by token is the gate.

**M3 — prefill + MTP.** Batch path parity, then `--mtp` on/off (a community distill's `nextn` block may be a
donor head: keep `--mtp` optional per model, never default on).

**M4 — serve + CLI.** `FAMILY_MODEL_NAMES`, model card sampling, chat template **read from the GGUF**
(`tokenizer.chat_template`, present: 7 756 chars) instead of a hardcoded jinja, `setup.py` family entry so
`strata launch --family dense35 --gguf-dir ~/models/qwen38-4b` writes a run config; then `pi` through
`~/.pi/agent/models.json`.

**M5 — shrink the diff.** Re-read upstream `CONTRIBUTING.md`, keep the patch upstream-shaped (one family, no
kernel changes), open a PR if it lands cleanly; otherwise this branch is the only owner.

## Reuse (prefix cache) — checked before the port

The `qwen35` engine path keeps conversation state **in RAM, per session/slot**: `match_len()` (`q36_engine.cu:3394`)
requires the held tokens to be a proper prefix of the new prompt *and* the same image keys; `pick_session()` (`:3407`)
takes the longest match, else an empty slot, else the LRU. Any divergence gives `reuse = 0` and `E.reset()` — the whole
prompt is read again. It is append-only by construction ("the recurrent state cannot step back", `:3509`).

No disk checkpoints and no `--prompt-cache*` flags on this path; those belong to the Flash-Next engine
(`src/core/conversation_*`, geometry-keyed). So a restart, an evicted slot, or a rewritten history (a harness's
compaction) is a cold read. Flash-Next's cache writes `geometry_key()` (18 fields) and refuses a zero in any field but
the layer count (`conversation_state.cpp:20-118`), so a dense family (`n_expert = 0`) cannot use that machinery without
a family-aware rule — noted, not planned.

The port touches none of this; the dense family inherits the same append-only, in-RAM reuse. Gate for M2/M3: a 3-turn
conversation through `pi` must show `reused` > 0 on turns 2..N (the server prints `prompt N tokens = M reused + K
read`); `tools/harness_reuse_probe.py` measures the interleaved-conversation case. The second-brain workload gets its
own benchmark, separate from this gate.

## Where it stands (M0+M1 done, first parity run)

`empero-ai/Qwen3.8-4B-Distill-GGUF` (`Q4_K_M`, 2.78 GB, `qwen35`, 32 layers + 1 MTP, tied head) runs on the q36 engine
path on the GTX 1650 (4 GB): **same tokens as llama.cpp**, greedy, from one prompt:

| prompt | llama.cpp (`279MB` server, temp 0) | strata |
|---|---|---|
| `The capital of France is` (760,6511,314,9338,369) | `11751 13 561 6511 314 9564 369 19241 …` = " Paris. The capital of Germany is Berlin." | **identical 16 tokens** |
| `The` (760) | top-1 `2614` | `2614` |

What the port needed, beyond M0's tables:

* loader: `ffn_gate/up/down.weight` into the shared-expert slot, no router/expert blobs, `router_ext` skipped; a tied head
  (no `output.weight`) uses `token_embd.weight` as `m.head` (same `[n_embd, n_vocab]`, no second copy); `p` (the metadata
  key prefix) follows the family's architecture string.
* `f32()` (host) now dequantizes Q4_K/Q5_K/Q6_K/IQ4_* blocks through `strata/artifact/dequant.hpp` — the 4B distill stores
  `ssm_alpha`/`ssm_beta` as Q4_K.
* layer: a dense branch after `add_rms_post` (SwiGLU + `sh_down`, then the new `k_dense_out`: `x += mlp`,
  `h = rms(x)·wn`); `k_moe_out` cannot serve it, it always reads its first expert's row.
* `place_experts()` returns at once for `n_expert == 0` (it divided by the expert count: SIGFPE).
* memory for a 4 GB card, all of it measured on the card: the auto-chunk rule budgets for expert weights that are not
  there (it picked 2048 rows and OOMed), so the dense family takes chunk 512; the prompt GEMM's 45 MB dequant scratch is
  not attempted below 640 MB free (`apply()`/`prep()` then read prompts on the native MMVQ — slower prefill, exact);
  the attention-score buffer starts at 16 MB instead of 256 MB. **The ceiling is ~2K cells** (ctx 2048: 0.02 GiB free,
  ctx 4096: OOM) — the weights are 2.59 GiB of the card.

Two open items, both found by the parity run and neither fixed:

1. **`warmup()` leaves state that `reset()` does not clear**: token 760 gives 89342 after a warm-up and 2614 from a clean
   start. The warm-up is therefore off for the dense family (`run()`), which is a correctness workaround, not a fix — the
   same `reset()` runs between two sequences, so this is a multi-turn/reuse item.
2. **CUDA graph capture** was suspected first and is not the cause (same logits with and without); the graph stays on.

Measured here (GTX 1650 4 GB, ctx 1024, chunk 512, prompt GEMM off): decode **37.5 tok/s**, prompt **87-91 tok/s**
(the GEMM path would raise prefill; there is no room for its scratch on a 4 GB card).

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
