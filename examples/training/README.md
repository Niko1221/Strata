# `examples/training/` - runnable samples for the training support

This tree is the **example layer** of the opt-in training support. Its sibling
[`training/`](../../training/README.md) holds the implementation, tests, probes,
docs and evaluated evidence; this directory holds only things a reader can copy
and run, or copy and replace with their own data.

Upstream Strata has no `examples/` tree. It is introduced here because shipping
sample `*.jsonl` inputs, a demo transcript and sample outputs inside the
implementation directory made `training/` a mixture of code and fixtures. The
split follows upstream's split-by-responsibility convention. Runtime asset
paths are resolved by `training/paths.py` from `__file__`; nothing here depends
on the caller's working directory.

All commands below assume the **repository root** as cwd, so sample paths are
written `examples/training/<name>`.

| file | what it is |
|---|---|
| `DEMO-IDENTITY.md` | measured before/after transcript of the 125B identity demo, its cost and the three silent failures |
| `data_developer.jsonl` | SFT sample: 10 identity pairs + 10 capability-replay pairs (`{dev}` placeholder resolved at runtime) |
| `data_dpo.jsonl` | 10 preference pairs (mechanism demo only - the `rejected` side is hand-written, **not** on-policy; see `PROTOCOL-FULLCTX-EVAL.md` section 10) |
| `data_dpo_128k.jsonl` | long-context DPO sample |
| `data_identity8.jsonl` | 8-item identity eval split derived from `data_developer.jsonl` by `training/bench/prep_eval_data.py` |
| `data_long_128k.jsonl` | 128K long-context SFT sample |
| `math_12.jsonl` | 12-item math eval split (4 train / 4 dev / 4 holdout) |
| `prompts_close.jsonl` | GRPO sample prompts with a shaped reward (word count) |
| `prompts_holdout.jsonl` | 12 held-out probe prompts used by the backprop SFT capability gate |
| `prompts_identity.jsonl` | 7 identity probe prompts used by both A/B scripts |
| `rollouts_close.jsonl` (+ `.meta.json`) | replayed GRPO rollouts sample |
| `rollouts_128k.jsonl` | long-context replayed rollouts sample |
| `mapping_q4.json` | default GGUF -> HF tensor-name map for `training/backward/loader_qwen4.py` |

Minimal zero-order run (see `training/README.md` for the full loop):

```bash
python3 training/train.py --list
python3 training/train.py --data examples/training/data_developer.jsonl \
    --steps 40 --out /tmp/lora.npz
```

Replacing any of these files with your own data is the intended usage: pass the
path explicitly (`--data`, `--dpo-data`, `--rollouts`, `--prompts`, ...). The
defaults only point here so `--list` and the no-torch test suite have something
concrete to validate.
