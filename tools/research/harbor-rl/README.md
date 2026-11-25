# Harbor + OpenEnv Closed-Loop RL on Strata (`+0 MiB` Extra VRAM)

Zero-extra-VRAM closed-loop RL trainer (`train_strata_harbor_rl.py`) that adapts a Strata-served model to real coding-agent harnesses (**Codex**, **OpenCode**, etc.) inside **Harbor** Docker sandboxes using **OpenEnv** token-level capture and **MeZO-LoRA**.

## How It Works

This trainer builds directly on the primitives introduced in `feat/lora-hip-gfx1151`:

1. **Rollout capture**: `POST /v1/chat/completions` (`return_token_ids=True`, `logprobs=True`) behind OpenEnv's `CaptureServer`, advertised to local Docker sandboxes over the Docker bridge gateway (`172.17.0.1`).
2. **Teacher-forced scoring (`+0 MiB` VRAM)**: `POST /v1/score` with opt-in frozen-layer frontier caching, skipping layers `0 .. first_adapted_layer - 1` bit-exactly when scoring MeZO perturbations.
3. **In-place LoRA hot reload**: `POST /lora-adapters` updates the active GGUF LoRA weights in-place (`~20–45 ms`, zero graph re-capture).
4. **Zero-order MeZO-LoRA update**: Supports `grpo`, `rft`, and `hybrid` (GRPO with RFT fallback) over the last `K` layers' LoRA `B` matrices.

## Files

- `train_strata_harbor_rl.py` — End-to-end closed-loop Harbor + OpenEnv + MeZO-LoRA trainer for Strata.
- `prepare_termgrade_local.py` — Patches `termgrade-train-harbor` `task.toml` files to bake `setup.sh` via per-task `environment/Dockerfile` on top of `codex-termgrade-base:v1` and switches `network_mode = "no-network"` to `"public"` for local Docker.
- `docker/Dockerfile.codex-termgrade` — Local base image (`ubuntu:24.04` + Python 3.12 + `/opt/verifier_venv` with `pytest==8.4.1` & `pytest-json-ctrf` + PEP 668 unblocked + Node 22 + `@openai/codex`).

## Quick Start

```bash
# 1. Build the local Codex base image
docker build -f tools/research/harbor-rl/docker/Dockerfile.codex-termgrade \
  -t codex-termgrade-base:v1 tools/research/harbor-rl/docker

# 2. Prepare local Harbor tasks (e.g. from harness-hub/termgrade-train-harbor)
python3 tools/research/harbor-rl/prepare_termgrade_local.py \
  --src /path/to/termgrade-train-harbor/tasks \
  --dst /path/to/termgrade-codex \
  --limit 100

# 3. Run closed-loop Harbor RL on Strata
python3 tools/research/harbor-rl/train_strata_harbor_rl.py \
  --dataset /path/to/termgrade-codex \
  --base-config /path/to/strata-config.json \
  --strata-exe /path/to/build/strata \
  --harness codex \
  --sandbox docker \
  --group-size 2 \
  --mode hybrid \
  --zo-dirs 1 \
  --max-score-turns 3 \
  --frontier auto \
  --frontier-mib 4096 \
  --reasoning-effort none \
  --adapt-layers 4 \
  --rank 4 \
  --lr 2e-4 \
  --work-dir ./harbor-rl-run
```

## Verification (AMD Strix Halo `gfx1151`, `Qwen3.8-Flash-Next-IQ3_XXS`, `--harness codex --sandbox docker`)

### 1. Synthetic smoke (`write-42`)
- Step 1: `rewards=[1.0, 1.0]`, `turns=[2, 2]`, `trainable_tokens=904`, `mode=rft_fallback`, `accepted=true`, `delta_j=+0.000309`, `reload_ms_total=38.4 ms`.
- Step 2: `rewards=[1.0, 1.0]`, `turns=[2, 2]`, `trainable_tokens=1066`, `reload_ms_total=24.3 ms`.

### 2. Real TermGrade 12-step run (`harness-hub/termgrade-train-harbor`, `G=2`, `mode=hybrid`, `r=4`, last 4 layers)

| Step | Task | Rewards | Mode | Trainable Tokens | Accepted | Delta J | `|B|` | LoRA Reload |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `task_00615_72cd3a` | `[1.0, 1.0]` | `rft_fallback` | 39,534 | True | `+0.000192` | `0.00021` | `38.6 ms` |
| 2 | `task_00699_a765e2` | `[0.0, 1.0]` | `grpo` | 14,444 | False | `0.0` | `0.00021` | `20.2 ms` |
| 4 | `task_00035_00b7e3` | `[1.0, 0.0]` | `grpo` | 29,121 | True | `+0.000094` | `0.00130` | `45.0 ms` |
| 5 | `task_00704_18a23b` | `[1.0, 1.0]` | `rft_fallback` | 10,108 | False | `0.0` | `0.00130` | `43.4 ms` |

- 4 of 8 built tasks produced non-zero rewards (50%), triggering both true `grpo` contrast updates and `rft_fallback` updates, with `+0 MiB` extra GPU VRAM and `20–45 ms` hot reloads (`final_adapter.gguf`, 545 KiB, `|B|=0.00130`).
- Subsequent PEP 668 fix (`PIP_BREAK_SYSTEM_PACKAGES=1` + removing `EXTERNALLY-MANAGED`) resolved the `pip install pytest` build failures on the remaining TermGrade `setup.sh` scripts.
