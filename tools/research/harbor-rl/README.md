# Harbor + OpenEnv Closed-Loop RL on Strata (`+0 MiB` Extra VRAM)

Zero-extra-VRAM closed-loop RL trainer (`train_strata_harbor_rl.py`) that adapts a Strata-served model to real coding-agent harnesses (**Codex**, **OpenCode**, etc.) inside **Harbor** Docker sandboxes using **OpenEnv** token-level capture and **MeZO-LoRA**.

## How It Works

This trainer builds directly on the primitives introduced in `feat/lora-hip-gfx1151`:

1. **Rollout capture**: `POST /v1/chat/completions` (`return_token_ids=True`, `logprobs=True`) behind OpenEnv's `CaptureServer`, advertised to local Docker sandboxes over the Docker bridge gateway (`172.17.0.1`).
2. **Teacher-forced scoring (`+0 MiB` VRAM)**: `POST /v1/score` with opt-in frozen-layer frontier caching, skipping layers `0 .. first_adapted_layer - 1` bit-exactly when scoring MeZO perturbations.
3. **In-place LoRA hot reload**: `POST /lora-adapters` updates the active GGUF LoRA weights in-place (`~24–38 ms`, zero graph re-capture).
4. **Zero-order MeZO-LoRA update**: Supports `grpo`, `rft`, and `hybrid` (GRPO with RFT fallback) over the last `K` layers' LoRA `B` matrices.

## Files

- `train_strata_harbor_rl.py` — End-to-end closed-loop Harbor + OpenEnv + MeZO-LoRA trainer for Strata.
- `prepare_termgrade_local.py` — Patches `termgrade-train-harbor` `task.toml` files to use a local prebuilt Codex image and `network_mode = "public"` for local Docker.
- `docker/Dockerfile.codex-termgrade` — Local base image (`ubuntu:24.04` + Python 3.12 + `pytest` + Node 22 + `@openai/codex`) so Harbor skips online CLI installation inside sandboxes.

## Quick Start

```bash
# 1. Build the local Codex base image
docker build -f tools/research/harbor-rl/docker/Dockerfile.codex-termgrade   -t codex-termgrade-base:v1 tools/research/harbor-rl/docker

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

## Smoke Verification (AMD Strix Halo `gfx1151`, `Qwen3.8-Flash-Next-IQ3_XXS`)

- `write-42` closed-loop smoke (`codex` + local Docker):
  - Step 1: `rewards=[1.0, 1.0]`, `turns=[2, 2]`, `trainable_tokens=904`, `mode=rft_fallback`, `accepted=true`, `delta_j=+0.000309`, `reload_ms_total=38.4 ms`.
  - Step 2: `rewards=[1.0, 1.0]`, `turns=[2, 2]`, `trainable_tokens=1066`, `reload_ms_total=24.3 ms`.
