#!/usr/bin/env bash
set -euo pipefail

# Launch Codex in WSL against the LAN-accessible dual RTX 5090 + RTX 3090
# llama.cpp server. This only changes the child Codex process; plain codex
# keeps its normal OpenAI configuration. This launcher intentionally runs the
# child with full host access and no approval prompts: use only for trusted work.

DUAL_CODEX_WORKDIR="${DUAL_CODEX_WORKDIR:-$PWD}"
DUAL_CODEX_SANDBOX="${DUAL_CODEX_SANDBOX:-danger-full-access}"
DUAL_CODEX_PROXY_URL="${DUAL_CODEX_PROXY_URL:-http://127.0.0.1:8090}"
DUAL_CODEX_MODEL="${DUAL_CODEX_MODEL:-}"
DUAL_CODEX_CONTEXT_WINDOW="${DUAL_CODEX_CONTEXT_WINDOW:-}"
DUAL_CODEX_AUTO_COMPACT_TOKEN_LIMIT="${DUAL_CODEX_AUTO_COMPACT_TOKEN_LIMIT:-}"
DUAL_CODEX_STREAM_IDLE_TIMEOUT_MS="${DUAL_CODEX_STREAM_IDLE_TIMEOUT_MS:-600000}"
DUAL_CODEX_STATE_DIR="${DUAL_CODEX_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/dual-codex}"
DUAL_CODEX_PROVIDER_ID="local_llama_cpp_proxy"

if [[ "${1:-}" == "resume" && $# -lt 2 ]]; then
  echo "usage: dual_codex_wsl.sh resume <session-id|--last> [prompt]" >&2
  exit 2
fi

for dependency in codex python3 curl; do
  if ! command -v "$dependency" >/dev/null 2>&1; then
    echo "dual_codex_wsl: missing required command: $dependency" >&2
    exit 1
  fi
done

if ! curl --max-time 3 --silent --fail "$DUAL_CODEX_PROXY_URL/health" >/dev/null; then
  echo "dual_codex_wsl: shared proxy is unavailable at $DUAL_CODEX_PROXY_URL" >&2
  exit 1
fi

runtime_payload="$(curl --max-time 5 --silent --fail "$DUAL_CODEX_PROXY_URL/props" 2>/dev/null || true)"
runtime_values=""
if [[ -n "$runtime_payload" ]]; then
  runtime_values="$(printf '%s' "$runtime_payload" | python3 -c '
import json, sys
try:
    payload = json.load(sys.stdin)
    model = payload.get("model_alias")
    context = payload.get("default_generation_settings", {}).get("n_ctx")
    ftype = payload.get("model_ftype", "unknown")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("missing model_alias")
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise ValueError("missing positive default_generation_settings.n_ctx")
    build = payload.get("build_info")
    backend = "strata" if isinstance(build, str) and build.startswith("Strata ") else "llama.cpp"
    print(f"{model}\t{context}\t{ftype}\t{backend}")
except (ValueError, TypeError, json.JSONDecodeError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    raise SystemExit(1)
')" || runtime_values=""
fi

if [[ -z "$runtime_values" ]]; then
  models_payload="$(curl --max-time 5 --silent --fail "$DUAL_CODEX_PROXY_URL/v1/models" 2>/dev/null || true)"
  runtime_values="$(printf '%s' "$models_payload" | python3 -c '
import json, sys
try:
    payload = json.load(sys.stdin)
    models = payload.get("data")
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError("expected exactly one model in /v1/models")
    model = models[0].get("id")
    context = models[0].get("meta", {}).get("n_ctx")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("missing data[0].id")
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise ValueError("missing positive data[0].meta.n_ctx")
    print(f"{model}\t{context}\tunknown\tunknown")
except (ValueError, TypeError, json.JSONDecodeError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    raise SystemExit(1)
')" || runtime_values=""
fi

if [[ -z "$runtime_values" ]]; then
  echo "dual_codex_wsl: could not discover model/context from $DUAL_CODEX_PROXY_URL/props or /v1/models" >&2
  exit 1
fi

IFS=$'\t' read -r discovered_model discovered_context discovered_ftype runtime_backend <<< "$runtime_values"
model="${DUAL_CODEX_MODEL:-$discovered_model}"
context_window="${DUAL_CODEX_CONTEXT_WINDOW:-$discovered_context}"
if ! [[ "$context_window" =~ ^[1-9][0-9]*$ ]]; then
  echo "dual_codex_wsl: invalid context window: $context_window" >&2
  exit 1
fi
if [[ -z "$DUAL_CODEX_AUTO_COMPACT_TOKEN_LIMIT" ]]; then
  if (( context_window >= 262144 )); then
    auto_compact=220000
  else
    auto_compact=$((context_window * 84 / 100))
  fi
else
  auto_compact="$DUAL_CODEX_AUTO_COMPACT_TOKEN_LIMIT"
fi
if ! [[ "$auto_compact" =~ ^[1-9][0-9]*$ ]] || (( auto_compact >= context_window )); then
  echo "dual_codex_wsl: auto-compaction limit must be positive and below context=$context_window" >&2
  exit 1
fi
if ! [[ "$DUAL_CODEX_STREAM_IDLE_TIMEOUT_MS" =~ ^[1-9][0-9]*$ ]]; then
  echo "dual_codex_wsl: invalid stream idle timeout: $DUAL_CODEX_STREAM_IDLE_TIMEOUT_MS" >&2
  exit 1
fi
if [[ "$model" != "$discovered_model" ]]; then
  echo "dual_codex_wsl: refusing model override '$model'; live stack exposes '$discovered_model'" >&2
  exit 1
fi
if [[ "$context_window" != "$discovered_context" ]]; then
  echo "dual_codex_wsl: refusing context override '$context_window'; live stack exposes '$discovered_context'" >&2
  exit 1
fi
catalog_args=()
if [[ "$runtime_backend" == "strata" ]]; then
  # Strata exposes an OpenAI model list, not Codex's ModelInfo catalog.
  # A per-child catalog avoids /models decode errors without another API proxy.
  catalog_setting="$(printf '%s' "$runtime_payload" | DUAL_CODEX_STATE_DIR="$DUAL_CODEX_STATE_DIR" python3 -c '
import hashlib, json, os, sys, tempfile
from pathlib import Path
props = json.load(sys.stdin)
model, context, compact, endpoint = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
modalities = ["text", "image"] if props.get("modalities", {}).get("vision") else ["text"]
# Codex requires an instruction template in a custom catalog. Reuse its cached
# agent instructions when available; do not load credentials or conversation history.
instructions = None
cache_path = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "models_cache.json"
try:
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    for cached_model in cached.get("models", []):
        candidate = cached_model.get("base_instructions") or (cached_model.get("model_messages") or {}).get("instructions_template")
        if isinstance(candidate, str) and candidate.strip():
            instructions = candidate
            break
except (OSError, ValueError, TypeError):
    pass
if not instructions:
    instructions = (
        "You are a coding agent running in Codex CLI. Complete the user task in the shared workspace using the provided tools. "
        "Follow the system, developer, user and applicable AGENTS.md instructions. Inspect relevant files before editing; "
        "preserve unrelated work. Use apply_patch for file edits and the shell tools for execution and verification. "
        "Do not invent tool results, tests or completion. For implementation tasks, implement and test the result, "
        "then report what changed and any remaining limitations. For questions or diagnostics, inspect and explain "
        "without making unrequested changes. Send concise progress updates during lengthy work. "
        "Do not stop after a plan or partial implementation when execution is requested."
    )
entry = {
    "slug": model, "display_name": model,
    "description": "Local Strata native Responses backend; metadata discovered from /props",
    "default_reasoning_level": "medium",
    "supported_reasoning_levels": [{"effort": effort, "description": effort} for effort in ("none", "low", "medium", "high")],
    "shell_type": "unified_exec", "visibility": "list", "supported_in_api": True, "priority": 0,
    "availability_nux": None, "upgrade": None,
    "supports_parallel_tool_calls": props.get("chat_template_caps", {}).get("supports_parallel_tool_calls", True),
    "supports_reasoning_summaries": True, "supports_reasoning_summary_parameter": True,
    "default_reasoning_summary": "none", "support_verbosity": False, "default_verbosity": None,
    "apply_patch_tool_type": "freeform", "truncation_policy": {"mode": "tokens", "limit": 10000},
    "context_window": context, "max_context_window": context, "auto_compact_token_limit": compact,
    "effective_context_window_percent": 95, "experimental_supported_tools": [], "input_modalities": modalities,
    "base_instructions": instructions,
}
catalog = {"models": [entry]}
digest = hashlib.sha256((endpoint + json.dumps(catalog, sort_keys=True)).encode()).hexdigest()[:20]
directory = Path(os.environ["DUAL_CODEX_STATE_DIR"])
directory.mkdir(parents=True, exist_ok=True)
target = directory / ("strata-model-catalog-" + digest + ".json")
with tempfile.NamedTemporaryFile("w", dir=directory, prefix=".catalog-", delete=False) as handle:
    json.dump(catalog, handle)
    temporary = Path(handle.name)
temporary.replace(target)
print("model_catalog_json=" + json.dumps(str(target)))
' "$model" "$context_window" "$auto_compact" "$DUAL_CODEX_PROXY_URL")"
  catalog_args=(-c "$catalog_setting")
fi
echo "dual_codex_wsl: model=$model context=$context_window backend_ftype=$discovered_ftype auto_compact=$auto_compact sandbox=$DUAL_CODEX_SANDBOX approvals=never" >&2

codex \
  --no-alt-screen \
  -C "$DUAL_CODEX_WORKDIR" \
  -s "$DUAL_CODEX_SANDBOX" \
  -a never \
  -c "model=\"$model\"" \
  -c "model_provider=\"$DUAL_CODEX_PROVIDER_ID\"" \
  -c "model_context_window=$context_window" \
  -c "model_auto_compact_token_limit=$auto_compact" \
  "${catalog_args[@]}" \
  -c "model_providers.$DUAL_CODEX_PROVIDER_ID={name=\"Local stack Responses API\",base_url=\"$DUAL_CODEX_PROXY_URL/v1\",wire_api=\"responses\",requires_openai_auth=false,stream_idle_timeout_ms=$DUAL_CODEX_STREAM_IDLE_TIMEOUT_MS}" \
  "$@"
