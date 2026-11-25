#!/usr/bin/env python3
"""Zero-extra-VRAM Closed-Loop Harbor + OpenEnv + Codex/OpenCode RL Trainer on Strata.

Architecture:
  1. Strata HTTP Server (`serve.server --engine strata`) with:
       - POST /v1/chat/completions (`return_token_ids=True`, `logprobs=True`)
       - POST /v1/score            (teacher-forced logprobs + frozen-layer frontier cache, +0 MiB VRAM)
       - POST /lora-adapters       (in-place GGUF LoRA hot reload, zero graph re-capture)
  2. OpenEnv `CaptureServer` (in-process intercept proxy on 0.0.0.0:<capture_port>,
     advertised over the Docker bridge IP e.g. 172.17.0.1:<capture_port>)
  3. Harbor Docker sandboxes running real coding harnesses (`codex`, `opencode`, etc.)
     with real Verifier rewards
  4. Zero-Order MeZO-LoRA (GRPO / RFT / Hybrid) updating the last-K-layer LoRA B matrices
     and hot-reloading into Strata every step with +0 MiB extra GPU VRAM.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import secrets
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# GGUF v3 constants (matching Strata's lora.cu / train_strata_rl.py)
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGML_TYPE_F32 = 0
ALIGNMENT = 32


# ---------------------------- Minimal GGUF LoRA State ----------------------------

class LoraAdapterState:
    """In-memory LoRA state for Qwen3.8-Flash-Next (48 layers, QSA at l % 4 == 3, GDN elsewhere).

    Freezes A (random orthogonal-ish init) and trains B (zero-init) on the last `adapt_last_n`
    layers so SCORE frontier caching skips layers `0 .. first_adapted_layer - 1` bit-exactly.
    """

    def __init__(
        self,
        n_layers_total: int = 48,
        adapt_last_n: int = 4,
        rank: int = 4,
        alpha: float | None = None,
        n_in: int = 6144,
        n_out: int = 2560,
        seed: int = 42,
    ) -> None:
        self.n_layers_total = int(n_layers_total)
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.layers = list(range(max(0, self.n_layers_total - int(adapt_last_n)), self.n_layers_total))

        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(self.n_in)
        self.A: dict[int, np.ndarray] = {}
        self.B: dict[int, np.ndarray] = {}
        for l in self.layers:
            self.A[l] = (rng.standard_normal((self.rank, self.n_in)) * scale).astype(np.float32)
            self.B[l] = np.zeros((self.n_out, self.rank), dtype=np.float32)

    @property
    def n_trainable_params(self) -> int:
        return sum(b.size for b in self.B.values())

    def param_norm(self) -> float:
        return float(math.sqrt(sum(float(np.sum(b * b)) for b in self.B.values())))

    def perturb_inplace(self, seed: int, scale: float) -> None:
        rng = np.random.default_rng(int(seed))
        for l in self.layers:
            z = rng.standard_normal(self.B[l].shape).astype(np.float32)
            self.B[l] += np.float32(scale) * z

    def _tensor_base(self, layer: int) -> str:
        return f"blk.{layer}.attn_output.weight" if (layer % 4 == 3) else f"blk.{layer}.ssm_out.weight"

    def write_gguf(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors: list[tuple[str, np.ndarray]] = []
        for l in self.layers:
            base = self._tensor_base(l)
            tensors.append((f"{base}.lora_a", np.ascontiguousarray(self.A[l], dtype=np.float32)))
            tensors.append((f"{base}.lora_b", np.ascontiguousarray(self.B[l], dtype=np.float32)))

        kvs = [
            ("general.type", GGUF_TYPE_STRING, "adapter"),
            ("adapter.type", GGUF_TYPE_STRING, "lora"),
            ("adapter.lora.alpha", GGUF_TYPE_FLOAT32, float(self.alpha)),
        ]

        def enc_str(s: str) -> bytes:
            b = s.encode("utf-8")
            return struct.pack("<Q", len(b)) + b

        header = bytearray()
        header += GGUF_MAGIC
        header += struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(kvs))
        for k, vtype, val in kvs:
            header += enc_str(k)
            header += struct.pack("<I", vtype)
            if vtype == GGUF_TYPE_STRING:
                header += enc_str(str(val))
            elif vtype == GGUF_TYPE_FLOAT32:
                header += struct.pack("<f", float(val))

        offset = 0
        for name, arr in tensors:
            # GGUF dims are reversed relative to numpy (ne0 = cols, ne1 = rows)
            ne0, ne1 = int(arr.shape[1]), int(arr.shape[0])
            header += enc_str(name)
            header += struct.pack("<IQQIQ", 2, ne0, ne1, GGML_TYPE_F32, offset)
            nbytes = arr.nbytes
            offset += (nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

        pad = (ALIGNMENT - (len(header) % ALIGNMENT)) % ALIGNMENT
        header += b"\x00" * pad

        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(header)
            for _, arr in tensors:
                raw = arr.tobytes()
                f.write(raw)
                rem = len(raw) % ALIGNMENT
                if rem:
                    f.write(b"\x00" * (ALIGNMENT - rem))
        tmp.replace(path)


# ---------------------------- RL Objective & Advantages ----------------------------

def group_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = math.sqrt(var_r)
    if std_r < 1e-8:
        return [0.0 for _ in rewards]
    return [(r - mean_r) / (std_r + 1e-8) for r in rewards]


def compute_episode_advantages(
    verifier_rewards: list[float],
    turn_counts: list[int],
    *,
    mode: str = "hybrid",
    w_eff: float = 0.1,
) -> tuple[list[float], list[float], str]:
    """Return (advantages, shaped_rewards, mode_used)."""
    shaped: list[float] = []
    for vr, n_turns in zip(verifier_rewards, turn_counts):
        eff = (w_eff / (1.0 + max(1, int(n_turns)) / 4.0)) if vr > 0.0 else 0.0
        shaped.append(float(vr) + eff)

    if mode == "rft":
        advs = [1.0 if vr > 0.0 else 0.0 for vr in verifier_rewards]
        return advs, shaped, "rft"

    advs = group_advantages(shaped)
    if any(abs(a) > 1e-6 for a in advs):
        return advs, shaped, "grpo"

    if mode == "hybrid" and any(vr > 0.0 for vr in verifier_rewards):
        return [1.0 if vr > 0.0 else 0.0 for vr in verifier_rewards], shaped, "rft_fallback"

    return [0.0 for _ in verifier_rewards], shaped, "no_signal"


def luspo_objective(
    cur_lps: list[float],
    old_lps: list[float],
    ref_lps: list[float],
    advantages: list[float],
    *,
    clip_eps: float = 0.2,
    kl_beta: float = 0.02,
) -> float:
    total = 0.0
    n = 0
    for lp, old_lp, ref_lp, adv in zip(cur_lps, old_lps, ref_lps, advantages):
        if abs(adv) < 1e-12:
            continue
        delta = max(-10.0, min(10.0, lp - old_lp))
        ratio = math.exp(delta)
        clipped = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio))
        surr = min(ratio * adv, clipped * adv)
        kl = max(-10.0, min(10.0, lp - ref_lp))
        # Schulman k3 approx (non-negative) or linear KL penalty
        kl_pen = math.exp(-kl) - 1.0 + kl
        total += surr - kl_beta * kl_pen
        n += 1
    return total / max(1, n)


# ---------------------------- Strata HTTP RL Client ----------------------------

class StrataHttpRLClient:
    """Talks to a running Strata HTTP server for /health, /v1/score, and /lora-adapters."""

    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _req(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Strata {method} {path} failed ({exc.code}): {detail}") from exc

    def health(self) -> dict[str, Any]:
        return self._req("GET", "/health")

    def lora_adapters(self) -> Any:
        return self._req("GET", "/lora-adapters")

    def lora_reload(self, gguf_path: Path, scale: float = 1.0) -> float:
        t0 = time.monotonic()
        resp = self._req("POST", "/lora-adapters", {"path": str(gguf_path), "scale": float(scale)})
        if isinstance(resp, dict) and resp.get("ms") is not None:
            return float(resp["ms"])
        return (time.monotonic() - t0) * 1000.0

    def score_sequence(
        self,
        tokens: list[int],
        from_pos: int,
        *,
        lora: bool = True,
        frontier: str | int | None = "auto",
    ) -> tuple[float, int, float]:
        payload: dict[str, Any] = {
            "tokens": [int(t) for t in tokens],
            "from": int(from_pos),
            "lora": bool(lora),
        }
        if frontier is not None and str(frontier) != "0":
            payload["frontier"] = frontier
        resp = self._req("POST", "/v1/score", payload)
        count = int(resp["count"])
        mean_lp = float(resp["sum_logprob"]) / max(1, count)
        ms = float(resp.get("ms", 0.0))
        return mean_lp, count, ms


# ---------------------------- Trace Extraction ----------------------------

@dataclass
class EpisodeTrace:
    """One Harbor rollout's scoreable turns + verifier reward."""

    task_name: str
    reward: float
    turns: list[tuple[list[int], int]]  # list of (full_token_ids, from_pos)
    n_trainable_tokens: int
    wall_s: float
    error: str | None = None


def extract_episode_turns(
    result: Any,
    *,
    max_context: int,
    max_score_turns: int,
) -> EpisodeTrace:
    """Convert OpenEnv HarborRolloutResult into exact (tokens, from_pos) tuples for /v1/score."""
    turns_out: list[tuple[list[int], int]] = []
    trainable_toks = 0
    raw_turns = list(getattr(result, "turns", None) or [])
    for t in raw_turns:
        if not getattr(t, "trainable", True):
            continue
        pids = [int(x) for x in (getattr(t, "prompt_token_ids", None) or [])]
        cids = [int(x) for x in (getattr(t, "completion_token_ids", None) or [])]
        if not pids or not cids:
            continue
        full = pids + cids
        if len(full) > max_context:
            # Skip turns exceeding engine context rather than corrupting prefix tokens
            continue
        from_pos = len(pids) - 1
        turns_out.append((full, from_pos))
        trainable_toks += len(cids)

    if max_score_turns > 0 and len(turns_out) > max_score_turns:
        # Keep first turn (initial plan/tool call) + most recent turns (final edits & finish)
        turns_out = [turns_out[0]] + turns_out[-(max_score_turns - 1) :]

    return EpisodeTrace(
        task_name=str(getattr(result, "task_name", "unknown")),
        reward=float(getattr(result, "reward", 0.0) or 0.0),
        turns=turns_out,
        n_trainable_tokens=trainable_toks,
        wall_s=float(getattr(result, "wall_s", 0.0) or 0.0),
        error=getattr(result, "error", None),
    )


def discover_tasks(dataset_path: Path) -> list[Path]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    if (dataset_path / "task.toml").is_file():
        return [dataset_path]
    found = sorted(p for p in dataset_path.iterdir() if p.is_dir() and (p / "task.toml").is_file())
    if not found:
        raise ValueError(f"No Harbor tasks (task.toml) found under {dataset_path}")
    return found


# ---------------------------- Managed Strata Server ----------------------------

def build_managed_config(
    base_config_path: Path,
    out_config_path: Path,
    *,
    strata_exe: str,
    lora_path: Path,
    host: str,
    port: int,
    max_context: int,
    frontier_mib: int,
    reasoning_effort: str,
    log_path: Path,
) -> dict[str, Any]:
    cfg = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    cfg["exe"] = strata_exe
    cfg["cwd"] = str(Path(strata_exe).resolve().parent.parent)
    cfg["log"] = str(log_path)
    cfg["host"] = host
    cfg["port"] = int(port)
    cfg.pop("api_key", None)
    cfg.pop("parallel", None)
    cfg.pop("reasoning_budget_tokens", None)
    if reasoning_effort:
        cfg["reasoning_effort"] = reasoning_effort

    args: list[str] = []
    raw = list(cfg.get("args") or [])
    i = 0
    while i < len(raw):
        a = raw[i]
        if a in ("--batch", "--max-context", "--lora", "--conversation-cache-mib"):
            i += 2
            continue
        if a == "--batch-mtp":
            i += 1
            continue
        args.append(a)
        i += 1
    args += [
        "--max-context",
        str(max_context),
        "--conversation-cache-mib",
        "2048",
        "--lora",
        str(lora_path),
    ]
    cfg["args"] = args
    env = dict(cfg.get("env") or {})
    env["STRATA_SCORE_FRONTIER_MIB"] = str(int(frontier_mib))
    cfg["env"] = env

    out_config_path.parent.mkdir(parents=True, exist_ok=True)
    out_config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def start_strata_server(
    *,
    strata_python: str,
    strata_terminal_dir: Path,
    config_path: Path,
    host: str,
    port: int,
    server_log: Path,
    ready_timeout_s: float = 180.0,
) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{strata_terminal_dir}:{existing_pp}" if existing_pp else str(strata_terminal_dir)
    log_fh = open(server_log, "wb")
    proc = subprocess.Popen(
        [
            strata_python,
            "-m",
            "serve.server",
            "--engine",
            "strata",
            "--config",
            str(config_path),
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    client = StrataHttpRLClient(f"http://{host}:{port}", timeout=10.0)
    t0 = time.monotonic()
    while time.monotonic() - t0 < ready_timeout_s:
        if proc.poll() is not None:
            log_fh.close()
            tail = server_log.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"Strata server exited early (rc={proc.returncode}):\n{tail}")
        try:
            client.health()
            return proc
        except Exception:
            time.sleep(1.0)
    proc.terminate()
    raise TimeoutError(f"Strata server did not become healthy within {ready_timeout_s}s")


def stop_strata_server(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


# ---------------------------- Unit Self-Checks ----------------------------

def run_unit_checks() -> None:
    # 1. Hybrid advantage fallback: mixed -> grpo, all-pass equal -> rft_fallback, all-fail -> no_signal
    adv_mix, _, m_mix = compute_episode_advantages([1.0, 0.0], [2, 3], mode="hybrid")
    assert m_mix == "grpo" and adv_mix[0] > 0 > adv_mix[1]

    adv_eff, _, m_eff = compute_episode_advantages([1.0, 1.0], [2, 6], mode="hybrid", w_eff=0.1)
    assert m_eff == "grpo" and adv_eff[0] > 0 > adv_eff[1]

    adv_eq, _, m_eq = compute_episode_advantages([1.0, 1.0], [2, 2], mode="hybrid", w_eff=0.1)
    assert m_eq == "rft_fallback" and adv_eq == [1.0, 1.0]

    adv_fail, _, m_fail = compute_episode_advantages([0.0, 0.0], [2, 4], mode="hybrid")
    assert m_fail == "no_signal" and adv_fail == [0.0, 0.0]

    # 2. Objective monotonicity
    old = [-0.4, -0.4]
    j0 = luspo_objective(old, old, old, [1.0, -1.0])
    j1 = luspo_objective([-0.3, -0.5], old, old, [1.0, -1.0])
    assert j1 > j0

    # 3. LoRA perturb + exact undo + GGUF header
    lora = LoraAdapterState(n_layers_total=48, adapt_last_n=4, rank=4)
    assert lora.layers == [44, 45, 46, 47]
    lora.perturb_inplace(777, 1e-3)
    assert lora.param_norm() > 0.0
    lora.perturb_inplace(777, -1e-3)
    assert lora.param_norm() < 1e-9

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.gguf"
        lora.write_gguf(p)
        assert p.read_bytes()[:4] == GGUF_MAGIC

    # 4. Episode extraction
    class FakeTurn:
        def __init__(self, pids, cids, trainable=True):
            self.prompt_token_ids = pids
            self.completion_token_ids = cids
            self.trainable = trainable

    class FakeResult:
        task_name = "demo"
        reward = 1.0
        wall_s = 12.3
        error = None
        turns = [FakeTurn([1, 2, 3], [4, 5]), FakeTurn([1, 2, 3, 4, 5], [6, 7, 8])]

    ep = extract_episode_turns(FakeResult(), max_context=128, max_score_turns=4)
    assert ep.reward == 1.0 and len(ep.turns) == 2 and ep.n_trainable_tokens == 5
    assert ep.turns[0] == ([1, 2, 3, 4, 5], 2)
    print("train_strata_harbor_rl unit checks: ALL PASS")


# ---------------------------- Main Closed-Loop Trainer ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Strata + OpenEnv + Harbor Closed-Loop RL Trainer")
    ap.add_argument("--check-unit", action="store_true")
    ap.add_argument("--dataset", required=True, help="Path to local Harbor tasks directory")
    ap.add_argument("--task-indices", default="", help="Comma-separated task indices, e.g. 0,1,2")
    ap.add_argument("--harness", default="codex")
    ap.add_argument("--sandbox", default="docker")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--mode", choices=("hybrid", "grpo", "rft"), default="hybrid")
    ap.add_argument("--w-eff", type=float, default=0.1)
    ap.add_argument("--max-score-turns", type=int, default=4, help="Max turns scored per rollout to bound ZO cost")
    ap.add_argument("--strata-url", default="", help="Attach to an existing Strata server URL instead of launching one")
    ap.add_argument("--base-config", required=True, help="Path to base Strata server JSON config")
    ap.add_argument("--strata-exe", default=str(Path(__file__).resolve().parents[3] / "build" / "strata"), help="Path to strata engine binary")
    ap.add_argument("--strata-python", default=sys.executable, help="Python executable for serve.server")
    ap.add_argument("--strata-terminal", default=str(Path(__file__).resolve().parents[3] / "terminal"), help="Path to Strata terminal/ directory")
    ap.add_argument("--model", default="qwen3.8-flash-next-iq3_xxs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--capture-port", type=int, default=8109)
    ap.add_argument("--advertise-host", default="172.17.0.1", help="Docker-bridge IP reachable from sandboxes")
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--frontier-mib", type=int, default=4096)
    ap.add_argument("--frontier", default="auto")
    ap.add_argument("--reasoning-effort", default="none", help="Strata default reasoning_effort (e.g. none, low, medium, high)")
    ap.add_argument("--adapt-layers", type=int, default=4)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--zo-dirs", type=int, default=1)
    ap.add_argument("--zo-eps", type=float, default=5e-3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-grad", type=float, default=2.0)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work-dir", default="./harbor-rl-run", help="Output directory for checkpoints, metrics, and trials")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_unit:
        run_unit_checks()
        return

    # Strata's /v1/score recomputes exact T=1.0 teacher-forced logprobs for old/ref/perturbed/new
    # policies consistently, so OpenEnv's raw-logprobs probe check is safe to allow here.
    os.environ.setdefault("OPENENV_ALLOW_RAW_LOGPROBS", "1")
    from openenv.core.harness.capture import CaptureServer
    from openenv.core.harness.capture.validate_llm import validate_llm
    from openenv.harbor.rollout import run_rollout

    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    trials_dir = work / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    live_gguf = work / "live_adapter.gguf"
    final_gguf = work / "final_adapter.gguf"
    metrics_path = work / "metrics.jsonl"

    tasks = discover_tasks(Path(args.dataset))
    if args.task_indices.strip():
        chosen_idx = [int(x.strip()) for x in args.task_indices.split(",") if x.strip()]
    else:
        chosen_idx = [i % len(tasks) for i in range(args.steps)]

    lora = LoraAdapterState(
        n_layers_total=48,
        adapt_last_n=args.adapt_layers,
        rank=args.rank,
        alpha=float(args.rank),
        seed=args.seed,
    )
    lora.write_gguf(live_gguf)
    print(
        f"[init] Harbor Closed-Loop RL | harness={args.harness} mode={args.mode} G={args.group_size} "
        f"tasks={len(tasks)} steps={len(chosen_idx)} | LoRA layers={lora.layers} r={lora.rank} "
        f"trainable_B={lora.n_trainable_params} ({live_gguf.stat().st_size / 1024:.1f} KiB) "
        f"frontier={args.frontier} frontier_mib={args.frontier_mib}",
        flush=True,
    )

    server_proc: subprocess.Popen[bytes] | None = None
    capture: Any = None
    try:
        if args.strata_url:
            strata_url = args.strata_url.rstrip("/")
        else:
            managed_cfg = work / "strata-managed.json"
            build_managed_config(
                Path(args.base_config),
                managed_cfg,
                strata_exe=args.strata_exe,
                lora_path=live_gguf,
                host=args.host,
                port=args.port,
                max_context=args.max_context,
                frontier_mib=args.frontier_mib,
                reasoning_effort=args.reasoning_effort,
                log_path=work / "strata-engine.log",
            )
            t_load = time.monotonic()
            server_proc = start_strata_server(
                strata_python=args.strata_python,
                strata_terminal_dir=Path(args.strata_terminal),
                config_path=managed_cfg,
                host=args.host,
                port=args.port,
                server_log=work / "strata-server.log",
            )
            strata_url = f"http://{args.host}:{args.port}"
            print(f"[init] Strata HTTP server ready at {strata_url} in {time.monotonic() - t_load:.1f}s", flush=True)

        client = StrataHttpRLClient(strata_url, timeout=600.0)
        # Ensure live adapter is active even when attaching to a running server
        init_reload_ms = client.lora_reload(live_gguf, scale=1.0)
        print(f"[init] Active LoRA synced via POST /lora-adapters ({init_reload_ms:.1f} ms)", flush=True)

        rep = validate_llm(strata_url, args.model, timeout=180.0)
        print(
            f"[init] OpenEnv validate_llm: ok={rep.ok} capture_level={rep.capture_level} "
            f"tool_support={rep.tool_support} n_prompt={rep.n_prompt_ids} n_comp={rep.n_completion_ids}",
            flush=True,
        )
        if not (rep.ok and rep.capture_level == "tokens"):
            raise RuntimeError(f"Strata endpoint failed OpenEnv tokens check: {rep}")

        capture = CaptureServer(
            llm_url=strata_url,
            model=args.model,
            port=args.capture_port,
            capture_level=rep.capture_level,
            admin_key=secrets.token_urlsafe(16),
        )
        capture.start()
        intercept_url = f"http://{args.advertise_host}:{args.capture_port}"
        print(f"[init] OpenEnv CaptureServer listening on :{args.capture_port} -> {intercept_url}", flush=True)

        def eval_episodes(episodes: list[EpisodeTrace], lora_on: bool) -> tuple[list[float], float]:
            ep_lps: list[float] = []
            tot_ms = 0.0
            for ep in episodes:
                if not ep.turns:
                    ep_lps.append(0.0)
                    continue
                turn_lps: list[float] = []
                for full_ids, fp in ep.turns:
                    mlp, _, ms = client.score_sequence(full_ids, fp, lora=lora_on, frontier=args.frontier)
                    turn_lps.append(mlp)
                    tot_ms += ms
                ep_lps.append(sum(turn_lps) / len(turn_lps))
            return ep_lps, tot_ms

        with open(metrics_path, "w", encoding="utf-8") as mf:
            for step, tidx in enumerate(chosen_idx):
                task_dir = tasks[tidx % len(tasks)]
                t_step = time.monotonic()
                t_roll = time.monotonic()

                episodes: list[EpisodeTrace] = []
                for g_idx in range(args.group_size):
                    print(
                        f"[step {step+1}/{len(chosen_idx)}] rollout {g_idx+1}/{args.group_size} "
                        f"task={task_dir.name} ...",
                        flush=True,
                    )
                    res = asyncio.run(
                        run_rollout(
                            task_dir=task_dir,
                            harness=args.harness,
                            sandbox=args.sandbox,
                            registry=capture.registry,
                            intercept_url=intercept_url,
                            model=args.model,
                            trials_dir=trials_dir,
                            dataset=str(Path(args.dataset).expanduser().resolve()),
                            capture_level=rep.capture_level,
                            inference=capture.inference,
                        )
                    )
                    ep = extract_episode_turns(
                        res, max_context=args.max_context, max_score_turns=args.max_score_turns
                    )
                    episodes.append(ep)
                    print(
                        f"  -> reward={ep.reward:.2f} turns={len(ep.turns)} "
                        f"trainable_toks={ep.n_trainable_tokens} wall={ep.wall_s:.1f}s err={ep.error}",
                        flush=True,
                    )

                rollout_s = time.monotonic() - t_roll
                verifier_rewards = [e.reward for e in episodes]
                turn_counts = [len(e.turns) for e in episodes]
                advs, shaped_rewards, mode_used = compute_episode_advantages(
                    verifier_rewards, turn_counts, mode=args.mode, w_eff=args.w_eff
                )
                # Zero out advantages for episodes that produced zero scoreable turns
                advs = [a if e.turns else 0.0 for a, e in zip(advs, episodes)]
                has_signal = any(abs(a) > 1e-6 for a in advs)

                t_zo = time.monotonic()
                score_ms_total = 0.0
                reload_ms_total = 0.0
                proj_grads: list[float] = []
                dir_seeds: list[int] = []
                accepted = False
                j_before = 0.0
                j_after = 0.0

                if has_signal:
                    # Only score episodes with non-zero advantage (huge speedup in RFT / partial pass!)
                    active_eps = [e for e, a in zip(episodes, advs) if abs(a) > 1e-6]
                    active_advs = [a for a in advs if abs(a) > 1e-6]

                    old_lps, ms0 = eval_episodes(active_eps, lora_on=True)
                    score_ms_total += ms0
                    if args.kl_beta > 0.0:
                        ref_lps, msr = eval_episodes(active_eps, lora_on=False)
                        score_ms_total += msr
                    else:
                        ref_lps = old_lps
                    j_before = luspo_objective(old_lps, old_lps, ref_lps, active_advs, kl_beta=args.kl_beta)

                    for d_idx in range(args.zo_dirs):
                        dseed = args.seed * 10000 + step * 100 + d_idx + 1
                        dir_seeds.append(dseed)

                        lora.perturb_inplace(dseed, +args.zo_eps)
                        lora.write_gguf(live_gguf)
                        reload_ms_total += client.lora_reload(live_gguf)
                        lps_plus, msp = eval_episodes(active_eps, lora_on=True)
                        score_ms_total += msp
                        j_plus = luspo_objective(lps_plus, old_lps, ref_lps, active_advs, kl_beta=args.kl_beta)

                        lora.perturb_inplace(dseed, -2.0 * args.zo_eps)
                        lora.write_gguf(live_gguf)
                        reload_ms_total += client.lora_reload(live_gguf)
                        lps_minus, msm = eval_episodes(active_eps, lora_on=True)
                        score_ms_total += msm
                        j_minus = luspo_objective(lps_minus, old_lps, ref_lps, active_advs, kl_beta=args.kl_beta)

                        lora.perturb_inplace(dseed, +args.zo_eps)
                        g_hat = max(-args.max_grad, min(args.max_grad, (j_plus - j_minus) / (2.0 * args.zo_eps)))
                        proj_grads.append(g_hat)

                    step_scale = args.lr / max(1, len(proj_grads))
                    for dseed, g_hat in zip(dir_seeds, proj_grads):
                        lora.perturb_inplace(dseed, step_scale * g_hat)
                    lora.write_gguf(live_gguf)
                    reload_ms_total += client.lora_reload(live_gguf)
                    new_lps, msn = eval_episodes(active_eps, lora_on=True)
                    score_ms_total += msn
                    j_after = luspo_objective(new_lps, old_lps, ref_lps, active_advs, kl_beta=args.kl_beta)

                    if j_after < j_before:
                        # 0.25x backtrack guard
                        for dseed, g_hat in zip(dir_seeds, proj_grads):
                            lora.perturb_inplace(dseed, -0.75 * step_scale * g_hat)
                        lora.write_gguf(live_gguf)
                        reload_ms_total += client.lora_reload(live_gguf)
                        new_lps, msb = eval_episodes(active_eps, lora_on=True)
                        score_ms_total += msb
                        j_after = luspo_objective(new_lps, old_lps, ref_lps, active_advs, kl_beta=args.kl_beta)
                        if j_after < j_before:
                            for dseed, g_hat in zip(dir_seeds, proj_grads):
                                lora.perturb_inplace(dseed, -0.25 * step_scale * g_hat)
                            lora.write_gguf(live_gguf)
                            reload_ms_total += client.lora_reload(live_gguf)
                            j_after = j_before
                        else:
                            accepted = True
                    else:
                        accepted = j_after > j_before

                    if accepted:
                        lora.write_gguf(work / f"checkpoint_step_{step + 1}.gguf")

                zo_s = time.monotonic() - t_zo
                step_s = time.monotonic() - t_step
                delta_j = j_after - j_before
                rec = {
                    "step": step + 1,
                    "task": task_dir.name,
                    "mode_used": mode_used,
                    "rewards": verifier_rewards,
                    "mean_reward": round(sum(verifier_rewards) / max(1, len(verifier_rewards)), 4),
                    "pass_rate": round(sum(1 for r in verifier_rewards if r > 0.0) / max(1, len(verifier_rewards)), 3),
                    "turns_per_ep": turn_counts,
                    "trainable_tokens_total": sum(e.n_trainable_tokens for e in episodes),
                    "has_signal": has_signal,
                    "accepted": accepted,
                    "proj_grads": [round(g, 5) for g in proj_grads],
                    "j_before": round(j_before, 6),
                    "j_after": round(j_after, 6),
                    "delta_j": round(delta_j, 6),
                    "b_norm": round(lora.param_norm(), 5),
                    "rollout_s": round(rollout_s, 2),
                    "zo_score_s": round(zo_s, 2),
                    "eng_score_ms": round(score_ms_total, 1),
                    "reload_ms_total": round(reload_ms_total, 1),
                    "step_s": round(step_s, 2),
                }
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                print(
                    f"[step {step+1}/{len(chosen_idx)}] task={rec['task']} mode={mode_used} "
                    f"pass={rec['pass_rate']:.2f} turns={turn_counts} toks={rec['trainable_tokens_total']} "
                    f"acc_step={accepted} dJ={delta_j:+.6f} |B|={rec['b_norm']:.5f} "
                    f"rollout={rollout_s:.1f}s zo={zo_s:.1f}s score_ms={score_ms_total:.0f} "
                    f"reload_ms={reload_ms_total:.1f} total={step_s:.1f}s",
                    flush=True,
                )

        lora.write_gguf(final_gguf)
        print(f"\n[done] saved final adapter -> {final_gguf} (|B|={lora.param_norm():.5f})", flush=True)
    finally:
        if capture is not None:
            try:
                capture.stop()
            except Exception:
                pass
        stop_strata_server(server_proc)


if __name__ == "__main__":
    main()
