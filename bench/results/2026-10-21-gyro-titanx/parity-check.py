#!/usr/bin/env python3
"""parity-check.py - the same prompt through the engine with and without a setting, and the answers compared.

One engine start per arm, the same fixed prompt, greedy (temperature 0), and the answer texts compared
character by character.  The engine's own timings are printed next to them, so the run also says whether the
setting is faster on this card.

    python parity-check.py --config strata-gyro-s.json --env STRATA_DENSE_F32=1
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/mo/strata-mainline/tools")
import gyro_ab  # noqa: E402  (the harness's server helpers)

ROOT = Path("/home/mo/strata-mainline")
OUT = Path(__file__).resolve().parent


def one_arm(cfg, env_extra, prompt, tag):
    """Start the engine with the config (plus the env), send the prompt, return the answer and the timings."""
    port = gyro_ab.free_port()
    run_cfg = dict(cfg)
    run_cfg["port"] = port
    run_cfg["host"] = "127.0.0.1"
    run_cfg["log"] = str(OUT / f"parity-{tag}-engine.log")
    run_cfg["cwd"] = str(ROOT)
    run_cfg["env"] = {**(cfg.get("env") or {}), **env_extra}
    cfg_path = OUT / f"parity-{tag}-config.json"
    cfg_path.write_text(json.dumps(run_cfg, indent=2))
    env = dict(os.environ)
    for k, v in env_extra.items():
        env[k] = v
    log = open(OUT / f"parity-{tag}-server.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(cfg_path),
                             "--port", str(port)], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        gyro_ab.wait_ready(f"http://127.0.0.1:{port}", cfg.get("api_key") or "", proc, 1800)
        r = gyro_ab.chat(f"http://127.0.0.1:{port}", cfg.get("api_key") or "", prompt, 64, False,
                         run_cfg.get("model_name"))
        return r
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "strata-gyro-s.json"
    env_extra = {}
    for a in sys.argv[2:]:
        k, _, v = a.partition("=")
        env_extra[k] = v
    cfg = json.loads(cfg_path.read_text())
    tok = gyro_ab.load_tokenizer(cfg.get("tokenizer"))
    prompt = gyro_ab.build_prompt(256, "parity", tok)
    print("arm 1: the default config", flush=True)
    a = one_arm(cfg, {}, prompt, "off")
    print(f"  {a['total_s']} s, ttft {a['ttft_s']} s, {len(a['answer'])} characters", flush=True)
    print(f"arm 2: with {env_extra}", flush=True)
    b = one_arm(cfg, env_extra, prompt, "on")
    print(f"  {b['total_s']} s, ttft {b['ttft_s']} s, {len(b['answer'])} characters", flush=True)
    same = a["answer"] == b["answer"]
    print(f"\nanswers identical: {same}")
    if not same:
        print("--- without:")
        print(a["answer"])
        print("--- with:")
        print(b["answer"])
    (OUT / "parity-answers.json").write_text(json.dumps({"same": same, "off": a["answer"], "on": b["answer"],
                                                          "env": env_extra}, indent=2))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
