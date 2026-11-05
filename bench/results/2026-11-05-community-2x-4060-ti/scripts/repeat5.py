"""Five-round statistical repeat: c70-s6 vs MTP, same three prompts, engine-log parsed."""
import json, re, sys, time
sys.path.insert(0, r"D:\agentworkspace\strata-research\tools")
from sweep_dflash2 import resolve, wait_ready, PROMPTS, BASE_URL, REPO, PY, PORT
import os, subprocess, urllib.request
from pathlib import Path

OUT = Path(r"DEV_DATA")
pat = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ \d+ read in \d+ ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)')
TASK = {44: 'code', 42: 'fiction', 57: 'reasoning'}

def run(arm, rounds=5):
    path = resolve(arm)
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    folder = OUT / arm
    folder.mkdir(parents=True, exist_ok=True)
    cfg["log"] = str(folder / "engine.log")
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    proc = subprocess.Popen([PY, str(REPO/"serve"/"server.py"), "--engine", "strata",
                             "--config", str(path), "--host", "127.0.0.1", "--port", str(PORT), "--no-api-key"],
                            cwd=str(REPO), env={**os.environ, "PYTHONUNBUFFERED": "1"},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready()
        time.sleep(2)
        # warm-up request (sentinel: 18-token prompt, identified by prompt_n=18)
        ask("Reply READY and nothing else.", 32)
        rows = []
        for rnd in range(1, rounds+1):
            for task in ("code", "fiction", "reasoning"):
                prompt, cap = PROMPTS[task]
                r = ask(prompt, cap)
                t = r.get("timings") or {}
                rows.append(dict(task=task, round=rnd, tps=t.get("predicted_per_second"),
                                 acc=t.get("draft_n_accepted"), off=t.get("draft_n")))
                time.sleep(1)
        (folder/"results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    finally:
        urllib.request.urlopen(urllib.request.Request(f"{BASE_URL}/unload", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST"), timeout=180).read()
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(30)
            except subprocess.TimeoutExpired: proc.kill()
    time.sleep(5)
    return rows

def ask(prompt, cap):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role":"user","content":prompt}],
            "max_tokens": cap, "temperature": 0, "seed": 1, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    with urllib.request.urlopen(urllib.request.Request(f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900) as resp:
        return json.loads(resp.read())

if __name__ == "__main__":
    import statistics as st
    for arm in ("df2-c70-s6", "mtp"):
        rows = run(arm)
        print(f"=== {arm} ===")
        for task in ("code","fiction","reasoning"):
            tps = [r["tps"] for r in rows if r["task"]==task and r["tps"]]
            print(f"  {task:9s} mean {st.mean(tps):5.1f} stdev {st.stdev(tps):4.1f} n={len(tps)} vals {[round(t,1) for t in tps]}")
