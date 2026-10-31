# Phase-2 P2 smoke: --batch-mtp with grammar slots. Two slots decode together: one constrained
# (json_schema), one plain, then both constrained - the batch windows carry a grammar slot's draft
# row through its ring chain-row (set_row_grammar_masks), and every constrained reply must still
# validate client-side. Run: python .zcode-obj/batch_mtp_smoke.py
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from grammar_acceptance import SCHEMAS  # noqa: E402

PORT = 8199
BASE = f"http://127.0.0.1:{PORT}"
CFG = ROOT / ".zcode-obj" / "batch-mtp.cfg.json"
LOG = ROOT / ".zcode-obj" / "batch-mtp.serve.log"

ENGINE = ["--pack", "F:/Strata-data/packs/swift-iq3_s",
          "--native", "F:/Strata-data/models/swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
          "--resident-budget-gib", "40", "--expert-profile", "F:/Strata/data/expert-profile.bin",
          "--expert-cache", "auto", "--prefill", "auto", "--spec", "2",
          "--max-context", "8192", "--kv", "int8",
          "--batch", "2", "--batch-mtp", "--mtp", "F:/Strata-data/mtp/rt"]

PROMPTS = [("List 3 classic novels with their publication years, as JSON.",
            {"type": "json_schema", "json_schema": {"name": "books", "schema": SCHEMAS["books"], "strict": True}}),
           ("Name five rivers in Europe with their countries, plain prose.", None),
           ("Give a short recipe for pancakes with exactly 5 steps, as JSON.",
            {"type": "json_schema", "json_schema": {"name": "recipe", "schema": SCHEMAS["recipe"], "strict": True}})]

try:
    import jsonschema
    def conforms(text, schema):
        jsonschema.validate(json.loads(text), schema)
        return True
except ImportError:
    def conforms(text, schema):
        return isinstance(json.loads(text), dict)


def post(payload):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def one(i):
    prompt, fmt = PROMPTS[i % len(PROMPTS)]
    payload = {"messages": [{"role": "user", "content": prompt + f" (set {i})"}],
               "max_tokens": 500, "reasoning_effort": "none", "temperature": 0.8, "seed": 31 + i}
    if fmt:
        payload["response_format"] = fmt
    t0 = time.time()
    d = post(payload)
    dt = time.time() - t0
    text = d["choices"][0]["message"]["content"]
    tok_s = d["usage"]["completion_tokens"] / dt
    if fmt:
        schema = fmt["json_schema"]["schema"]
        ok = conforms(text, schema)
        tag = "constrained"
    else:
        ok = len(text.strip()) > 0
        tag = "plain"
    print(f"  request {i} ({tag}): {'OK' if ok else 'SCHEMA FAIL'} {tok_s:.1f} tok/s", flush=True)
    if not ok and fmt:
        print("    text:", text[:300], flush=True)
    return tag, ok, tok_s


def main():
    cfg = {"exe": str(ROOT / "build" / "strata.exe"), "args": ENGINE, "cwd": str(ROOT),
           "tokenizer": "F:/Strata-data/packs/swift-iq3_s/tokenizer",
           "model_name": "Qwen3.8-Flash-Next", "log": str(LOG),
           "port": PORT, "host": "127.0.0.1", "gpu": 0,
           "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}}
    CFG.write_text(json.dumps(cfg), encoding="utf-8")
    srv = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                            "--config", str(CFG), "--port", str(PORT)],
                           stdout=open(LOG, "ab"), stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            time.sleep(5)
            try:
                with urllib.request.urlopen(BASE + "/v1/status", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
        else:
            raise RuntimeError("the server did not come up")
        print("server up; firing concurrent requests", flush=True)
        results = [None] * 6
        threads = [threading.Thread(target=lambda i=i: results.__setitem__(i, one(i))) for i in range(6)]
        for t in threads: t.start()
        for t in threads: t.join()
        bad = [i for i, r in enumerate(results) if r and not r[1]]
        fails = [i for i, r in enumerate(results) if r is None]
        print(f"smoke: {sum(1 for r in results if r and r[1])}/6 OK, failures {bad + fails}", flush=True)
        return 0 if not bad and not fails else 1
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    sys.exit(main())
