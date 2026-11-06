"""Status page for the Ascend box, bound to 127.0.0.1:8188 so an ssh -L tunnel can reach it.

Shows what the download is doing, whether llama-server is up, what the NPUs are doing, and lets
you send one prompt to the model once it is serving. Standard library only.

    python3 /data/status_page.py &
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PAGE_PORT", "8188"))
MODEL_DIR = "/data/models/IQ3_XXS"
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
SERVICE = "http://127.0.0.1:%d" % LLAMA_PORT


def sh(cmd: str, timeout: float = 10.0) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def download() -> dict:
    out = {"bytes": 0, "files": [], "done": False}
    for root, _dirs, files in os.walk(MODEL_DIR):
        for f in files:
            p = os.path.join(root, f)
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            out["files"].append({"name": f, "bytes": size, "complete": not f.endswith(".incomplete")})
            out["bytes"] += size
    out["files"].sort(key=lambda x: -x["bytes"])
    total_expected = 75_800_000_000
    out["expected"] = total_expected
    out["percent"] = round(100.0 * out["bytes"] / total_expected, 1)
    # a shard still being written ends in .incomplete, so "every .gguf is complete" is not the test
    out["done"] = bool(out["files"]) and not any(x["name"].endswith(".incomplete") for x in out["files"])
    # rate over the last two samples
    global _last
    now = time.time()
    if _last and now - _last[0] > 3:
        out["rate_bps"] = (out["bytes"] - _last[1]) / (now - _last[0])
    _last = (now, out["bytes"])
    if out.get("rate_bps") and out["rate_bps"] > 0:
        out["eta_s"] = int((total_expected - out["bytes"]) / out["rate_bps"])
    return out


def service() -> dict:
    try:
        with urllib.request.urlopen(SERVICE + "/health", timeout=3) as r:
            return {"up": r.status == 200, "health": json.loads(r.read() or b"{}")}
    except Exception:
        return {"up": False}


def npus() -> list:
    text = sh("npu-smi info 2>/dev/null")
    cards = []
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        m = re.fullmatch(r"(\d+)\s+(\S+)", cells[0]) if cells else None
        if m and len(cells) > 2:
            vals = re.findall(r"\d+(?:\.\d+)?", cells[2])
            if len(vals) >= 2:
                cur = {"id": int(m.group(1)), "name": m.group(2), "power_w": float(vals[0]),
                       "temp_c": float(vals[1])}
                cards.append(cur)
            continue
        if cur is not None and len(cells) > 2 and re.search(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}", cells[1]):
            vals = re.findall(r"\d+(?:\.\d+)?", cells[2])
            if len(vals) >= 3:
                cur["util_pct"] = float(vals[0])
                cur["hbm_used_mb"] = float(vals[-2])
                cur["hbm_total_mb"] = float(vals[-1])
    return cards


_last = None


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>昇腾 910B2C · Qwen3.8-Flash-Next</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#232a33;--fg:#e6edf3;--dim:#8b949e;--ok:#3fb950;--bad:#f85149;--acc:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.6 -apple-system,"Segoe UI",system-ui,sans-serif}
header{padding:20px 28px 6px}h1{margin:0;font-size:19px;font-weight:600}
.sub{color:var(--dim);font-size:12px}
.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:16px;padding:16px 28px 28px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card h2{margin:0 0 12px;font-size:13px;font-weight:600;color:var(--dim);letter-spacing:.04em;text-transform:uppercase}
.bar{height:8px;background:#21262d;border-radius:4px;overflow:hidden;margin:10px 0 6px}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,#1f6feb,#58a6ff)}
.kv{display:flex;justify-content:space-between;padding:3px 0;font-size:13px}
.kv span:first-child{color:var(--dim)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--bad);margin-right:6px}
.dot.on{background:var(--ok);box-shadow:0 0 8px var(--ok)}
pre{margin:8px 0 0;padding:10px;background:#0b0f14;border:1px solid var(--line);border-radius:6px;
font:12px/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap;max-height:220px;overflow:auto}
input,button{font:13px inherit;padding:8px 10px;border-radius:6px;border:1px solid var(--line);
background:#0b0f14;color:var(--fg)}button{cursor:pointer;background:#21262d}
button:hover{border-color:var(--acc)}input{width:100%;margin-bottom:8px}
</style></head><body>
<header><h1>昇腾 910B2C · Qwen3.8-Flash-Next</h1>
<div class="sub" id="stamp">加载中…</div></header>
<div class="grid">
  <div class="card"><h2>模型下载</h2><div id="dl"></div></div>
  <div class="card"><h2>推理服务</h2><div id="svc"></div>
    <input id="q" placeholder="模型就绪后在这里提问，回车发送">
    <pre id="ans"></pre>
  </div>
  <div class="card" style="grid-column:1/-1"><h2>NPU</h2><div id="npu"></div></div>
</div>
<script>
const gb=b=>(b/1e9).toFixed(2)+" GB";
const mm=b=>(b/1e6).toFixed(0)+" MB";
function render(d){
  document.getElementById('stamp').textContent =
    new Date().toLocaleTimeString() + " · " + d.host;
  const dl=d.download;
  let h=`<div class="bar"><i style="width:${Math.min(100,dl.percent)}%"></i></div>`;
  h+=`<div class="kv"><span>${gb(dl.bytes)} / ${gb(dl.expected)}</span><span>${dl.percent}%</span></div>`;
  if(dl.rate_bps) h+=`<div class="kv"><span>速率</span><span>${mm(dl.rate_bps)}/s</span></div>`;
  if(dl.eta_s) h+=`<div class="kv"><span>预计剩余</span><span>${Math.round(dl.eta_s/60)} 分钟</span></div>`;
  h+=`<div class="kv"><span>状态</span><span>${dl.done?"已就绪":"下载中"}</span></div>`;
  for(const f of dl.files) h+=`<div class="kv"><span style="max-width:60%;overflow:hidden;text-overflow:ellipsis">${f.name}</span><span>${gb(f.bytes)}${f.complete?" ✓":""}</span></div>`;
  document.getElementById('dl').innerHTML=h;

  const s=d.service;
  document.getElementById('svc').innerHTML =
    `<div class="kv"><span><span class="dot ${s.up?'on':''}"></span>${s.up?"运行中":"未启动"}</span><span>127.0.0.1:${d.llama_port}</span></div>`+
    (s.up&&s.health?`<div class="kv"><span>模型</span><span>${(s.health.slots||s.health.model||"—")}</span></div>`:"");

  let n="";
  for(const c of d.npus){
    n+=`<div class="kv"><span>NPU ${c.id} ${c.name}</span>
        <span>${c.hbm_used_mb?mm(c.hbm_used_mb*1024*1024)+" / "+mm(c.hbm_total_mb*1024*1024):"—"} · ${c.util_pct??0}% · ${c.temp_c}°C</span></div>`;
  }
  document.getElementById('npu').innerHTML=n||"（npu-smi 无输出）";
}
async function tick(){ const r=await fetch('/api/status'); render(await r.json()); }
document.getElementById('q').addEventListener('keydown', async e=>{
  if(e.key!=='Enter'||!e.target.value) return;
  const out=document.getElementById('ans'); out.textContent='生成中…';
  const r=await fetch('/api/chat',{method:'POST',body:JSON.stringify({q:e.target.value})});
  const d=await r.json();
  out.textContent=d.ok?d.text:('失败: '+d.error);
});
tick(); setInterval(tick, 2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            payload = {
                "host": os.uname().nodename,
                "download": download(),
                "service": service(),
                "llama_port": LLAMA_PORT,
                "npus": npus(),
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
        elif self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self.path.startswith("/api/chat"):
            self._send(404, b"{}", "application/json")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            q = json.loads(self.rfile.read(n) or b"{}").get("q", "")
            body = json.dumps({
                "messages": [{"role": "user", "content": q}],
                "max_tokens": 256, "temperature": 0,
            }).encode()
            req = urllib.request.Request(SERVICE + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            msg = d["choices"][0].get("message") or {}
            text = msg.get("content") or ""
            think = msg.get("reasoning_content") or ""
            out = {"ok": True, "text": text or ("（思考中，没到最终回答）\n" + think if think else "(空)")}
        except Exception as e:
            out = {"ok": False, "error": str(e)}
        self._send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json")


if __name__ == "__main__":
    print("status page on http://127.0.0.1:%d" % PORT, flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
