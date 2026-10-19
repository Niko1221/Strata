"""serve/test_web_mcp_retry.py - the web page's GET /mcp retries, with backoff, after a failure (0.1.45, #3113).

A single failed fetch used to disable MCP tools for the life of the tab: `loadMcp` returned on `!r.ok` or an
exception and only retried while a server's status was literally `starting`.  Pins, with node, the schedule, the
stop on 404, the API-key message on 401, and the success reset.  Skipped without node.

    python -m unittest serve.test_web_mcp_retry -v
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parent / "web" / "app.js"


def slice_between(src: str, start: str, end: str) -> str:
    a = src.index(start)
    return src[a:src.index(end, a)]


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class WebMcpRetry(unittest.TestCase):
    def test_schedule_stop_and_reset(self):
        block = slice_between(APP.read_text(encoding="utf-8"), "let mcpInfo", "const MCP_STATE")
        js = """
let timers = [], nextTimer = 1, renders = 0;
global.setTimeout = (fn, ms) => { const id = nextTimer++; timers.push({id, fn, ms}); return id; };
global.clearTimeout = (id) => { timers = timers.filter((t) => t.id !== id); };
function headers() { return {}; }
function renderMcp() { renders++; }
let reply = async () => ({ok: true, status: 200, json: async () => ({servers: [], tools: 0})});
global.fetch = (...a) => reply(...a);
""" + block + """
(async () => {
  const out = {}, delay = () => timers.length ? timers[timers.length - 1].ms : null;
  const fail = () => { reply = async () => { throw new Error("down"); }; };
  const ok = (data) => { reply = async () => ({ok: true, status: 200, json: async () => data}); };
  const status = (code) => { reply = async () => ({ok: false, status: code, json: async () => ({})}); };

  fail();
  for (const k of ["fail1", "fail2", "fail3", "fail4", "fail5"]) { await loadMcp(); out[k] = delay(); }

  ok({servers: [], tools: 0});
  await loadMcp(); out.afterOk = delay();
  fail(); await loadMcp(); out.reset = delay();

  mcpInfo = {servers: [{status: "ready"}], tools: 1};
  mcpTries = 0; timers = []; status(404); await loadMcp();
  out.after404 = timers.length; out.error404 = mcpError; out.info404 = mcpInfo.tools;
  mcpTries = 0; timers = []; status(401); await loadMcp();
  out.after401 = {delay: delay(), error: mcpError};

  mcpTries = 0; timers = []; ok({servers: [{status: "starting"}], tools: 0}); await loadMcp();
  out.starting = delay();
  mcpTries = 0; timers = []; ok({servers: [{status: "ready"}], tools: 1}); await loadMcp();
  out.ready = timers.length;

  mcpTries = 0; timers = [];
  reply = async () => ({ok: true, status: 200, json: async () => { throw new Error("cut"); }});
  await loadMcp(); out.jsonFail = {delay: delay(), error: mcpError};

  out.renders = renders;
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        r = json.loads(out.stdout)
        self.assertEqual([r["fail1"], r["fail2"], r["fail3"], r["fail4"], r["fail5"]],
                         [3000, 6000, 12000, 30000, 30000])
        self.assertIsNone(r["afterOk"])                     # a success clears the timer and schedules none
        self.assertEqual(r["reset"], 3000)                  # ... and resets the backoff
        # an older server: no retry, no message, and no stale list
        self.assertEqual((r["after404"], r["error404"], r["info404"]), (0, None, 0))
        self.assertEqual(r["after401"]["delay"], 3000)
        self.assertIn("API key", r["after401"]["error"])
        self.assertEqual(r["starting"], 3000)               # npx is still loading: poll as before
        self.assertEqual(r["ready"], 0)
        self.assertEqual(r["jsonFail"]["delay"], 3000)
        self.assertIn("stopped arriving", r["jsonFail"]["error"])
        self.assertGreater(r["renders"], 0)


if __name__ == "__main__":
    unittest.main()
