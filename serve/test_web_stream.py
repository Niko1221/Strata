"""The web app's long-lived streams (#2687) and its language packs (#2686).

The page used to ask for /metrics once a second (the API monitor every 2 s), which rewrote its own text and fought
Chrome's page translation.  These tests open the streams as a browser does, with no model or GPU: the engine is
the scripted one (serve/server.py's MockEngine).
"""
import json
import re
import time
import unittest
import urllib.error
import socket
import urllib.request
from http.client import HTTPConnection
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

WEB = Path(__file__).parent / "web"


def new_service(api_monitor=False):
    tok = ByteTokenizer()
    engine = MockEngine(tok, "Hello.", max_context=4096)
    svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
    svc.api_monitor = api_monitor
    return svc


class Stream(unittest.TestCase):
    """GET /metrics/stream: one response the server writes `metrics` events to, ending when the client hangs up."""

    def setUp(self):
        self.svc = new_service()
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def open_stream(self, path="/metrics/stream"):
        """-> (conn, response): an EventSource-like client with the connection left open."""
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", path, headers={"Accept": "text/event-stream"})
        return conn, conn.getresponse()

    def test_stream_is_event_stream(self):
        conn, r = self.open_stream()
        try:
            self.assertEqual(r.status, 200)
            self.assertIn("text/event-stream", r.headers.get("Content-Type", ""))
        finally:
            conn.close()

    def test_events_keep_coming_on_one_connection(self):
        """The Monitor is a live dashboard: one GET, then events on their own - no request per second."""
        conn, r = self.open_stream()
        try:
            self.assertEqual(r.readline().strip(), b"event: metrics")
            data = json.loads(r.readline()[5:])
            for key in ("live", "engine", "hardware", "requests", "time"):
                self.assertIn(key, data)
        finally:
            conn.close()

    def test_a_live_monitor_keeps_getting_events(self):
        """#2687: the stream must not throttle.  An idle Monitor still scrolls (the hardware sampler runs every
        second), so more events have to arrive on the one connection without the page asking again."""
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        try:
            sock.sendall(b"GET /metrics/stream HTTP/1.0\r\nAccept: text/event-stream\r\n\r\n")
            reader = sock.makefile("rb")
            while reader.readline().strip():                  # the headers, up to the blank line
                pass
            seen = 0
            deadline = time.time() + 8                        # the sampler's own tick is one second
            while time.time() < deadline and seen < 3:
                line = reader.readline()
                if not line:
                    break
                if line.strip() == b"event: metrics":
                    reader.readline()                        # the data line
                    seen += 1
            self.assertGreaterEqual(seen, 3, "the stream throttled: a live Monitor stopped getting events")
        finally:
            sock.close()

    def test_stream_ends_when_the_client_hangs_up(self):
        """A closed tab must not leave a thread waiting on a tick: the next tick notices the socket is gone."""
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        sock.sendall(b"GET /metrics/stream HTTP/1.0\r\nAccept: text/event-stream\r\n\r\n")
        self.assertIn(b"200", sock.recv(200))
        self.assertEqual(len(self.svc.metrics_streams), 1)
        sock.close()                                  # the browser's tab was closed
        for _ in range(40):                           # at most ~4 s: one tick plus the socket's own timeout
            if not self.svc.metrics_streams:
                break
            time.sleep(0.1)
        self.assertEqual(self.svc.metrics_streams, [], "the stream stayed registered after the client hung up")

    def test_server_close_stops_every_open_stream(self):
        conn, r = self.open_stream()
        try:
            r.readline()
            r.readline()
            self.assertEqual(len(self.svc.metrics_streams), 1)
            self.httpd.server_close()
            self.assertEqual(self.svc.metrics_streams, [], "an open stream outlived the server")
        finally:
            conn.close()

    def test_stream_needs_the_api_key(self):
        self.svc.api_key = "secret"
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/metrics/stream", headers={"Accept": "text/event-stream"})
        try:
            self.assertEqual(conn.getresponse().status, 401)
        finally:
            conn.close()
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/metrics/stream", headers={"Accept": "text/event-stream", "Authorization": "Bearer secret"})
        try:
            self.assertEqual(conn.getresponse().status, 200)
        finally:
            conn.close()

    def test_show_all_keeps_every_request(self):
        """?requests=all: the flag the page's "Show all" sends."""
        conn, r = self.open_stream("/metrics/stream?requests=all")
        try:
            r.readline()
            self.assertIn("requests", json.loads(r.readline()[5:]))
        finally:
            conn.close()


class ApiMonitorStream(unittest.TestCase):
    """GET /api/requests/stream: the API monitor's two reads in one response (#2687)."""

    def setUp(self):
        self.svc = new_service(api_monitor=True)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_stream_sends_status_and_requests(self):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/api/requests/stream", headers={"Accept": "text/event-stream"})
        try:
            r = conn.getresponse()
            self.assertEqual(r.status, 200)
            self.assertEqual(r.readline().strip(), b"event: requests")
            data = json.loads(r.readline()[5:])
            self.assertIn("status", data)
            self.assertIsInstance(data["requests"], list)
        finally:
            conn.close()

    def test_stream_is_off_with_the_monitor(self):
        self.svc.api_monitor = False
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/api/requests/stream", headers={"Accept": "text/event-stream"})
        try:
            self.assertEqual(conn.getresponse().status, 404)
        finally:
            conn.close()


class LanguagePacks(unittest.TestCase):
    """#2686: one UTF-8 pack per language; the page lists them and fetches one by name."""

    def setUp(self):
        self.svc = new_service()
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()

    def test_every_pack_is_utf8_and_has_the_same_keys(self):
        packs = sorted(p.name for p in (WEB / "i18n").glob("*.json"))
        self.assertIn("en.json", packs)
        self.assertIn("zh-CN.json", packs)
        keys = None
        for name in packs:
            raw = (WEB / "i18n" / name).read_bytes()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), f"{name} has a UTF-8 BOM")
            raw.decode("utf-8")                        # never GBK/cp936: it must be valid UTF-8
            pack = json.loads(raw.decode("utf-8"))
            self.assertIn("meta", pack, name)
            if name == "en.json":
                keys = set(pack)
            else:
                self.assertEqual(set(pack), keys, f"{name} and en.json differ")

    def test_packs_are_served_as_utf8_json(self):
        status, ctype, body = self.get("/web/i18n/zh-CN.json")
        self.assertEqual(status, 200)
        self.assertIn("charset=utf-8", ctype)
        pack = json.loads(body.decode("utf-8"))        # UTF-8 on the wire
        self.assertIn("tab.monitor", pack)

    def test_a_path_outside_the_folder_is_not_served(self):
        for path in ("/web/i18n/../app.js", "/web/i18n/app.js"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.get(path)
            self.assertEqual(e.exception.code, 404, path)

    def test_the_server_lists_them(self):
        status, ctype, body = self.get("/i18n")
        self.assertEqual(status, 200)
        listed = json.loads(body.decode("utf-8"))["languages"]
        codes = {l["code"] for l in listed}
        self.assertIn("en", codes)
        self.assertIn("zh-CN", codes)
        for l in listed:
            self.assertTrue(l["name"], l)

    def test_the_pages_are_utf8_and_ask_not_to_be_machine_translated(self):
        """#2687: the pages bring their own translation now, so the browser must not re-translate them."""
        for page in ("index.html", "monitor.html"):
            text = (WEB / page).read_text(encoding="utf-8")
            self.assertIn('charset="utf-8"', text, page)
            self.assertIn('translate="no"', text, page)

    def test_the_api_monitor_page_is_translated_too(self):
        """#2686: /api-monitor is a page of its own; translate="no" is only fair if it is actually translated."""
        html = (WEB / "monitor.html").read_text(encoding="utf-8")
        js = (WEB / "monitor.js").read_text(encoding="utf-8")
        self.assertGreaterEqual(html.count("data-i18n"), 40, "the API monitor's markup carries almost no keys")
        self.assertIn("data-i18n", html)
        self.assertIn('t("apimon.', js)
        keys = set(re.findall(r'data-i18n(?:-aria|-title|-placeholder)?="([^"]+)"', html))
        keys |= set(re.findall(r'\bt\("([a-z]+\.[a-zA-Z]+)"', js))
        for name in ("en.json", "zh-CN.json"):
            pack = json.loads((WEB / "i18n" / name).read_text(encoding="utf-8"))
            missing = sorted(k for k in keys if k not in pack)
            self.assertEqual(missing, [], f"{name} is missing {missing[:8]}")

    def test_every_key_the_page_uses_exists(self):
        """A key the pack lacks would show as the raw key on screen."""
        html = (WEB / "index.html").read_text(encoding="utf-8")
        js = (WEB / "app.js").read_text(encoding="utf-8")
        used = set(re.findall(r'data-i18n(?:-aria|-title|-placeholder)?="([^"]+)"', html))
        used |= set(re.findall(r'[^\w.]t\("([a-zA-Z][\w.]*)"', js))
        used |= set(re.findall(r'[^\w.]H\("([a-zA-Z][\w.]*)"', js))
        # the Monitor's eight cards: built as "mon.metric." + the card's key
        used |= {f"mon.metric.{k}" for k in ("speed", "gpu", "vram", "temp", "power", "pcie", "cpu", "disk")}
        namespaces = ("mon.", "chat.", "fact.", "drawer.", "tool.", "mcp.", "cc.", "toast.", "pill.", "tab.",
                      "img.", "about.", "cfg.", "code.", "file.", "chip.", "export.", "theme.", "app.")
        used = {k for k in used if k.startswith(namespaces) and not k.endswith(".")}
        for name in ("en.json", "zh-CN.json"):
            pack = json.loads((WEB / "i18n" / name).read_text(encoding="utf-8"))
            missing = sorted(k for k in used if k not in pack)
            self.assertEqual(missing, [], f"{name} is missing {missing[:10]}")

    def test_tokens_are_not_translated_into_something_else(self):
        """#2686: the wording is translated, the terms are not - a token is not a 代币 or a 令牌."""
        en = json.loads((WEB / "i18n" / "en.json").read_text(encoding="utf-8"))
        zh = json.loads((WEB / "i18n" / "zh-CN.json").read_text(encoding="utf-8"))
        for key, value in en.items():
            if "token" in value:                         # "token(s)" stays "token(s)", never 代币/令牌
                self.assertIn("token", zh.get(key, ""), key)
            for wrong in ("代币", "令牌"):
                self.assertNotIn(wrong, zh.get(key, ""), key)
        joined = " ".join(v for v in zh.values() if isinstance(v, str))
        for term in ("VRAM", "PCIe", "Prefill", "Decode", "MTP", "KV cache", "Top-p", "Top-k", "seed"):
            self.assertIn(term, joined, term)

    def test_the_chinese_pack_is_actually_translated(self):
        en = json.loads((WEB / "i18n" / "en.json").read_text(encoding="utf-8"))
        zh = json.loads((WEB / "i18n" / "zh-CN.json").read_text(encoding="utf-8"))
        same = [k for k in en if k != "meta" and en[k] == zh[k] and not re.fullmatch(r"[^\u4e00-\u9fff]*", en[k]) is None]
        translated = [k for k in en if k != "meta" and en[k] != zh[k]]
        self.assertGreater(len(translated), 100, "the Chinese pack hardly translates anything")

class Translation(unittest.TestCase):
    """#2686: the pack's own logic - t() with its variables, H() markers, and the browser language match.

    Run without a browser: the block between the language markers in serve/web/app.js is loaded with the DOM and
    the network stubbed out, so a broken t() or a renamed key fails here as it would on screen.
    """

    def block(self):
        js = (WEB / "app.js").read_text(encoding="utf-8")
        start = js.index("// ------------------------------------------------------------------ language (#2686)")
        end = js.index("// ------------------------------------------------------------------ toasts")
        return js[start:end]

    def run_js(self, script):
        """-> the script's stdout (node is only needed for this test; skip when it is not installed)."""
        import subprocess
        import tempfile
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("node is not installed")
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as f:
            f.write(script)
            path = f.name
        try:
            done = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(done.returncode, 0, done.stderr[:500])
        return done.stdout

    def harness(self):
        en = json.loads((WEB / "i18n" / "en.json").read_text(encoding="utf-8"))
        zh = json.loads((WEB / "i18n" / "zh-CN.json").read_text(encoding="utf-8"))
        return "\n".join([
            'const document = { documentElement: {lang: "en", dir: "ltr"}, title: "", querySelectorAll: () => [] };',
            'const navigator = {languages: ["zh-CN", "zh"], language: "zh-CN"};',
            'const store = {get: (k, d) => d, set: () => {}};',
            'const esc = (s) => String(s);',
            'const fetch = async () => { throw new Error("no network"); };',
            'const lastMetrics = null, messages = [], busy = null;',
            'function render(m) {}', 'function renderMcp() {}', 'function renderChat() {}',
            'function setBusy() {}', 'function outputs() {}', 'function loadConfig() {}',
            'function headers() { return {}; }',
            'let segEl = null;',
            'const $ = (id) => (id === "lang-seg" && segEl ? segEl : null);',
            'const METRICS = [{key: "speed", label: "Speed"}, {key: "vram", label: "VRAM"}];',
            'const FALLBACK = new Map();',
            'for (const m of METRICS) FALLBACK.set("mon.metric." + m.key, m.label);',
            self.block(),
            "const out = [];",
            'function check(name, got, want) { out.push((got === want ? "ok" : "FAIL") + " " + name + ": " + JSON.stringify(got) + (got === want ? "" : " want " + JSON.stringify(want))); }',
            "pack = " + json.dumps(en, ensure_ascii=False) + ";",
            'check("en tab.chat", t("tab.chat"), "Chat");',
            'check("en VRAM stays VRAM", t("mon.metric.vram"), "VRAM");',
            "pack = " + json.dumps(zh, ensure_ascii=False) + ";",
            'check("zh tab.chat", t("tab.chat"), "对话");',
            'check("zh VRAM stays VRAM", t("mon.metric.vram"), "VRAM");',
            'check("zh tokens stay tokens", t("mon.tokens", {n: "1,234"}), "1,234 tokens");',
            'check("zh variables", t("pill.generating", {rate: "42.1"}), "正在生成 · 42.1 tok/s");',
            'check("zh an empty variable drops", t("chat.meta", {n: "50", rate: "", stopped: ""}), "50 tokens");',
            'check("a missing key shows the key", t("nope.nope"), "nope.nope");',
            'check("an H() marker is read when drawn", t(H("mcp.ready")), "已连接");',
            'LANGS.length = 0;',
            'LANGS.push({code: "en", name: "English", dir: "ltr"}, {code: "zh-CN", name: "简体中文", dir: "ltr"});',
            'check("a Chinese browser gets Chinese", browserLang(), "zh-CN");',
            'navigator.languages = ["en-US", "en"];',
            'check("an English browser gets English", browserLang(), "en");',
            'navigator.languages = ["de-DE"];',
            'check("a language with no pack falls back to English", browserLang(), "en");',
            '// the switch is the drawer\'s own .seg control, one radio button per language, wired to setLang',
            'segEl = { children: [], set innerHTML(h) { this._h = h; this.children = [...h.matchAll(/data-lang="([^"]*)" aria-checked="([^"]*)"/g)].map(m => ({dataset: {lang: m[1]}, checked: m[2], onclick: null})); }, get innerHTML() { return this._h; } };',
            'segEl = { children: [], set innerHTML(h) { this._h = h; this.children = [...h.matchAll(/data-lang="([^"]*)" aria-checked="([^"]*)"/g)].map(m => ({dataset: {lang: m[1]}, checked: m[2], onclick: null})); }, get innerHTML() { return this._h; } };',
            'fillLangs();',
            'check("seg has a button per language", segEl.children.length, 2);',
            'check("the current language is the checked radio", segEl.children.map(b => b.checked).join(","), "true,false");',
            'check("every button is wired", segEl.children.every(b => typeof b.onclick === "function"), true);',
            'console.log(out.join("\\n"));',
            'console.log(out.some(l => l.startsWith("FAIL")) ? "RESULT: FAIL" : "RESULT: ALL OK");',
        ])

    def test_the_pack_logic_in_javascript(self):
        out = self.run_js(self.harness())
        self.assertIn("RESULT: ALL OK", out, out)

class NoRewrite(unittest.TestCase):
    """#2687: the flicker's cause.  Chrome translates a word, and a script that wrote the English back every second

    flipped the page to English until Chrome translated again.  The fix is not fewer updates (a live Monitor must
    still scroll) - it is never writing a node whose text did not change.  Run in node with the DOM stubbed.
    """

    def run_js(self, script):
        import subprocess
        import tempfile
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("node is not installed")
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as f:
            f.write(script)
            path = f.name
        try:
            done = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(done.returncode, 0, done.stderr[:500])
        return done.stdout

    def the_helpers(self):
        js = (WEB / "app.js").read_text(encoding="utf-8")
        start = js.index("// ------------------------------------------------------------------ writing text (#2687)")
        end = js.index("// ------------------------------------------------------------------ toasts")
        return js[start:end]

    def test_a_node_is_not_rewritten_when_its_text_did_not_change(self):
        script = "\n".join([
            'let writes = 0;',
            'const node = { get textContent() { return this._t; }, set textContent(v) { writes++; this._t = v; }, dataset: {} };',
            'const other = { get innerHTML() { return this._h; }, set innerHTML(v) { writes++; this._h = v; }, dataset: {} };',
            self.the_helpers(),
            'setText(node, "42.1"); setText(node, "42.1"); setText(node, "42.1");',
            'if (writes !== 1) { console.log("FAIL setText wrote " + writes + " times for the same value"); process.exit(1); }',
            'setText(node, "43.0");',
            'if (writes !== 2) { console.log("FAIL setText did not write a changed value"); process.exit(1); }',
            'const before = writes;',
            'setWords(node, "Decode now"); setWords(node, "Decode now");',
            'if (writes !== before + 1) { console.log("FAIL setWords rewrote an unchanged label"); process.exit(1); }',
            '// a number changes, a word does not: the Monitor may update values every tick but must not',
            '// touch the words it showed, or a translated page flickers back to English',
            'console.log("RESULT: ALL OK");',
        ])
        self.assertIn("RESULT: ALL OK", self.run_js(script))

class RenderedPages(unittest.TestCase):
    """#2686: what the three pages really show in a Chinese browser - not "the keys exist", but the rendered words.

    These need a headless browser and skip without one, so the rest of the suite still runs anywhere. Each opens
    the page as a user does and reads back the text that is on screen.
    """

    @classmethod
    def setUpClass(cls):
        import shutil
        if not shutil.which("chromium") and not shutil.which("google-chrome"):
            raise unittest.SkipTest("no headless browser installed")
        cls.browser = shutil.which("chromium") or shutil.which("google-chrome")
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, "Hello.", max_context=4096)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = "http://127.0.0.1:" + str(cls.httpd.server_address[1])

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def dom(self, path="/"):
        import subprocess
        done = subprocess.run([self.browser, "--headless", "--no-sandbox", "--disable-gpu", "--lang=zh-CN",
                               "--virtual-time-budget=12000", "--dump-dom", self.base + path],
                              capture_output=True, text=True, timeout=120)
        return done.stdout

    def test_the_html_is_chinese(self):
        self.assertIn('lang="zh-CN"', self.dom("/"))

    def test_the_chat_page_is_chinese(self):
        dom = self.dom("/")
        for word in ("对话", "监控", "关于", "随便问点什么"):
            self.assertIn(word, dom, word)
        self.assertNotIn("代币", dom)
        self.assertNotIn("令牌", dom)

    def test_the_monitor_page_is_chinese(self):
        dom = self.dom("/#monitor")
        for word in ("模型状态", "上下文占用", "最近的请求", "时间", "状态", "耗时"):
            self.assertIn(word, dom, word)
        self.assertNotIn("mon.metric.", dom, "a card showed a raw key instead of its label")

    def test_the_about_page_is_chinese(self):
        dom = self.dom("/#about")
        for word in ("模型与引擎", "这台电脑", "接入你的工具", "设置", "界面语言"):
            self.assertIn(word, dom, word)
        self.assertIn("简体中文", dom)

    def test_the_language_switch_is_the_drawer_is_segmented_control(self):
        """#2686: the language switch uses the same .seg component as the drawer's "Thinking", as radio buttons."""
        import re
        dom = self.dom("/#about")
        self.assertIn('class="seg seg--fit"', dom)
        seg = re.search(r'<div class="seg seg--fit" id="lang-seg"[^>]*>(.*?)</div>', dom, re.S)
        self.assertIsNotNone(seg, "the language switch is not there")
        buttons = re.findall(r'<button type="button" role="radio" data-lang="([^"]+)" aria-checked="([^"]+)">([^<]+)</button>', seg.group(1))
        self.assertEqual([b[0] for b in buttons], ["en", "zh-CN"])
        self.assertEqual([b[1] for b in buttons], ["false", "true"], "a Chinese browser must show 简体中文 as chosen")
        self.assertEqual(buttons[1][2], "简体中文")

    def test_no_raw_i18n_key_reaches_the_screen(self):
        """A key the pack lacks shows as the key - a bug a user can see, on any of the three pages."""
        import re
        for path in ("/", "/#monitor", "/#about"):
            dom = self.dom(path)
            shown = re.findall(r">((?:mon|chat|fact|drawer|tool|mcp|cc|toast|pill|tab|img|about|cfg|code|file|chip|export|theme|app)\.[a-zA-Z.]+)<", dom)
            self.assertEqual(shown, [], "%s showed raw keys: %s" % (path, shown[:5]))

    def test_the_console_is_clean(self):
        """An exception during start used to leave the page untranslated, silently."""
        import subprocess
        done = subprocess.run([self.browser, "--headless", "--no-sandbox", "--disable-gpu", "--lang=zh-CN",
                               "--enable-logging=stderr", "--log-level=0", "--virtual-time-budget=10000",
                               "--dump-dom", self.base + "/"],
                              capture_output=True, text=True, timeout=120)
        errors = [line for line in done.stderr.splitlines()
                  if "Uncaught" in line or ("CONSOLE" in line and ("Error" in line or "Exception" in line))]
        self.assertEqual(errors, [], "\n".join(errors[:3]))

class Wording(unittest.TestCase):
    """#2686: the wording is translated, the terms are not - and the other way round: an ordinary word must not

    be left in English just because it sits next to a term.  Three labels were (Seed, Temperature, Prompt).
    """

    def pack(self, name):
        return json.loads((WEB / "i18n" / name).read_text(encoding="utf-8"))

    def test_a_term_is_never_translated(self):
        zh = self.pack("zh-CN.json")
        joined = " ".join(v for v in zh.values() if isinstance(v, str))
        for term in ("token", "VRAM", "PCIe", "Prefill", "Decode", "MTP", "KV cache", "Top-p", "Top-k", "seed"):
            self.assertIn(term, joined, term)
        for wrong in ("代币", "令牌"):
            self.assertNotIn(wrong, joined, wrong)

    def test_an_ordinary_word_is_translated(self):
        """These are plain nouns, not terms - leaving them in English is just a missed translation."""
        zh = self.pack("zh-CN.json")
        en = self.pack("en.json")
        for key, chinese in (("drawer.seed", "随机种子"), ("drawer.temp", "温度"), ("mon.th.prompt", "提示词")):
            self.assertEqual(zh[key], chinese, key)
            self.assertNotEqual(zh[key], en[key], key)

    def test_what_the_pages_show_is_translated(self):
        """Rendered, not just in the pack: the three labels a user actually reads."""
        import shutil
        if not shutil.which("chromium") and not shutil.which("google-chrome"):
            self.skipTest("no headless browser installed")
        browser = shutil.which("chromium") or shutil.which("google-chrome")
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            import subprocess
            base = "http://127.0.0.1:" + str(httpd.server_address[1])
            env = dict(__import__("os").environ, LANG="zh_CN.UTF-8")
            def dom(path):
                return subprocess.run([browser, "--headless", "--no-sandbox", "--disable-gpu",
                                       "--user-data-dir=/tmp/crW" + path.replace("/", "_"), "--lang=zh-CN",
                                       "--virtual-time-budget=12000", "--dump-dom", base + path],
                                      capture_output=True, text=True, timeout=120, env=env).stdout
            chat, monitor = dom("/"), dom("/#monitor")
            self.assertIn("随机种子", chat)
            self.assertIn("温度", chat)
            self.assertIn("提示词", monitor)
            # and the terms survived the trip
            self.assertNotIn("代币", chat + monitor)
            self.assertIn("Top-p", chat)
        finally:
            httpd.shutdown()
            httpd.server_close()
