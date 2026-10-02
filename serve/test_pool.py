"""serve/test_pool.py - the pool's server side without a GPU: the pool config, the worker supervisor, discovery,
and switching a running server between the roles (a scripted engine stands in for `strata`).

    python -m unittest serve.test_pool -v
"""
from __future__ import annotations

import json
import os
import socket
import stat
import sys
import tempfile
import textwrap
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.pool import (Discovery, PoolConfig, PoolManager, WorkerSupervisor, config_problems,  # noqa: E402
                        coordinator_args, normalize_addr, parse_engine_pool, strip_pool_args, worker_args)
from serve.server import ByteTokenizer, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# A stand-in for `strata --serve`: a worker prints its POOL lines; a chat engine speaks the stdin protocol.
FAKE = textwrap.dedent('''\
    #!{python}
    import os, sys, time
    args = sys.argv[1:]
    log = sys.stderr
    if "--pool-listen" in args:
        print("POOL LISTEN 7701", flush=True)
        if os.environ.get("FAKE_RELOAD_ONCE") and not os.path.exists(os.environ["FAKE_RELOAD_ONCE"]):
            open(os.environ["FAKE_RELOAD_ONCE"], "w").close()
            print("POOL COORD 10.0.0.1:5555", flush=True)
            print("POOL RELOAD", flush=True)
            sys.exit(3)
        print("POOL COORD 10.0.0.1:5555", flush=True)
        print("POOL CONFIG 24 48 4096", flush=True)
        print("POOL READY lb=24 le=48 slots=900 cache_mib=1800 chunk=2048 vram_free_mib=600 arena_mib=12000", flush=True)
        print("POOL REQ windows=10 tokens=17 prompt=100 ms_verify=55.5 ms_prompt=200.0 hits=90 lookups=100", flush=True)
        while True:
            time.sleep(1)
    if "--pool-peers" in args:
        if os.environ.get("FAKE_POOL_FAIL"):
            print("strata pool: pool worker 10.0.0.9:7701 is not reachable: no answer (timed out)", file=log, flush=True)
            sys.exit(1)
        peers = args[args.index("--pool-peers") + 1]
        if os.environ.get("STRATA_POOL_SECRET", "") == "":
            print("strata pool: no secret", file=log, flush=True)
            sys.exit(1)
        print("INFO pool=coordinator pool_layers=0-23 pool_workers=" + peers.split(",")[0] + "@24-47/900 "
              "pool_wire=f32 pool_slots_here=1000", flush=True)
    print("INFO kv=int8 expert_slots=1900 engine=0.1.30", flush=True)
    print("READY 4096 stop", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line == "QUIT":
            break
        if line.startswith("GEN"):
            print("RESUME 0", flush=True)
            print("REUSED 0", flush=True)
            for t in (60, 47, 116, 104, 105, 110, 107, 62, 10, 10, 111, 107, 257):
                print(f"T {{t}}", flush=True)
            print("DONE 13 5 1.0 1.0 stop 0 0 0 0 0", flush=True)
''')


def fake_engine(tmp: Path) -> str:
    p = tmp / "strata-fake"
    p.write_text(FAKE.format(python=sys.executable))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def wait_for(cond, timeout=20.0, step=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return None


class Config(unittest.TestCase):
    def test_defaults_and_secret(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.pool.json")
            pc = PoolConfig.load(path)
            self.assertEqual(pc.role, "off")
            self.assertTrue(len(pc["secret"]) >= 16)            # made once, then kept
            self.assertEqual(PoolConfig.load(path)["secret"], pc["secret"])

    def test_validation(self):
        ok = PoolConfig.clean({"role": "coordinator", "peers": ["laptop", {"addr": "10.0.0.5:7800", "enabled": False}],
                               "split": " 30 "})
        self.assertEqual(ok["peers"][0]["addr"], "laptop:7701")
        self.assertEqual(ok["peers"][1]["addr"], "10.0.0.5:7800")
        self.assertEqual(ok["split"], "30")
        for bad in ({"role": "boss"}, {"worker_port": 0}, {"split": "abc"}, {"wire": "int4"},
                    {"peers": ["bad host!"]}, {"role": "coordinator", "peers": ["a", "b"], "split": "30"},
                    {"split": "30,20"}, {"split": "1"}, {"secret": "a\nb"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                PoolConfig.clean(bad)
        # duplicates collapse, unknown keys go
        c = PoolConfig.clean({"peers": ["a", "a:7701"], "zzz": 1})
        self.assertEqual(len(c["peers"]), 1)
        self.assertNotIn("zzz", c)

    def test_addresses(self):
        self.assertEqual(normalize_addr("pc", 7701), "pc:7701")
        self.assertEqual(normalize_addr("pc:80", 7701), "pc:80")
        self.assertEqual(normalize_addr("fe80::1", 7701), "[fe80::1]:7701")
        self.assertEqual(normalize_addr("[fe80::1]:9", 7701), "[fe80::1]:9")
        for bad in ("", "a b", "pc:x", "pc:70000", "[zz"):
            self.assertIsNone(normalize_addr(bad, 7701), bad)

    def test_engine_arguments(self):
        pc = PoolConfig(None, {"role": "coordinator", "peers": ["a", {"addr": "b", "enabled": False}, "c"],
                               "split": "auto", "wire": "f16"})
        self.assertEqual(coordinator_args(pc), ["--pool-peers", "a:7701,c:7701", "--pool-split", "auto",
                                                "--pool-wire", "f16", "--pool-draft-wire", "f16"])
        self.assertEqual(worker_args(pc), ["--pool-listen", "0.0.0.0:7701"])
        self.assertEqual(strip_pool_args(["--x", "1", "--pool-peers", "a", "--pool-wire=f16", "--y"]), ["--x", "1", "--y"])

    def test_problems(self):
        pc = PoolConfig(None, {"role": "worker"})
        self.assertEqual(config_problems({"args": []}, pc), [])
        self.assertEqual(len(config_problems({"args": [], "vision": {"exe": "v"}}, pc)), 1)
        self.assertEqual(len(config_problems({"args": ["--control-vector-scaled", "x:1"], "gpu": "0,1"}, pc)), 2)
        self.assertEqual(config_problems({"args": [], "vision": {}}, PoolConfig(None, {"role": "off"})), [])
        # an engine without the layer split's code: the split roles say so, routing does not need it
        with tempfile.TemporaryDirectory() as d:
            old, new = Path(d) / "old.exe", Path(d) / "new.exe"
            old.write_bytes(b"MZ... --layer-split ...")
            new.write_bytes(b"MZ... --pool-listen [H:]P ...")
            self.assertIn("built without", config_problems({"args": [], "exe": str(old)}, pc)[0])
            self.assertEqual(config_problems({"args": [], "exe": str(new)}, pc), [])
            self.assertEqual(config_problems({"args": [], "exe": str(old)}, PoolConfig(None, {"role": "router"})), [])

    def test_engine_pool_info(self):
        p = parse_engine_pool({"pool": "coordinator", "pool_layers": "0-21",
                               "pool_workers": "10.0.0.5:7701@22-35/800,10.0.0.6:7701@36-47/700", "pool_wire": "f32"})
        self.assertEqual(p["le"], 22)
        self.assertEqual([(w["lb"], w["le"], w["slots"]) for w in p["workers"]], [(22, 36, 800), (36, 48, 700)])
        self.assertIsNone(parse_engine_pool({}))


class Supervisor(unittest.TestCase):
    def test_states_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            exe = fake_engine(Path(d))
            marker = os.path.join(d, "reloaded")
            env = dict(os.environ, FAKE_RELOAD_ONCE=marker)
            msgs = []
            w = WorkerSupervisor(exe, ["--pool-listen", "0.0.0.0:7701"], log=os.path.join(d, "w.log"), env=env,
                                 echo=msgs.append)
            w.start()
            try:
                st = wait_for(lambda: w.snapshot()["state"] == "serving" and w.snapshot())
                self.assertTrue(st, w.snapshot())
                self.assertEqual(st["restarts"], 1)              # exit 3 = reload: started again at once
                self.assertEqual((st["lb"], st["le"]), (24, 48))
                self.assertEqual(st["ready"]["slots"], 900)
                self.assertTrue(wait_for(lambda: w.snapshot()["requests"] == 1))
                self.assertAlmostEqual(w.snapshot()["last_request"]["ms_verify"], 55.5)
                self.assertTrue(any("serving layers 24-47" in m for m in msgs))
            finally:
                w.stop()
            self.assertEqual(w.snapshot()["state"], "stopped")


class Beacons(unittest.TestCase):
    def test_two_pcs_hear_each_other(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        a = Discovery(lambda: {"name": "desktop", "role": "coordinator", "http_port": 8080, "worker_port": 7701},
                      port=port, dests=("127.0.0.1",))
        b = Discovery(lambda: {"name": "laptop", "role": "worker", "http_port": 8081, "worker_port": 7701},
                      port=port, dests=("127.0.0.1",))
        a.start()
        b.start()
        try:
            # both bind the same port (SO_REUSEADDR); unicast to 127.0.0.1 reaches one of them - enough to see the
            # parsing, the self-filter and freshness
            seen = wait_for(lambda: a.peers() + b.peers(), timeout=10)
            self.assertTrue(seen)
            names = {p["name"] for p in seen}
            self.assertTrue(names <= {"desktop", "laptop"})
            for p in seen:
                self.assertTrue(p["fresh"])
                self.assertEqual(p["ip"], "127.0.0.1")
        finally:
            a.stop()
            b.stop()


class Switching(unittest.TestCase):
    """A server that starts as a worker, becomes a PC alone, then a coordinator - without restarting."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.exe = fake_engine(d)
        self.cfg = {"exe": self.exe, "args": ["--max-context", "4096", "--spec", "4"], "cwd": None,
                    "log": str(d / "engine.log")}
        self.pc = PoolConfig.load(str(d / "x.pool.json"))
        self.pc.data["role"] = "worker"
        self.pc.data["discovery"] = False
        self.pc.save()
        env = dict(os.environ)
        env.pop("FAKE_POOL_FAIL", None)
        self.engine = StrataEngine.deferred(self.exe, self.cfg["args"], log=self.cfg["log"], env=env)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.pool = PoolManager(self.svc, self.cfg, self.pc, self.cfg["args"], env, 0, echo=lambda m: None)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.svc.pool.start()

    def tearDown(self):
        self.svc.pool.close()
        try:
            self.svc.engine.close()
        except Exception:  # noqa: BLE001
            pass
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def req(self, method, path, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        r = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                   headers=h, method=method)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read().decode() or "{}")

    def chat(self):
        return self.req("POST", "/v1/chat/completions",
                        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20})

    def test_roles(self):
        code, st = self.req("GET", "/pool")
        self.assertEqual(code, 200)
        self.assertEqual(st["config"]["role"], "worker")
        self.assertTrue(st["config"]["secret"])              # this PC itself sees the secret
        self.assertTrue(wait_for(lambda: self.req("GET", "/pool")[1]["worker"]["state"] == "serving"))
        code, node = self.req("GET", "/pool/node")
        self.assertEqual((code, node["role"], node["layers"]), (200, "worker", "24-47"))
        self.assertNotIn("secret", json.dumps(node))
        # a worker does not chat
        code, body = self.chat()
        self.assertEqual(code, 503)
        self.assertIn("pool coordinator", body["error"]["message"])
        # from another page: refused
        code, _ = self.req("POST", "/pool/config", {"role": "off"}, headers={"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        # bad settings: 400 with the reason
        code, body = self.req("POST", "/pool/config", {"role": "coordinator", "peers": []})
        self.assertEqual(code, 400)
        self.assertIn("worker", body["error"]["message"])
        # -> this PC alone: the worker stops, the chat engine loads
        code, body = self.req("POST", "/pool/config", {"role": "off"})
        self.assertEqual((code, body["status"]), (200, "applying"))
        self.assertTrue(wait_for(lambda: self.svc.loaded() and not self.svc.pool.applying), self.svc.pool.last_error)
        self.assertIsNone(self.svc.pool.worker)
        code, body = self.chat()
        self.assertEqual(code, 200, body)
        self.assertEqual(body["choices"][0]["message"]["content"], "ok")
        # -> coordinator of one worker: the engine restarts with the pool's arguments and its secret
        code, body = self.req("POST", "/pool/config", {"role": "coordinator", "peers": ["10.0.0.9"], "split": "24"})
        self.assertEqual(code, 200, body)
        self.assertTrue(wait_for(lambda: self.svc.loaded() and not self.svc.pool.applying and
                                 self.svc.engine.info.get("pool") == "coordinator"), self.svc.pool.last_error)
        self.assertIn("--pool-peers", self.svc.engine.spawn[1])
        self.assertEqual(self.svc.engine.spawn[4]["STRATA_POOL_SECRET"], self.pc["secret"])
        code, st = self.req("GET", "/pool")
        self.assertEqual(st["engine"]["pool"]["workers"][0]["addr"], "10.0.0.9:7701")
        self.assertEqual(st["engine"]["pool"]["le"], 24)
        code, body = self.chat()
        self.assertEqual(code, 200, body)
        # the saved file has the new role (the next start keeps it)
        self.assertEqual(json.loads(Path(self.pc.path).read_text())["role"], "coordinator")
        # a LAN neighbour does not get the secret
        self.assertEqual(self.svc.pool.state(reveal_secret=False)["config"]["secret"], "")

    def test_coordinator_that_cannot_reach_its_worker(self):
        self.svc.pool.base_env["FAKE_POOL_FAIL"] = "1"
        code, _ = self.req("POST", "/pool/config", {"role": "coordinator", "peers": ["10.0.0.9"]})
        self.assertEqual(code, 200)
        err = wait_for(lambda: self.svc.pool.last_error)
        self.assertIn("not reachable", err or "")
        code, st = self.req("GET", "/pool")
        self.assertIn("not reachable", st["error"])


# ------------------------------------------------------------------------------------------------ routing mode
from serve import route  # noqa: E402


class RoutingPieces(unittest.TestCase):
    def test_chain_follows_a_conversation(self):
        a = {"messages": [{"role": "user", "content": "hi"}]}
        b = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}]}
        b2 = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]},
                           {"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]}
        ha, ca = route.chain("/v1/chat/completions", a)
        hb, _ = route.chain("/v1/messages", b)
        hb2, cb2 = route.chain("/v1/messages", b2)
        self.assertEqual(len(ha), 1)
        self.assertEqual(hb[0], hb2[0])                    # the caching marker does not change the conversation
        self.assertNotEqual(ha[0], hb[0])                  # another API's prompt is another conversation
        self.assertEqual(ca, [len("user") + len("hi")])
        self.assertEqual(cb2[-1], len("user" + "text" + "hi" + "assistant" + "hello" + "user" + "more"))
        t = route.Table(cap=3)
        t.add(hb)
        self.assertEqual(t.match(hb2), 1)
        t.add(hb2)
        self.assertEqual(t.match(hb2), 3)
        t.add(["x", "y", "z"])                             # the oldest go first
        self.assertEqual(t.match(hb2), 0)

    def test_signatures(self):
        v = route.Verifier(lambda: "s3cret")
        sig = route.sign("s3cret", "/v1/messages", b"{}")
        self.assertTrue(v.ok(sig, "/v1/messages", b"{}"))
        self.assertFalse(v.ok(sig, "/v1/messages", b"{}"))           # a replay
        self.assertFalse(v.ok(route.sign("other", "/v1/messages", b"{}"), "/v1/messages", b"{}"))
        self.assertFalse(v.ok(route.sign("s3cret", "/v1/messages", b"{}"), "/v1/messages", b"{ }"))
        self.assertFalse(v.ok(route.sign("s3cret", "/v1/messages", b"{}", now=time.time() - 600), "/v1/messages", b"{}"))
        self.assertFalse(v.ok("garbage", "/v1/messages", b"{}"))
        self.assertNotEqual(route.secret_id("a"), route.secret_id("b"))

    def test_choices(self):
        chars = [1000, 50000, 350000]                       # ~100K tokens
        me = {"local": True, "match": 0, "busy": False, "queued": 0, "max_context": 262144, "loaded": True}
        peer = {"local": False, "addr": "lap:8080", "match": 0, "busy": False, "queued": 0, "max_context": 196608,
                "loaded": True}
        tok = route.est_tokens(chars[-1])
        # a new conversation: the primary when idle, else this PC
        d = route.pick([me, {**peer, "primary": True}], chars, tok, 8000)
        self.assertEqual(d.get("addr"), "lap:8080")
        d = route.pick([me, peer], chars, tok, 8000)
        self.assertTrue(d["local"])
        d = route.pick([{**me, "busy": True}, peer], chars, tok, 8000)
        self.assertEqual(d.get("addr"), "lap:8080")
        # it stays where its context is
        d = route.pick([me, {**peer, "match": 2}], chars, tok, 8000)
        self.assertEqual(d.get("addr"), "lap:8080")
        # that PC is busy: a long conversation waits there, a short one moves
        d = route.pick([me, {**peer, "match": 2, "busy": True}], chars, tok, 8000)
        self.assertEqual(d.get("addr"), "lap:8080")
        short = [100, 200, 3000]
        d = route.pick([me, {**peer, "match": 2, "busy": True}], short, route.est_tokens(3000), 8000)
        self.assertTrue(d["local"])
        # a peer whose window is too small is never chosen
        big = [1000, 50000, 800000]                         # ~228K tokens
        d = route.pick([{**me, "busy": True}, {**peer, "primary": True, "match": 2}], big, route.est_tokens(big[-1]), 8000)
        self.assertTrue(d["local"])


def router_server(tmp: Path, name: str, secret: str, primary=False):
    exe = fake_engine(tmp)
    cfg = {"exe": exe, "args": ["--max-context", "4096", "--spec", "4"], "cwd": None, "log": str(tmp / f"{name}.log")}
    pc = PoolConfig.load(str(tmp / f"{name}.pool.json"))
    pc.data.update(role="router", name=name, secret=secret, discovery=False, primary=primary)
    env = dict(os.environ)
    engine = StrataEngine(exe, cfg["args"], log=cfg["log"], env=env)
    svc = Service(engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.pool = PoolManager(svc, cfg, pc, cfg["args"], env, 0, echo=lambda m: None)
    httpd = serve(svc, port=0)
    return svc, httpd, pc


class Routing(unittest.TestCase):
    """Two PCs in routing mode, each a server with a scripted engine: a request reaching either runs where it should,
    and a PC whose peer is gone answers on its own."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "a").mkdir()
        (d / "b").mkdir()
        self.a, self.ha, pa = router_server(d / "a", "desk", "pw", primary=True)
        self.b, self.hb, pb = router_server(d / "b", "lap", "pw")
        self.pa, self.pb = self.ha.server_address[1], self.hb.server_address[1]
        pa.data["route_peers"] = [{"addr": f"127.0.0.1:{self.pb}", "enabled": True, "label": ""}]
        pb.data["route_peers"] = [{"addr": f"127.0.0.1:{self.pa}", "enabled": True, "label": ""}]
        self.a.pool.start()
        self.b.pool.start()
        ok = wait_for(lambda: all(p.get("usable") for p in self.a.pool.router.peers.values()) and
                      all(p.get("usable") for p in self.b.pool.router.peers.values()) and
                      self.a.pool.router.peers and self.b.pool.router.peers, timeout=15)
        self.assertTrue(ok, (self.a.pool.router.state(), self.b.pool.router.state()))

    def tearDown(self):
        for svc, h in ((self.a, self.ha), (self.b, self.hb)):
            svc.pool.close()
            try:
                svc.engine.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                h.shutdown()
                h.server_close()
            except Exception:  # noqa: BLE001
                pass
        self.tmp.cleanup()

    def chat(self, port, messages, headers=None):
        r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                   data=json.dumps({"model": "m", "messages": messages, "max_tokens": 20}).encode(),
                                   headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, resp.headers.get("X-Strata-Pool-Ran-On"), json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, None, json.loads(e.read().decode() or "{}")

    def test_routes_and_islands(self):
        conv = [{"role": "user", "content": "a new question"}]
        # a new conversation reaching the laptop runs on the idle primary (the desktop)
        code, ran, body = self.chat(self.pb, conv)
        self.assertEqual((code, ran), (200, "desk"), body)
        self.assertEqual(body["choices"][0]["message"]["content"], "ok")
        self.assertEqual(self.b.pool.router.counts["routed_out"], 1)
        self.assertEqual(self.a.pool.router.counts["routed_in"], 1)
        # its next turn, reaching the laptop again, goes back to the desktop: it holds the conversation
        conv2 = conv + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "and then?"}]
        code, ran, _ = self.chat(self.pb, conv2)
        self.assertEqual((code, ran), (200, "desk"))
        # a conversation the laptop holds stays there, also when it reaches the desktop
        self.b.pool.router.table.add(route.chain("/v1/chat/completions", [{"role": "user", "content": "lap's"}] and
                                                 {"messages": [{"role": "user", "content": "lap's"}]})[0])
        code, ran, _ = self.chat(self.pa, [{"role": "user", "content": "lap's"}, {"role": "assistant", "content": "x"},
                                           {"role": "user", "content": "go on"}])
        self.assertEqual((code, ran), (200, "lap"))
        # a stream stays a stream through the other PC
        r = urllib.request.Request(f"http://127.0.0.1:{self.pb}/v1/chat/completions",
                                   data=json.dumps({"model": "m", "stream": True, "max_tokens": 20,
                                                    "messages": [{"role": "user", "content": "stream it"}]}).encode(),
                                   headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(r, timeout=30) as resp:
            self.assertEqual(resp.headers.get("X-Strata-Pool-Ran-On"), "desk")
            self.assertIn("text/event-stream", resp.headers.get("Content-Type", ""))
            text = resp.read().decode()
        self.assertIn("data: [DONE]", text)
        # routing off and on again: the same engine keeps running (no reload)
        proc = self.a.engine.proc
        self.a.pool.apply({"role": "off"})
        self.assertTrue(wait_for(lambda: not self.a.pool.applying and self.a.pool.router is None))
        self.a.pool.apply({"role": "router"})
        self.assertTrue(wait_for(lambda: not self.a.pool.applying and self.a.pool.router is not None))
        self.assertIs(self.a.engine.proc, proc)
        self.assertTrue(wait_for(lambda: any(p.get("usable") for p in self.a.pool.router.peers.values()), timeout=15))
        # a request signed with another secret is turned away; an unsigned "routed" one too
        code, _, _ = self.chat(self.pa, conv, headers={route.ROUTED_HEADER: "x",
                                                       route.AUTH_HEADER: route.sign("nope", "/v1/chat/completions", b"")})
        self.assertEqual(code, 503)
        # the laptop goes away: the desktop answers its own requests at once and keeps looking
        self.b.pool.close()
        self.hb.shutdown()
        self.hb.server_close()
        code, ran, body = self.chat(self.pa, [{"role": "user", "content": "while alone"}])
        self.assertEqual((code, ran), (200, None), body)
        self.assertTrue(wait_for(lambda: self.a.pool.router.state()["islanded"], timeout=15))
        code, _, st = 200, None, self.a.pool.state(reveal_secret=False)
        self.assertTrue(st["router"]["islanded"])


if __name__ == "__main__":
    unittest.main()
