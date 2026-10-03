"""serve/test_workspace.py - the web app's workspace (serve/workspace.py): projects, chats, search, the read-only
file explorer and its limits, and the HTTP guards (API key, JSON, own page). No GPU.

    python -m unittest serve.test_workspace -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.workspace import MAX_TEXT, Workspace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(Path(self.tmp.name) / "ws")

    def tearDown(self):
        self.tmp.cleanup()

    def ok(self, op, **req):
        code, obj = self.ws.handle(op, req)
        self.assertEqual(code, 200, obj)
        return obj

    def err(self, op, code, **req):
        got, obj = self.ws.handle(op, req)
        self.assertEqual(got, code, obj)
        return obj["error"]["message"]

    def test_projects_files_context(self):
        p = self.ok("project.save", name="Demo app", instructions="Answer in German.")["project"]
        self.ok("project.save", id=p["id"], instructions="Answer in English.")
        f = self.ok("project.file.add", project=p["id"], name="notes.md", text="# Notes\nport 8000")["file"]
        ctx = self.ok("project.context", project=p["id"])
        self.assertEqual(ctx["instructions"], "Answer in English.")
        self.assertEqual(ctx["files"], [{"name": "notes.md", "text": "# Notes\nport 8000"}])
        self.ok("project.file.delete", project=p["id"], id=f["id"])
        self.assertEqual(self.ok("project.context", project=p["id"])["files"], [])
        self.err("project.save", 400, name="  ")
        self.err("project.file.add", 413, project=p["id"], name="big", text="x" * (MAX_TEXT + 1))

    def test_chats_meta_search_and_project_delete(self):
        p = self.ok("project.save", name="P")["project"]
        msgs = [{"role": "user", "text": "Where does the demo service listen?"}, {"role": "assistant", "text": "Port 8000."}]
        self.ok("chat.save", id="abc123def", project=p["id"], title="Demo port", messages=msgs, time=2)
        self.ok("chat.save", id="zzz999yyy", title="Other", messages=[{"role": "user", "text": "hello"}], time=1)
        st = self.ok("state")
        self.assertEqual([c["id"] for c in st["chats"]], ["abc123def", "zzz999yyy"])   # newest first
        self.assertEqual(self.ok("chat.get", id="abc123def")["chat"]["messages"], msgs)
        self.ok("chat.meta", id="zzz999yyy", title="Renamed")
        self.assertTrue(self.ok("chat.get", id="zzz999yyy")["chat"]["named"])
        hits = self.ok("search", q="8000")["hits"]
        self.assertEqual([h["id"] for h in hits], ["abc123def"])
        self.assertIn("8000", hits[0]["snippet"])
        self.assertEqual([h["id"] for h in self.ok("search", q="renamed")["hits"]], ["zzz999yyy"])   # titles too
        self.ok("project.delete", id=p["id"])                     # its chats stay, without a project
        self.assertIsNone(self.ok("chat.get", id="abc123def")["chat"]["project"])
        self.ok("chat.delete", id="abc123def")
        self.err("chat.get", 404, id="abc123def")

    def test_bad_ids_and_ops(self):
        self.err("chat.get", 400, id="../../etc/passwd")
        self.err("chat.save", 400, id="ABC", messages=[])
        self.err("chat.save", 404, id="abcdef1", project="nosuchproj1", messages=[])
        code, _ = self.ws.handle("__init__", {})
        self.assertEqual(code, 404)
        code, _ = self.ws.handle("../x", {})
        self.assertEqual(code, 404)


class Explorer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "shared"
        (self.root / "src").mkdir(parents=True)
        (self.root / "src" / "app.py").write_text("print('hi')\n")
        (self.root / "blob.bin").write_bytes(b"\x00\x01\x02")
        (self.root / "huge.txt").write_text("x" * (MAX_TEXT + 1))
        self.secret = base / "secret.txt"
        self.secret.write_text("do not read")
        os.symlink(self.secret, self.root / "link-out.txt")
        self.ws = Workspace(base / "ws", [self.root])

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_and_read_inside(self):
        code, top = self.ws.handle("fs.list", {})
        self.assertEqual((code, [e["path"] for e in top["entries"]]), (200, [str(self.root.resolve())]))
        code, d = self.ws.handle("fs.list", {"path": str(self.root)})
        self.assertEqual(code, 200)
        self.assertEqual(d["entries"][0]["name"], "src")                       # folders first
        self.assertIsNone(d["parent"])                                         # a root has no way up
        code, f = self.ws.handle("fs.read", {"path": str(self.root / "src" / "app.py")})
        self.assertEqual((code, f["text"]), (200, "print('hi')\n"))

    def test_refused(self):
        for path in (str(self.secret), str(self.root / ".." / "secret.txt"), str(self.root / "link-out.txt"), "/etc/passwd"):
            code, obj = self.ws.handle("fs.read", {"path": path})
            self.assertEqual(code, 403, (path, obj))
        self.assertEqual(self.ws.handle("fs.read", {"path": str(self.root / "blob.bin")})[0], 415)
        self.assertEqual(self.ws.handle("fs.read", {"path": str(self.root / "huge.txt")})[0], 413)
        self.assertEqual(self.ws.handle("fs.list", {"path": str(self.root.parent)})[0], 403)

    def test_off_without_roots(self):
        ws = Workspace(Path(self.tmp.name) / "ws2")
        code, obj = ws.handle("fs.read", {"path": str(self.root / "src" / "app.py")})
        self.assertEqual(code, 403)
        self.assertIn("off", obj["error"]["message"])

    def test_add_to_project_from_explorer(self):
        _, p = self.ws.handle("project.save", {"name": "P"})
        code, obj = self.ws.handle("project.file.add", {"project": p["project"]["id"], "path": str(self.root / "src" / "app.py")})
        self.assertEqual(code, 200, obj)
        self.assertEqual(self.ws.handle("project.file.add", {"project": p["project"]["id"], "path": str(self.secret)})[0], 403)


class Roots(unittest.TestCase):
    """The shared folders set in the Settings tab: kept in settings.json, checked, beside the start command's."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        self.fixed, self.a, self.b = (self.base / n for n in ("fixed", "a", "b"))
        for d in (self.fixed, self.a, self.b):
            d.mkdir()
        (self.a / "note.md").write_text("# hi\n")
        (self.a / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_keep_and_reload(self):
        ws = Workspace(self.base / "ws", [self.fixed])
        code, obj = ws.handle("roots.save", {"roots": [str(self.a), str(self.a), str(self.fixed), str(self.b)]})
        self.assertEqual(code, 200, obj)
        self.assertEqual(obj["roots"], [str(self.fixed), str(self.a), str(self.b)])     # once each, the fixed one first
        self.assertEqual([r["path"] for r in obj["root_settings"]["saved"]], [str(self.a), str(self.b)])
        self.assertEqual(ws.handle("fs.read", {"path": str(self.a / "note.md")})[0], 200)
        again = Workspace(self.base / "ws", [self.fixed])                               # a restart keeps them
        self.assertEqual(again.handle("state", {})[1]["roots"], [str(self.fixed), str(self.a), str(self.b)])
        code, obj = again.handle("roots.save", {"roots": [str(self.b)]})                 # removing one
        self.assertEqual((code, obj["roots"]), (200, [str(self.fixed), str(self.b)]))
        self.assertEqual(again.handle("fs.read", {"path": str(self.a / "note.md")})[0], 403)

    def test_refused_folders(self):
        ws = Workspace(self.base / "ws")
        (self.b / ".ssh").mkdir()
        (self.base / "credentials").mkdir()
        for bad in ("relative/path", "", str(self.base / "missing"), "/", str(Path.home()), str(self.b),
                    str(self.base), str(self.a / "note.md")):
            code, obj = ws.handle("roots.save", {"roots": [bad]})
            self.assertEqual(code, 400, (bad, obj))
        self.assertEqual(ws.handle("roots.save", {"roots": "x"})[0], 400)
        self.assertEqual(ws.handle("roots.save", {"roots": [str(self.a)] * 21})[0], 400)
        self.assertEqual(ws.handle("state", {})[1]["roots"], [])                        # nothing was kept

    def test_gone_folder_stays_listed_and_projects_outside(self):
        ws = Workspace(self.base / "ws")
        ws.handle("roots.save", {"roots": [str(self.a), str(self.b)]})
        ws.handle("project.save", {"name": "B", "folder": str(self.b)})
        self.b.rmdir()
        self.assertEqual(ws.handle("state", {})[1]["roots"], [str(self.a)])
        code, obj = ws.handle("roots.save", {"roots": [str(self.b)]})                    # gone, but was saved
        self.assertEqual(code, 200, obj)
        self.assertEqual(obj["root_settings"]["saved"], [{"path": str(self.b), "ok": False}])
        self.assertEqual(obj["outside"], ["B"])

    def test_unshared_folder_stops_the_tools(self):
        ws = Workspace(self.base / "ws")
        ws.handle("roots.save", {"roots": [str(self.a)]})
        _, p = ws.handle("project.save", {"name": "A", "folder": str(self.a), "mode": "auto"})
        run = {"project": p["project"]["id"], "name": "read_file", "arguments": {"path": "note.md"}}
        self.assertEqual(ws.handle("tool.run", run)[0], 200)
        ws.handle("roots.save", {"roots": []})
        code, obj = ws.handle("tool.run", run)
        self.assertEqual(code, 403)
        self.assertIn("not shared", obj["error"]["message"])

    def test_not_editable(self):
        ws = Workspace(self.base / "ws", [], editable=False)
        self.assertEqual(ws.handle("roots.save", {"roots": [str(self.a)]})[0], 403)
        self.assertFalse(ws.handle("state", {})[1]["root_settings"]["editable"])

    def test_image(self):
        ws = Workspace(self.base / "ws", [self.a])
        code, obj = ws.handle("fs.image", {"path": str(self.a / "pic.png")})
        self.assertEqual((code, obj["type"]), (200, "image/png"))
        self.assertEqual(ws.handle("fs.image", {"path": str(self.a / "note.md")})[0], 415)
        self.assertEqual(ws.handle("fs.image", {"path": str(self.fixed / "x.png")})[0], 403)


class Skills(unittest.TestCase):
    """A project's agent skills reach the model's system message: name, description, path; links out are skipped."""
    def test_listed_once_and_confined(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            repo, outside = base / "repo", base / "outside"
            (repo / ".agents" / "skills" / "plan").mkdir(parents=True)
            (repo / ".agents" / "skills" / "plan" / "SKILL.md").write_text(
                "---\nname: plan\ndescription: >-\n  Plan the work\n  in small pieces.\n---\n# Plan\n")
            (repo / ".agents" / "skills" / "nodesc").mkdir()
            (repo / ".agents" / "skills" / "nodesc" / "SKILL.md").write_text("---\nname: nodesc\n---\n")
            (outside / "evil").mkdir(parents=True)
            (outside / "evil" / "SKILL.md").write_text("---\nname: evil\ndescription: 'from outside'\n---\n")
            os.symlink(outside / "evil", repo / ".agents" / "skills" / "evil")
            (repo / ".claude").mkdir()
            os.symlink("../.agents/skills", repo / ".claude" / "skills")       # Claude Code's place, the same skills
            ws = Workspace(base / "ws", [base])
            _, p = ws.handle("project.save", {"name": "R", "folder": str(repo)})
            code, ctx = ws.handle("project.context", {"project": p["project"]["id"]})
            self.assertEqual(code, 200, ctx)
            self.assertEqual(ctx["skills"], [{"name": "plan", "description": "Plan the work in small pieces.",
                                              "path": ".agents/skills/plan/SKILL.md"}])


class Tools(unittest.TestCase):
    """The coding tools of a project with a folder (tool.run)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "shared"
        self.repo = self.root / "repo"
        (self.repo / "src" / "deep").mkdir(parents=True)
        (self.repo / "node_modules" / "x").mkdir(parents=True)
        (self.repo / "AGENTS.md").write_text("Run cargo test before you commit.\n")
        (self.repo / "main.py").write_text("def hello():\n    return 'hi'\n\nprint(hello())\n")
        (self.repo / "src" / "deep" / "util.py").write_text("X = 1\nX = 1\n")
        (self.repo / "node_modules" / "x" / "skip.py").write_text("pass\n")
        (base / "secret.txt").write_text("no")
        self.ws = Workspace(base / "ws", [self.root])
        _, obj = self.ws.handle("project.save", {"name": "repo", "folder": str(self.repo)})
        self.pid = obj["project"]["id"]

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, name, **args):
        code, obj = self.ws.handle("tool.run", {"project": self.pid, "name": name, "arguments": args})
        return code, obj

    def text(self, name, **args):
        code, obj = self.run_tool(name, **args)
        self.assertEqual(code, 200, obj)
        return obj

    def test_read_list_search_find(self):
        r = self.text("read_file", path="main.py")
        self.assertTrue(r["ok"])
        self.assertIn("     2\t    return 'hi'", r["text"])
        self.assertIn("     4\tprint", self.text("read_file", path="main.py", offset=4)["text"])
        self.assertIn("src/", self.text("list_dir")["text"])
        self.assertIn("main.py:1:def hello", self.text("search", pattern="def hello")["text"])
        found = self.text("find_files", pattern="**/*.py")["text"].splitlines()
        self.assertEqual(found, ["main.py", "src/deep/util.py"])               # top level too, node_modules skipped

    def test_write_and_edit(self):
        self.assertIn("created", self.text("write_file", path="new/dir/a.txt", content="one\n")["text"])
        self.assertEqual((self.repo / "new/dir/a.txt").read_text(), "one\n")
        r = self.text("edit_file", path="main.py", old_string="return 'hi'", new_string="return 'hello'")
        self.assertTrue(r["ok"], r)
        self.assertIn("return 'hello'", (self.repo / "main.py").read_text())
        r = self.text("edit_file", path="src/deep/util.py", old_string="X = 1", new_string="X = 2")
        self.assertFalse(r["ok"])
        self.assertIn("occurs 2 times", r["text"])
        self.assertTrue(self.text("edit_file", path="src/deep/util.py", old_string="X = 1", new_string="X = 2", replace_all=True)["ok"])
        self.assertFalse(self.text("edit_file", path="main.py", old_string="nope", new_string="x")["ok"])

    def test_run_command(self):
        r = self.text("run_command", command="pwd && echo out && exit 3")
        self.assertIn(str(self.repo.resolve()), r["text"])
        self.assertIn("exit code 3", r["text"])
        t0 = time.time()
        r = self.text("run_command", command="sleep 30 & sleep 30; echo never", timeout_s=1)
        self.assertLess(time.time() - t0, 10)                                 # the whole group was killed
        self.assertIn("stopped after 1 s", r["text"])

    def test_live_output_and_stop_from_the_page(self):
        import threading
        got = {}
        # the page names the job, so it can follow it while tool.run is still running
        t = threading.Thread(target=lambda: got.update(r=self.ws.handle("tool.run", {"project": self.pid, "name": "run_command",
                             "arguments": {"command": "echo first; sleep 30; echo never"}, "job": "jobpage001"})))
        t.start()
        for _ in range(50):
            code, live = self.ws.handle("job.output", {"id": "jobpage001", "since": 0})
            if code == 200 and "first" in live["text"]:
                break
            time.sleep(0.1)
        self.assertIn("first", live["text"])
        self.assertFalse(live["done"])
        self.ws.handle("job.stop", {"id": "jobpage001"})
        t.join(10)
        self.assertFalse(t.is_alive())
        text = got["r"][1]["text"]
        self.assertIn("stopped by the user", text)
        self.assertNotIn("never", text.split("\n", 1)[1])                   # the output, not the command line

    def test_background_jobs(self):
        r = self.text("run_command", command="echo started; sleep 3; echo later; sleep 30", background=True)
        self.assertIn("started in the background as job", r["text"])
        self.assertIn("started", r["text"].split("First output:", 1)[1])     # what it printed in the first 2 s
        jid = r["text"].split("as job ")[1].split(" ")[0]
        out = self.text("job_output", job_id=jid, wait_s=3)["text"]
        head, new = out.split("\n", 1)
        self.assertIn("still running", head)
        self.assertEqual(new.strip(), "later")                                # only what is new since the last read
        stopped = self.text("job_stop", job_id=jid)["text"]
        self.assertIn("stopped by the user", stopped)
        quick = self.text("run_command", command="exit 7", background=True)["text"]   # fails at once: said at once
        self.assertIn("exit code 7", quick)
        self.assertFalse(self.text("job_output", job_id="jobnope00")["ok"])

    def test_changes_and_undo(self):
        chat = "chatcp0001"
        run = lambda name, **args: self.ws.handle("tool.run", {"project": self.pid, "name": name, "arguments": args, "chat": chat})[1]
        orig = (self.repo / "main.py").read_text()
        run("edit_file", path="main.py", old_string="return 'hi'", new_string="return 'hello'")
        run("edit_file", path="main.py", old_string="return 'hello'", new_string="return 'hey'")   # a second edit
        run("write_file", path="docs/new.md", content="# New\n")
        files = {f["path"]: f for f in self.ws.handle("changes.list", {"project": self.pid, "chat": chat})[1]["files"]}
        self.assertEqual(files["main.py"]["status"], "modified")
        self.assertIn("-    return 'hi'", files["main.py"]["diff"])                 # against the FIRST original
        self.assertIn("+    return 'hey'", files["main.py"]["diff"])
        self.assertEqual((files["docs/new.md"]["status"], files["docs/new.md"]["added"]), ("added", 1))
        self.ws.handle("changes.undo", {"project": self.pid, "chat": chat, "path": "main.py"})
        self.assertEqual((self.repo / "main.py").read_text(), orig)
        self.assertTrue((self.repo / "docs/new.md").exists())
        self.ws.handle("changes.undo", {"project": self.pid, "chat": chat})
        self.assertFalse((self.repo / "docs/new.md").exists())                  # new before the chat: removed
        self.assertEqual(self.ws.handle("changes.list", {"project": self.pid, "chat": chat})[1]["files"], [])
        other = self.ws.handle("changes.list", {"project": self.pid, "chat": "chatnone01"})[1]["files"]
        self.assertEqual(other, [])                                             # another chat changed nothing

    def test_confined_to_the_folder(self):
        for path in ("../secret.txt", str(Path(self.tmp.name) / "secret.txt"), "/etc/passwd"):
            r = self.text("read_file", path=path)
            self.assertFalse(r["ok"], path)
            self.assertIn("outside the project folder", r["text"])
        self.assertFalse(self.text("write_file", path="../escape.txt", content="x")["ok"])
        self.assertFalse((Path(self.tmp.name) / "shared" / "escape.txt").exists())
        code, _ = self.ws.handle("project.save", {"name": "out", "folder": self.tmp.name})   # outside the roots
        self.assertEqual(code, 403)

    def test_allow_rules_are_kept_clean(self):
        code, obj = self.ws.handle("project.save", {"id": self.pid, "allow": ["cargo   test", "git status", "cargo test", ""]})
        self.assertEqual((code, obj["project"]["allow"]), (200, ["cargo test", "git status"]))   # trimmed, no duplicates
        self.assertEqual(self.ws.handle("project.save", {"id": self.pid, "max_rounds": "0"})[1]["project"]["max_rounds"], 0)
        self.assertEqual(self.ws.handle("project.save", {"id": self.pid, "max_rounds": "lots"})[0], 400)
        code, _ = self.ws.handle("project.save", {"id": self.pid, "allow": ["ok", 5]})
        self.assertEqual(code, 400)                                                            # not a string: refused
        self.assertEqual(self.ws.handle("project.context", {"project": self.pid})[0], 200)

    def test_read_only_mode_and_context(self):
        self.ws.handle("project.save", {"id": self.pid, "mode": "read"})
        self.assertTrue(self.text("read_file", path="main.py")["ok"])
        for name, args in (("write_file", {"path": "x", "content": "y"}), ("edit_file", {"path": "main.py", "old_string": "a", "new_string": "b"}),
                           ("run_command", {"command": "touch y"})):
            code, obj = self.run_tool(name, **args)
            self.assertEqual((code, obj["ok"]), (200, False), name)
            self.assertIn("read only", obj["text"])
        self.assertFalse((self.repo / "y").exists())
        self.assertEqual(self.ws.handle("project.save", {"id": self.pid, "mode": "yolo"})[0], 400)
        ctx = self.ws.handle("project.context", {"project": self.pid})[1]
        self.assertEqual((ctx["mode"], ctx["guide"]["name"]), ("read", "AGENTS.md"))
        self.assertEqual(ctx["folder"], str(self.repo.resolve()))


class Http(unittest.TestCase):
    """POST /workspace/<op> needs the key, JSON and Strata's own page - like /settings."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.api_key = "k3y"
        cls.svc.workspace = Workspace(Path(cls.tmp.name) / "ws")
        cls.httpd = serve(cls.svc, port=0)
        cls.host = f"127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def post(self, op, body, key="k3y", ctype="application/json", origin=None):
        h = {"Content-Type": ctype}
        if key:
            h["Authorization"] = "Bearer " + key
        if origin:
            h["Origin"] = origin
        req = urllib.request.Request(f"http://{self.host}/workspace/{op}", data=json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read() or b"{}")

    def test_guards(self):
        self.assertEqual(self.post("state", {}, key=None)[0], 401)
        self.assertEqual(self.post("state", {}, ctype="text/plain")[0], 415)
        self.assertEqual(self.post("state", {}, origin="http://evil.example")[0], 403)
        code, obj = self.post("state", {}, origin=f"http://{self.host}")
        self.assertEqual(code, 200, obj)
        self.assertEqual(obj["roots"], [])
        code, obj = self.post("project.save", {"name": "Via HTTP"})
        self.assertEqual((code, obj["project"]["name"]), (200, "Via HTTP"))
        self.assertEqual(self.post("nope", {})[0], 404)


if __name__ == "__main__":
    unittest.main()
