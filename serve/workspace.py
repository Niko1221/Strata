"""serve/workspace.py - the web app's workspace (#361): projects, chats kept on this server, search across them, a
read-only file explorer over shared folders, and the coding tools a project with a folder gets (the agent loop itself
runs in the page, serve/web/app.js).

Everything lives in one folder (default: Strata-data/workspace next to the Strata folder):

    projects.json            {"projects": [{id, name, instructions, files: [{id, name, size, source, added}], time}]}
    chats.json               {"chats": [{id, project, title, named, time}]}   the list, newest first
    chats/<id>.json          {id, project, title, named, time, messages}      one chat, as the web app keeps it
    files/<project>/<id>.txt the text of a project file
    settings.json            {"roots": [folder, ...]}   the shared folders set in the web app's Settings tab

The web app reaches it with POST /workspace/<op> (server.py): the API key and Strata's own page are required, like
/settings. The file explorer only reads, only inside the folders given as roots, and is off when no roots are set. Roots come
from the start command (--workspace-root, fixed) and from the Settings tab (settings.json, changeable there).
"""
from __future__ import annotations

import base64
import difflib
import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

ID = re.compile(r"^[a-z0-9]{6,40}$")             # chat ids come from the page (base36), project/file ids from here
MAX_TEXT = 512 * 1024                             # one file read or added (the chat's own limit for attachments)
MAX_PROJECT_TEXT = 4 * 1024 * 1024                # all files of one project together
MAX_ENTRIES = 2000                                # one folder listing
MAX_HITS = 50
MODES = ("read", "ask", "edit", "auto")           # read only · ask before changes and commands · edits auto · all auto
READ_TOOLS = {"list_dir", "read_file", "search", "find_files"}
WRITE_TOOLS = {"write_file", "edit_file"}
RUN_TOOLS = {"run_command", "job_output", "job_stop"}
JOB_KEEP = 512 * 1024                             # the last bytes of a command's output kept for the live view
JOB_TTL = 3600                                    # a finished job is forgotten after an hour
MAX_TOOL_OUT = 30_000                             # characters of a tool's result the model reads
MAX_IMAGE = 5 * 1024 * 1024                       # one picture shown in a Markdown file
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp", ".avif": "image/avif", ".bmp": "image/bmp", ".ico": "image/x-icon",
               ".svg": "image/svg+xml"}            # an <img> does not run an SVG's scripts
MAX_SKILLS = 100                                  # a project's agent skills listed to the model
MAX_ROOTS = 20                                    # shared folders set in the Settings tab
# a folder holding one of these is not shared from the Settings tab: it would open keys to the page and the agent
SECRET_DIRS = (".ssh", ".gnupg", ".aws", ".kube", ".password-store", "credentials")
SKIP_DIRS = {".git", "node_modules", "target", "dist", "build", ".next", "__pycache__", ".venv", "venv"}
NO_LOCK = {"tool.run", "job.output", "job.stop"}  # a command can take minutes: it must not hold up the other ops


class WorkspaceError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _now() -> int:
    return int(time.time() * 1000)                # milliseconds, like the page's Date.now()


def _text(v, name: str, limit: int = 100_000) -> str:
    if v is None:
        return ""
    if not isinstance(v, str):
        raise WorkspaceError(400, f"{name} must be a string")
    if len(v) > limit:
        raise WorkspaceError(413, f"{name} is longer than {limit:,} characters")
    return v


def _id(v, name: str = "id") -> str:
    if not isinstance(v, str) or not ID.match(v):
        raise WorkspaceError(400, f"{name} is not a valid id")
    return v


class Job:
    """One command in a project folder: its output is collected as it comes (for the page's live view and for
    job_output), the whole process group is killed on stop or timeout."""

    def __init__(self, jid, folder, command, timeout):
        self.id, self.command, self.started = jid, command, time.time()
        self.buf = bytearray()                    # the output's last JOB_KEEP bytes
        self.total = 0                            # bytes ever written (offsets for the readers)
        self.read_by_model = 0                    # where job_output left off
        self.code, self.status, self.done = None, "running", threading.Event()
        env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PAGER": "cat", "GIT_PAGER": "cat", "CI": "1"}
        self.proc = subprocess.Popen(["bash", "-lc", command], cwd=folder, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()
        self.timer = threading.Timer(timeout, self.stop, args=(f"stopped after {timeout} s (timeout_s)",))
        self.timer.daemon = True
        self.timer.start()

    def _read(self):
        while True:
            chunk = self.proc.stdout.read1(65536)
            if not chunk:
                break
            with self.lock:
                self.buf += chunk
                self.total += len(chunk)
                if len(self.buf) > JOB_KEEP:
                    del self.buf[:len(self.buf) - JOB_KEEP]
        self.proc.wait()
        self.timer.cancel()
        with self.lock:
            self.code = self.proc.returncode
            if self.status == "running":
                self.status = f"exit code {self.code}"
        self.done.set()

    def stop(self, why="stopped by the user"):
        with self.lock:
            if self.done.is_set() or self.status != "running":
                return
            self.status = why
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)   # the whole group: a build's children too
        except ProcessLookupError:
            pass

    def since(self, offset):
        """(text written after `offset`, the new offset); output older than what is kept is skipped."""
        with self.lock:
            start = self.total - len(self.buf)
            offset = max(offset, start)
            data = bytes(self.buf[offset - start:])
            return data.decode("utf-8", errors="replace"), self.total

    def text(self):
        out, _ = self.since(0)
        dropped = self.total - len(self.buf)
        return (f"[... the first {dropped:,} bytes are not kept ...]\n" if dropped > 0 else "") + out


def _frontmatter(text: str) -> dict:
    """name and description from a SKILL.md's front matter (plain, quoted or folded with >- / |), no YAML library."""
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    out, key = {}, None
    for line in (m.group(1).splitlines() if m else []):
        kv = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if kv:
            key, val = kv.group(1), kv.group(2).strip()
            out[key] = "" if val in (">", ">-", "|", "|-") else val.strip("\"'")
        elif key and line.startswith((" ", "\t")):
            out[key] = (out[key] + " " + line.strip()).strip()
    return out


def _skills(folder: Path) -> list:
    """A project's agent skills (.agents/skills/<name>/SKILL.md, also .claude/skills): name, description and path, for
    the model to read the one a task needs. Only files inside the folder (a link out of it is skipped)."""
    found, seen = [], set()
    try:
        root = folder.resolve()
    except OSError:
        return found
    for base in (".agents/skills", ".claude/skills"):
        d = folder / base
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir(), key=lambda x: x.name):
            f = sub / "SKILL.md"
            try:
                real = f.resolve()
                if not (f.is_file() and real.is_relative_to(root)) or real.stat().st_size > 256 * 1024:
                    continue
                meta = _frontmatter(real.read_text(encoding="utf-8", errors="replace")[:8192])
            except OSError:
                continue
            name = meta.get("name") or sub.name
            if name in seen or not meta.get("description"):
                continue
            seen.add(name)
            found.append({"name": name[:80], "description": meta["description"][:400],
                          "path": f.relative_to(folder).as_posix()})
            if len(found) >= MAX_SKILLS:
                return found
    return found


class Workspace:
    def __init__(self, folder, roots=(), editable=True):
        self.dir = Path(folder)
        (self.dir / "chats").mkdir(parents=True, exist_ok=True)
        (self.dir / "files").mkdir(parents=True, exist_ok=True)
        self.fixed_roots = []                     # from the start command: shown in Settings, not changed there
        for r in roots or ():
            p = Path(os.path.expandvars(os.path.expanduser(str(r)))).resolve()
            if p.is_dir() and p not in self.fixed_roots:
                self.fixed_roots.append(p)
        self.editable = editable                  # False: no shared folders from Settings (see server.py)
        self.saved_roots = []                     # from the Settings tab (settings.json), kept even while missing
        if editable:
            try:
                saved = self._load("settings.json", {}).get("roots", [])
            except (WorkspaceError, AttributeError) as e:
                print(f"[strata] workspace: {self.dir / 'settings.json'} is ignored: {e}", flush=True)
                saved = []
            for r in saved if isinstance(saved, list) else []:
                if isinstance(r, str) and r and Path(r) not in self.saved_roots:
                    self.saved_roots.append(Path(r))
        self._set_roots()
        self.lock = threading.Lock()
        self.jobs = {}                            # job id -> Job (commands, also in the background)
        self.jobs_lock = threading.Lock()

    def _set_roots(self):
        roots = list(self.fixed_roots)
        for p in self.saved_roots:
            try:
                r = p.resolve()
            except OSError:
                continue
            if r.is_dir() and r not in roots:
                roots.append(r)
        self.roots = roots                        # one assignment: readers see the old list or the new one

    def _root_settings(self):
        return {"editable": self.editable, "fixed": [str(r) for r in self.fixed_roots],
                "saved": [{"path": str(p), "ok": p.is_dir()} for p in self.saved_roots], "max": MAX_ROOTS}

    # ------------------------------------------------------------------ storage
    def _load(self, name, default):
        try:
            return json.loads((self.dir / name).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except ValueError as e:
            raise WorkspaceError(500, f"{self.dir / name} does not parse: {e}") from None

    def _save(self, name, obj):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)   # also when the folder was removed while Strata runs
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)                      # atomic: a crash leaves the old or the new file, never half of one

    def _projects(self):
        return self._load("projects.json", {"projects": []})["projects"]

    def _chats(self):
        return self._load("chats.json", {"chats": []})["chats"]

    def _project(self, projects, pid):
        for p in projects:
            if p["id"] == pid:
                return p
        raise WorkspaceError(404, "no such project")

    # ------------------------------------------------------------------ the ops
    def handle(self, op: str, req: dict):
        """(HTTP status, JSON object) for POST /workspace/<op>."""
        fn = getattr(self, "op_" + op.replace(".", "_").replace("/", "_"), None) if re.match(r"^[a-z.]+$", op) else None
        if fn is None:
            return 404, {"error": {"message": f"no workspace operation {op!r}"}}
        try:
            if op in NO_LOCK:
                return 200, fn(req)
            with self.lock:
                return 200, fn(req)
        except WorkspaceError as e:
            return e.code, {"error": {"message": str(e)}}
        except OSError as e:
            return 500, {"error": {"message": f"{type(e).__name__}: {e}"}}

    def op_state(self, req):
        self._set_roots()                         # a saved folder made (again) or removed since
        return {"projects": self._projects(), "chats": self._chats(), "roots": [str(r) for r in self.roots],
                "root_settings": self._root_settings(), "limits": {"text": MAX_TEXT, "project_text": MAX_PROJECT_TEXT}}

    # the shared folders (Settings tab)
    def _check_root(self, raw) -> Path:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
            raise WorkspaceError(400, "a folder must be a non-empty path")
        given = Path(os.path.expandvars(os.path.expanduser(raw.strip())))
        if not given.is_absolute():
            raise WorkspaceError(400, f"{raw}: give the full path (for example /home/me/projects or ~/projects)")
        p = given.resolve()
        if not p.is_dir():
            raise WorkspaceError(400, f"{p} is not a folder on this server")
        if p == Path(p.anchor):
            raise WorkspaceError(400, f"{p} is the whole disk: share a folder inside it")
        home = Path.home().resolve()
        if home == p or home.is_relative_to(p):
            raise WorkspaceError(400, f"{p} holds your whole home folder and its keys: share a folder inside it")
        for name in SECRET_DIRS:
            if name in p.parts or (p / name).exists():
                raise WorkspaceError(400, f"{p} holds {name} (keys or passwords): share a folder without it")
        return p

    def op_roots_save(self, req):
        """{roots: [folder, ...]}: the shared folders set in the Settings tab (besides the start command's)."""
        if not self.editable:
            raise WorkspaceError(403, "shared folders cannot be set here: other devices reach this server and it has "
                                      "no API key")
        raw = req.get("roots")
        if not isinstance(raw, list) or len(raw) > MAX_ROOTS:
            raise WorkspaceError(400, f"roots must be a list of at most {MAX_ROOTS} folders")
        keep = {str(p) for p in self.saved_roots}     # a saved folder that is gone for now may stay in the list
        clean = []
        for r in raw:
            p = Path(r) if isinstance(r, str) and r in keep and not Path(r).is_dir() else self._check_root(r)
            if p not in clean and p not in self.fixed_roots:
                clean.append(p)
        self._save("settings.json", {**self._load("settings.json", {}), "roots": [str(p) for p in clean]})
        self.saved_roots = clean
        self._set_roots()
        outside = [p["name"] for p in self._projects() if p.get("folder") and not any(
            Path(p["folder"]) == r or Path(p["folder"]).is_relative_to(r) for r in self.roots)]
        return {"roots": [str(r) for r in self.roots], "root_settings": self._root_settings(), "outside": outside}

    # projects
    def op_project_save(self, req):
        projects = self._projects()
        name = _text(req.get("name"), "name", 120).strip()
        instructions = _text(req.get("instructions"), "instructions")
        if req.get("id"):
            p = self._project(projects, _id(req["id"]))
            if name:
                p["name"] = name
            if "instructions" in req:
                p["instructions"] = instructions
        else:
            if not name:
                raise WorkspaceError(400, "a project needs a name")
            p = {"id": uuid.uuid4().hex[:12], "name": name, "instructions": instructions, "files": [],
                 "folder": None, "mode": "ask"}
            projects.append(p)
        if "folder" in req:                       # the folder its tools work in (inside the shared folders)
            p["folder"] = str(self._inside(req["folder"])) if req["folder"] else None
            if p["folder"] and not Path(p["folder"]).is_dir():
                raise WorkspaceError(400, "not a folder")
        if "allow" in req:                        # command prefixes that run without asking (the page applies them)
            rules = req["allow"] if isinstance(req["allow"], list) else []
            clean = []
            for r in rules:
                r = " ".join(_text(r, "allow rule", 200).split())
                if r and r not in clean:
                    clean.append(r)
            p["allow"] = clean[:50]
        if "compact_at" in req:                   # context tokens at which a long session is summarized (0: never)
            try:
                p["compact_at"] = max(0, min(int(req["compact_at"]), 10_000_000))
            except (TypeError, ValueError):
                raise WorkspaceError(400, "compact_at must be a whole number (0: never)") from None
        if "max_rounds" in req:                   # tool rounds before an answer pauses and offers Continue (0: none)
            try:
                p["max_rounds"] = max(0, min(int(req["max_rounds"]), 100_000))
            except (TypeError, ValueError):
                raise WorkspaceError(400, "max_rounds must be a whole number (0: no limit)") from None
        if "mode" in req:
            if req["mode"] not in MODES:
                raise WorkspaceError(400, f"mode must be one of {', '.join(MODES)}")
            p["mode"] = req["mode"]
        p["time"] = _now()
        self._save("projects.json", {"projects": projects})
        return {"project": p}

    def op_project_delete(self, req):
        pid = _id(req.get("id"))
        projects = self._projects()
        self._project(projects, pid)
        self._save("projects.json", {"projects": [p for p in projects if p["id"] != pid]})
        chats = self._chats()                     # its chats stay, without a project
        for c in chats:
            if c.get("project") == pid:
                c["project"] = None
                self._set_chat_meta(c["id"], project=None)
        self._save("chats.json", {"chats": chats})
        folder = self.dir / "files" / pid
        if folder.is_dir():
            for f in folder.iterdir():
                f.unlink()
            folder.rmdir()
        return {"ok": True}

    def op_project_file_add(self, req):
        projects = self._projects()
        p = self._project(projects, _id(req.get("project"), "project"))
        if req.get("path"):                       # from the file explorer
            got = self._read_file(req["path"])
            name, text, source = got["name"], got["text"], got["path"]
        else:                                     # uploaded from the browser
            name = _text(req.get("name"), "name", 255).strip() or "file.txt"
            text = _text(req.get("text"), "text", MAX_TEXT)
            source = "upload"
        total = sum(f["size"] for f in p["files"]) + len(text)
        if total > MAX_PROJECT_TEXT:
            raise WorkspaceError(413, f"the project's files would exceed {MAX_PROJECT_TEXT // 1048576} MB of text")
        fid = uuid.uuid4().hex[:12]
        folder = self.dir / "files" / p["id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{fid}.txt").write_text(text, encoding="utf-8")
        entry = {"id": fid, "name": name, "size": len(text), "source": source, "added": _now()}
        p["files"].append(entry)
        p["time"] = _now()
        self._save("projects.json", {"projects": projects})
        return {"file": entry, "project": p}

    def op_project_file_delete(self, req):
        projects = self._projects()
        p = self._project(projects, _id(req.get("project"), "project"))
        fid = _id(req.get("id"))
        p["files"] = [f for f in p["files"] if f["id"] != fid]
        f = self.dir / "files" / p["id"] / f"{fid}.txt"
        if f.exists():
            f.unlink()
        p["time"] = _now()
        self._save("projects.json", {"projects": projects})
        return {"project": p}

    def op_project_context(self, req):
        """The instructions and the files' text, for the chat's system message."""
        p = self._project(self._projects(), _id(req.get("project"), "project"))
        files = []
        for f in p["files"]:
            path = self.dir / "files" / p["id"] / f"{f['id']}.txt"
            if path.exists():
                files.append({"name": f["name"], "text": path.read_text(encoding="utf-8")})
        guide = None                              # the repository's own notes for agents, like Claude Code reads them
        if p.get("folder"):
            for n in ("AGENTS.md", "CLAUDE.md"):
                f = Path(p["folder"]) / n
                if f.is_file() and f.stat().st_size <= 64 * 1024:
                    guide = {"name": n, "text": f.read_text(encoding="utf-8", errors="replace")}
                    break
        return {"name": p["name"], "instructions": p.get("instructions", ""), "files": files,
                "folder": p.get("folder"), "mode": p.get("mode", "ask"), "guide": guide,
                "skills": _skills(Path(p["folder"])) if p.get("folder") else []}

    # chats
    def _set_chat_meta(self, cid, **kw):
        path = self.dir / "chats" / f"{cid}.json"
        if path.exists():
            chat = json.loads(path.read_text(encoding="utf-8"))
            chat.update(kw)
            self._save(f"chats/{cid}.json", chat)

    def op_chat_get(self, req):
        cid = _id(req.get("id"))
        chat = self._load(f"chats/{cid}.json", None)
        if chat is None:
            raise WorkspaceError(404, "no such chat")
        return {"chat": chat}

    def op_chat_save(self, req):
        cid = _id(req.get("id"))
        messages = req.get("messages")
        if not isinstance(messages, list):
            raise WorkspaceError(400, "messages must be a list")
        pid = req.get("project") or None
        if pid is not None:
            self._project(self._projects(), _id(pid, "project"))
        chats = self._chats()
        entry = next((c for c in chats if c["id"] == cid), None)
        if entry is None:
            entry = {"id": cid}
            chats.append(entry)
        entry.update(project=pid, title=_text(req.get("title"), "title", 200) or "New chat",
                     named=bool(req.get("named")), time=int(req.get("time") or _now()))
        chats.sort(key=lambda c: c.get("time", 0), reverse=True)
        self._save(f"chats/{cid}.json", {**entry, "messages": messages})
        self._save("chats.json", {"chats": chats})
        return {"chat": entry}

    def op_chat_meta(self, req):
        """Rename or move a chat without sending its messages again."""
        cid = _id(req.get("id"))
        chats = self._chats()
        entry = next((c for c in chats if c["id"] == cid), None)
        if entry is None:
            raise WorkspaceError(404, "no such chat")
        kw = {}
        if "title" in req:
            kw.update(title=_text(req["title"], "title", 200).strip() or entry["title"], named=True)
        if "project" in req:
            pid = req["project"] or None
            if pid is not None:
                self._project(self._projects(), _id(pid, "project"))
            kw["project"] = pid
        entry.update(kw)
        self._set_chat_meta(cid, **kw)
        self._save("chats.json", {"chats": chats})
        return {"chat": entry}

    def op_chat_delete(self, req):
        cid = _id(req.get("id"))
        self._save("chats.json", {"chats": [c for c in self._chats() if c["id"] != cid]})
        path = self.dir / "chats" / f"{cid}.json"
        if path.exists():
            path.unlink()
        shutil.rmtree(self._cp_dir(cid), ignore_errors=True)   # its kept originals go with it
        return {"ok": True}

    def op_search(self, req):
        q = _text(req.get("q"), "q", 200).strip().casefold()
        if not q:
            return {"hits": []}
        hits = []
        for c in self._chats():
            snippet = None
            if q in (c.get("title") or "").casefold():
                snippet = ""
            chat = self._load(f"chats/{c['id']}.json", None) if snippet is None else None
            for m in (chat or {}).get("messages", []):
                text = m.get("text") or ""
                i = text.casefold().find(q)
                if i >= 0:
                    a = max(0, i - 50)
                    snippet = ("…" if a else "") + " ".join(text[a:i + len(q) + 70].split()) + "…"
                    break
            if snippet is not None:
                hits.append({**c, "snippet": snippet})
                if len(hits) >= MAX_HITS:
                    break
        return {"hits": hits}

    # ------------------------------------------------------------------ the file explorer (read only)
    def _inside(self, raw) -> Path:
        if not self.roots:
            raise WorkspaceError(403, "the file explorer is off (no folders are shared: add them under Settings)")
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise WorkspaceError(400, "path must be a non-empty string")
        p = Path(raw).resolve()                   # follows links: a link out of a root is refused below
        if not any(p == r or p.is_relative_to(r) for r in self.roots):
            raise WorkspaceError(403, "outside the shared folders")
        return p

    def op_fs_list(self, req):
        if not req.get("path"):
            return {"path": "", "parent": None,
                    "entries": [{"name": str(r), "path": str(r), "dir": True} for r in self.roots]}
        p = self._inside(req["path"])
        if not p.is_dir():
            raise WorkspaceError(400, "not a folder")
        entries = []
        try:
            items = list(os.scandir(p))
        except PermissionError:
            raise WorkspaceError(403, "this folder cannot be read") from None
        for e in items:
            try:
                is_dir = e.is_dir()
                st = e.stat()
            except OSError:
                continue
            entries.append({"name": e.name, "path": str(Path(e.path)), "dir": is_dir,
                            "size": None if is_dir else st.st_size, "mtime": int(st.st_mtime * 1000)})
        entries.sort(key=lambda x: (not x["dir"], x["name"].casefold()))
        root = any(p == r for r in self.roots)
        return {"path": str(p), "parent": None if root else str(p.parent), "entries": entries[:MAX_ENTRIES],
                "more": max(0, len(entries) - MAX_ENTRIES)}

    def _read_file(self, raw):
        p = self._inside(raw)
        if not p.is_file():
            raise WorkspaceError(400, "not a file")
        if p.stat().st_size > MAX_TEXT:
            raise WorkspaceError(413, f"{p.name} is over {MAX_TEXT // 1024} KB")
        data = p.read_bytes()
        if b"\x00" in data:
            raise WorkspaceError(415, f"{p.name} looks like a binary file")
        return {"path": str(p), "name": p.name, "size": len(data), "text": data.decode("utf-8", errors="replace")}

    def op_fs_read(self, req):
        return self._read_file(req.get("path"))

    def op_fs_image(self, req):
        """A picture in the shared folders, for the Markdown view (a README's images): {type, data (base64)}."""
        p = self._inside(req.get("path"))
        kind = IMAGE_TYPES.get(p.suffix.lower())
        if not kind or not p.is_file():
            raise WorkspaceError(415, f"{p.name} is not a picture")
        if p.stat().st_size > MAX_IMAGE:
            raise WorkspaceError(413, f"{p.name} is over {MAX_IMAGE // 1048576} MB")
        return {"path": str(p), "type": kind, "data": base64.b64encode(p.read_bytes()).decode()}

    # ------------------------------------------------------------------ the coding tools (a project with a folder)
    def _tool_path(self, folder: Path, raw, must_exist=True) -> Path:
        if not isinstance(raw, str) or "\x00" in raw:
            raise WorkspaceError(400, "path must be a string")
        p = (folder / raw).resolve() if raw else folder
        if not (p == folder or p.is_relative_to(folder)):
            raise WorkspaceError(403, f"{raw} is outside the project folder {folder}")
        if must_exist and not p.exists():
            raise WorkspaceError(404, f"{raw} does not exist")
        return p

    def op_tool_run(self, req):
        """Run one of the coding tools in a project's folder: {project, name, arguments} -> {ok, text}."""
        with self.lock:
            p = self._project(self._projects(), _id(req.get("project"), "project"))
        if not p.get("folder"):
            raise WorkspaceError(400, "this project has no folder")
        folder = Path(p["folder"]).resolve()
        if not folder.is_dir():
            raise WorkspaceError(400, f"the project folder {folder} is gone")
        if not any(folder == r or folder.is_relative_to(r) for r in self.roots):
            raise WorkspaceError(403, f"the project folder {folder} is not shared any more (Settings > Shared folders)")
        name, args = req.get("name"), req.get("arguments") or {}
        if not isinstance(args, dict):
            raise WorkspaceError(400, "arguments must be an object")
        if name not in READ_TOOLS | WRITE_TOOLS | RUN_TOOLS:
            raise WorkspaceError(400, f"no tool {name!r}")
        if p.get("mode") == "read" and name not in READ_TOOLS:   # enforced here, whatever the page does
            return {"ok": False, "text": "error: the project is read only: the user can switch its mode to let you change "
                                         "files or run commands; until then, suggest the change in your answer"}
        if name in WRITE_TOOLS and req.get("chat"):  # the original, before this chat's first change to the file
            try:
                self._checkpoint(_id(req["chat"], "chat"), folder, self._tool_path(folder, args.get("path"), must_exist=False))
            except WorkspaceError:
                pass                              # the tool itself reports a bad path
        try:
            fn = getattr(self, "_t_" + name)
            text = fn(folder, args, req.get("job")) if name == "run_command" else fn(folder, args)
            ok = True
        except WorkspaceError as e:
            text, ok = f"error: {e}", False
        if len(text) > MAX_TOOL_OUT:
            half = MAX_TOOL_OUT // 2
            text = text[:half] + f"\n\n[... {len(text) - MAX_TOOL_OUT:,} characters cut ...]\n\n" + text[-half:]
        return {"ok": ok, "text": text}

    def _t_list_dir(self, folder, a):
        d = self._tool_path(folder, a.get("path") or "")
        if not d.is_dir():
            raise WorkspaceError(400, f"{a.get('path')} is not a folder")
        rows = []
        for e in sorted(os.scandir(d), key=lambda e: (not e.is_dir(), e.name.casefold())):
            try:
                rows.append(e.name + "/" if e.is_dir() else f"{e.name}  ({e.stat().st_size:,} B)")
            except OSError:
                continue
        rel = d.relative_to(folder).as_posix() if d != folder else "."
        return f"{rel}:\n" + ("\n".join(rows[:500]) or "(empty)") + (f"\n... {len(rows) - 500} more" if len(rows) > 500 else "")

    def _t_read_file(self, folder, a):
        f = self._tool_path(folder, a.get("path"))
        if not f.is_file():
            raise WorkspaceError(400, f"{a.get('path')} is not a file")
        data = f.read_bytes()
        if b"\x00" in data[:8192]:
            raise WorkspaceError(415, f"{a.get('path')} looks like a binary file")
        lines = data.decode("utf-8", errors="replace").splitlines()
        start = max(1, int(a.get("offset") or 1))
        count = max(1, min(int(a.get("limit") or 2000), 5000))
        part = lines[start - 1:start - 1 + count]
        out = "\n".join(f"{i:>6}\t{l}" for i, l in enumerate(part, start))
        if start - 1 + count < len(lines):
            out += f"\n... {len(lines) - (start - 1 + count):,} more lines (read on with offset={start + count})"
        return out or "(empty file)"

    def _t_search(self, folder, a):
        pattern = a.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise WorkspaceError(400, "pattern is required")
        where = self._tool_path(folder, a.get("path") or "")
        if shutil.which("rg"):
            cmd = ["rg", "--line-number", "--no-heading", "--color", "never", "-S", "--max-columns", "300"]
            if a.get("glob"):
                cmd += ["--glob", str(a["glob"])]
            cmd += ["--", pattern, str(where)]
        else:
            cmd = ["grep", "-rnIE", *[f"--exclude-dir={x}" for x in SKIP_DIRS]]
            if a.get("glob"):
                cmd += [f"--include={a['glob']}"]
            cmd += ["--", pattern, str(where)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, errors="replace")
        out = r.stdout.replace(str(folder) + os.sep, "")
        lines = out.splitlines()
        if not lines:
            return "no matches" if r.returncode == 1 else f"error: {r.stderr.strip()[:500]}"
        return "\n".join(lines[:300]) + (f"\n... {len(lines) - 300} more matches" if len(lines) > 300 else "")

    def _t_find_files(self, folder, a):
        pattern = a.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise WorkspaceError(400, "pattern is required (e.g. **/*.rs)")
        # a walk that never enters .git, node_modules, target, ...: glob("**") would read them all first
        pats = [pattern] + ([pattern[3:]] if pattern.startswith("**/") else [])
        found = []
        for root, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            rel = Path(root).relative_to(folder)
            for name in [d + "/" for d in dirs] + files:
                path = (rel / name.rstrip("/")).as_posix() if rel != Path(".") else name.rstrip("/")
                if any(fnmatch.fnmatchcase(path, x) for x in pats):
                    found.append(path + ("/" if name.endswith("/") else ""))
            if len(found) > 1000:
                break
        found.sort()
        return "\n".join(found[:500]) + (f"\n... more than 500" if len(found) > 500 else "") if found else "no files match"

    def _t_write_file(self, folder, a):
        f = self._tool_path(folder, a.get("path"), must_exist=False)
        content = a.get("content")
        if not isinstance(content, str):
            raise WorkspaceError(400, "content must be a string")
        new = not f.exists()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        return f"{'created' if new else 'overwrote'} {f.relative_to(folder).as_posix()} ({len(content.encode()):,} bytes)"

    def _t_edit_file(self, folder, a):
        f = self._tool_path(folder, a.get("path"))
        old, new = a.get("old_string"), a.get("new_string")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise WorkspaceError(400, "old_string (not empty) and new_string are required")
        text = f.read_text(encoding="utf-8")
        n = text.count(old)
        if n == 0:
            raise WorkspaceError(400, "old_string was not found: read the file again and copy the text exactly")
        if n > 1 and not a.get("replace_all"):
            raise WorkspaceError(400, f"old_string occurs {n} times: add more context to make it unique, or set replace_all")
        f.write_text(text.replace(old, new) if a.get("replace_all") else text.replace(old, new, 1), encoding="utf-8")
        line = text[:text.index(old)].count("\n") + 1
        return f"edited {f.relative_to(folder).as_posix()}: {n if a.get('replace_all') else 1} replacement(s), at line {line}"

    def _job_new(self, jid, folder, command, timeout):
        with self.jobs_lock:
            now = time.time()
            for k in [k for k, j in self.jobs.items() if j.done.is_set() and now - j.started > JOB_TTL]:
                del self.jobs[k]
            if jid in self.jobs:
                raise WorkspaceError(409, f"job {jid} exists")
            running = sum(1 for j in self.jobs.values() if not j.done.is_set())
            if running >= 8:
                raise WorkspaceError(429, "8 commands are running already: stop one (job_stop) first")
            job = self.jobs[jid] = Job(jid, folder, command, timeout)
            return job

    def _job(self, jid):
        job = self.jobs.get(jid) if isinstance(jid, str) else None
        if job is None:
            raise WorkspaceError(404, f"no job {jid!r} (finished jobs are kept for an hour)")
        return job

    def _t_run_command(self, folder, a, job_id=None):
        command = a.get("command")
        if not isinstance(command, str) or not command.strip():
            raise WorkspaceError(400, "command is required")
        timeout = max(1, min(int(a.get("timeout_s") or (3600 if a.get("background") else 120)), 4 * 3600))
        jid = job_id if isinstance(job_id, str) and ID.match(job_id) else "job" + uuid.uuid4().hex[:10]
        job = self._job_new(jid, folder, command, timeout)
        if a.get("background"):
            job.done.wait(2)                      # a command that fails right away says so now
            out, job.read_by_model = job.since(0)
            if job.done.is_set():
                return f"$ {command}\n{out.rstrip()}\n[{job.status}, {time.time() - job.started:.1f} s]"
            return (f"$ {command}\nstarted in the background as job {jid} (it runs up to {timeout} s).\n"
                    f"First output:\n{out.rstrip() or '(none yet)'}\n"
                    f"Read more with job_output, stop it with job_stop.")
        job.done.wait()
        return f"$ {command}\n{job.text().rstrip()}\n[{job.status}, {time.time() - job.started:.1f} s]"

    def _t_job_output(self, folder, a):
        job = self._job(a.get("job_id"))
        job.done.wait(min(max(float(a.get("wait_s") or 0), 0), 60))   # optionally wait for it a little
        out, job.read_by_model = job.since(job.read_by_model)
        state = job.status if job.done.is_set() else f"still running ({time.time() - job.started:.0f} s)"
        return f"job {job.id} ($ {job.command}): {state}\n{out.rstrip() or '(no new output)'}"

    def _t_job_stop(self, folder, a):
        job = self._job(a.get("job_id"))
        job.stop()
        job.done.wait(5)
        out, job.read_by_model = job.since(job.read_by_model)
        return f"job {job.id}: {job.status}\n{out.rstrip()}"

    # the page's live view of a command (not tools of the model)
    def op_job_output(self, req):
        job = self._job(req.get("id"))
        out, offset = job.since(int(req.get("since") or 0))
        return {"text": out, "offset": offset, "done": job.done.is_set(), "status": job.status,
                "seconds": round(time.time() - job.started, 1)}

    def op_job_stop(self, req):
        self._job(req.get("id")).stop()
        return {"ok": True}

    # ------------------------------------------------------------------ what a chat changed, and undo
    # checkpoints/<chat>/index.json: {"folder": ..., "files": {relative path: {"existed": bool, "orig": "<n>.orig"}}}
    def _cp_dir(self, chat):
        return self.dir / "checkpoints" / chat

    def _cp_index(self, chat):
        return self._load(f"checkpoints/{chat}/index.json", {"folder": None, "files": {}})

    def _checkpoint(self, chat, folder, path: Path):
        with self.lock:
            idx = self._cp_index(chat)
            rel = path.relative_to(folder).as_posix()
            if rel in idx["files"]:
                return                            # kept from the first change: undo goes back to before the chat
            idx["folder"] = str(folder)
            entry = {"existed": path.is_file()}
            if entry["existed"]:
                if path.stat().st_size > 8 * 1024 * 1024:
                    return                        # too big to keep a copy of: not undoable
                name = f"{len(idx['files'])}.orig"
                d = self._cp_dir(chat)
                d.mkdir(parents=True, exist_ok=True)
                (d / name).write_bytes(path.read_bytes())
                entry["orig"] = name
            idx["files"][rel] = entry
            self._save(f"checkpoints/{chat}/index.json", idx)

    def _project_folder(self, req):
        p = self._project(self._projects(), _id(req.get("project"), "project"))
        if not p.get("folder"):
            raise WorkspaceError(400, "this project has no folder")
        return Path(p["folder"]).resolve()

    def op_changes_list(self, req):
        chat = _id(req.get("chat"), "chat")
        folder = self._project_folder(req)
        idx = self._cp_index(chat)
        out = []
        for rel, e in sorted(idx["files"].items()):
            path = folder / rel
            before = (self._cp_dir(chat) / e["orig"]).read_bytes() if e.get("orig") else b""
            now = path.read_bytes() if path.is_file() else None
            if now is None:
                status = "deleted" if e["existed"] else "unchanged"
            elif not e["existed"]:
                status = "added"
            else:
                status = "unchanged" if now == before else "modified"
            if status == "unchanged":
                continue
            a = before.decode("utf-8", errors="replace").splitlines(keepends=True)
            b = (now or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
            diff = "".join(difflib.unified_diff(a, b, f"a/{rel}", f"b/{rel}"))
            lines = diff.splitlines(keepends=True)
            if len(lines) > 1500:
                diff = "".join(lines[:1500]) + f"\n... {len(lines) - 1500} more lines\n"
            plus = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
            minus = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
            out.append({"path": rel, "status": status, "added": plus, "removed": minus, "diff": diff})
        return {"files": out}

    def op_changes_undo(self, req):
        """Put files back as they were before this chat's first change (one path, or all)."""
        chat = _id(req.get("chat"), "chat")
        folder = self._project_folder(req)
        idx = self._cp_index(chat)
        which = [req["path"]] if req.get("path") else list(idx["files"])
        undone = []
        for rel in which:
            e = idx["files"].get(rel)
            if e is None:
                continue
            path = self._tool_path(folder, rel, must_exist=False)
            if e.get("orig"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((self._cp_dir(chat) / e["orig"]).read_bytes())
            elif not e["existed"] and path.is_file():
                path.unlink()                     # it did not exist before the chat
            undone.append(rel)
            del idx["files"][rel]
        self._save(f"checkpoints/{chat}/index.json", idx)
        return {"undone": undone}
