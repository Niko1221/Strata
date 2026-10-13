"""Strata manager: management console, chat playground, system monitoring,
inference metrics dashboard and API key management.

Design follows app/manager (the SGLang console): FastAPI + docker.sock for
control / logs / terminal, with a streaming proxy that records metrics.
Adds SQLite time-series storage and an API key store that backs the
nginx auth_request authorization.

Authorization model: nginx is the only external entry point and calls
/api/auth/verify per path. A regular key may only reach the inference API
(/v1/*); an admin key reaches everything. While no user exists yet there is a
one-time bootstrap window that lets the first administrator be created.
"""
import asyncio
import base64
import collections
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import docker
import httpx
from docker.errors import NotFound
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

STRATA_CONTAINER = os.environ.get("STRATA_CONTAINER", "strata")
STRATA_URL = os.environ.get("STRATA_URL", "http://strata:8080")
DB_PATH = os.environ.get("DB_PATH", "/data/manager.db")
SAMPLE_INTERVAL = float(os.environ.get("SAMPLE_INTERVAL", "10"))
KEEP_DAYS = float(os.environ.get("KEEP_DAYS", "7"))
PORT = int(os.environ.get("PORT", "8088"))
LOCAL_TZ = ZoneInfo(os.environ.get("TZ_NAME", "Asia/Shanghai"))

cli = docker.APIClient(base_url="unix:///var/run/docker.sock", version="auto")

app = FastAPI()
db_lock = threading.Lock()


# ---------------------------------------------------------------- SQLite

def db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with db_lock, db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS sys_samples(
            ts INTEGER PRIMARY KEY,
            host_cpu REAL, mem_total_gb REAL, mem_used_gb REAL,
            gpu_util REAL, vram_used_mb REAL, vram_total_mb REAL,
            gpu_temp REAL, gpu_power_w REAL,
            cont_cpu REAL, cont_mem_gb REAL);
        CREATE TABLE IF NOT EXISTS llm_requests(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER, model TEXT, endpoint TEXT,
            prompt_tokens INTEGER, completion_tokens INTEGER,
            ttft_ms REAL, total_ms REAL,
            prefill_tps REAL, decode_tps REAL);
        CREATE INDEX IF NOT EXISTS idx_req_ts ON llm_requests(ts);
        CREATE TABLE IF NOT EXISTS api_keys(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            name TEXT, role TEXT DEFAULT 'user',
            enabled INTEGER DEFAULT 1,
            created_at INTEGER);
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            pw_hash TEXT NOT NULL,
            role TEXT DEFAULT 'admin',
            created_at INTEGER);
        CREATE TABLE IF NOT EXISTS sessions(
            token TEXT PRIMARY KEY,
            user_id INTEGER,
            expires INTEGER);
        """)


# ---------------------------------------------------------------- users & sessions

SESSION_DAYS = 7


def hash_pw(pw: str, salt: str = None) -> str:
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 60000).hex()
    return salt + "$" + h


def check_pw(pw: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return secrets.compare_digest(hash_pw(pw, salt).split("$", 1)[1], stored.split("$", 1)[1])


def find_user(username: str):
    with db_lock, db() as c:
        return c.execute("SELECT id,username,pw_hash,role,created_at FROM users WHERE username=?",
                         (username,)).fetchone()


def user_count() -> int:
    with db_lock, db() as c:
        return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def _is_bootstrap() -> bool:
    """True while no user account exists yet: the first-run setup window."""
    return user_count() == 0


def find_session(token: str):
    with db_lock, db() as c:
        row = c.execute(
            "SELECT u.id,u.username,u.role,s.expires FROM sessions s "
            "JOIN users u ON u.id=s.user_id WHERE s.token=?", (token,)).fetchone()
    if row and row[3] > time.time():
        return row
    return None


def _session_role(request: Request):
    tok = request.cookies.get("session")
    row = find_session(tok) if tok else None
    return row[2] if row else None


def _is_admin(request: Request) -> bool:
    """Administrative access: an admin session, an admin API key, or the
first-run window before any user exists."""
    if _session_role(request) == "admin":
        return True
    if _is_bootstrap():
        return True
    k = _extract_key(request)
    row = find_key(k) if k else None
    return bool(row and row[3] == "admin" and row[4])


@app.post("/api/login")
def api_login(req: Request):
    body = asyncio.run(req.json())
    u = str(body.get("username", "")).strip()
    p = str(body.get("password", ""))
    if not u or not p:
        return JSONResponse({"error": "username and password are required"}, status_code=400)
    if _is_bootstrap():                       # the first account becomes the administrator
        with db_lock, db() as c:
            c.execute("INSERT INTO users(username,pw_hash,role,created_at) VALUES(?,?,?,?)",
                      (u, hash_pw(p), "admin", int(time.time())))
        role = "admin"
        uid = find_user(u)[0]
        log_event("sys", line=f"first administrator account created: {u}")
    else:
        row = find_user(u)
        if not row or not check_pw(p, row[2]):
            return JSONResponse({"error": "incorrect username or password"}, status_code=401)
        uid, role = row[0], row[3]
    token = secrets.token_urlsafe(32)
    with db_lock, db() as c:
        c.execute("INSERT INTO sessions(token,user_id,expires) VALUES(?,?,?)",
                  (token, uid, int(time.time() + SESSION_DAYS * 86400)))
    resp = JSONResponse({"ok": True, "username": u, "role": role,
                         "bootstrap": user_count() == 1})
    resp.set_cookie("session", token, max_age=SESSION_DAYS * 86400,
                    httponly=True, samesite="lax", secure=True, path="/")
    return resp


@app.post("/api/logout")
def api_logout(req: Request):
    tok = req.cookies.get("session")
    if tok:
        with db_lock, db() as c:
            c.execute("DELETE FROM sessions WHERE token=?", (tok,))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("session", path="/")
    return resp


@app.get("/api/me")
def api_me(req: Request):
    tok = req.cookies.get("session")
    row = find_session(tok) if tok else None
    if not row:
        return JSONResponse({"error": "not logged in"}, status_code=401)
    return {"username": row[1], "role": row[2]}


@app.get("/api/setup-status")
def api_setup_status():
    """Public: whether the console is still in first-run setup, so the login
page can offer to create the administrator."""
    return {"need_setup": _is_bootstrap()}


# ---------------------------------------------------------------- API keys

def _mask(k: str) -> str:
    return k[:8] + "…" + k[-4:] if len(k) > 14 else k[:4] + "…"


def key_count() -> int:
    with db_lock, db() as c:
        return c.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0]


def find_key(k: str):
    with db_lock, db() as c:
        return c.execute("SELECT id,key,name,role,enabled,created_at FROM api_keys WHERE key=?",
                         (k,)).fetchone()


def _is_bootstrap_keys() -> bool:
    """Whether the key store is empty. Drives a UI hint only, not authorization."""
    return key_count() == 0


def _extract_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "") or request.query_params.get("api_key", "")


# nginx auth_request callback: session cookie (browser login) or API key (external apps)
ADMIN_PATH_RE = re.compile(r"^/(api|ws|static)(/|$)|^/$")


@app.get("/api/auth/verify")
def auth_verify(request: Request):
    uri = request.headers.get("x-original-uri", "")
    need_admin = ADMIN_PATH_RE.match(uri or "/") is not None
    # 1) browser session (the Cookie is forwarded through nginx auth_request)
    role = _session_role(request)
    if role and (role == "admin" or not need_admin):
        return Response(status_code=204)
    # 2) API key used by external applications
    key = _extract_key(request)
    row = find_key(key) if key else None
    if row and row[4] and (row[3] == "admin" or not need_admin):
        return Response(status_code=204)
    # 3) first-run window (no user account yet)
    if _is_bootstrap() and need_admin:
        return Response(status_code=204)
    return Response(status_code=401)


@app.get("/api/keys")
def keys_list(request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "admin required"}, status_code=401)
    with db_lock, db() as c:
        rows = c.execute("SELECT id,name,role,enabled,created_at FROM api_keys ORDER BY id").fetchall()
    return {"bootstrap_keys": _is_bootstrap_keys(),
            "keys": [{"id": r[0], "name": r[1], "role": r[2], "enabled": bool(r[3]),
                      "created_at": r[4]} for r in rows]}


@app.post("/api/keys")
def keys_create(request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "admin key required"}, status_code=401)
    body = asyncio.run(request.json())
    name = str(body.get("name") or "").strip() or "key"
    role = "admin" if body.get("role") == "admin" else "user"
    k = "sk-" + secrets.token_urlsafe(24)
    with db_lock, db() as c:
        c.execute("INSERT INTO api_keys(key,name,role,enabled,created_at) VALUES(?,?,?,1,?)",
                  (k, name, role, int(time.time())))
    log_event("sys", line=f"api key created: {name} ({role})")
    return {"ok": True, "key": k, "name": name, "role": role}


@app.post("/api/keys/{kid}/toggle")
def keys_toggle(kid: int, request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "admin key required"}, status_code=401)
    with db_lock, db() as c:
        c.execute("UPDATE api_keys SET enabled = 1 - enabled WHERE id=?", (kid,))
    return {"ok": True}


@app.delete("/api/keys/{kid}")
def keys_delete(kid: int, request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "admin key required"}, status_code=401)
    with db_lock, db() as c:
        c.execute("DELETE FROM api_keys WHERE id=?", (kid,))
    log_event("sys", line=f"api key #{kid} deleted")
    return {"ok": True}



def insert_sample(s: dict):
    with db_lock, db() as c:
        c.execute("INSERT OR REPLACE INTO sys_samples VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                  (s["ts"], s.get("host_cpu"), s.get("mem_total_gb"), s.get("mem_used_gb"),
                   s.get("gpu_util"), s.get("vram_used_mb"), s.get("vram_total_mb"),
                   s.get("gpu_temp"), s.get("gpu_power_w"),
                   s.get("cont_cpu"), s.get("cont_mem_gb")))


def insert_request(r: dict):
    with db_lock, db() as c:
        c.execute("INSERT INTO llm_requests(ts,model,endpoint,prompt_tokens,completion_tokens,"
                  "ttft_ms,total_ms,prefill_tps,decode_tps) VALUES(?,?,?,?,?,?,?,?,?)",
                  (r["ts"], r.get("model"), r.get("endpoint"), r.get("prompt_tokens"),
                   r.get("completion_tokens"), r.get("ttft_ms"), r.get("total_ms"),
                   r.get("prefill_tps"), r.get("decode_tps")))


def prune():
    cutoff = int(time.time() - KEEP_DAYS * 86400)
    with db_lock, db() as c:
        c.execute("DELETE FROM sys_samples WHERE ts < ?", (cutoff,))
        c.execute("DELETE FROM llm_requests WHERE ts < ?", (cutoff,))


# ---------------------------------------------------------------- host stats

_proc_last = {"t": None, "idle": 0, "total": 0}


def host_cpu_pct():
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()[1:]
        vals = list(map(int, parts[:8]))
        idle = vals[3] + vals[4]
        total = sum(vals)
        if _proc_last["t"] is not None:
            di, dt = idle - _proc_last["idle"], total - _proc_last["total"]
            pct = (1 - di / dt) * 100 if dt > 0 else 0.0
        else:
            pct = 0.0
        _proc_last.update(t=time.time(), idle=idle, total=total)
        return round(pct, 1)
    except Exception:
        return None


def host_mem():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = float(v.split()[0]) / 1e6  # GB
        return round(info.get("MemTotal", 0), 1), round(info.get("MemTotal", 0) - info.get("MemAvailable", 0), 1)
    except Exception:
        return None, None


# ---------------------------------------------------------------- sampler

def _exec_in_strata(argv):
    eid = cli.exec_create(STRATA_CONTAINER, argv)["Id"]
    return cli.exec_start(eid).decode("utf-8", "replace")


def sample_once():
    s = {"ts": int(time.time())}
    s["host_cpu"] = host_cpu_pct()
    s["mem_total_gb"], s["mem_used_gb"] = host_mem()
    try:
        st = cli.stats(STRATA_CONTAINER, stream=False)
        cd = (st["cpu_stats"]["cpu_usage"]["total_usage"] - st["precpu_stats"]["cpu_usage"]["total_usage"])
        sd = (st["cpu_stats"]["system_cpu_usage"] - st["precpu_stats"]["system_cpu_usage"])
        ncpu = st["cpu_stats"].get("online_cpus") or 8
        s["cont_cpu"] = round(cd / sd * ncpu * 100, 1) if sd > 0 else None
        s["cont_mem_gb"] = round(st["memory_stats"].get("usage", 0) / 1e9, 2)
    except Exception:
        pass
    try:
        raw = _exec_in_strata(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
                               "temperature.gpu,power.draw", "--format=csv,noheader,nounits"])
        p = [x.strip() for x in raw.strip().split(",")]
        if len(p) >= 5:
            s.update(gpu_util=float(p[0]), vram_used_mb=float(p[1]), vram_total_mb=float(p[2]),
                     gpu_temp=float(p[3]), gpu_power_w=float(p[4]))
    except Exception:
        pass
    return s


def sampler():
    while True:
        try:
            st, _ = container_state()
            if st == "running":
                insert_sample(sample_once())
        except Exception:
            pass
        time.sleep(SAMPLE_INTERVAL)


# ---------------------------------------------------------------- logs

access_log = collections.deque(maxlen=2000)
log_subscribers: set = set()
loop_holder: dict = {}


def log_event(kind: str, **data):
    evt = {"ts": datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), "kind": kind, **data}
    access_log.append(evt)
    dead = []
    for ws in list(log_subscribers):
        try:
            asyncio.run_coroutine_threadsafe(ws.send_json(evt), loop_holder["loop"])
        except Exception:
            dead.append(ws)
    for ws in dead:
        log_subscribers.discard(ws)


def _container_log_follower():
    last = None
    while True:
        try:
            ins = cli.inspect_container(STRATA_CONTAINER)
            st = ins["State"]["Status"]
            if st != last:
                log_event("sys", line=f"container state: {last} → {st}")
                last = st
            if st != "running":
                time.sleep(2)
                continue
            for chunk in cli.logs(container=STRATA_CONTAINER, stdout=True, stderr=True,
                                  stream=True, follow=True, tail=0):
                if isinstance(chunk, bytes):
                    line = chunk.decode("utf-8", "replace").rstrip("\n")
                    if line and "/health" not in line:
                        log_event("container", line=line)
        except NotFound:
            if last != "missing":
                log_event("sys", line="container state: → missing")
                last = "missing"
            time.sleep(5)
        except Exception:
            time.sleep(3)


@app.on_event("startup")
def _init():
    init_db()
    loop_holder["loop"] = asyncio.get_running_loop()
    threading.Thread(target=sampler, daemon=True).start()
    threading.Thread(target=_container_log_follower, daemon=True).start()
    threading.Thread(target=_prune_loop, daemon=True).start()


def _prune_loop():
    while True:
        time.sleep(3600)
        try:
            prune()
        except Exception:
            pass


# ---------------------------------------------------------------- status / control

def container_state():
    try:
        ins = cli.inspect_container(STRATA_CONTAINER)
        return ins["State"]["Status"], ins
    except NotFound:
        return "missing", None
    except Exception:
        return "unknown", None


async def strata_health():
    try:
        async with httpx.AsyncClient(timeout=3.0) as ac:
            r = await ac.get(f"{STRATA_URL}/health")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


@app.get("/api/status")
async def api_status():
    st, ins = container_state()
    started = None
    if ins:
        t = (ins.get("State") or {}).get("StartedAt", "")
        try:
            started = t[:19].replace("T", " ")
        except Exception:
            pass
    return {
        "container": STRATA_CONTAINER,
        "container_status": st,
        "started_at": started,
        "health": await strata_health() if st == "running" else None,
    }


@app.post("/api/control/{action}")
def api_control(action: str):
    try:
        if action == "start":
            cli.start(STRATA_CONTAINER)
        elif action == "stop":
            cli.stop(STRATA_CONTAINER, timeout=10)
        elif action == "restart":
            cli.restart(STRATA_CONTAINER, timeout=10)
        else:
            return JSONResponse({"error": "unknown action"}, status_code=400)
    except NotFound:
        return JSONResponse({"error": "container not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    st, _ = container_state()
    return {"ok": True, "status": st}


# ---------------------------------------------------------------- sys history API

def _bucket(rows, bucket_s):
    """rows: list of (ts, val...) -> bucketed averages."""
    out = {}
    for row in rows:
        b = row[0] // bucket_s * bucket_s
        agg = out.setdefault(b, [0] * (len(row) - 1))
        for i, v in enumerate(row[1:]):
            if v is not None:
                agg[i] = (agg[i][0] + v, agg[i][1] + 1) if isinstance(agg[i], tuple) else (v, 1)
    res = []
    for b in sorted(out):
        vals = []
        for acc in out[b]:
            vals.append(round(acc[0] / acc[1], 1) if isinstance(acc, tuple) else None)
        res.append([b] + vals)
    return res


@app.get("/api/sys/latest")
async def api_sys_latest():
    st, _ = container_state()
    with db_lock, db() as c:
        row = c.execute("SELECT * FROM sys_samples ORDER BY ts DESC LIMIT 1").fetchone()
    keys = ["ts", "host_cpu", "mem_total_gb", "mem_used_gb", "gpu_util", "vram_used_mb",
            "vram_total_mb", "gpu_temp", "gpu_power_w", "cont_cpu", "cont_mem_gb"]
    latest = dict(zip(keys, row)) if row else None
    return {"container_status": st, "latest": latest,
            "health": await strata_health() if st == "running" else None}


@app.get("/api/sys/history")
def api_sys_history(minutes: int = 60):
    since = int(time.time()) - min(minutes, 7 * 24 * 60) * 60
    with db_lock, db() as c:
        rows = c.execute("SELECT ts,host_cpu,mem_used_gb,gpu_util,vram_used_mb,gpu_temp,gpu_power_w,"
                         "cont_cpu,cont_mem_gb FROM sys_samples WHERE ts >= ? ORDER BY ts", (since,)).fetchall()
    bucket_s = 60 if minutes > 24 * 60 else (30 if minutes > 60 else 10)
    return {"bucket_s": bucket_s, "series": _bucket(rows, bucket_s)}


# ---------------------------------------------------------------- llm metrics API

def _pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return round(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f), 1)


@app.get("/api/llm/stats")
def api_llm_stats(hours: int = 24):
    since = int(time.time()) - min(hours, 7 * 24) * 3600
    with db_lock, db() as c:
        rows = c.execute("SELECT ts,prompt_tokens,completion_tokens,ttft_ms,total_ms,prefill_tps,decode_tps"
                         " FROM llm_requests WHERE ts >= ?", (since,)).fetchall()
    ttfts = sorted(r[3] for r in rows if r[3] is not None)
    decodes = sorted(r[6] for r in rows if r[6] is not None)
    prefills = sorted(r[5] for r in rows if r[5] is not None)
    agg = {
        "count": len(rows),
        "prompt_tokens": sum(r[1] or 0 for r in rows),
        "completion_tokens": sum(r[2] or 0 for r in rows),
        "ttft_avg": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
        "ttft_p50": _pct(ttfts, 50), "ttft_p95": _pct(ttfts, 95), "ttft_max": round(ttfts[-1], 1) if ttfts else None,
        "decode_avg": _pct(decodes, 50), "prefill_avg": _pct(prefills, 50),
        "total_ms_avg": round(sum(r[4] for r in rows if r[4]) / max(len(rows), 1), 1),
    }
    bucket_s = 3600 if hours > 24 else 300
    series = _bucket([(r[0], r[3], r[6], None) for r in rows], bucket_s)
    # series rows: [ts, ttft_avg, decode_avg, dummy]
    return {"agg": agg, "bucket_s": bucket_s, "series": series}


@app.get("/api/llm/requests")
def api_llm_requests(limit: int = 100, offset: int = 0):
    with db_lock, db() as c:
        rows = c.execute("SELECT * FROM llm_requests ORDER BY id DESC LIMIT ? OFFSET ?",
                         (min(limit, 500), offset)).fetchall()
        total = c.execute("SELECT COUNT(*) FROM llm_requests").fetchone()[0]
    keys = ["id", "ts", "model", "endpoint", "prompt_tokens", "completion_tokens",
            "ttft_ms", "total_ms", "prefill_tps", "decode_tps"]
    return {"total": total, "requests": [dict(zip(keys, r)) for r in rows]}


# ---------------------------------------------------------------- config / rebuild

def strata_config():
    try:
        out = _exec_in_strata(["sh", "-c", "cat /data/config/strata-*.json"])
        return json.loads(out)
    except Exception:
        return None


def container_env():
    try:
        ins = cli.inspect_container(STRATA_CONTAINER)
        env = {}
        for e in (ins.get("Config") or {}).get("Env") or []:
            k, _, v = e.partition("=")
            env[k] = v
        return env
    except Exception:
        return {}


@app.get("/api/config")
def api_config_get():
    return {"config": strata_config(), "env": container_env()}


CONFIG_KEYS = ("api_key", "max_tokens", "fit_max_tokens", "allowed_hosts", "threads", "sampling")


@app.post("/api/config")
def api_config_save(req: Request):
    body = asyncio.run(req.json())
    patch = {k: v for k, v in body.items() if k in CONFIG_KEYS}
    if not patch:
        return JSONResponse({"error": "no valid keys"}, status_code=400)
    script = ("import json,sys,glob\n"
              "p=glob.glob('/data/config/strata-*.json')[0]\n"
              "cfg=json.load(open(p))\n"
              "patch=json.loads(sys.argv[1])\n"
              "for k,v in patch.items():\n"
              "    if v is None or v=='': cfg.pop(k,None)\n"
              "    else: cfg[k]=v\n"
              "json.dump(cfg,open(p,'w'),indent=1)\n"
              "print('config saved:',p)\n")
    try:
        _exec_in_strata(["/opt/strata/.venv/bin/python", "-c", script, json.dumps(patch)])
    except Exception as e:
        return JSONResponse({"error": f"write config failed: {e}"}, status_code=500)
    try:
        cli.restart(STRATA_CONTAINER, timeout=10)
    except Exception as e:
        return JSONResponse({"error": f"saved but restart failed: {e}"}, status_code=500)
    return {"ok": True, "msg": "saved; restarting the container (model reloads in about 1-2 minutes)"}


# A rebuild inherits the current container configuration, merges the new
# environment (including REINSTALL=1) and recreates the container. A background
# thread waits for setup to finish, then recreates it once more without
# REINSTALL so the container boots normally again.
REBUILD_ENV_KEYS = ("FAMILY", "MODEL", "CONTEXT", "VISION", "KV", "GPU", "GPUS",
                    "LAYER_SPLIT", "LOW_RAM", "HOST", "PORT", "API_KEY", "HF_ENDPOINT")


def _merged_env(ins, overrides: dict):
    env = {}
    for e in (ins.get("Config") or {}).get("Env") or []:
        k, _, v = e.partition("=")
        env[k] = v
    for k, v in overrides.items():
        if v is None or v == "":
            env.pop(k, None)
        else:
            env[k] = str(v)
    return [f"{k}={v}" for k, v in env.items()]


def _recreate(ins, env_list):
    name = STRATA_CONTAINER
    image = ins["Config"]["Image"]
    hc = ins.get("HostConfig") or {}
    binds = [b["Source"] + ":" + b["Destination"] + (":ro" if not b.get("RW", True) else "")
             for b in (ins.get("Mounts") or []) if b.get("Type") == "bind"]
    port_bindings = {}
    for cport, binds2 in (hc.get("PortBindings") or {}).items():
        if binds2:
            port_bindings[cport] = [{"HostIp": b.get("HostIp") or "", "HostPort": b.get("HostPort") or ""}
                                    for b in binds2]
    new_hc = cli.create_host_config(
        binds=binds or None,
        port_bindings=port_bindings or None,
        device_requests=hc.get("DeviceRequests"),
        ipc_mode=hc.get("IpcMode") or None,
        ulimits=hc.get("Ulimits") or None,
        restart_policy={"Name": "unless-stopped"},
    )
    nets = [n for n in ((ins.get("NetworkSettings") or {}).get("Networks") or {})
            if n not in ("bridge", "host", "none")]
    cli.remove_container(name, force=True)
    resp = cli.create_container(image=image, name=name, host_config=new_hc,
                                environment=env_list)
    cli.start(name)
    for net in nets:
        try:
            cli.connect_container_to_network(name, net)   # note: docker-py takes the container name first, then the network
        except Exception as e:
            log_event("sys", line=f"rebuild: failed to attach network {net}: {e}")
    return resp


def _wait_healthy(timeout_s):
    import httpx as _hx
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = _hx.get(f"{STRATA_URL}/health", timeout=3.0)
            if r.status_code == 200 and r.json().get("loaded"):
                return True
        except Exception:
            pass
        time.sleep(15)
    return False


def _rebuild_flow(overrides: dict):
    try:
        ins = cli.inspect_container(STRATA_CONTAINER)
        env1 = _merged_env(ins, {**overrides, "REINSTALL": "1"})
        log_event("sys", line=f"rebuild: recreating the container (REINSTALL=1), env: {overrides}")
        _recreate(ins, env1)
        log_event("sys", line="rebuild: container recreated, setup running (fast when the model is already downloaded)…")
        if not _wait_healthy(timeout_s=6 * 3600):
            log_event("sys", line="rebuild: timed out waiting for healthy (6h); REINSTALL=1 is kept, check the logs")
            return
        log_event("sys", line="rebuild: setup finished, recreating once more to drop REINSTALL…")
        ins2 = cli.inspect_container(STRATA_CONTAINER)
        env2 = [e for e in _merged_env(ins2, {}) if not e.startswith("REINSTALL=")]
        _recreate(ins2, env2)
        log_event("sys", line="rebuild: done, container starting with the new configuration")
    except Exception as e:
        log_event("sys", line=f"rebuild failed: {e}")


@app.post("/api/rebuild")
def api_rebuild(req: Request):
    body = asyncio.run(req.json())
    overrides = {k.upper(): v for k, v in (body.get("env") or {}).items()
                 if k.upper() in REBUILD_ENV_KEYS and v is not None}
    if "MODEL" not in overrides:
        return JSONResponse({"error": "MODEL is required"}, status_code=400)
    threading.Thread(target=_rebuild_flow, args=(overrides,), daemon=True).start()
    return {"ok": True, "msg": "rebuild started: watch progress on the Logs tab (long if a model still has to be downloaded)"}


# ---------------------------------------------------------------- websockets

def _ws_admin_ok(ws: WebSocket) -> bool:
    """WebSocket endpoints check the credential themselves: an admin session
cookie (sent automatically by the browser) or an admin API key (?api_key=)."""
    tok = ws.cookies.get("session")
    row = find_session(tok) if tok else None
    if row and row[2] == "admin":
        return True
    if _is_bootstrap():
        return True
    k = ws.headers.get("x-api-key", "") or ws.query_params.get("api_key", "")
    auth = ws.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        k = auth[7:].strip()
    kr = find_key(k) if k else None
    return bool(kr and kr[3] == "admin" and kr[4])


@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    if not _ws_admin_ok(ws):
        await ws.send_json({"kind": "sys", "ts": "-", "line": "unauthorized: an admin key is required (?api_key=)"})
        await ws.close(code=4401)
        return
    log_subscribers.add(ws)
    try:
        for evt in list(access_log)[-200:]:
            await ws.send_json(evt)
        while True:
            await asyncio.sleep(3600)
    except WebSocketDisconnect:
        pass
    finally:
        log_subscribers.discard(ws)


@app.websocket("/ws/console")
async def ws_console(ws: WebSocket):
    await ws.accept()
    if not _ws_admin_ok(ws):
        await ws.send_json({"type": "error", "data": "unauthorized: an admin key is required (?api_key=)"})
        await ws.close(code=4401)
        return
    st, _ = container_state()
    if st != "running":
        await ws.send_json({"type": "error", "data": f"container '{STRATA_CONTAINER}' is {st}. Start it first."})
        await ws.close()
        return
    exec_id = cli.exec_create(STRATA_CONTAINER, ["bash", "-l"], tty=True, stdin=True,
                              user="root", workdir="/")["Id"]
    sock = cli.exec_start(exec_id, socket=True, demux=False)
    raw = sock._sock
    raw.settimeout(0.2)
    loop = asyncio.get_running_loop()

    async def _send(obj):
        try:
            await ws.send_json(obj)
        except Exception:
            pass

    def reader():
        pending = b""
        while True:
            try:
                data = raw.recv(65536)
                if not data:
                    break
                pending += data
                while len(pending) >= 8 and pending[0] in (0, 1, 2) and pending[1:4] == b"\x00\x00\x00":
                    n = int.from_bytes(pending[4:8], "big")
                    if len(pending) < 8 + n:
                        break
                    payload, pending = pending[8:8 + n], pending[8 + n:]
                    asyncio.run_coroutine_threadsafe(
                        _send({"type": "output", "b64": base64.b64encode(payload).decode()}), loop)
                if pending and not (len(pending) >= 8 and pending[0] in (0, 1, 2) and pending[1:4] == b"\x00\x00\x00"):
                    chunk, pending = pending, b""
                    asyncio.run_coroutine_threadsafe(
                        _send({"type": "output", "b64": base64.b64encode(chunk).decode()}), loop)
            except Exception:
                try:
                    if not cli.exec_inspect(exec_id).get("Running", False):
                        break
                except Exception:
                    break
        try:
            code = cli.exec_inspect(exec_id).get("ExitCode", 0) or 0
        except Exception:
            code = 0
        asyncio.run_coroutine_threadsafe(_send({"type": "exit", "code": code}), loop)

    threading.Thread(target=reader, daemon=True).start()
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "input":
                try:
                    raw.sendall(base64.b64decode(msg.get("data", "")))
                except Exception:
                    break
            elif msg.get("type") == "resize":
                try:
                    cli.exec_resize(exec_id, height=int(msg["rows"]), width=int(msg["cols"]))
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        try:
            raw.close()
        except Exception:
            pass


# ---------------------------------------------------------------- strata proxy (records inference metrics)

GEN_PATHS = ("v1/chat/completions", "v1/completions", "v1/messages")


def _extract_obj(text: str, key: str):
    """Find the last "key": { ... } with proper brace matching (handles nesting)."""
    idx = text.rfind(f'"{key}"')
    while idx != -1:
        start = text.find("{", idx)
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        idx = text.rfind(f'"{key}"', 0, idx)
    return None


def _parse_tail(tail: bytes):
    try:
        text = tail.decode("utf-8", "replace")
    except Exception:
        return None, None
    usage = timings = None
    u = _extract_obj(text, "usage")
    if u:
        pt, ct = u.get("prompt_tokens") or u.get("input_tokens"), \
                 u.get("completion_tokens") or u.get("output_tokens")
        if pt is not None and ct is not None:
            usage = (pt, ct)
    t = _extract_obj(text, "timings")
    if t:
        timings = t
    return usage, timings


@app.api_route("/{full_path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(full_path: str, request: Request):
    if full_path == "":
        return FileResponse("/app/static/index.html",
                            headers={"Cache-Control": "no-cache, must-revalidate"})
    if full_path.startswith("static/"):
        fp = os.path.normpath(os.path.join("/app/static", full_path[len("static/"):]))
        if fp.startswith("/app/static") and os.path.isfile(fp):
            return FileResponse(fp, headers={"Cache-Control": "no-cache, must-revalidate"})
        return JSONResponse({"error": "not found"}, status_code=404)
    if full_path in ("static", "favicon.ico") or full_path.startswith(("api/", "ws/")):
        return JSONResponse({"error": "not found"}, status_code=404)

    url = f"{STRATA_URL}/{full_path}"
    if request.url.query:
        url += "?" + request.url.query
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
    body = await request.body()

    log_event("access", method=request.method, path="/" + full_path, detail="request", status="-")
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    try:
        req = client.build_request(request.method, url, headers=headers, content=body or None)
        resp = await client.send(req, stream=True)
    except Exception as e:
        await client.aclose()
        log_event("access", method=request.method, path="/" + full_path,
                  detail=f"upstream error: {e}", status="ERR")
        return JSONResponse({"error": f"strata unreachable: {e}"}, status_code=502)

    status_code = resp.status_code
    ctype = resp.headers.get("content-type", "")
    is_gen = full_path in GEN_PATHS and request.method == "POST"
    t0 = time.time()
    t_first = None

    async def stream_and_close():
        nonlocal t_first
        nbytes = 0
        tail = b""
        try:
            async for chunk in resp.aiter_raw():
                if t_first is None:
                    t_first = time.time()
                nbytes += len(chunk)
                tail = (tail + chunk)[-8192:]
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()
            log_event("access", method=request.method, path="/" + full_path,
                      detail=f"{nbytes} bytes", status=status_code)
            if is_gen and status_code == 200:
                record_inference(t0, t_first, full_path, tail)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        stream_and_close(), status_code=status_code,
        media_type=ctype or "application/octet-stream",
        headers={k: v for k, v in resp.headers.items()
                 if k.lower() in ("content-type", "cache-control")})


def record_inference(t0, t_first, endpoint, tail):
    t_end = time.time()
    usage, timings = _parse_tail(tail)
    if not usage or usage[0] is None or usage[1] is None:
        return
    pt, ct = int(usage[0]), int(usage[1])
    ttft = (t_first - t0) * 1000 if t_first else None
    total_ms = (t_end - t0) * 1000
    prefill_tps = decode_tps = None
    if timings:
        try:
            pn, pm = timings.get("prompt_n"), timings.get("prompt_ms")
            if pn and pm and pn >= 32:
                prefill_tps = round(pn / pm * 1000, 1)
            dn, dm = timings.get("predicted_n"), timings.get("predicted_ms")
            if dn and dm and dn >= 8:
                decode_tps = round(dn / dm * 1000, 1)
        except Exception:
            pass
    if decode_tps is None and ttft and ct and total_ms > ttft:
        decode_tps = round(ct / (total_ms - ttft) * 1000, 1)
    rec = {"ts": int(t0), "model": None, "endpoint": endpoint,
           "prompt_tokens": pt, "completion_tokens": ct,
           "ttft_ms": round(ttft, 1) if ttft else None,
           "total_ms": round(total_ms, 1),
           "prefill_tps": prefill_tps, "decode_tps": decode_tps}
    try:
        insert_request(rec)
    except Exception:
        pass
    log_event("metric", line=f"{endpoint}: ttft={rec['ttft_ms']}ms decode={decode_tps}tok/s "
                             f"({pt}→{ct} tok)")


@app.get("/")
def index():
    return FileResponse("/app/static/index.html")


app.mount("/static", StaticFiles(directory="/app/static"), name="static")
