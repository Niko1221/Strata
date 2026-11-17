#!/usr/bin/env python3
"""gateway2.py — session gateway (transparent proxy + snapshot orchestration)

Position: clients -> this gateway (default 127.0.0.1:8191) -> strata --serve.
Responsibilities:
  1. Transparent forwarding of every request (SSE streaming included).
  2. Session identity is a family+chain dual key (aligned with the vLLM
     chained-block-hash convention):
       family = sha1(tools+system) -> borrowing group (the first user message
                is excluded so client-side rewrites cannot churn the family)
       chain  = chained hash over the full message list -> snapshot identity
     Same-chain extension is a message-level prefix match; a fork becomes a
     new chain/new archive and coexists inside the family.
  3. After an engine restart sessions are restored from snapshots, so
     prefill only pays for the delta.
  4. Idle refresh: dirty sessions are re-snapshoned while the engine idles.
Design rules: snapshot-based / serialized commit points / atomic publish /
fingerprint binding / fail-closed / disk cold tier / prefer existing standards.
"""
import hashlib
import json
import os
import re
import shutil
import threading
import signal as _sig, tracemalloc as _tm
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- Deployment endpoints (override via environment) -----------------------
ENGINE = (os.environ.get("STRATA_GW_ENGINE_HOST", "127.0.0.1"),
          int(os.environ.get("STRATA_GW_ENGINE_PORT", "8080")))
LISTEN = (os.environ.get("STRATA_GW_BIND", "127.0.0.1"),
          int(os.environ.get("STRATA_GW_PORT", "8191")))

# --- Storage layout (relative defaults; override per deployment) -----------
_VAR_DIR = os.environ.get("STRATA_GW_VAR_DIR", "./var")
SNAP_DIR = os.environ.get("STRATA_GW_SNAP_DIR", os.path.join(_VAR_DIR, "snapshots"))
ENGINE_SNAP_DIR = os.environ.get("STRATA_ENGINE_SNAP_DIR",
                                 os.path.join(_VAR_DIR, "engine-slots"))
HOT_DIR = os.environ.get("STRATA_GW_HOT_DIR", os.path.join(_VAR_DIR, "hot"))
SNAPSHOT_INTERVAL_S = 300                        # tmpfs -> SSD snapshot cadence
_LOG_DIR = os.environ.get("STRATA_GW_LOG_DIR", os.path.join(_VAR_DIR, "logs"))
DECISIONS_LOG = os.path.join(_LOG_DIR, "decisions.jsonl")   # per-switch tiering decisions
REQUESTS_LOG = os.path.join(_LOG_DIR, "requests.jsonl")     # per-request latency timeline
SAVEINC_LOG = os.path.join(_LOG_DIR, "saveinc-bytes.jsonl")
SAVE_COMMIT_LOG = os.path.join(_LOG_DIR, "save-commits.jsonl")
LIFECYCLE_LOG = os.path.join(_LOG_DIR, "l2-lifecycle.jsonl")
RESTORE_GUARD_LOG = os.path.join(_LOG_DIR, "restore-guards.jsonl")
MEMORY_EVENTS_LOG = os.path.join(_LOG_DIR, "memory-events.jsonl")
ENGINE_CGROUP_LOG = os.path.join(_LOG_DIR, "engine-cgroup.jsonl")

# --- Optional behaviours (defaults are the conservative posture) ------------
# Rewrite \uXXXX escapes inside tool_calls args to raw UTF-8. Off by default:
# a proxy should not mutate response bodies unless the deployment asks.
TOOLJSON_UNESCAPE = os.environ.get("STRATA_GW_TOOLJSON_UNESCAPE", "0") == "1"
# FIFO passthrough: disable SJF/live reordering and admit in arrival order
# (L2 orchestration still runs; only the queue policy changes).
SCHED_FIFO = os.environ.get("STRATA_GW_SCHED", "sjf") == "off"
# Startup fingerprint manifest (md5 of gateway/server sources + engine argv);
# meant for self-hosted daily-report joins. Off by default.
STARTUP_MANIFEST = os.environ.get("STRATA_GW_STARTUP_MANIFEST", "0") == "1"
# Managed-role mode: register this gateway with the Control process group
# (strata_control/procs.py) so its lifecycle is tracked and leftovers are
# reclaimed by the next Control start. Requires running from the repo tree.
# Unset (default): standalone mode, no registration.
GW_ROLE = os.environ.get("STRATA_GW_ROLE", "") == "1"
GW_ROLE_TOKEN = None

def _role_register() -> None:
    """Best-effort registration with the Control process group; never fatal."""
    global GW_ROLE_TOKEN
    if not GW_ROLE:
        return
    try:
        from strata_control import procs
        GW_ROLE_TOKEN = procs.write_record(os.getpid(), "gateway", cmd=list(__import__("sys").argv))
        if GW_ROLE_TOKEN is None:
            print("[gw] role: registration declined (no token); running unmanaged", flush=True)
    except Exception as e:  # ImportError outside the repo tree, or any I/O hiccup
        print(f"[gw] role: unavailable ({e}); running unmanaged", flush=True)

def _role_unregister() -> None:
    if GW_ROLE_TOKEN is None:
        return
    try:
        from strata_control import procs
        procs.drop_record(os.getpid(), GW_ROLE_TOKEN)
    except Exception:
        pass
# Engine conversation_file_v2 reader hard limit. A larger file must never
# reach restorepp/restorep/full-fallback: an oversized snapshot can
# desynchronize the protocol and make the server exit(-9).
ENGINE_RESTORE_LIMIT_BYTES = int(os.environ.get(
    "STRATA_ENGINE_RESTORE_LIMIT_BYTES", "15804775000"))
STAGE_LEDGER_KEYS = (
    "embed_steps_ms", "hc_read_ms", "gdn_ms", "qsa_projection_ms", "qsa_indexer_ms", "qsa_select_ms",
    "qsa_attention_ms", "router_shared_ms", "host_grouping_ms", "gather_ms", "wait_copy_ms", "dequant_ms",
    "moe_gate_up_ms", "moe_down_ms", "combine_ms", "ple_gpu_ms", "kv_stage_ms", "gdn_conv_gates_ms",
    "gdn_recurrence_ms", "gdn_output_projection_ms", "hc_write_ms", "hc_down_silu_ms", "hc_up_ms",
    "hc_inject_ms", "hc_mix_ms", "qsa_indexer_gemms_ms", "qsa_prepare_ms",
    "prefill_gpu_timeline_ms", "prefill_wall_ms", "prefill_host_staging_ms", "prefill_host_setup_ms",
    "prefill_host_sync_ms", "prefill_host_chunk_ms", "prefill_host_ple_ms",
    "prefill_ledger_runs", "stage_ledger_version",
)
SAVE_RESERVE_MIB = int(os.environ.get("STRATA_SESSION_SAVE_RESERVE_MIB", "4096"))
SMALL_BORROW_MAX_TOKENS = int(os.environ.get("STRATA_L2_SMALL_BORROW_MAX_TOKENS", "98304"))
SMALL_BORROW_MIN_RATIO = float(os.environ.get("STRATA_L2_SMALL_BORROW_MIN_RATIO", "2.0"))
SMALL_BORROW_MIN_ABS_DELTA = int(os.environ.get("STRATA_L2_SMALL_BORROW_MIN_ABS_DELTA", "65536"))
SMALL_BORROW_MIN_REUSE = float(os.environ.get("STRATA_L2_SMALL_BORROW_MIN_REUSE", "0.5"))
BORROW_HOTSPOT_HITS = int(os.environ.get("STRATA_L2_BORROW_HOTSPOT_HITS", "3"))
COMMIT_WAIT_S = float(os.environ.get("STRATA_L2_COMMIT_WAIT_S", "60"))
PENDING_MAX_S = float(os.environ.get("STRATA_L2_PENDING_MAX_S", "300"))
BOOTSTRAP_STALE_S = 600  # v2.0: 引导存档条件 — 无档 或 档龄超过此值 (秒)
KV_POOL_BYTES_PER_TOKEN = 50600      # 池记账口径 (含 slab/检查点), 文件口径≈6.5KB/tok 差8倍

def _pool_budget_mib() -> int:
    """Same source as the engine config; the save reserve is not pool capacity."""
    env = os.environ.get("STRATA_POOL_BUDGET_MIB")
    if env:
        return int(env)
    else:
        cfg = os.environ.get("STRATA_ENGINE_CONFIG")
        try:
            with open(cfg) as f:
                args = (json.load(f) or {}).get("user_args") or []
            i = args.index("--conversation-cache-mib")
            configured = int(args[i + 1])
        except (OSError, ValueError, IndexError, TypeError, json.JSONDecodeError):
            configured = 24576
    return max(0, configured - max(0, SAVE_RESERVE_MIB))

POOL_BUDGET_BYTES = _pool_budget_mib() * 1024**2
GC_DRY_RUN = os.environ.get("STRATA_L2_GC_DRY_RUN", "1") == "1"
GC_PLAN_LOG = os.path.join(_LOG_DIR, "l2-gc-plan.jsonl")
WORKSET_WINDOW_S = 600               # 近 10 分钟活跃会话 = 工作集
WORKSET_PRESSURE = 0.8               # 工作集+本人 ≥ 池帽 80% → 驱逐几乎必然 → 溢出恢复
REFRESH_S = 20          # 同一会话最快刷快照间隔 (v2: 60→20 — saveinc 瘦身后高频小存, 流量做平)
IDLE_GRACE_S = 10       # 空闲宽限 (v2: 30→10 — 连发间隙稍歇即存, 丢失窗口缩小; saveinc 400ms 级可承受撞车)
IDLE_POLL_S = 20        # 空闲刷新轮询
HEALTH_POLL_S = 5

# ===== v1.6 分层代价模型 (本机实测权值, 2026-10-27) =====
# A live:0 | B 池:0.009ms/tok+50ms | C L2:0.020ms/tok+300ms | D 冷读:0.46ms/tok
# 交叉点: C>D 需复用>~700tok; B>D 需复用>12%N; A 永远最优
# 2026-11-12 标定: 冷 prefill 实测 2224-2650 tok/s (0.38-0.45ms/tok), 0.46 略保守保留;
# decode 实测 gen_tps p50=77.3 => 12.9ms/tok (原 20ms 低估长生成 1.5x)
PREFILL_MS_PER_TOK = 0.46
RAM_MS_PER_TOK = 0.009
L2_MS_PER_TOK = 0.020
L2_FIXED_MS = 300
DECODE_MS_PER_TOKEN = float(os.environ.get("STRATA_SCHED_DECODE_MS_PER_TOKEN", "13"))
SAVE_MARGIN = 0.20        # restorep 只在省 ≥20% 时触发

def layer_costs(n, live_lcp, pool_best, l2_est, l2_read=None):
    """A/B/C/D(ms)：l2_est=真实可复用前缀；l2_read=实际读取量，默认整档模式下可大于复用量。"""
    a = (n - live_lcp) * PREFILL_MS_PER_TOK
    b = 50 + pool_best * RAM_MS_PER_TOK + (n - pool_best) * PREFILL_MS_PER_TOK
    read = l2_est if l2_read is None else l2_read
    c = L2_FIXED_MS + read * L2_MS_PER_TOK + (n - l2_est) * PREFILL_MS_PER_TOK
    d = n * PREFILL_MS_PER_TOK
    return {"A": a, "B": b, "C": c, "D": d}

def engine_probe(body: dict, rid: str = "", prefix_messages: list = None):
    """Cache probe plus optional exact, full-IDs-verified message-prefix token watermarks."""
    pb = {"messages": body.get("messages") or []}
    if prefix_messages:
        pb["prefix_messages"] = prefix_messages[:8]
    if body.get("tools") is not None: pb["tools"] = body["tools"]
    if isinstance(body.get("chat_template_kwargs"), dict):
        pb["chat_template_kwargs"] = body["chat_template_kwargs"]
    t0 = time.time()
    st, js, raw = engine_req("POST", "/v1/cache/probe", pb, timeout=3, rid=rid)   # v2.2.8: 60→3s — 引擎忙时探针排队会挂满 state_lock
    ms = round((time.time() - t0) * 1000)
    if st == 200 and js:
        return (js.get("reusable_tokens", 0), js.get("live_lcp", 0), js.get("prompt_tokens", 0), ms,
                js.get("prefix_tokens") or {}, js.get("parked_best"), js.get("parked_serial"))
    print(f"[gw] 探针失败 {ms}ms HTTP{st}: {str(raw or js)[:120]}", flush=True)
    return None, None, None, ms, {}, None, None

def cache_stats():
    st, js, _ = engine_req("GET", "/v1/cache/stats", timeout=5)
    return js if st == 200 and isinstance(js, dict) else None

def cache_list():
    st, js, _ = engine_req("GET", "/v1/cache/list", timeout=5)
    return (js.get("entries") or []) if st == 200 and isinstance(js, dict) else []

def cache_has_source(snap: str, min_tokens: int = 0):
    """Return a PARKED serial only when the entry covers the requested waterline.

    A source filename alone is not enough: the same L2 file can be preloaded
    first as a shallow branch prefix (e.g. a small fork) and later as the full
    parent.  Reusing the shallow entry for the parent silently drops its suffix.
    """
    for e in cache_list():
        source = str(e.get("source") or "")
        tokens = int(e.get("tokens") or 0)
        if os.path.basename(source) == snap and tokens >= int(min_tokens or 0):
            return e.get("serial")
    return None

def cache_pin(serial, pinned=True):
    if serial is None:
        return False
    st, js, _ = engine_req("POST", "/v1/cache/pin" if pinned else "/v1/cache/unpin",
                           {"serial": int(serial)}, timeout=5)
    return st == 200

def cache_preload(snap: str, body: dict, rid: str = "", pin: bool = True):
    rb = {"filename": snap, "messages": (body or {}).get("messages") or [], "pin": bool(pin)}
    if (body or {}).get("tools") is not None:
        rb["tools"] = body["tools"]
    if isinstance((body or {}).get("chat_template_kwargs"), dict):
        rb["chat_template_kwargs"] = body["chat_template_kwargs"]
    st, js, raw = engine_req("POST", "/v1/cache/preload", rb, timeout=180, rid=rid)
    if st == 200 and isinstance(js, dict):
        return True, js, st
    print(f"[gw] parked preload miss {st} {str(raw or js)[:180]}", flush=True)
    return False, js if isinstance(js, dict) else None, st

def release_preload_pin(decision: dict | None) -> None:
    if not decision:
        return
    serial = decision.get("preload_serial")
    if serial is not None:
        try:
            cache_pin(serial, False)
        except Exception:
            pass

last_seen = {}          # fp → 最近一次请求时刻 (工作集/溢出门控用)
_borrow_hotspots = {}   # family -> {common, tokens, hits, last_ts} (进程内，重启后重新学习)
_borrow_hotspots_lock = threading.Lock()

def append_event_log(path: str, obj: dict) -> None:
    """Best-effort structured event sink; never block inference on logging."""
    try:
        with open(path, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        pass


state_lock = threading.RLock()   # save finalization may advance inline while the decision thread owns it
live_fp = None          # 引擎当前持有的会话族 (头哈希)
live_chain = None       # v2.1: 当前链哈希 (全消息链式哈希, 快照身份)
live_hashes = []        # v2.1: 当前会话消息哈希列 (前缀判定)
live_dirty = False      # 自上次 save 后有新轮次
last_save_ts = 0.0
last_done_ts = 0.0       # 最后一轮完成时刻 (空闲宽限用)
stats = {"saves": 0, "restores": 0, "switches": 0, "requests": 0, "restore_fail": 0}
_saveinc_ok = None      # v2.0: None=未探测, True/False=引擎栈是否支持 action=saveinc
_pending_saves = {}     # (family, target) -> pending commit job
_pending_saves_lock = threading.Lock()
_save_start_lock = threading.Lock()   # covers target check -> engine enqueue -> pending registration
save_wakeup = threading.Event()

def engine_req(method: str, path: str, body: dict | None = None, timeout: float = 60, rid: str = ""):
    """对引擎的短请求; 返回 (status, json|None, raw|None)"""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    if rid:
        headers["X-Strata-Request-Id"] = rid
    req = urllib.request.Request(f"http://{ENGINE[0]}:{ENGINE[1]}{path}", method=method,
                                 data=data, headers=headers)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        raw = r.read()
        try:
            return r.status, json.loads(raw), None
        except Exception:
            return r.status, None, raw
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw), None
        except Exception:
            return e.code, None, raw
    except Exception:
        return 0, None, None

def fingerprint(body: dict) -> str | None:
    """会话指纹: tools + system (v2.2.5: 剔除首条 user)
    实证 (2026-10-30): 客户端会改写首条 user (编辑历史/注入变动) → 同会话 family 弹跳
    (bf75↔f2c1, system 哈希全程相同) → live-continuation 断裂 → 400K 全量 prefill 220s 惨案。
    族内区分由 chain 链式哈希承担 (精确到内容), 首条 user 不再参与路由身份;
    同 system+tools 的多会话共存同族, 链档各存各的 (FAM_KEEP LRU), 活跃链不会被踢。"""
    try:
        msgs = body.get("messages") or []
        head = {"tools": body.get("tools") or []}
        for m in msgs:
            role = m.get("role")
            if role in ("system", "developer"):
                head.setdefault("sys", []).append(str(m.get("content"))[:2000])
            elif role == "user":
                break   # v2.2.5: 首条 user 即止, 不纳入指纹
        return hashlib.sha1(json.dumps(head, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    except Exception:
        return None

def main_path(fp: str) -> str:
    return os.path.join(ENGINE_SNAP_DIR, fp)

def tok_path(fp: str) -> str:
    return os.path.join(SNAP_DIR, fp) + ".tok"

def snap_path(fp: str) -> str:
    """Backward-compatible main-file view (never metadata)."""
    return main_path(fp)

def archive_is_v3(path: str) -> bool:
    return os.path.isdir(path) and not os.path.islink(path) and \
           os.path.isfile(os.path.join(path, "manifest"))

def archive_exists(path: str) -> bool:
    return os.path.isfile(path) or archive_is_v3(path)

def archive_bytes(path: str) -> int:
    """Physical bytes, including superseded/orphaned files pending cleanup."""
    try:
        if not os.path.isdir(path):
            return os.path.getsize(path)
        total = 0
        for root, dirs, files in os.walk(path, followlinks=False):
            dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
            for name in files:
                p = os.path.join(root, name)
                if not os.path.islink(p):
                    try:
                        total += os.path.getsize(p)
                    except OSError:
                        pass
        return total
    except OSError:
        return 0

def archive_time(path: str, atime: bool = False) -> float:
    """Latest archive member timestamp; V3 writes do not touch the root directory."""
    try:
        if not os.path.isdir(path):
            return os.path.getatime(path) if atime else os.path.getmtime(path)
        value = os.path.getatime(path) if atime else os.path.getmtime(path)
        for root, dirs, files in os.walk(path, followlinks=False):
            for name in files:
                p = os.path.join(root, name)
                try:
                    value = max(value, os.path.getatime(p) if atime else os.path.getmtime(p))
                except OSError:
                    pass
        return value
    except OSError:
        return 0

def archive_unlink(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)

# ===== v2.1 链式身份 (对齐 vLLM 链式块哈希: 身份=内容本身, 分叉天然不撞) =====
def msg_hash(m: dict) -> str:
    """单条消息哈希 (role+content+tool_calls 全量, 序列化稳定)"""
    return hashlib.sha1(json.dumps(m, sort_keys=True, ensure_ascii=False)
                        .encode()).hexdigest()[:12]

def chain_hashes(messages) -> list:
    return [msg_hash(m) for m in (messages or [])]

def chain_id(hashes: list) -> str:
    """链哈希 = 全消息哈希的链式合成 — 同链每轮追加都会变, 身份跟内容走"""
    return hashlib.sha1(("".join(hashes)).encode()).hexdigest()[:16]

FAM_KEEP = 3          # 每族保留最近 N 条链档 (LRU)
L2_BUDGET_BYTES = int(os.environ.get("STRATA_L2_BUDGET_GB", "200")) * 1024**3
                       # v2.1.3: L2 全局字节预算 — 超限按全局 LRU 驱逐 (vLLM 前缀池模型)
INDEX_FILE = os.path.join(SNAP_DIR, "index.json")
fam_idx_lock = threading.RLock()   # [giphoo] issue#4-C: RLock — do_save→fam_gc→l2_reclaim 链内可重入
fam_idx = {}          # family -> {chain_id: {"file": 名, "ts": 秒, "tok": n, "hashes": [...]}}

def fam_idx_load():
    """启动装载: index.json + 目录扫描 (旧头哈希档天然自成一族)"""
    global fam_idx
    try:
        fam_idx = json.load(open(INDEX_FILE))
    except Exception:
        fam_idx = {}
    try:
        for f in os.listdir(SNAP_DIR):
            if "." in f or len(f) != 16 or not all(c in "0123456789abcdef" for c in f):
                continue
            fam = f   # 旧 scheme: 文件名即头哈希 = 族
            fam_idx.setdefault(fam, {})
            if f not in fam_idx[fam]:
                try:
                    tok = int(open(tok_path(f)).read() or 0)
                except OSError:
                    tok = 0
                fam_idx[fam][f] = {"file": f, "ts": os.path.getmtime(snap_path(f)), "tok": tok,
                                   "hashes": None}   # 旧档无消息哈希: 不做祖先匹配
    except OSError:
        pass

def fam_idx_save():
    try:
        with open(INDEX_FILE + ".tmp", "w") as f:
            json.dump(fam_idx, f, ensure_ascii=False)
        os.replace(INDEX_FILE + ".tmp", INDEX_FILE)
    except OSError:
        pass

def normalize_inplace_aliases() -> int:
    """Any physical file referenced by multiple chains can have grown in place."""
    changed = 0
    with fam_idx_lock:
        for fam in fam_idx.values():
            counts = {}
            for entry in fam.values():
                f = entry.get("file")
                if f:
                    counts[f] = counts.get(f, 0) + 1
            for entry in fam.values():
                f = entry.get("file")
                if f and counts.get(f, 0) > 1 and not entry.get("inplace"):
                    entry["inplace"] = True
                    changed += 1
        if changed:
            fam_idx_save()
    return changed

def chain_prefix_length(left: list, right: list) -> int:
    common = 0
    for a, b in zip(left or [], right or []):
        if a != b:
            break
        common += 1
    return common

def _ancestor_tok_watermark(entry: dict) -> int | float:
    """Unknown watermarks are least attractive when an equal-depth tie is cost-aware."""
    try:
        tok = int(entry.get("tok") or 0)
    except (TypeError, ValueError):
        tok = 0
    return tok if tok > 0 else float("inf")

def _ancestor_replaces(candidate: dict, common: int, best: dict | None, best_len: int) -> bool:
    if best is None or common > best_len:
        return True
    if common != best_len:
        return False
    # Equal lineage depth means both archives cover the same reusable messages; their private
    # suffixes are equally useless to this fork.  The smaller sufficient archive is cheaper.
    cand_tok, best_tok = _ancestor_tok_watermark(candidate), _ancestor_tok_watermark(best)
    if cand_tok != best_tok:
        return cand_tok < best_tok
    return float(candidate.get("ts") or 0) > float(best.get("ts") or 0)

def ancestor_snap(family: str, hashes: list):
    """v2.1.6: 本分支的上一档 — 按【公共消息哈希前缀长度】评分 (entry ⊂ incoming 的
    特例); 分叉后各自的轮次不再互为前缀, 但公共祖先仍是最该用的档"""
    fam = fam_idx.get(family) or {}
    best, best_len = None, 0
    for e in fam.values():
        eh = e.get("hashes")
        if not eh:
            continue
        common = chain_prefix_length(eh, hashes)
        if _ancestor_replaces(e, common, best, best_len):
            best, best_len = e, common
    return best if best_len > 0 else None

def ancestor_snap_global(hashes: list):
    """v2.2.5: 跨族血缘兜底 — 族键算法变更 (剔首user) 后旧族档仍可按消息前缀血缘命中,
    避免迁移期全量 prefill; 安全性: common>0 要求 hashes[0](system)相同, 异 system 天然不互借"""
    best, best_len = None, 0
    for fam in fam_idx.values():
        for e in fam.values():
            eh = e.get("hashes")
            if not eh:
                continue
            common = chain_prefix_length(eh, hashes)
            if _ancestor_replaces(e, common, best, best_len):
                best, best_len = e, common
    return best if best_len > 0 else None

def resolve_snap(family: str, chain: str = None, hashes: list = None):
    """快照解析 (v2.1.6 血缘优先): 精确链 → 消息哈希祖先(本分支上次存档) → 族内最深档。
    tok-max 曾把上一轮的旧档(答案已变, 首答即分歧)顶在本轮自档之上 → 全冷读"""
    fam = fam_idx.get(family) or {}
    if chain and chain in fam:
        return fam[chain]["file"]
    if hashes:
        anc = ancestor_snap(family, hashes)
        if not anc:
            anc = ancestor_snap_global(hashes)   # v2.2.5: 跨族兜底 (族键迁移期)
        if anc:
            return anc["file"]
    if not fam:
        return None
    best = max(fam.values(), key=lambda e: (e.get("tok", 0), e.get("ts", 0)))
    return best["file"]

def snapshot_prefix_plan(family: str, snap: str, hashes: list):
    """Return the selected entry, common message depth and a previously verified token watermark."""
    with fam_idx_lock:
        entries = [e for fam in fam_idx.values() for e in fam.values()]
        entry = next((e for e in entries if e.get("file") == snap and (not family or e in (fam_idx.get(family) or {}).values())), None)
        if entry is None:
            entry = next((e for e in entries if e.get("file") == snap), None)
        if not entry:
            return None, 0, None
        eh = entry.get("hashes") or []
        common = chain_prefix_length(eh, hashes)
        if common == len(eh) and common > 0:
            return entry, common, entry.get("tok")
        cached = (entry.get("prefix_tokens") or {}).get(str(common))
        if cached is not None:
            return entry, common, int(cached)
        target = (hashes or [])[:common]
        if target:
            exact = next((e for e in entries if e.get("hashes") == target and e.get("tok") is not None), None)
            if exact:
                return entry, common, int(exact["tok"])
        return entry, common, None

def pool_workset_metrics(now: float | None = None) -> dict:
    """Read-only working-set estimate; never feeds GC or eviction decisions."""
    now = time.time() if now is None else now
    families, tokens = 0, 0
    for fp, seen in last_seen.items():
        if now - seen >= WORKSET_WINDOW_S:
            continue
        snap = resolve_snap(fp)
        if not snap:
            continue
        try:
            with open(os.path.join(SNAP_DIR, snap) + ".tok") as tf:
                tok = int(tf.read() or 0)
        except (OSError, ValueError):
            continue
        families += 1
        tokens += tok
    est_bytes = tokens * KV_POOL_BYTES_PER_TOKEN
    return {"families": families, "tokens": tokens, "est_bytes": est_bytes,
            "budget_bytes": POOL_BUDGET_BYTES,
            "pressure": round(est_bytes / POOL_BUDGET_BYTES, 3) if POOL_BUDGET_BYTES else None}

def engine_main_metrics() -> dict:
    """Actual engine-root usage for observability only; existing GC paths are unchanged."""
    total, files, temp_bytes = 0, 0, 0
    try:
        for name in os.listdir(ENGINE_SNAP_DIR):
            path = os.path.join(ENGINE_SNAP_DIR, name)
            if not archive_exists(path):
                continue
            size = archive_bytes(path)
            files += 1
            total += size
            if name.endswith((".l2tmp", ".tmp")):
                temp_bytes += size
    except OSError:
        pass
    return {"files": files, "bytes": total, "temp_bytes": temp_bytes}

def restore_cost_policy(costs: dict, pool_best_eff: int, n: int, legacy_costs: dict = None):
    """Old gate+costs versus P1 min(A,B,D) with read/reuse-separated costs."""
    legacy = legacy_costs or costs
    old = pool_best_eff < n * 0.5 and legacy["C"] < legacy["D"] * (1 - SAVE_MARGIN)
    best_without_l2 = min(costs["A"], costs["B"], costs["D"])
    new = costs["C"] < best_without_l2 * (1 - SAVE_MARGIN)
    return old, new, best_without_l2

def _classify_small_borrow(n: int, snapshot_tokens: int, reuse_tokens: int,
                           common_messages: int, snapshot_messages: int) -> str | None:
    """A short fork that can reuse a proven prefix of a much larger archive."""
    if not n or not snapshot_tokens or not reuse_tokens or not common_messages:
        return None
    if common_messages >= max(1, snapshot_messages):
        return None
    if n > SMALL_BORROW_MAX_TOKENS:
        return None
    if snapshot_tokens < max(int(n * SMALL_BORROW_MIN_RATIO),
                             n + SMALL_BORROW_MIN_ABS_DELTA):
        return None
    reuse_floor = min(n, SMALL_BORROW_MIN_ABS_DELTA) * SMALL_BORROW_MIN_REUSE
    if reuse_tokens < reuse_floor:
        return None
    return "small-borrow-big"

def _remember_borrow_hotspot(family: str, common_messages: int, reuse_tokens: int) -> dict:
    now = time.time()
    with _borrow_hotspots_lock:
        old = _borrow_hotspots.get(family)
        if old and old.get("common") == common_messages:
            old["hits"] = int(old.get("hits", 0)) + 1
            old["last_ts"] = now
            if reuse_tokens > 0:
                old["tokens"] = reuse_tokens
        else:
            old = {"common": common_messages, "tokens": reuse_tokens, "hits": 1, "last_ts": now}
            _borrow_hotspots[family] = old
        return dict(old)

def _inject_borrow_prefix_hint(body: dict | None, family: str, hotspot: dict,
                               n: int) -> dict | None:
    """Create/pin the proven shared boundary through the server's existing strata_prefix."""
    if not isinstance(body, dict) or body.get("strata_prefix") is not None:
        return None
    hits = int(hotspot.get("hits") or 0)
    if hits < BORROW_HOTSPOT_HITS:
        return None
    tokens = int(hotspot.get("tokens") or 0)
    if 0 < tokens < int(n) - 1:
        body["strata_prefix"] = {"tokens": tokens}
        return {"kind": "tokens", "tokens": tokens, "hits": hits}
    common = int(hotspot.get("common") or 0)
    messages = body.get("messages")
    if isinstance(messages, list) and 0 < common < len(messages):
        body["strata_prefix"] = {"messages": common}
        return {"kind": "messages", "messages": common, "hits": hits}
    return None

def fam_gc(family: str):
    """族内 LRU: 保留最近 FAM_KEEP 条链档。

    In-place generations share one physical file across aliases.  An old alias
    must be removed from the index without deleting the shared main file; only
    the last reference may reclaim it.
    """
    fam = fam_idx.get(family)
    if not fam or len(fam) <= FAM_KEEP:
        return
    victims = sorted(fam.values(), key=lambda x: x.get("ts", 0))[:-FAM_KEEP]
    for e in victims:
        key = next((k for k, v in fam.items() if v is e), None)
        shared = sum(1 for v in fam.values() if v.get("file") == e.get("file"))
        if shared > 1:
            # Drop the stale alias key only; the physical file is still the
            # current generation for the remaining aliases.
            if key is not None:
                fam.pop(key, None)
            continue
        if l2_reclaim(e["file"], "fam_lru"):
            fam.pop(key, None)

_gc_last_plan_s = 0.0
_cold_dry_last_s = 0.0
_gc_plan_seen = {}

def _log_gc_plan(target: str, why: str, scope: str = "all") -> None:
    now = time.time()
    if now - _gc_plan_seen.get(target, 0) < 3600:
        return
    _gc_plan_seen[target] = now
    try:
        size = archive_bytes(main_path(target)) if os.path.exists(main_path(target)) else 0
        with open(GC_PLAN_LOG, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%F %T"), "dry_run": True,
                                "target": target, "why": why, "scope": scope,
                                "main_bytes": size}, ensure_ascii=False) + "\n")
    except OSError:
        pass

def l2_total_bytes():
    """Charge each physical archive once, however many chain aliases name it.

    In-place generations intentionally leave several index entries pointing at
    the same file/directory.  Counting every alias inflates the L2 budget by
    tens of GiB and triggers global LRU eviction too early.
    """
    total = 0
    seen_inodes = set()

    def add_once(path: str) -> None:
        nonlocal total
        try:
            st = os.stat(path)
        except OSError:
            return
        identity = (st.st_dev, st.st_ino)
        if identity in seen_inodes:
            return
        seen_inodes.add(identity)
        total += archive_bytes(path)

    for fam in fam_idx.values():
        for e in fam.values():
            add_once(snap_path(e.get("file", "")))
    try:
        for f in os.listdir(ENGINE_SNAP_DIR):
            if f.endswith(".stale"):
                add_once(main_path(f))
    except OSError:
        pass
    return total

def l2_global_gc(protect: str = None):
    """v2.1.4: 全局 L2 池治理 — 一切驱逐只由【预算 LRU】裁决, 无启动即清。
    .stale 不再直接删: 注册为最低优先级候选, 仅在超预算时最先被逐
    (stale 可能是误判的好档 — d941.stale 教训; 盘上留着就有恢复机会)"""
    global _gc_last_plan_s
    now = time.time()
    if GC_DRY_RUN and now - _gc_last_plan_s < 300:
        return
    stale_files = []
    try:
        for f in os.listdir(ENGINE_SNAP_DIR):
            if f.endswith(".stale"):
                stale_files.append(f)
    except OSError:
        pass
    if l2_total_bytes() <= L2_BUDGET_BYTES:
        return
    if GC_DRY_RUN:
        _gc_last_plan_s = now
    # 驱逐序: .stale (按 mtime 旧→新) → 正常档全局 LRU; protect 永不先逐
    for f in sorted(stale_files, key=lambda x: archive_time(main_path(x)) if os.path.exists(main_path(x)) else 0):
        if l2_total_bytes() <= L2_BUDGET_BYTES:
            break
        if GC_DRY_RUN:
            _log_gc_plan(f, "stale", "global_stale")
            continue
        try:
            archive_unlink(main_path(f))
        except OSError:
            pass
    victims = []
    for fam_name, fam in fam_idx.items():
        for chain, e in fam.items():
            if e["file"] != protect:
                victims.append((e.get("ts", 0), fam_name, chain, e))
    victims.sort()
    for _, fam_name, chain, e in victims:
        if l2_total_bytes() <= L2_BUDGET_BYTES:
            break
        # Never reclaim a physical file that another, newer alias still uses.
        with fam_idx_lock:
            same_file = [v.get("ts", 0) for fam in fam_idx.values()
                         for v in fam.values() if v.get("file") == e.get("file")]
        if same_file and e.get("ts", 0) < max(same_file):
            continue
        if l2_reclaim(e["file"], "global_budget_lru", protect={protect} if protect else set()):
            fam_idx[fam_name].pop(chain, None)
    fam_idx_save()

def commit_matches(commit: dict, prev: str, gen: int) -> bool:
    """Match an engine L2C record by chain id + generation.

    The engine reports its own session root (for example
    <engine-slot-dir>/<chain>), while the gateway's hot/cold roots are
    different.  The chain id, not the absolute root, is the stable identity.
    """
    try:
        if int(commit.get("gen", -1)) != int(gen):
            return False
    except (TypeError, ValueError):
        return False
    path = os.path.basename(str(commit.get("path") or ""))
    if path.endswith(".state"):
        path = path[:-6]
    return path == os.path.basename(str(prev))

def commit_lookup(prev: str, gen: int, rid: str = "") -> tuple[str, int, dict | None]:
    """Return (state, wait_ms, record) for one L2 commit generation."""
    st, js, _ = engine_req("GET", "/slots", timeout=5, rid=rid)
    if st == 200 and isinstance(js, list) and js:
        for c in js[0].get("l2_commits", []):
            if commit_matches(c, prev, gen):
                status = c.get("status")
                if status == 1:
                    return "committed", 0, c
                if status == 2:
                    return "failed", 0, c
                return "pending", 0, c
    return "missing", 0, None

def wait_l2_commit(prev: str, gen: int, rid: str = "", wait_s: float | None = None
                    ) -> tuple[str, int, dict | None]:
    """Poll the exact (chain, generation) commit without comparing roots."""
    start = time.time()
    timeout_s = COMMIT_WAIT_S if wait_s is None else wait_s
    last = None
    while True:
        state, _, rec = commit_lookup(prev, gen, rid=rid)
        if rec is not None:
            last = rec
        if state in ("committed", "failed"):
            return state, round((time.time() - start) * 1000), last
        if time.time() - start >= timeout_s:
            return "timeout", round((time.time() - start) * 1000), last
        time.sleep(0.25)

def _record_save_decision(decision: dict | None, action: str, st: int, js: dict | None,
                          enqueue_ms: int | None = None) -> None:
    if decision is None:
        return
    decision["save_action"] = action
    decision["save_status"] = st
    if enqueue_ms is not None:
        decision["save_enqueue_ms"] = enqueue_ms
    if isinstance(js, dict):
        decision["save_n_saved"] = js.get("n_saved")
        # Engine field is not a byte count on both paths: sync = file size,
        # async WriterPool = job-buffer count. Keep the legacy name for
        # compatibility and expose the semantic flag separately.
        decision["save_n_written"] = js.get("n_written")
        decision["save_incremental"] = bool(js.get("incremental"))
        decision["save_fallback"] = action == "saveinc" and not js.get("incremental")
        decision["save_n_reused"] = js.get("n_reused")
        decision["save_n_kept"] = js.get("n_kept")
        decision["save_ms"] = (js.get("timings") or {}).get("save_ms")
        decision["save_gen"] = js.get("gen")
        decision["save_async_pending"] = js.get("async_pending")
        decision["save_async_mode"] = bool(js.get("async_mode"))

def _advance_snapshot(family: str, chain: str, prev: str | None, hashes: list, n_tok: int,
                      move_prev: bool = True, published_name: str | None = None) -> None:
    published = published_name or chain
    if move_prev and prev and prev != chain:
        # L2-2 writes the authoritative main file under the engine session root;
        # SNAP_DIR only holds .tok/index/mini metadata.  Rename the engine file,
        # not a gateway-root file that does not exist.
        engine_prev = os.path.join(ENGINE_SNAP_DIR, prev)
        engine_chain = os.path.join(ENGINE_SNAP_DIR, chain)
        moved = False
        try:
            if os.path.exists(engine_prev):
                os.replace(engine_prev, engine_chain)
                moved = True
        except OSError as e:
            print(f"[gw] engine roll-forward {prev[:12]}->{chain[:12]} failed: {e}", flush=True)
        if not moved:
            try:
                os.replace(snap_path(prev), snap_path(chain))
            except OSError as e:
                print(f"[gw] roll-forward {prev[:12]}->{chain[:12]} failed: {e}", flush=True)
        # L2 2.0 组装对齐: mini 随主档前滚同步改名
        try:
            os.replace(os.path.join(HOT_DIR, prev + ".state"),
                       os.path.join(HOT_DIR, chain + ".state"))
        except OSError:
            pass
        published = chain
    try:
        with open(tok_path(published), "w") as f:
            f.write(str(n_tok))
    except OSError:
        pass
    if published != chain:
        # A saveinc fallback keeps the engine file under `published`; the chain
        # name has no main file and must not leave a ghost .tok behind.
        try:
            os.unlink(tok_path(chain))
        except OSError:
            pass
    with fam_idx_lock:
        fam = fam_idx.setdefault(family, {})
        if move_prev and prev:
            for entry in fam.values():
                if entry.get("file") == prev:
                    entry["file"] = chain
        if prev and prev in fam:
            del fam[prev]                     # the ancestor key is superseded
        fam[chain] = {"file": published, "ts": time.time(), "tok": n_tok, "hashes": hashes,
                      "inplace": True}
        for entry in fam.values():
            if entry.get("file") in (prev, published, chain):
                entry["inplace"] = True
        fam_gc(family)
        fam_idx_save()

def _log_save_commit(job: dict, state: str, wait_ms: int, n_tok: int, error: str = "",
                     rec: dict | None = None) -> None:
    try:
        row = {
            "ts": time.strftime("%F %T"), "rid": job.get("rid", ""),
            "family": job.get("family", ""), "chain": job.get("chain", "")[:16],
            "prev": (job.get("prev") or "")[:16], "gen": job.get("gen"),
            "state": state, "wait_ms": wait_ms, "n_saved": n_tok,
            "error": error[:200],
        }
        for key in ("copy_bytes", "copy_ms", "saveinc_ms", "writer_ms", "growth_bytes", "copy_used", "inplace"):
            if rec and rec.get(key) is not None:
                row[key] = rec[key]
        with open(SAVE_COMMIT_LOG, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass

def _release_enqueue_placeholder(key: tuple) -> None:
    with _pending_saves_lock:
        job = _pending_saves.get(key)
        if job is not None and job.get("state") == "enqueueing":
            _pending_saves.pop(key, None)

def _finalize_pending_save(job: dict) -> None:
    """Wait for an already-enqueued saveinc commit, then roll forward.

    This runs outside the request decision lock.  It never performs a full
    engine save after a switch: at that point the engine slot may belong to a
    different session, so a rescue save would capture the wrong state.

    A healthy writer can spend tens of seconds behind the FIFO queue.  Keep the
    pending slot reserved and poll until a hard deadline instead of releasing
    it after the short commit window and enqueuing duplicate generations.
    """
    started = time.time()
    deadline = started + PENDING_MAX_S
    state, wait_ms, rec = wait_l2_commit(
        job["prev"], job["gen"], rid=job.get("rid", ""),
        wait_s=min(COMMIT_WAIT_S, PENDING_MAX_S))
    while state == "timeout" and time.time() < deadline:
        state, _, rec2 = wait_l2_commit(
            job["prev"], job["gen"], rid=job.get("rid", ""),
            wait_s=min(COMMIT_WAIT_S, max(0.25, deadline - time.time())))
        if rec2 is not None:
            rec = rec2
    wait_ms = round((time.time() - started) * 1000)
    job["state"] = state
    n_tok = job.get("n_tok", 0)
    if rec and rec.get("tokens") is not None:
        n_tok = int(rec.get("tokens") or 0)
    with _pending_saves_lock:
        current = _pending_saves.get(job["key"]) is job
    if current and state == "committed":
        with state_lock:
            with _pending_saves_lock:
                current = _pending_saves.get(job["key"]) is job
            if current:
                _advance_snapshot(job["family"], job["chain"], job["prev"],
                                  job["hashes"], n_tok, move_prev=True)
        if current:
            try:
                l2_global_gc(protect=job["chain"])
            except Exception as e:
                print(f"[gw] commit GC {job['chain'][:12]}: {e}", flush=True)
    with _pending_saves_lock:
        if _pending_saves.get(job["key"]) is job:
            _pending_saves.pop(job["key"], None)
    _log_save_commit(job, state, wait_ms, n_tok,
                     "" if state == "committed" else f"commit {state}", rec=rec)
    print(f"[gw] commit {job['chain'][:12]} gen={job['gen']} state={state} "
          f"wait={wait_ms}ms", flush=True)

def _read_int_file(path: str):
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _read_memory_events(path: str) -> dict:
    out = {}
    try:
        with open(path) as handle:
            for line in handle:
                key, _, value = line.strip().partition(" ")
                try:
                    out[key] = int(value)
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _read_memory_psi() -> dict:
    try:
        with open("/proc/pressure/memory") as handle:
            text = handle.read()
    except OSError:
        return {}
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(" ")
        for item in rest.split():
            name, _, value = item.partition("=")
            if name.startswith("avg"):
                try:
                    out[f"{key}_{name}"] = float(value)
                except ValueError:
                    pass
    return out


def _cgroup_memory_view(path: str) -> dict:
    current = _read_int_file(path + "/memory.current")
    if current is None:
        return {}
    high = _read_int_file(path + "/memory.high")
    maximum = _read_int_file(path + "/memory.max")
    # cgroup v2 uses "max" for unlimited; keep None instead of伪造 0.
    limits = [x for x in (high, maximum) if x is not None and x > 0]
    return {"current": current, "high": high, "max": maximum,
            "headroom": (min(limits) - current) if limits else None,
            "events": _read_memory_events(path + "/memory.events")}


def memory_pressure_snapshot() -> dict:
    """Read service/container/host memory state without adding GPU synchronization.

    This is deliberately cheap: a handful of sysfs/procfs reads.  It is sampled
    per request and may be sampled before background restore/save/warm work.
    """
    service = _cgroup_memory_view("/sys/fs/cgroup/system.slice/strata-flash.service")
    container = _cgroup_memory_view("/sys/fs/cgroup")
    host_available = None
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    host_available = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    return {"service": service, "container": container,
            "host_available": host_available, "psi": _read_memory_psi()}


def _memory_pressure_reasons(snapshot: dict, est_bytes: int = 0) -> list:
    reasons = []
    service = snapshot.get("service") or {}
    container = snapshot.get("container") or {}
    host_available = snapshot.get("host_available")
    service_headroom = service.get("headroom")
    container_headroom = container.get("headroom")
    reserve = int(os.environ.get("STRATA_MEM_MIN_HEADROOM_MIB", "1024")) * 1024**2
    need = max(reserve, int(est_bytes or 0))
    if host_available is not None and host_available < max(
            reserve, int(os.environ.get("STRATA_MEM_MIN_HOST_AVAILABLE_MIB", "4096")) * 1024**2):
        reasons.append("host_available_low")
    if service_headroom is not None and service_headroom < need:
        reasons.append("service_headroom_low")
    if container_headroom is not None and container_headroom < need:
        reasons.append("container_headroom_low")
    psi = float((snapshot.get("psi") or {}).get("some_avg10") or 0)
    if psi > float(os.environ.get("STRATA_MEM_PSI_SOME_AVG10", "20")):
        reasons.append("memory_psi_high")
    return reasons


def memory_backpressure(reason: str, est_bytes: int = 0, rid: str = "") -> dict:
    """Return an immediate memory-pressure decision for optional background work."""
    snapshot = memory_pressure_snapshot()
    reasons = _memory_pressure_reasons(snapshot, est_bytes)
    decision = {"ok": not reasons, "reasons": reasons, "est_bytes": int(est_bytes or 0)}
    if reasons:
        append_event_log(MEMORY_EVENTS_LOG, {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "backpressure",
            "operation": reason, "rid": rid, **decision, "snapshot": snapshot})
    return decision


def wait_for_memory_admission(est_bytes: int, reason: str, rid: str = "",
                              max_wait_s: float | None = None) -> dict:
    """Fail-closed admission for direct restore-sized memory jumps.

    Optional background work should use memory_backpressure(); this path guards
    a request-facing restore after PARKED preload has failed.  If the cgroup or
    host still cannot fit the restore after the bounded wait, return ok=False
    and let the caller take engine-native prefill instead of risking an OOM.
    """
    started = time.time()
    max_wait_s = float(os.environ.get("STRATA_L2_MEM_WAIT_S", "120") if max_wait_s is None else max_wait_s)
    last = memory_pressure_snapshot()
    while True:
        reasons = _memory_pressure_reasons(last, est_bytes)
        if not reasons:
            return {"ok": True, "waited_ms": round((time.time() - started) * 1000), "reasons": [], "snapshot": last}
        if time.time() - started >= max_wait_s:
            out = {"ok": False, "waited_ms": round((time.time() - started) * 1000),
                   "reasons": reasons, "snapshot": last}
            append_event_log(MEMORY_EVENTS_LOG, {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "restore_memory_rejected",
                "operation": reason, "rid": rid, **out})
            return out
        if int(time.time() - started) % 10 == 0:
            print(f"[gw] {reason} waits memory: est={est_bytes/2**30:.1f}G "
                  f"reasons={','.join(reasons)}", flush=True)
        time.sleep(2)
        last = memory_pressure_snapshot()


def _inline_finalize_pending(rid: str, deadline: float) -> bool:
    """Finish commits while the caller already owns ``state_lock``.

    The background finalizer needs ``state_lock`` to roll the index forward.
    If the serial wait blocks while holding the same lock, the committed
    generation is never consumed (the observed 300 s Yx stall).
    RLock is reentrant, so this path polls and advances inline instead.
    """
    while True:
        with _pending_saves_lock:
            jobs = list(_pending_saves.values())
        if not jobs:
            return True
        progressed = False
        for job in jobs:
            state = job.get("state")
            rec = None
            if state not in ("committed", "failed"):
                remaining = max(0.25, deadline - time.time())
                state, _, rec = wait_l2_commit(
                    job["prev"], job["gen"], rid=job.get("rid", ""),
                    wait_s=min(COMMIT_WAIT_S, remaining))
                if state == "committed":
                    job["state"] = "committed"
                elif state == "failed":
                    job["state"] = "failed"
            with _pending_saves_lock:
                current = _pending_saves.get(job["key"]) is job
            if not current:
                progressed = True
                continue
            if state == "committed":
                with _pending_saves_lock:
                    if _pending_saves.get(job["key"]) is not job:
                        continue
                    _pending_saves.pop(job["key"], None)
                n_tok = job.get("n_tok", 0)
                if rec and rec.get("tokens") is not None:
                    n_tok = int(rec.get("tokens") or 0)
                # state_lock is held and reentrant here.
                _advance_snapshot(job["family"], job["chain"], job["prev"],
                                  job["hashes"], n_tok, move_prev=True)
                progressed = True
            elif state == "failed":
                with _pending_saves_lock:
                    if _pending_saves.get(job["key"]) is job:
                        _pending_saves.pop(job["key"], None)
                progressed = True
        if progressed:
            continue
        if time.time() >= deadline:
            with _pending_saves_lock:
                pending = len(_pending_saves)
            print(f"[gw] save {rid[:12] or '-'} waits for {pending} pending generation(s): "
                  "serial timeout", flush=True)
            return False
        time.sleep(0.05)

def _wait_no_pending_saves(rid: str = "") -> tuple[bool, int]:
    """Serialize save starts until the previous generation's commit/finalize.

    WriterPool is single-threaded, but gateway roll-forward renames the target
    before a queued job is visible everywhere.  Starting another save in that
    window can make the engine's post-write verifier open a path that was just
    renamed or reclaimed.  One outstanding save at a time matches the writer's
    actual concurrency and keeps the path stable through verify.
    """
    started = time.time()
    timeout = float(os.environ.get("STRATA_L2_SAVE_SERIAL_WAIT_S", str(PENDING_MAX_S)))
    deadline = started + timeout
    if getattr(state_lock, "_is_owned", lambda: False)():
        ok = _inline_finalize_pending(rid, deadline)
        return ok, round((time.time() - started) * 1000)
    while True:
        with _pending_saves_lock:
            pending = len(_pending_saves)
        if pending == 0:
            return True, round((time.time() - started) * 1000)
        if time.time() - started >= timeout:
            print(f"[gw] save {rid[:12] or '-'} waits for {pending} pending generation(s): "
                  "serial timeout", flush=True)
            return False, round((time.time() - started) * 1000)
        time.sleep(0.05)

def do_save(family: str, chain: str, hashes: list = None, rid: str = "", decision: dict = None) -> bool:
    """One save may be between target selection and pending registration at a time."""
    with _save_start_lock:
        return _do_save_locked(family, chain, hashes, rid, decision)

def _do_save_locked(family: str, chain: str, hashes: list = None, rid: str = "",
                    decision: dict = None) -> bool:
    """Queue an incremental save and finalize its commit in the background.

    The fast path keeps saveinc close to the engine's enqueue latency.  The
    old code blocked the request while polling a path that belongs to a
    different session root, then attempted a full rescue after every timeout.
    """
    global last_save_ts, _saveinc_ok
    hashes = hashes or []
    if decision is not None:
        decision["save_reserve_mib"] = SAVE_RESERVE_MIB
    prev = None
    with fam_idx_lock:
        anc = ancestor_snap(family, hashes)
        prev = anc["file"] if anc and anc["file"] != chain else None
    # A duplicate request for the exact pending target joins it rather than
    # waiting for the first commit and then appending a second generation.
    if prev and _saveinc_ok is not False:
        same_key = (family, prev)
        with _pending_saves_lock:
            if same_key in _pending_saves:
                _record_save_decision(decision, "saveinc", 0, None)
                if decision is not None:
                    decision["save_commit_state"] = "pending-existing"
                print(f"[gw] save {family}/{chain}: saveinc already pending for {prev[:12]}", flush=True)
                return True
    serial_ok, serial_ms = _wait_no_pending_saves(rid)
    if decision is not None:
        decision["save_serial_wait_ms"] = serial_ms
    if not serial_ok:
        if decision is not None:
            decision["save_commit_state"] = "serial-wait-timeout"
        return False
    action, target = "save", chain
    if prev and _saveinc_ok is not False:   # v2.4: 引擎已修复二次扩展+写后校验, saveinc 恢复
        action, target = "saveinc", prev          # 扩展祖先档, 成功后改名前滚
    if action == "saveinc":
        try:
            target_ok = archive_exists(main_path(target)) and archive_bytes(main_path(target)) > 0
        except OSError:
            target_ok = False
        if not target_ok:
            print(f"[gw] save {family}/{chain}: saveinc target {target[:12]} is stale/missing; "
                  "fall back to full save", flush=True)
            if decision is not None:
                decision["save_missing_target"] = True
            action, target = "save", chain
    key = (family, target)
    reserved = False
    if action == "saveinc":
        with _pending_saves_lock:
            if key in _pending_saves:
                _record_save_decision(decision, action, 0, None)
                if decision is not None:
                    decision["save_commit_state"] = "pending-existing"
                print(f"[gw] save {family}/{chain}: saveinc already pending for {target[:12]}", flush=True)
                return True
            # Reserve before the engine POST so two threads cannot enqueue
            # duplicate generations for the same target.
            _pending_saves[key] = {"key": key, "family": family, "chain": chain,
                                   "prev": prev, "hashes": list(hashes), "rid": rid,
                                   "state": "enqueueing"}
            reserved = True
    t_enqueue = time.time()
    st, js, _ = engine_req("POST", f"/slots/0?action={action}", {"filename": target},
                           timeout=180 if action == "saveinc" else 120, rid=rid)
    if action == "saveinc" and st == 400 and "action" in str(js):
        _release_enqueue_placeholder(key)
        reserved = False
        _saveinc_ok = False
        action, target = "save", chain
        st, js, _ = engine_req("POST", "/slots/0?action=save", {"filename": chain}, timeout=120, rid=rid)
    enqueue_ms = round((time.time() - t_enqueue) * 1000)
    ok = st == 200
    n_tok = (js or {}).get("n_saved", 0) if isinstance(js, dict) else 0
    _record_save_decision(decision, action, st, js if isinstance(js, dict) else None, enqueue_ms)
    actual_incremental = bool(isinstance(js, dict) and js.get("incremental"))
    if ok:
        stats["saves"] += 1
        if actual_incremental:
            stats["saveinc"] = stats.get("saveinc", 0) + 1
        last_save_ts = time.time()
        n_w = (js or {}).get("n_written", 0) if isinstance(js, dict) else 0   # sync=file size / async=job buffers; not write volume
        s_ms = ((js or {}).get("timings") or {}).get("save_ms", 0) if isinstance(js, dict) else 0
        if n_w:
            try:
                with open(SAVEINC_LOG, "a") as bf:
                    bf.write(f"{time.strftime('%F %T')} chain={chain[:12]} action={action} incremental={int(actual_incremental)} n_saved={n_tok} n_written={n_w} save_ms={s_ms:.0f}\n")
            except OSError:
                pass
        if action == "saveinc" and prev and actual_incremental:
            gen = (js or {}).get("gen", 0) if isinstance(js, dict) else 0
            pending = (js or {}).get("async_pending", 0) if isinstance(js, dict) else 0
            async_mode = bool((js or {}).get("async_mode")) if isinstance(js, dict) else False
            if (pending or async_mode) and gen:
                job = {"key": key, "family": family, "chain": chain, "prev": prev,
                       "hashes": list(hashes), "gen": gen, "n_tok": n_tok, "rid": rid,
                       "state": "pending"}
                with _pending_saves_lock:
                    _pending_saves[key] = job
                reserved = False
                if decision is not None:
                    decision["save_commit_state"] = "pending"
                threading.Thread(target=_finalize_pending_save, args=(job,), daemon=True).start()
                print(f"[gw] save {family}/{chain}: saveinc@{target[:8]} queued gen={gen} "
                      f"enqueue={enqueue_ms}ms", flush=True)
                return True
        # The engine response is already committed (sync saveinc or full fallback).
        if reserved:
            _release_enqueue_placeholder(key)
            reserved = False
        # A saveinc that declined inside the engine falls back to a full save,
        # but that full save is published under the requested ancestor file
        # (`target`), not under the gateway's new chain id.
        published = target if (action == "saveinc" and not actual_incremental) else chain
        _advance_snapshot(family, chain, prev, hashes, n_tok,
                          move_prev=(actual_incremental and bool(prev)),
                          published_name=published)
        if decision is not None:
            decision["save_published_file"] = published
        try:
            l2_global_gc(protect=published)
        except Exception as e:
            print(f"[gw] save GC {published[:12]}: {e}", flush=True)
        if decision is not None:
            decision["save_commit_state"] = "committed"
    else:
        if reserved:
            _release_enqueue_placeholder(key)
        if decision is not None:
            decision["save_commit_state"] = "failed"
        if st == 503 and "not enough RAM" in str(js):
            if decision is not None:
                decision["save_memory_rejected"] = True
            append_event_log(MEMORY_EVENTS_LOG, {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "save_memory_rejected",
                "operation": action, "rid": rid, "family": family, "chain": chain,
                "http_status": st, "snapshot": memory_pressure_snapshot()})
    print(f"[gw] save {family}/{chain}: {action}{'@'+target[:8] if target != chain else ''} {st} "
          f"{str(js)[:300]}", flush=True)
    return ok

def hot_ensure(snap: str) -> bool:
    """L2 v3 复活器: tmpfs 无 .state → 从 L2-2 备份区 cp 回."""
    import shutil
    hot = os.path.join(HOT_DIR, snap + ".state")
    main = os.path.join(ENGINE_SNAP_DIR, snap)
    if os.path.exists(hot) or not os.path.exists(main):
        return os.path.exists(main)
    bk = os.path.join(ARCHIVE_DIR, snap + ".state")
    if os.path.exists(bk):
        try:
            shutil.copyfile(bk, hot)
            print(f"[gw2] mini revived from archive: {snap[:12]}", flush=True)
        except OSError:
            pass
    return True

SLOT_KV_CAPACITY_G = float(__import__('os').environ.get("STRATA_SLOT_KV_G", "7.2"))   # L2 2.0: 槽 RAM KV 容量 (X300 6.3G 进槽/Z400 8.4G parked 校准)
STATE_OVERHEAD_G = 1.5    # 档内 state/checkpoint 开销近似

RESTORE_TRUNC_TOKENS = int(__import__('os').environ.get("STRATA_RESTORE_TRUNC_K", "200"))  # K token — [giphoo] 350→200: 277K 档也截断 (24G 池双 PARKED 装得下)

def restore_will_park(snap: str) -> bool:
    """零损耗预判: restorep 恒 parked (P 语义); 实测 >350K 档 parked 借用失效 (全量 prefill 218s).
    判定改为 token 水位 (.tok 直读) — 超限走截断深借."""
    try:
        n = int(open(os.path.join(SNAP_DIR, snap) + ".tok").read() or 0)
        return n > RESTORE_TRUNC_TOKENS * 1000
    except (OSError, ValueError):
        return False

def truncate_messages_for_slot(msgs: list) -> list:
    """parked 预判 → 截断消息列到槽容量内 (每token≈4.5B KV q4, 留10%余量, 消息边界保守截)"""
    budget_tokens = RESTORE_TRUNC_TOKENS * 1000   # 截到借用安全水位 (实测借用可用的深度)
    est, cut = 0, len(msgs)
    for i, m in enumerate(msgs):
        est += len(str(m.get("content", ""))) // 2 + 8
        if est > budget_tokens:
            cut = max(i, 1)
            break
    return msgs[:cut]

def _evict_generation(snap: str, why: str) -> None:
    """Drop a semantically bad generation from every family index; keep files for audit."""
    with fam_idx_lock:
        for fam in fam_idx.values():
            for key, entry in list(fam.items()):
                if entry.get("file") == snap:
                    fam.pop(key, None)
        fam_idx_save()
    print(f"[gw] generation {snap[:12]} evicted ({why})", flush=True)

def _wait_restore_file_idle(snap: str, rid: str = "") -> tuple[bool, int]:
    """Do not read a main file while its gateway generation is being appended."""
    started = time.time()
    while True:
        with _pending_saves_lock:
            job = next((j for j in _pending_saves.values() if j.get("prev") == snap), None)
        if job is None:
            return True, round((time.time() - started) * 1000)
        gen = job.get("gen")
        if gen:
            state, _, _ = wait_l2_commit(snap, gen, rid=rid,
                                         wait_s=max(0.25, COMMIT_WAIT_S - (time.time() - started)))
            if state == "committed":
                return True, round((time.time() - started) * 1000)
            print(f"[gw] restore waits {snap[:12]} gen={gen}: {state}; use self-heal", flush=True)
            return False, round((time.time() - started) * 1000)
        if time.time() - started >= COMMIT_WAIT_S:
            print(f"[gw] restore waits {snap[:12]} enqueue: timeout; use self-heal", flush=True)
            return False, round((time.time() - started) * 1000)
        time.sleep(0.05)

def _snapshot_size_metrics(snap: str):
    """Return physical/logical bytes; V2's reader limit is its current payload."""
    try:
        path = snap_path(snap)
        physical = archive_bytes(path)
        logical = physical
        file_format = "unknown"
        header_path = os.path.join(path, "manifest") if archive_is_v3(path) else path
        with open(header_path, "rb") as handle:
            header = handle.read(64)
        if len(header) >= 40 and header[:8] in (b"STRSESS\x02", b"STRSESS\x03"):
            logical = int.from_bytes(header[32:40], "little")
            file_format = "v3" if header[:8] == b"STRSESS\x03" else "v2"
        elif len(header) >= 8 and header[:7] == b"STRSESS":
            file_format = "v1"
        return {"physical_bytes": physical, "logical_bytes": logical,
                "format": file_format, "limit_basis": "logical_payload" if file_format in ("v2", "v3") else "physical_file"}
    except OSError:
        return None


def _snapshot_size_bytes(snap: str):
    metrics = _snapshot_size_metrics(snap)
    return metrics.get("physical_bytes") if metrics else None


def _eligible_restore_alternative(family: str, hashes: list, excluded: str):
    """Pick the deepest lineage candidate below the engine hard size limit."""
    if not hashes:
        return None
    candidates = []
    with fam_idx_lock:
        entries = []
        for fam_name, fam in fam_idx.items():
            for entry in fam.values():
                # Same-family lineage is preferred; global lineage remains a fallback.
                entries.append((0 if fam_name == family else 1, entry))
        for family_rank, entry in entries:
            name = entry.get("file")
            if not name or name == excluded:
                continue
            entry_hashes = entry.get("hashes") or []
            common = chain_prefix_length(entry_hashes, hashes)
            if common <= 0:
                continue
            metrics = _snapshot_size_metrics(name)
            if metrics is None or metrics["logical_bytes"] > ENGINE_RESTORE_LIMIT_BYTES:
                continue
            candidates.append((family_rank, common, int(entry.get("tok") or 0), name, metrics))
    if not candidates:
        return None
    family_rank, common, _tok, name, metrics = min(candidates, key=lambda x: (x[0], -x[1], -x[2]))
    return {"file": name, "common_messages": common, **metrics}


def _record_restore_size_guard(family: str, snap: str, metrics: dict, rid: str = "",
                               decision: dict | None = None, alternative: dict | None = None,
                               stage: str = "pre-restore") -> None:
    physical = int(metrics.get("physical_bytes") or 0)
    logical = int(metrics.get("logical_bytes") or physical)
    event = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "restore_size_exceeded",
        "stage": stage, "family": family, "snapshot": snap, "rid": rid,
        "size_bytes": logical, "physical_bytes": physical, "logical_bytes": logical,
        "limit_basis": metrics.get("limit_basis"), "limit_bytes": ENGINE_RESTORE_LIMIT_BYTES,
        "alternative": alternative,
    }
    append_event_log(RESTORE_GUARD_LOG, event)
    if decision is not None:
        decision["restore_guard"] = "size_exceeded"
        decision["restore_snapshot_bytes"] = physical
        decision["restore_logical_bytes"] = logical
        decision["restore_limit_basis"] = metrics.get("limit_basis")
        decision["restore_limit_bytes"] = ENGINE_RESTORE_LIMIT_BYTES
        if alternative and alternative.get("file"):
            decision["restore_guard_alternative"] = alternative.get("file")
        else:
            decision["restore_full_fallback_suppressed"] = True
    print(f"[gw] restore guard {snap[:12]} logical={logical/2**30:.2f}G "
          f"physical={physical/2**30:.2f}G limit={ENGINE_RESTORE_LIMIT_BYTES/2**30:.2f}G alt="
          f"{(alternative or {}).get('file', 'none')[:12]}", flush=True)


def _record_restore_metrics(decision: dict | None, js: dict | None) -> None:
    if decision is None or not isinstance(js, dict):
        return
    n_read = js.get("n_read")
    timings = js.get("timings") or {}
    restore_ms = timings.get("restore_ms")
    decision["restore_n_read"] = n_read
    decision["restore_n_restored"] = js.get("n_restored")
    decision["restore_engine_ms"] = restore_ms
    decision["restore_partial"] = bool(js.get("partial"))
    decision["restore_checkpoint_tokens"] = js.get("prefix_checkpoint_tokens")
    decision["restore_image_bytes"] = js.get("prefix_image_bytes")
    decision["restore_payload_read_bytes"] = js.get("prefix_payload_read_bytes")
    decision["restore_io_read_bytes"] = js.get("prefix_block_read_bytes")
    decision["restore_archive_format"] = js.get("archive_format")
    decision["restore_fallback"] = js.get("prefix_fallback")
    io_read = js.get("prefix_block_read_bytes")
    if (bool(js.get("partial")) and isinstance(io_read, (int, float)) and
            isinstance(restore_ms, (int, float)) and restore_ms > 0):
        decision["restore_io_read_mib_s"] = round(io_read / 2**20 / (restore_ms / 1000), 1)
    if isinstance(n_read, (int, float)) and isinstance(restore_ms, (int, float)) and restore_ms > 0:
        decision["restore_read_mib_s"] = round(n_read / 2**20 / (restore_ms / 1000), 1)

def do_restore(family: str, chain: str = None, body: dict | None = None, rid: str = "",
               decision: dict | None = None) -> bool:
    """v2.1.2: 优先 restorepp (M3.1 按检查点部分恢复 — 读按借用量计), 失败回落 restorep 全档"""
    hashes = chain_hashes((body or {}).get("messages")) if body is not None else None
    snap = resolve_snap(family, chain, hashes)   # v2.1.6: 血缘祖先优先于 tok-max
    # v2.2.0: 选档取证 — top3 血缘候选 (depth, tok) 随每次恢复落日志, 选档正确性不再靠手算
    if hashes:
        cands = []
        with fam_idx_lock:
            for c, e in (fam_idx.get(family) or {}).items():
                eh = e.get("hashes") or []
                d = chain_prefix_length(eh, hashes)
                cands.append((d, int(e.get("tok") or 0), c, float(e.get("ts") or 0)))
        def candidate_order(item):
            depth, tok, _chain, ts = item
            return (-depth, tok if tok > 0 else float("inf"), -ts)
        cands.sort(key=candidate_order)
        if cands:
            top = [{"depth": d, "tokens": t, "chain": c[:12]} for d, t, c, _ts in cands[:3]]
            if decision is not None:
                decision["restore_candidates"] = top
            print(f"[gw] resolve {family[:8]} cand: " +
                  " ".join(f"{c[:10]}(d{d},t{t})" for d, t, c, _ts in cands[:3]) +
                  f" pick={snap[:10] if snap else 'none'}", flush=True)
    if not snap:
        return False
    # P0 size guard: V2 is judged by its current logical payload. Physical
    # append-only slack is reclaimed by save compaction, not misread as live KV.
    snap_metrics = _snapshot_size_metrics(snap)
    if snap_metrics is not None and snap_metrics["logical_bytes"] > ENGINE_RESTORE_LIMIT_BYTES:
        alternative = _eligible_restore_alternative(family, hashes or [], snap)
        _record_restore_size_guard(family, snap, snap_metrics, rid=rid, decision=decision,
                                   alternative=alternative, stage="pre-restore")
        if not alternative:
            restore_outcome(False)
            return False
        snap = alternative["file"]
        snap_metrics = alternative
    if decision is not None and snap_metrics is not None:
        decision["restore_snapshot_bytes"] = snap_metrics["physical_bytes"]
        decision["restore_logical_bytes"] = snap_metrics["logical_bytes"]
        decision["restore_limit_basis"] = snap_metrics["limit_basis"]
    if body is not None:
        _msgs = body.get("messages") or []
        # [giphoo] v2.3.0: restore 模式不截断 (完整恢复到活跃槽, prefill 处理增量)
        if os.environ.get("STRATA_RESTORE_MODE", "restore") == "restorepp" and restore_will_park(snap) and len(_msgs) > 4:
            _msgs = truncate_messages_for_slot(_msgs)
            print(f"[gw] L2 池感知: {snap[:10]} 超槽容量({SLOT_KV_CAPACITY_G}G) → restorepp 截断深借 (msgs→{len(_msgs)})", flush=True)
        rb = {"filename": snap, "messages": _msgs}
        if body.get("tools") is not None:
            rb["tools"] = body["tools"]
        if isinstance(body.get("chat_template_kwargs"), dict):
            rb["chat_template_kwargs"] = body["chat_template_kwargs"]
        idle, pending_wait_ms = _wait_restore_file_idle(snap, rid=rid)
        if decision is not None:
            decision["restore_pending_wait_ms"] = pending_wait_ms
        if not idle:
            restore_outcome(False)
            return False
        hot_ensure(snap)   # L2 v3: mini 复活器 (备份区→tmpfs, 组装级恢复就位)
        # [giphoo] RAM control plane: consume an existing PARKED target instead of
        # letting RESTORE replace active live (which would discard the outgoing
        # PARKED state). Pin only until the request consumes it; proxy finally
        # unpins the serial.
        # The same physical L2 file can have a shallow partial PARKED image
        # from a small branch.  Only consume it if it covers this request's
        # waterline; otherwise preload the target again.
        try:
            need_tokens = int((decision or {}).get("l2_est") or 0)
        except (TypeError, ValueError):
            need_tokens = 0
        if not need_tokens:
            try:
                need_tokens = int(open(tok_path(snap)).read() or 0)
            except OSError:
                need_tokens = 0
        try:
            prompt_tokens = int((decision or {}).get("n") or 0)
        except (TypeError, ValueError):
            prompt_tokens = 0
        if prompt_tokens:
            need_tokens = min(need_tokens or prompt_tokens, prompt_tokens)
        min_tokens = max(0, need_tokens - max(1024, need_tokens // 100))
        parked_serial = cache_has_source(snap, min_tokens=min_tokens)
        if parked_serial is not None:
            cache_pin(parked_serial, True)
            if decision is not None:
                decision["restore_parked"] = True
                decision["parked_cached"] = True
                decision["preload_serial"] = parked_serial
                decision["preload_status"] = "parked-hit"
                decision["preload_http_status"] = 200
                decision["restore_source"] = "parked-hit"
            restore_outcome(True)
            print(f"[gw] parked hit {snap[:12]} serial={parked_serial}; use engine-native", flush=True)
            return True
        preload_ok, preload_js, preload_status = cache_preload(snap, body or {}, rid=rid, pin=True)
        if decision is not None:
            decision["preload_status"] = "success" if preload_ok else "failed"
            decision["preload_http_status"] = preload_status
        if preload_ok:
            _record_restore_metrics(decision, preload_js)
            stats["restores"] += 1
            if isinstance(preload_js, dict) and preload_js.get("partial"):
                stats["restorepp"] = stats.get("restorepp", 0) + 1
            restore_outcome(True)
            if decision is not None:
                decision["restore_parked"] = True
                decision["preload_serial"] = (preload_js or {}).get("serial")
                decision["restore_source"] = "parked-preload"
            with fam_idx_lock:
                for fam in fam_idx.values():
                    for e in fam.values():
                        if e["file"] == snap:
                            e["ts"] = time.time()
            print(f"[gw] preloaded {snap[:12]} as PARKED serial={(preload_js or {}).get('serial')}", flush=True)
            return True
        if decision is not None:
            message = str(preload_js or {})
            if "exceeds this engine's limit" in message or "exceeds the limit" in message:
                decision["preload_error_kind"] = "size_exceeded"
            elif "budget or slots" in message:
                decision["preload_error_kind"] = "budget_or_slots"
            elif preload_status == 507:
                decision["preload_error_kind"] = "insufficient_storage"
            else:
                decision["preload_error_kind"] = f"http_{preload_status}"
        print(f"[gw] parked preload failed for {snap[:12]}; fallback direct restore", flush=True)
        # P0/P1 memory admission: direct restore can add a snapshot-sized RSS
        # jump.  Wait briefly, then fail closed to engine-native instead of
        # triggering reclaim storms or an engine restart.
        est = max(1, int(((snap_metrics or {}).get("logical_bytes") or 0) * 1.2))
        memory_admission = wait_for_memory_admission(est, "direct_restore", rid=rid)
        if decision is not None:
            decision["restore_memory_admission"] = {
                "ok": memory_admission["ok"], "waited_ms": memory_admission.get("waited_ms"),
                "reasons": memory_admission.get("reasons")}
        if not memory_admission["ok"]:
            restore_outcome(False)
            return False
        # [giphoo] v2.3.0: restore 统一模式 + 引擎 LCP 借用 (BPE 边界差修复)
        # restore 完整恢复 KV 到活跃槽; 引擎 starts_with 失败时用 LCP 借用最长公共前缀
        # 同会话 100% 复用; 跨会话 LCP 借用 (不再退化全量 prefill)
        st, js, _ = engine_req("POST", f"/slots/0?action={os.environ.get('STRATA_RESTORE_MODE', 'restore')}", rb, timeout=60, rid=rid)
        if st == 200:
            _record_restore_metrics(decision, js)
            stats["restores"] += 1
            if decision is not None:
                decision["restore_source"] = "direct-engine"
            thrash_check(snap)         # [giphoo] issue#3: thrashing 检测
            restore_outcome(True)      # [giphoo] issue#3: 成功率
            if isinstance(js, dict) and js.get("partial"):
                stats["restorepp"] = stats.get("restorepp", 0) + 1
            with fam_idx_lock:   # 借用即热度: touch LRU
                for fam in fam_idx.values():
                    for e in fam.values():
                        if e["file"] == snap:
                            e["ts"] = time.time()
            print(f"[gw] restorepp {family} -> {snap}: {st} {str(js)[:300]}", flush=True)
            return True
        # A semantic checkpoint/live mismatch is deterministic.  Reading the
        # same file again through restorep only doubles I/O; fail to the normal
        # engine-native/self-heal path instead.
        if st == 400 and "checkpoint is not a live token prefix" in str(js):
            restore_outcome(False)
            _evict_generation(snap, "checkpoint/live mismatch")
            print(f"[gw] restore rejected incompatible mini/main for {snap[:12]}; self-heal", flush=True)
            return False
        # P0: even if a file passed the local stat guard, an engine-reported
        # size-limit rejection is deterministic.  Never retry it with full restorep.
        if st == 400 and ("exceeds this engine's limit" in str(js) or "exceeds the limit" in str(js)):
            metrics = _snapshot_size_metrics(snap) or {"physical_bytes": 0, "logical_bytes": 0, "limit_basis": "unknown"}
            _record_restore_size_guard(family, snap, metrics, rid=rid, decision=decision,
                                       alternative=None, stage="engine-restorepp-400")
            restore_outcome(False)
            return False
        # 400=旧栈/渲染失败 等 → 全档回落 (200/SERR 由引擎内部处理)
        restore_outcome(False)   # [giphoo] issue#3: 失败计数
        print(f"[gw] restorepp miss ({st} {str(js)[:200]}), full fallback", flush=True)
    st, js, _ = engine_req("POST", "/slots/0?action=restorep", {"filename": snap}, timeout=60, rid=rid)   # v2.2.8
    ok = st == 200
    # v2.1.9: 结构性坏档同自愈 — 400 且错误文本属永久性 (段序/格式/指纹) 时除名;
    # 503 (RAM) 与渲染类瞬时错误保留重试。761cbd 事故: 08:54 双bug合谋写坏, 400 循环选中。
    _perm = ("payload order", "another engine", "another model", "not the v2", "unsupported version",
             "directory", "header", "block", "footer", "bad v2")   # [giphoo] issue#4: footer 损坏属永久
    if st == 400 and any(k in str(js) for k in _perm):
        l2_reclaim(snap, "corrupt_400_selfheal", scope="all")   # [giphoo] issue#4-C: 坏档整组回收 (含索引)
        print(f"[gw] corrupt snapshot {snap[:12]} reclaimed (400 structural self-heal)", flush=True)
    if st == 404:
        # Remove every alias of this physical file, not just the current key.
        _evict_generation(snap, "404 self-heal")
        print(f"[gw] dead snapshot {snap[:12]} evicted from index (404 self-heal)", flush=True)
    if ok:
        _record_restore_metrics(decision, js)
        stats["restores"] += 1
        with fam_idx_lock:   # 借用即热度: touch LRU
            for fam in fam_idx.values():
                for e in fam.values():
                    if e["file"] == snap:
                        e["ts"] = time.time()
    else:
        stats["restore_fail"] += 1
        # v2.1.4: 仅【身份不匹配】才标记废弃 (模型/配置指纹变) — RAM 拒绝等瞬态
        # 失败绝不 stale, 否则真实大会话档被误杀 (d941.stale 5.86G 教训)
        msg = json.dumps(js) if js else ""
        if ("fingerprint" in msg or "another model" in msg or "configuration" in msg) \
           and os.path.exists(snap_path(snap)):
            try:
                os.rename(snap_path(snap), snap_path(snap) + ".stale")
            except OSError:
                pass
    print(f"[gw] restore {family} -> {snap}: {st} {str(js)[:80]}", flush=True)
    return ok

def log_decision(decision: dict):
    try:
        with open(DECISIONS_LOG, "a") as f:
            f.write(json.dumps(decision, ensure_ascii=False) + "\n")
    except OSError:
        pass

def intercept(family: str | None, chain: str | None = None, hashes: list | None = None,
              body: dict | None = None, req_hash: str = "", rid: str = ""):
    """请求进入前的会话切换编排 (持锁串行)。
    v2.1 身份: family(族,借用分组) + chain(链,快照身份,全消息链式哈希);
    同链延长 = 消息哈希前缀判定 (radix 语义的消息级实现, 对齐 vLLM 链式块哈希)。
    每次决策落 decisions.jsonl: {ts, fp=family, chain, req_hash, 探针读数, 四层代价, 动作}"""
    global live_fp, live_chain, live_hashes, live_dirty, _last_body
    _last_body = body
    if family is None:
        return
    hashes = hashes or []
    with state_lock:
        # 同族同链延长: 新消息列 = live 消息列的追加 → 续聊, 链身份随内容延长
        if live_fp == family and live_hashes and len(hashes) >= len(live_hashes) \
           and hashes[:len(live_hashes)] == live_hashes:
            live_hashes, live_chain = hashes, chain
            live_dirty = True
            decision = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "fp": family, "chain": chain,
                        "req_hash": req_hash, "rid": rid, "action": "live-continuation"}
            log_decision(decision)
            return decision
        now = time.time()
        last_seen[family] = now
        decision = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "fp": family, "chain": chain,
                    "req_hash": req_hash, "rid": rid}
        decision["pool_workset"] = pool_workset_metrics(now)
        if live_fp == family and live_hashes:
            decision["fork_of"] = live_chain   # 族内分叉: 同头不同枝, 各存各档
        snap = resolve_snap(family, chain, hashes)
        if live_fp is None and snap:
            t0 = time.time()
            ok = do_restore(family, chain, _last_body, rid=rid, decision=decision)     # 引擎重启: 池全灭, 磁盘是唯一来源
            decision["restorep_ms"] = round((time.time() - t0) * 1000)
            if ok:
                stats["switches"] += 1
                decision["action"] = "restore@restart"
        else:
            # v2.0/2.1.5: 引导式切换存档 — 新鲜度按【本分支祖先链】判, 不按族:
            # 同族小分支的存档/触碰不该掩护大分支 (实测: B1(4.5K) 刚存 → X(332K)
            # bootstrap 被族粒度门跳过 → 全程冷读)。按离场链名存档, 分叉互不覆盖
            if live_fp and live_dirty and live_chain:
                with fam_idx_lock:
                    anc = ancestor_snap(live_fp, live_hashes)
                    anc_ts = anc["ts"] if anc else 0
                    anc_cov = chain_prefix_length(anc.get("hashes") or [], live_hashes) if anc else 0
                # v2.1.7: 内容门 — 恢复会 touch ts (LRU 热度), 浅档被恢复后时间门被蒙蔽,
                # 深度活会话的存档被跳过 → park 又被 RAM 准入拒 → 全冷读 (Z400 442K 218s 事故)。
                # 活会话已超出祖先档覆盖 2 条消息 → 祖先护不住, 无论多新必存。
                if now - anc_ts > BOOTSTRAP_STALE_S or len(live_hashes) > anc_cov + 2:
                    t0 = time.time()
                    ok = do_save(live_fp, live_chain, live_hashes, rid=rid, decision=decision)
                    decision["bootstrap_save"] = ("ok" if ok else "fail")
                    decision["bootstrap_save_ms"] = round((time.time() - t0) * 1000)
            # 无快照 → 无决策可做 → 免探针 (省渲染+分词)
            if not snap:
                decision["action"] = "engine-native"
            else:
                snap_entry, snap_common, prefix_reuse = snapshot_prefix_plan(family, snap, hashes)
                need_prefix = bool(snap_entry and snap_common > 0 and prefix_reuse is None
                                   and snap_common < len(snap_entry.get("hashes") or []))
                (pool_best, live_lcp, n, probe_ms, prefix_tokens,
                 parked_best, parked_serial) = engine_probe(
                    _last_body or {}, rid=rid, prefix_messages=[snap_common] if need_prefix else [])
                decision["probe_ms"] = probe_ms
                decision["parked_best"] = parked_best
                decision["parked_serial"] = parked_serial
                if pool_best is None:
                    # v2.2.8: 探针失败(引擎忙/超时) → 本地索引兜底: 档覆盖估算≥门槛即恢复。
                    # 实证 (2026-10-31 12:35): 30万会话 probe 排队60s超时 → action=null 全量213s,
                    # 而池内有档、下一轮 1.8s 全命中 — 探针超时不该废掉 L2 救命路径
                    try: l2_tok = int(open(tok_path(snap)).read() or 0)
                    except OSError: l2_tok = 0
                    est_tok = sum(len(str(m.get("content", ""))) for m in
                                  ((_last_body or {}).get("messages") or [])) // 4
                    if l2_tok and est_tok and l2_tok >= est_tok * SAVE_MARGIN:
                        t0 = time.time(); ok = do_restore(family, chain, _last_body, rid=rid, decision=decision)
                        decision["restorep_ms"] = round((time.time() - t0) * 1000)
                        if ok:
                            stats["poolidx_restores"] = stats.get("poolidx_restores", 0) + 1
                            decision["action"] = "restore@pool-idx"
                    # v2.2.8 结束 — 原 v1.4 工作集压力启发式仅在本地兜底未触发时走
                    if decision.get("action") is None:
                        def pool_equiv(f):
                            s = resolve_snap(f)
                            if not s:
                                return 0
                            try: return int(open(tok_path(s)).read() or 0) * KV_POOL_BYTES_PER_TOKEN
                            except OSError: return 0
                        workset = sum(pool_equiv(f2) for f2, t in last_seen.items()
                                      if f2 != family and now - t < WORKSET_WINDOW_S)
                        if (pool_equiv(family) + workset) >= POOL_BUDGET_BYTES * WORKSET_PRESSURE:
                            t0 = time.time(); ok = do_restore(family, chain, _last_body, rid=rid, decision=decision)
                            decision["restorep_ms"] = round((time.time() - t0) * 1000)
                            if ok:
                                stats["overflow_restores"] = stats.get("overflow_restores", 0) + 1
                                decision["action"] = "restore@workset-fallback"
                else:
                    try: l2_tok = int(open(tok_path(snap)).read() or 0)
                    except OSError: l2_tok = 0
                    reuse_source = "whole-snapshot"
                    if snap_entry and snap_common < len(snap_entry.get("hashes") or []):
                        reuse_source = "index-prefix"
                        if prefix_reuse is None and snap_common > 0:
                            verified = prefix_tokens.get(str(snap_common))
                            if verified is not None:
                                prefix_reuse = int(verified)
                                with fam_idx_lock:
                                    snap_entry.setdefault("prefix_tokens", {})[str(snap_common)] = prefix_reuse
                                    fam_idx_save()
                                reuse_source = "probe-prefix"
                        if prefix_reuse is None:
                            reuse_source = "unverified-divergent"
                    elif not snap_entry:
                        reuse_source = "no-lineage"
                    l2_est = min(max(0, int(prefix_reuse or 0)), l2_tok, n or 0)
                    if snap_entry and snap_entry.get("inplace") and l2_tok:
                        # An in-place generation keeps growing under alias names;
                        # lineage can undercount what the current file now holds.
                        l2_est = min(l2_tok, n or l2_tok)
                        reuse_source = "inplace-latest"
                    l2_read = l2_est if os.environ.get("STRATA_RESTORE_MODE", "restore") == "restorepp" else l2_tok
                    # L2 池感知 v3 (正解, 先于成本计算): 大档自动借用不可靠, pool_best 打折喂模型
                    POOL_TRUST_K = int(os.environ.get("STRATA_POOL_TRUST_K", "150"))
                    pool_best_eff = (pool_best or 0)
                    if n and n > POOL_TRUST_K * 1000:
                        pool_best_eff = int(pool_best_eff * 0.3)
                    costs = layer_costs(n or 0, live_lcp or 0, pool_best_eff, l2_est, l2_read=l2_read)
                    legacy_costs = layer_costs(n or 0, live_lcp or 0, pool_best_eff, min(l2_tok, n or 0))
                    old_restore, new_restore, best_without_l2 = restore_cost_policy(
                        costs, pool_best_eff, n or 0, legacy_costs=legacy_costs)
                    use_min_abd = os.environ.get("STRATA_RESTORE_MIN_ABD", "1") == "1"
                    restore_now = new_restore if use_min_abd else old_restore
                    # [giphoo] PARKED-first: if the engine already has a matching
                    # parked image, let the normal GEN path take/borrow it. Do not
                    # run RESTORE and destroy the outgoing live state.
                    if parked_best and n and parked_best >= n * 0.80 and parked_serial:
                        if cache_pin(parked_serial, True):
                            decision["preload_serial"] = parked_serial
                        decision["parked_hit"] = True
                        decision["action"] = "engine-native"
                        restore_now = False
                    decision.update({"n": n, "live_lcp": live_lcp, "pool_best": pool_best,
                                     "parked_best": parked_best, "parked_serial": parked_serial,
                                     "l2_est": l2_est, "l2_reuse": l2_est, "l2_read": l2_read,
                                     "l2_common_messages": snap_common, "l2_reuse_source": reuse_source,
                                     "costs": {k: round(v) for k, v in costs.items()},
                                     "restore_shadow": {"old_restore": old_restore, "new_restore": new_restore,
                                                        "new_action": "restorep@cost" if new_restore else "engine-native",
                                                        "old_costs": {k: round(v) for k, v in legacy_costs.items()},
                                                        "best_without_l2": round(best_without_l2)},
                                     "restore_rule": "min_abd" if use_min_abd else "legacy_pool50"})
                    decision["pool_trust_discounted"] = (pool_best_eff != (pool_best or 0))
                    if snap_entry:
                        snapshot_messages = len(snap_entry.get("hashes") or [])
                        borrow_kind = _classify_small_borrow(
                            int(n or 0), int(l2_tok or 0), int(l2_est or 0),
                            int(snap_common or 0), snapshot_messages)
                        if borrow_kind:
                            decision["borrow_kind"] = borrow_kind
                            hotspot = _remember_borrow_hotspot(family, int(snap_common or 0),
                                                               int(l2_est or 0))
                            decision["borrow_hotspot_hits"] = hotspot.get("hits")
                            hint = _inject_borrow_prefix_hint(body, family, hotspot, int(n or 0))
                            if hint:
                                decision["prefix_hint"] = hint
                                decision["prefix_hint_injected"] = True
                    if restore_now:
                        t0 = time.time(); ok = do_restore(family, chain, _last_body, rid=rid, decision=decision)
                        decision["restorep_ms"] = round((time.time() - t0) * 1000)
                        if ok:
                            stats["overflow_restores"] = stats.get("overflow_restores", 0) + 1
                            stats["restore_cost_restores"] = stats.get("restore_cost_restores", 0) + 1
                            decision["action"] = "restorep@cost"
                    else:
                        decision["action"] = "engine-native"
        if "action" not in decision and "restorep_ms" in decision and decision.get("restorep_ms") is not None:
            decision.setdefault("action", "engine-native")
        log_decision(decision)
        live_fp = family
        live_chain, live_hashes = chain, hashes
        live_dirty = False
        return decision

ARCHIVE_DIR = SNAP_DIR + "/.state-archive"
ARCHIVE_DEBOUNCE_S = float(os.environ.get("STRATA_L2_ARCH_DEBOUNCE_S", "30"))
ARCHIVE_MIN_GAP_S = float(os.environ.get("STRATA_L2_ARCH_GAP_S", "600"))

def _archived_marker(hot_name: str) -> str:
    return HOT_DIR + "/." + hot_name + ".archived"

def archive_debounced():
    import time as _t, shutil
    now = _t.time()
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    try:
        states = [f for f in os.listdir(HOT_DIR) if f.endswith(".state")]
    except OSError:
        return
    for f in states:
        p = os.path.join(HOT_DIR, f)
        mk = _archived_marker(f)
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        if now - mt < ARCHIVE_DEBOUNCE_S:
            continue
        try:
            if os.path.exists(mk) and os.path.getmtime(mk) >= mt:
                continue
        except OSError:
            continue
        try:
            if os.path.exists(mk) and now - os.path.getmtime(mk) < ARCHIVE_MIN_GAP_S:
                continue
        except OSError:
            continue
        tmp = os.path.join(ARCHIVE_DIR, f + ".tmp")
        dst = os.path.join(ARCHIVE_DIR, f)
        try:
            shutil.copyfile(p, tmp)
            os.replace(tmp, dst)
            with open(mk, "w") as kf:
                kf.write(str(int(now)))
            with open(LIFECYCLE_LOG, "a") as lf:
                lf.write(json.dumps({"ts": _t.strftime("%F %T"), "archive": f,
                                     "size_mb": os.path.getsize(dst) // 1048576}) + "\n")
        except OSError as e:
            print(f"[gw2] archive {f}: {e}", flush=True)

def l2_reclaim(chain_file: str, why: str, protect: set = None, scope: str = "all") -> bool:
    """[giphoo] issue#4-C: 统一回收入口 — 候选保护校验 → 删整组文件 → 同步索引 → 记录原因.
    所有实际删除必须经此函数 (hot/cold/fam/global 四入口合一).
    chain_file: 主档文件名 (不含路径; hot 淘汰传 .state 名). protect: 额外保护集合.
    scope: "all"=整组(主档+mini+archive+tok+marker) / "hot"=仅 hot mini (主档保留)."""
    if not chain_file or chain_file.startswith("."):
        return False
    protect = protect or set()
    base = chain_file[:-6] if chain_file.endswith(".state") else chain_file
    main = main_path(base)
    tok = tok_path(base)
    # ---- 保护校验 ----
    if base in protect or chain_file in protect:
        return False
    # 非 writer pending: 该档有未完成提交则拒删 (引擎未就绪时跳过 — 启动期不阻塞)
    try:
        st, js, _ = engine_req("GET", "/slots", timeout=1)
        if st == 200 and isinstance(js, list) and js:
            for c in js[0].get("l2_commits", []):
                if os.path.basename(str(c.get("path") or "")) == base and c.get("status") == 0:
                    return False   # writer pending, 拒删
    except Exception:
        pass   # 引擎未就绪: 跳过 pending 检查 (启动期回收安全 — 无 writer 活动)
    hot = os.path.join(HOT_DIR, base + ".state")
    if scope == "all":
        # 整组删除时 hot mini 必须有有效已提交备份才可删 (唯一最新态保护)
        if os.path.exists(hot) and not os.path.exists(_archived_marker(base + ".state")):
            return False
    else:
        # hot-only: 同样要求已存档
        if not os.path.exists(_archived_marker(base + ".state")):
            return False
    if GC_DRY_RUN:
        _log_gc_plan(base, why, scope)
        return False
    # ---- 删文件组 (按 scope) ----
    size = 0
    paths = []
    if scope == "all":
        paths = [main, tok, hot, hot + ".tok",
                 os.path.join(ARCHIVE_DIR, base + ".state"), _archived_marker(base + ".state")]
    else:   # hot-only: 只删 hot mini 与其 tok (archive 备份保留供复活)
        paths = [hot, hot + ".tok"]
    for p in paths:
        try:
            if os.path.exists(p):
                if p == main:
                    size = archive_bytes(p)
                    archive_unlink(p)
                else:
                    os.unlink(p)
        except OSError:
            pass
    # ---- 同步索引 (scope=all 才动 fam_idx) ----
    if scope == "all":
        with fam_idx_lock:
            for fam_name, fam in list(fam_idx.items()):
                for chain, e in list(fam.items()):
                    if e.get("file") == base:
                        fam.pop(chain, None)
        fam_idx_save()
    # ---- 记录 ----
    try:
        with open(LIFECYCLE_LOG, "a") as lf:
            lf.write(json.dumps({"ts": time.strftime("%F %T"), "reclaim": base, "scope": scope,
                                 "why": why, "size_mb": size // 1048576}) + "\n")
    except OSError:
        pass
    return True

def hot_evict_lru():
    """L2 2.0: RAM 活层 LRU 淘汰 — 超 48G 时删最旧档 (SSD 有快照, read-through 可回补)"""
    import time as _t
    HOT_AGE_MAX_S = int(os.environ.get("STRATA_L2_HOT_AGE_S", "7200"))   # .state 年龄上限 (默认 2h)
    try:
        now = _t.time()
        snaps = [f for f in os.listdir(HOT_DIR) if not f.endswith(".tok") and not f.startswith(".")]
        files = [(os.path.getmtime(os.path.join(HOT_DIR, f)), f) for f in snaps]
        files.sort()
        total = sum(os.path.getsize(os.path.join(HOT_DIR, f)) for _, f in files)
        limit = int(os.environ.get('STRATA_L2_HOT_LIMIT_G', '8')) * 1073741824
        kept = 0
        for mt, f in files:
            p = os.path.join(HOT_DIR, f)
            size = os.path.getsize(p)
            age_s = now - mt
            why = None
            archived = os.path.exists(_archived_marker(f)) and \
                       os.path.getmtime(_archived_marker(f)) >= mt   # 标记新鲜 = SSD 有份
            if not archived:
                kept += 1
                continue                                             # 未存档坚决不删 (唯一最新态)
            if age_s > HOT_AGE_MAX_S and len(files) - kept > 3:
                why = f"age {age_s/60:.0f}min archived"
            elif total > limit and len(files) - kept > 3:
                why = f"cap {total/2**30:.1f}G archived"
            if not why:
                kept += 1
                continue
            # [giphoo] issue#4-C: 统一回收 (hot mini 已 archived, 只删 hot 层; 主档不在此)
            if l2_reclaim(f, why, scope="hot"):
                total -= size
                with open(LIFECYCLE_LOG, "a") as lf:   # 生命周期审计
                    lf.write(json.dumps({"ts": _t.strftime("%F %T"), "evict": f, "why": why,
                                         "size_mb": size // 1048576, "lived_min": round(age_s / 60)}) + "\n")
        with open(LIFECYCLE_LOG, "a") as lf:       # 水位快照
            lf.write(json.dumps({"ts": _t.strftime("%F %T"), "watermark": True,
                                 "files": len(files), "total_g": round(total / 2**30, 2)}) + "\n")
    except OSError as e:
        print(f"[gw2] hot lifecycle: {e}", flush=True)

def engine_cgroup_io() -> dict:
    """Cumulative block-device I/O for all processes in the engine cgroup."""
    cg = "/sys/fs/cgroup/system.slice/strata-flash.service"
    totals = {"processes": 0, "process_read_bytes": 0, "process_write_bytes": 0}
    try:
        with open(cg + "/cgroup.procs") as pf:
            pids = [p.strip() for p in pf if p.strip()]
    except OSError:
        return totals
    for pid in pids:
        try:
            vals = {}
            with open(f"/proc/{pid}/io") as io_file:
                for line in io_file:
                    key, _, value = line.partition(":")
                    if key in ("read_bytes", "write_bytes"):
                        vals[key] = int(value.strip())
            if vals:
                totals["processes"] += 1
                totals["process_read_bytes"] += vals.get("read_bytes", 0)
                totals["process_write_bytes"] += vals.get("write_bytes", 0)
        except (OSError, ValueError):
            continue
    return totals

def engine_cgroup_metrics_loop():
    """P1: 30s engine cgroup IO + memory baseline for daily attribution."""
    import time as _t
    while True:
        _t.sleep(30)
        append_event_log(ENGINE_CGROUP_LOG, {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "io": engine_cgroup_io(),
            "memory": memory_pressure_snapshot(),
        })

def cold_l2_lru():
    """L2-2 main-file LRU in the engine root; dry-run by default."""
    global _cold_dry_last_s
    now = time.time()
    if GC_DRY_RUN and now - _cold_dry_last_s < 300:
        return
    if GC_DRY_RUN:
        _cold_dry_last_s = now
    limit = int(os.environ.get('STRATA_L2_COLD_LIMIT_G', '150')) * 1073741824
    try:
        snaps = [f for f in os.listdir(ENGINE_SNAP_DIR)
                 if len(f) == 16 and all(c in "0123456789abcdef" for c in f)
                 and archive_exists(main_path(f))]
        with fam_idx_lock:
            protected = {e.get("file") for fam in fam_idx.values() for e in fam.values()}
        files = [(archive_time(main_path(f), atime=True), f) for f in snaps]
        files.sort()
        total = sum(archive_bytes(main_path(f)) for _, f in files)
        dropped = 0
        for at, f in files:
            if total <= limit or len(files) - dropped <= 20:
                break
            if f in protected:
                continue
            sz = archive_bytes(main_path(f))
            if l2_reclaim(f, "cold_lru", scope="all"):
                total -= sz
                dropped += 1
        if dropped:
            print(f"[gw2] cold L2 LRU evicted {dropped}, now {total/2**30:.1f}G", flush=True)
    except OSError as e:
        print(f"[gw2] cold LRU: {e}", flush=True)

def ssd_snapshotter():
    """L2 v3: L2-1 状态机巡查 (防抖备份 → 按标记回收) + 心跳"""
    while True:
        time.sleep(20)
        archive_debounced()
        hot_evict_lru()
        cold_l2_lru()
        with open(HOT_DIR + "/.snapshot-heartbeat", "w") as hb:
            hb.write(str(int(time.time())))

def on_request_done():
    global live_dirty, last_done_ts
    with state_lock:
        live_dirty = True
        last_done_ts = time.time()
    save_wakeup.set()

def idle_refresher():
    """引擎空闲时为脏会话刷快照 (限制频率)。
    v1.9: save 在 state_lock 外执行 — 174K 会话存档 1.6-2.9s, 锁内执行会
    堵住并发请求的决策阶段 (实测 decision_ms 1660-2947ms 异常)。锁内只做
    资格复查+占位, 保存代际校验防与切换竞态。"""
    global live_dirty
    while True:
        delay = max(0.5, IDLE_GRACE_S - (time.time() - last_done_ts))
        save_wakeup.wait(min(IDLE_POLL_S, delay))
        save_wakeup.clear()
        with state_lock:
            fp, ch, hs, dirty, elapsed = live_fp, live_chain, list(live_hashes), live_dirty, time.time() - last_save_ts
        if fp and ch and dirty and elapsed >= REFRESH_S and time.time() - last_done_ts >= IDLE_GRACE_S:
            # SAVE has its own reserve and the engine evicts parked entries before
            # refusing.  Do not let generic background backpressure starve the
            # archive first: the engine's fail-closed admission owns this decision.
            pressure = memory_backpressure("idle_save")
            if not pressure["ok"]:
                print(f"[gw] idle save under pressure: {','.join(pressure['reasons'])}; "
                      "allow engine save admission/eviction", flush=True)
            st, js, _ = engine_req("GET", "/slots", timeout=10)
            busy = False
            if st == 200 and isinstance(js, list) and js:
                busy = bool(js[0].get("is_processing"))
            if not busy:
                with state_lock:
                    if live_fp != fp or live_chain != ch or not live_dirty:   # 窗口期变了主人: 放弃
                        continue
                    live_dirty = False                        # 占位: 防重复触发
                do_save(fp, ch, hs)                           # 锁外! 引擎FIFO自会与请求排队

# v2.2.1 全量消息留存: 每请求落消息哈希索引行, 新哈希的完整内容存 sidecar —
# 事后可 diff 任意两次请求的消息级差异 (前缀失效/压缩重写取证), 磁盘只存唯一内容
def eng_mem() -> dict:
    """v2.2.2: 引擎 cgroup 内存构成采样 (请求级) — file_mapped 增长归因到请求"""
    try:
        st = {}
        for l in open("/sys/fs/cgroup/system.slice/strata-flash.service/memory.stat"):
            k2, _, v2 = l.partition(" ")
            if k2 in ("anon", "file_mapped", "shmem", "active_file", "inactive_file"):
                st[k2] = round(int(v2) / 2**30, 1)
        st["cur"] = round(int(open("/sys/fs/cgroup/system.slice/strata-flash.service/memory.current").read()) / 2**30, 1)
        return st
    except OSError:
        return {}

def health_watcher():
    """引擎重启检测 → 快照成为唯一真相, 下个请求按需恢复"""
    global live_fp, live_chain, live_hashes, live_dirty
    was_ok = True
    while True:
        time.sleep(HEALTH_POLL_S)
        st, _, _ = engine_req("GET", "/health", timeout=5)
        if st != 200 and was_ok:
            with state_lock:
                print(f"[gw] engine down: live_fp={live_fp} 清空, 快照接管", flush=True)
                live_fp, live_dirty = None, False
                live_chain, live_hashes = None, []
        was_ok = st == 200

# ===================== [giphoo] issue#3: 流水线调度器 (单放行门 + SJF 队列) =====================
# P1 2026-11-09: decode cost uses observed family completion history.  The old
# max_tokens*20ms value was 327.68s for almost every request (~56x the observed
# wall), so SJF carried pseudo-precision and reordered only once in a full day.
SCHED_COMPLETION_HISTORY = {}
SCHED_COMPLETION_HISTORY_N = int(os.environ.get("STRATA_SCHED_COMPLETION_HISTORY_N", "32"))

def sched_expected_completion_tokens(family: str | None, max_tokens: int) -> int:
    values = SCHED_COMPLETION_HISTORY.get(family or "", [])
    cap = max(1, int(max_tokens or 2048))
    if len(values) >= 8:
        expected = max(256, int(percentile_local(values, 0.9)))
    elif len(values) >= 3:
        expected = max(256, int(percentile_local(values, 0.5) * 1.5))
    else:
        expected = int(os.environ.get("STRATA_SCHED_DECODE_EXPECT_TOKENS", "512"))
    return min(cap, expected)

def percentile_local(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))]

def sched_observe_completion(family: str | None, completion_tokens) -> None:
    try:
        value = int(completion_tokens or 0)
    except (TypeError, ValueError):
        return
    if value <= 0:
        return
    values = SCHED_COMPLETION_HISTORY.setdefault(family or "", [])
    values.append(value)
    if len(values) > SCHED_COMPLETION_HISTORY_N:
        del values[:-SCHED_COMPLETION_HISTORY_N]


def classify_gateway_wait(decision: dict | None, queue_wait_ms: int,
                          gateway_wait_ms: int, decision_ms: int) -> str:
    """Low-cardinality wait attribution stored in the existing request row."""
    decision = decision or {}
    nonqueue_ms = max(0, gateway_wait_ms - queue_wait_ms)
    if decision.get("save_commit_state") == "serial-wait-timeout":
        return "save_serial_timeout"
    # [giphoo] waitfix: queue 主导必须先归因。原实现先扫 nonqueue 阶段, 只要
    # bootstrap_save/probe >= nonqueue*0.5 即返回, 即使 queue_wait 占 99% 也被
    # 掩盖 (11-11 巡检 171 条 bootstrap_save / 73 条 probe 实为单放行排队)。
    # 排队占绝对多数时 nonqueue 阶段只是伴随噪声, 不得抢占 scheduler_queue。
    if gateway_wait_ms > 0 and queue_wait_ms >= gateway_wait_ms * 0.8:
        return "scheduler_queue"
    candidates = (
        ("save_serial", "save_serial_wait_ms"),
        ("restore_pending", "restore_pending_wait_ms"),
        ("restore", "restorep_ms"),
        ("bootstrap_save", "bootstrap_save_ms"),
        ("probe", "probe_ms"),
    )
    for label, key in candidates:
        value = decision.get(key)
        if isinstance(value, (int, float)) and value > 0 and value >= nonqueue_ms * 0.5:
            return label
    if decision_ms > 0 and decision_ms >= nonqueue_ms * 0.8:
        return "decision_other"
    return "mixed" if gateway_wait_ms > 0 else "none"


class SchedRequest:
    """调度请求: 优先级 = est_cost − aging×wait_s; live 高优先但受有界公平约束."""
    __slots__ = ("rid", "fp", "chain", "body", "hashes", "est_cost", "enqueue_ts", "max_tokens",
                 "turn_event", "done_event", "sched_action", "admit_ts", "wfile", "est_prompt",
                 "cover", "prefill_est", "restore_est", "decode_est", "is_live", "is_starved", "sched_cost",
                 "queue_depth_at_enqueue", "queue_depth_max", "engine_busy_at_enqueue", "blocking_request_ids",
                 "live_admit_baseline", "live_overtakes", "live_forced")
    def __init__(self, fp, chain, body, hashes, rid=""):
        self.rid = rid; self.fp = fp; self.chain = chain; self.body = body; self.hashes = hashes
        self.enqueue_ts = time.time()
        self.turn_event = threading.Event()   # 放行信号
        self.done_event = threading.Event()   # 完成信号
        self.max_tokens = (body or {}).get("max_tokens", 2048)
        self.est_prompt = self.cover = self.prefill_est = self.restore_est = self.decode_est = 0
        self.is_live = self.is_starved = False
        self.est_cost = self._estimate()
        self.sched_action = ""
        self.sched_cost = self.est_cost
        self.admit_ts = 0
        self.wfile = None
        self.queue_depth_at_enqueue = None
        self.queue_depth_max = None
        self.engine_busy_at_enqueue = None
        self.blocking_request_ids = []
        self.live_admit_baseline = 0
        self.live_overtakes = 0
        self.live_forced = False
    def _estimate(self):
        """零探测代价模型: 本地 .tok 索引 + 消息长度估算 (issue#3 §2)"""
        msgs = (self.body or {}).get("messages") or []
        # 2026-11-12: 改 UTF-8 字节口径。中文 1 字≈3 字节≈1.1-1.3 tok, 英文 4 字符≈4 字节≈1 tok,
        # 字节//4 在两种语言都接近; 旧字符//4 在中文流量下实测中位低估 3.63x (p90 5.58x, max 14.9x),
        # 导致 est_prompt 与 prefill_est 系统性偏小、SJF 把大 prefill 请求当成小请求。
        self.est_prompt = sum(len(str(m.get("content", "")).encode("utf-8")) for m in msgs) // 4
        expected_completion = sched_expected_completion_tokens(self.fp, self.max_tokens)
        self.decode_est = min(expected_completion * DECODE_MS_PER_TOKEN,
                              int(os.environ.get("STRATA_SCHED_DECODE_EST_MAX_MS", "180000")))
        # live-continuation retains the lowest normal bucket, but Scheduler's
        # fairness gate can temporarily exclude it when a non-live request ages.
        if live_fp == self.fp and live_hashes and len(self.hashes) >= len(live_hashes) \
           and self.hashes[:len(live_hashes)] == live_hashes:
            self.is_live = True
            return -1e9
        snap = resolve_snap(self.fp, self.chain, self.hashes)
        if snap:
            try: l2_tok = int(open(tok_path(snap)).read() or 0)
            except OSError: l2_tok = 0
            self.cover = min(l2_tok, self.est_prompt)
            # 2026-11-12: cover 是 L2 快照口径, 引擎实际可复用前缀可能远小于它
            # (实测 cover/actual_cached 最高 3433x; pool-idx 案例 cover=78889 实际 cached=26103),
            # 全覆盖时不允许 prefill_est=0, 至少按 MIN_FRESH_RATIO 计 fresh, 防止过度自信。
            min_fresh_ratio = float(os.environ.get("STRATA_SCHED_MIN_FRESH_RATIO", "0.15"))
            fresh = max(self.est_prompt - self.cover, int(self.est_prompt * min_fresh_ratio))
            self.prefill_est = fresh * PREFILL_MS_PER_TOK
            self.restore_est = 5000
        else:
            self.prefill_est = self.est_prompt * PREFILL_MS_PER_TOK
        return self.prefill_est + self.restore_est + self.decode_est
    def priority(self, now):
        if self.is_live:
            return self.est_cost
        wait_s = now - self.enqueue_ts
        self.is_starved = wait_s > float(os.environ.get("STRATA_SCHED_STARVE_S", "60"))
        if self.is_starved:
            return -1e10
        aging = float(os.environ.get("STRATA_SCHED_AGING", "1000"))   # ms/s
        return self.est_cost - aging * wait_s

class Scheduler:
    """单放行门: 一次一个请求到引擎; SJF+aging 重排; 队列异常=直通透传"""
    def __init__(self):
        self.cond = threading.Condition()
        self.queue = []   # list[SchedRequest]
        self.busy = False
        self.current = None
        self.enabled = os.environ.get("STRATA_SCHED_ENABLED", "1") == "1"
        self.stats = {"enqueued": 0, "reordered": 0, "starved_rescue": 0, "live_admitted": 0,
                      "bypassed": 0, "fairness_rescue": 0, "memory_warm_deferred": 0, "live_forced": 0}
        self.live_burst = 0
        self.live_admit_counter = 0
        self.wait_window = []   # [giphoo] issue#3 观测: 最近 100 条 sched_wait_ms (滚动)
        self.depth_samples = []  # queue_depth 时序
    def submit(self, req: SchedRequest):
        """入队并等放行; 返回后请求线程执行 intercept+forward; 完成后调 done()"""
        if not self.enabled:
            self.stats["bypassed"] += 1
            return   # 直通
        with self.cond:
            req.queue_depth_at_enqueue = len(self.queue)
            req.engine_busy_at_enqueue = self.busy
            req.blocking_request_ids = [self.current.rid] if self.busy and self.current else []
            req.live_admit_baseline = self.live_admit_counter
            req.sched_action = "enqueued"
            self.queue.append(req)
            self.stats["enqueued"] += 1
            self.cond.notify_all()
        # [giphoo] issue#3: wait for admission without writing SSE bytes.
        # A queued request has not sent its HTTP status line yet; a raw
        # ": heartbeat" makes OpenAI/aiohttp clients fail with BadStatusLine.
        # LiteLLM already keeps the downstream SSE client alive.
        req.turn_event.wait()
    def done(self, req: SchedRequest):
        req.done_event.set()
    def _warm_next(self):
        """issue#2 WARM: 当前请求执行时, 对队列里下一个请求的档做 fadvise WILLNEED 预热"""
        if not self.queue:
            return
        # P1: readahead deliberately grows page cache; skip it under pressure.
        if memory_backpressure("warm_fadvise")["ok"] is False:
            self.stats["memory_warm_deferred"] += 1
            return
        now = time.time()
        best = min(self.queue, key=lambda r: r.priority(now))
        snap = resolve_snap(best.fp, best.chain, best.hashes)
        if not snap:
            return
        path = snap_path(snap)
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                print(f"[warm] {os.path.basename(path)} fd={fd} (next={best.fp[:8]})", flush=True)
            finally:
                os.close(fd)
        except OSError:
            pass

    def gate_loop(self):
        while True:
            with self.cond:
                while not self.queue or self.busy:
                    self.cond.wait(timeout=1)
                    if not self.queue:
                        continue
                    if self.busy:
                        continue
                now = time.time()
                # P1 bounded live priority: live continuation remains preferred,
                # but not infinitely.  Aged non-live work or a live burst forces
                # one non-live admission so interactive switches cannot wait 300s.
                # H2 fix: the original implementation, when fairness_due, fully
                # 完全踢出候选 (eligible_idx=nonlive_idx)。长生成积压令
                # oldest_nonlive_wait 恒 > LIVE_MAX_WAIT_S, fairness_due 永久为真,
                # live(sched_cost=-1e9) 被无限期排除 —— 12:08 实测 live 排队 286s。
                # 现改"限流"而非"排除": live 等待 >= LIVE_FORCE_S 时强制让其参与竞争,
                # 其余时间仍只放 non-live; non-live 侧另有 starve(-1e10) 兜底防饿死。
                nonlive_idx = [i for i, r in enumerate(self.queue) if not r.is_live]
                live_idx = [i for i, r in enumerate(self.queue) if r.is_live]
                live_available = bool(live_idx)
                if nonlive_idx:
                    oldest_nonlive_wait = max(now - self.queue[i].enqueue_ts for i in nonlive_idx)
                else:
                    oldest_nonlive_wait = 0
                if live_idx:
                    oldest_live_wait = max(now - self.queue[i].enqueue_ts for i in live_idx)
                else:
                    oldest_live_wait = 0.0
                max_burst = int(os.environ.get("STRATA_SCHED_LIVE_MAX_BURST", "4"))
                max_live_wait = float(os.environ.get("STRATA_SCHED_LIVE_MAX_WAIT_S", "10"))
                live_force_s = float(os.environ.get("STRATA_SCHED_LIVE_FORCE_S", "30"))
                live_forced = bool(live_idx and oldest_live_wait >= live_force_s)
                fairness_due = bool(nonlive_idx and live_available and not live_forced and (
                self.live_burst >= max_burst or oldest_nonlive_wait >= max_live_wait))
                eligible_idx = nonlive_idx if fairness_due else range(len(self.queue))
                # SJF: 按 priority 取最优 (含 aging); SCHED_FIFO -> 队首直通
                best_idx = next(iter(eligible_idx))
                best_pri = 0.0
                if not SCHED_FIFO:
                    best_pri = self.queue[best_idx].priority(now)
                    for i in list(eligible_idx)[1:]:
                        p = self.queue[i].priority(now)
                        if p < best_pri:
                            best_pri = p; best_idx = i
                req = self.queue.pop(best_idx)
                req.sched_cost = best_pri
                req.live_overtakes = self.live_admit_counter - req.live_admit_baseline
                req.live_forced = bool(live_forced and req.is_live)
                fairness_triggered = bool(fairness_due and req.sched_action == "enqueued")
                if req.is_live:
                    req.sched_action = "live_continuation"
                    self.stats["live_admitted"] += 1
                    if req.live_forced:
                        self.stats["live_forced"] += 1
                    self.live_burst += 1
                    self.live_admit_counter += 1
                else:
                    self.live_burst = 0
                    if fairness_triggered:
                        req.sched_action = "fairness_rescue"
                        self.stats["fairness_rescue"] += 1
                    elif req.is_starved:
                        req.sched_action = "starve_rescue"
                        self.stats["starved_rescue"] += 1
                    elif best_idx != 0 and req.est_cost > 0:
                        req.sched_action = "sjf_reorder"
                        self.stats["reordered"] += 1
                self.busy = True
                self.current = req
                req.queue_depth_max = max(req.queue_depth_at_enqueue or 0, len(self.queue))
            # 调度决策落日志
            try:
                with open(DECISIONS_LOG, "a") as f:
                    f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                        "scheduler": True, "rid": req.rid, "fp": req.fp, "chain": req.chain,
                                        "action": req.sched_action, "est_cost": round(req.est_cost),
                                        "sched_cost": round(req.sched_cost), "est_prompt": req.est_prompt,
                                        "cover": req.cover, "prefill_est": round(req.prefill_est),
                                        "restore_est": round(req.restore_est), "decode_est": round(req.decode_est),
                                        "is_live": req.is_live, "is_starved": req.is_starved,
                                         "live_burst": self.live_burst,
                                         "fairness_due": fairness_due,
                                         "live_forced": live_forced,
                                         "oldest_live_wait_s": round(oldest_live_wait, 3),
                                         "oldest_nonlive_wait_s": round(oldest_nonlive_wait, 3),
                                         "live_overtakes": req.live_overtakes,
                                         "wait_s": round(now - req.enqueue_ts, 1),
                                         "queue_depth": len(self.queue),
                                         "queue_depth_at_enqueue": req.queue_depth_at_enqueue,
                                         "queue_depth_max": req.queue_depth_max,
                                         "engine_busy_at_enqueue": req.engine_busy_at_enqueue,
                                         "blocking_request_ids": req.blocking_request_ids}) + "\n")
            except OSError:
                pass
            req.admit_ts = time.time() # 先记放行时刻，再唤醒，避免请求线程读到 0/负 queue_wait
            req.turn_event.set()       # 放行
            self._warm_next()          # issue#2: 预热队列里下一个请求的档
            req.done_event.wait()      # 等完成 (锁外)
            with self.cond:
                self.busy = False
                self.current = None
                self.cond.notify_all()
            # [giphoo] issue#3 观测: 等待时延滚动窗口 (纯排队时间, 不含执行)
            w_ms = ((req.admit_ts or time.time()) - req.enqueue_ts) * 1000
            self.wait_window.append(w_ms)
            if len(self.wait_window) > 100:
                self.wait_window.pop(0)

scheduler = Scheduler()

def sched_metrics_loop():
    """[giphoo] issue#3 观测: 调度器指标 30s 聚合落盘 — queue_depth/sched_wait P50P90/重排率"""
    import time as _t
    while True:
        _t.sleep(30)
        try:
            w = sorted(scheduler.wait_window)
            p50 = w[len(w)//2] if w else 0
            p90 = w[int(len(w)*0.9)] if w else 0
            with open(os.path.join(_LOG_DIR, "sched-metrics.jsonl"), "a") as f:
                f.write(json.dumps({"ts": time.strftime("%F %T"),
                                    "queue_depth": len(scheduler.queue),
                                    "busy": scheduler.busy,
                                    "wait_p50_ms": round(p50), "wait_p90_ms": round(p90),
                                    "wait_samples": len(w),
                                    **scheduler.stats}) + "\n")
        except OSError:
            pass

# [giphoo] issue#3 thrashing 检测: 同档 5min 内 restorep >2 次告警
_restore_history = {}   # {snap: [ts, ...]}

def thrash_check(snap: str):
    now = time.time()
    h = _restore_history.setdefault(snap, [])
    h.append(now)
    h[:] = [t for t in h if now - t < 300]   # 5min 窗
    if len(h) > 2:
        print(f"[gw] THRASH 告警: {snap[:12]} 5min 内恢复 {len(h)} 次 (池驱逐循环)", flush=True)
        try:
            with open(os.path.join(_LOG_DIR, "l2-alarms.log"), "a") as f:
                f.write(f"{time.strftime('%F %T')} THRASH {snap[:12]} restores={len(h)} in 5min\n")
        except OSError:
            pass

# restorep 成功率趋势 (30s 聚合)
_restore_outcomes = {"ok": 0, "fail": 0}

def restore_outcome(ok: bool):
    _restore_outcomes["ok" if ok else "fail"] += 1

def restore_metrics_loop():
    import time as _t
    while True:
        _t.sleep(30)
        try:
            o, f_ = _restore_outcomes["ok"], _restore_outcomes["fail"]
            if o + f_ > 0:
                with open(os.path.join(_LOG_DIR, "restore-metrics.jsonl"), "a") as f:
                    f.write(json.dumps({"ts": time.strftime("%F %T"), "ok": o, "fail": f_,
                                        "ok_rate": round(o / (o + f_), 3)}) + "\n")
                _restore_outcomes["ok"] = _restore_outcomes["fail"] = 0
        except OSError:
            pass

def usage_from_tail(tail: bytes):
    """响应尾部的 usage（L2 请求 timeline 观测；不做用量统计持久化）"""
    try:
        m = re.search(rb'"usage"\s*:\s*', tail)
        if not m:
            return None
        u, _ = json.JSONDecoder().raw_decode(tail[m.end():].decode(errors="replace"))
        return u if isinstance(u, dict) and ("prompt_tokens" in u or "completion_tokens" in u) else None
    except Exception:
        return None

def finish_reason_from_tail(tail: bytes):
    """响应尾部最后一个非空 finish_reason（流式末块/非流式 choices 均可）。"""
    try:
        text = tail.decode(errors="replace")
        found = None
        for m in re.finditer(r'"finish_reason"\s*:\s*("(?:[^"\\]|\\.)*"|null)', text):
            value = json.loads(m.group(1))
            if value is not None:
                found = value
        return found
    except Exception:
        return None

def timings_from_tail(tail: bytes):
    """响应尾部最后一个引擎 timings（流式末块与非流式同口径）。"""
    try:
        matches = list(re.finditer(rb'"timings"\s*:\s*', tail))
        for m in reversed(matches):
            try:
                t, _ = json.JSONDecoder().raw_decode(tail[m.end():].decode(errors="replace"))
                if isinstance(t, dict) and ("prompt_ms" in t or "predicted_ms" in t or "stage_ledger" in t):
                    return t
            except Exception:
                continue
        return None
    except Exception:
        return None


def first_model_delta_kind(buf: bytes):
    """Return the first generated SSE delta kind in a buffered prefix.

    Legacy `ttft_ms` marks any `data:` frame, including an initial role-only
    delta.  This helper parses SSE JSON until content/reasoning/tool output is
    actually present, giving a user-meaningful first-token timestamp without a
    GPU sync.  Parsing stops once detected and the buffer is bounded.
    """
    try:
        text = buf.decode(errors="replace")
        for match in re.finditer(r"(?im)^data:\s*(.*?)[\r\n]+$", text):
            payload = match.group(1).strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            for choice in obj.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                for key in ("reasoning_content", "content", "tool_calls", "function_call"):
                    value = delta.get(key)
                    if isinstance(value, str) and value.strip():
                        return key
                    if isinstance(value, (list, dict)) and value:
                        return key
                text_value = choice.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    return "text"
        return None
    except Exception:
        return None


def rewrite_tool_args(buf: bytes) -> bytes:
    """v2.2.9 (ported back from the legacy session-gateway; lost in the L2-2.0 rewrite):
    strata 引擎 tool_calls.arguments 把中文输出为 \\uXXXX 转义串 (合法 JSON 但客户端直显乱码) —
    非流式整包解析重写为 UTF-8 直出; 流式由 StreamToolUnescaper 增量转换。
    双层 parse→re-dump (非字节替换), 结构/控制转义由序列化器自动保留, 任何异常原样返回 (fail-open)。"""
    try:
        d = json.loads(buf)
        hit = False
        for ch in d.get("choices") or []:
            for tc in (ch.get("message") or {}).get("tool_calls") or []:
                f = tc.get("function") or {}
                a = f.get("arguments")
                if isinstance(a, str) and "\\u" in a:
                    try:
                        f["arguments"] = json.dumps(json.loads(a), ensure_ascii=False)
                        hit = True
                    except Exception:
                        pass
        if hit:
            return json.dumps(d, ensure_ascii=False).encode("utf-8")
    except Exception:
        pass
    return buf


def _unescape_feed(s: str, pend: str):
    """v2.3 (B1) 状态机单次喂入: 返回 (本帧可放行文本, 未决尾巴)。
    只解码 \\uXXXX 中 cp>=0x20 且非引号/反斜杠的码点 (与 ensure_ascii=False 再序列化语义一致);
    结构/控制/孤立低位代理转义与内层 JSON 自身短转义原样保留;
    转义或高位代理在片尾不完整时挂起与下帧拼接重扫; 绝不输出孤立代理字符 (UTF-8 编码会炸)。"""
    if pend:
        s = pend + s
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:                      # 帧尾孤立反斜杠: 挂起
            break
        if s[i + 1] != "u":                 # 内层 JSON 自身转义: 原样放行
            out.append(s[i:i + 2])
            i += 2
            continue
        if i + 6 > n:                       # \uXXXX 十六进制不完整: 挂起
            break
        try:
            cp = int(s[i + 2:i + 6], 16)
        except ValueError:                  # 非法 \u 序列: 原样交下游解析器处置
            out.append(s[i:i + 2])
            i += 2
            continue
        if cp < 0x20 or cp == 0x22 or cp == 0x5C or 0xDC00 <= cp <= 0xDFFF:
            out.append(s[i:i + 6])          # 结构/控制/孤立低位代理: 保留转义
            i += 6
            continue
        if 0xD800 <= cp <= 0xDBFF:          # 高位代理: 必须等低位配对
            if n < i + 12:
                break                       # 低位信息不全: 挂起重扫
            if s[i + 6] == "\\" and s[i + 7] == "u":
                try:
                    lo = int(s[i + 8:i + 12], 16)
                except ValueError:
                    lo = -1
                if 0xDC00 <= lo <= 0xDFFF:
                    out.append(chr(0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)))
                    i += 12
                    continue
            out.append(s[i:i + 6])          # 判定不配对: 保留原转义
            i += 6
            continue
        out.append(chr(cp))                 # 普通字符 (中文等): 解码直出
        i += 6
    return "".join(out), s[i:]


class StreamToolUnescaper:
    """v2.3 (B1): 流式增量把 tool_calls.function.arguments 片段中的 \\uXXXX 转为 UTF-8 原文。
    行级 SSE 重组 (read1 网络块与 SSE 帧不对齐) + 按工具 index 的挂起状态机 (转义/代理对可跨 delta);
    无 tool_calls 的帧字节直通 (零解析开销); 任何异常该流自动退化为直通 (fail-open);
    流尾未决尾巴以合成 delta 帧补发 (先于 [DONE]) → 客户端聚合结果与直通逐字符等价。"""

    def __init__(self):
        self.buf = bytearray()   # 不完整行缓冲
        self.pend = {}           # tool index -> 未决尾巴
        self.off = False         # fail-open: 后续直通
        self.shell = None        # 最近工具帧外壳 (id/object/...), 合成帧复用

    def feed(self, chunk: bytes) -> bytes:
        if self.off:
            return chunk
        out = []
        try:
            self.buf += chunk
            while True:
                nl = self.buf.find(b"\n")
                if nl < 0:
                    break
                line = bytes(self.buf[:nl])
                del self.buf[:nl + 1]
                out.append(self._line(line))
                out.append(b"\n")
            return b"".join(out)
        except Exception:
            self.off = True
            rest = bytes(self.buf)
            self.buf = bytearray()
            return b"".join(out) + rest

    def finish(self) -> bytes:
        if self.off:
            return b""
        out = []
        if self.buf:
            line = bytes(self.buf)
            self.buf = bytearray()
            out.append(self._line(line))
            out.append(b"\n")
        out.append(self._flush_pend())
        return b"".join(out)

    def _line(self, line: bytes) -> bytes:
        try:
            s = line.rstrip(b"\r")
            if not s.startswith(b"data:"):
                return line
            payload = s[5:].strip()
            if payload == b"[DONE]":
                return self._flush_pend() + line    # 未决尾巴先于 [DONE] 补发
            if not payload:
                return line
            if b'"tool_calls"' not in s:
                return line                        # 内容/用量帧零开销直通
            try:
                obj = json.loads(payload)
            except Exception:
                return line
            if not isinstance(obj, dict):
                return line
            changed = False
            for ch in obj.get("choices") or []:
                if not isinstance(ch, dict):
                    continue
                delta = ch.get("delta")
                if not isinstance(delta, dict):
                    continue
                tcs = delta.get("tool_calls")
                if not isinstance(tcs, list):
                    continue
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    f = tc.get("function")
                    if not isinstance(f, dict):
                        continue
                    a = f.get("arguments")
                    if not isinstance(a, str) or not a:
                        continue
                    idx = tc.get("index")
                    if not isinstance(idx, int):
                        idx = 0
                    new_a, pend = _unescape_feed(a, self.pend.get(idx, ""))
                    self.pend[idx] = pend
                    if new_a != a:
                        f["arguments"] = new_a
                        changed = True
            self._note_shell(obj)
            if not changed:
                return line
            return b"data: " + json.dumps(obj, ensure_ascii=False,
                                          separators=(",", ":")).encode("utf-8")
        except Exception:
            return line

    def _note_shell(self, obj):
        shell = {k: obj[k] for k in ("id", "object", "created", "model") if k in obj}
        if shell.get("id") is not None:
            self.shell = shell

    def _flush_pend(self) -> bytes:
        frames = [{"index": idx, "function": {"arguments": tail}}
                  for idx, tail in sorted(self.pend.items()) if tail]
        self.pend = {}
        if not frames:
            return b""
        obj = dict(self.shell) if self.shell else {}
        obj["choices"] = [{"index": 0, "delta": {"tool_calls": frames}}]
        try:
            return (b"data: " + json.dumps(obj, ensure_ascii=False,
                                           separators=(",", ":")).encode("utf-8") + b"\n\n")
        except Exception:
            return b""


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True   # TCP_NODELAY: 小包立即发, 流式低延迟

    def log_message(self, *a):
        pass

    def _admin(self):
        # P1: read-only trace lookup for one rid/chain. It returns only gateway
        # metric records; request/completion text is never loaded here.
        if self.path.startswith("/__gw/trace"):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            rid = (query.get("rid") or [""])[0][:64]
            chain = (query.get("chain") or [""])[0][:64]
            n = 50
            try:
                n = max(1, min(500, int((query.get("n") or ["50"])[0])))
            except ValueError:
                pass
            def metric_rows(path):
                rows = []
                try:
                    with open(path) as handle:
                        tail = handle.readlines()[-5000:]
                    for line in tail:
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if (rid and row.get("rid") == rid) or (chain and row.get("chain") == chain):
                            rows.append(row)
                except OSError:
                    pass
                return rows[-n:]
            body = json.dumps({"rid": rid, "chain": chain,
                               "requests": metric_rows(REQUESTS_LOG),
                               "decisions": metric_rows(DECISIONS_LOG)},
                              ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True
        # v2.2.0: 决策尾窗 — 面板拉最近 N 条 decisions (含动作/恢复时长/存档耗时)
        if self.path.startswith("/__gw/decisions"):
            n = 30
            for part in self.path.split("?"):
                if part.startswith("n="):
                    try: n = max(1, min(200, int(part[2:])))
                    except ValueError: pass
            rows = []
            try:
                with open(DECISIONS_LOG) as f:
                    tail = f.readlines()[-n:]
                rows = [json.loads(l) for l in tail if l.strip()]
            except OSError:
                pass
            body = json.dumps({"decisions": rows}, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True
        if self.path == "/__gw/status":
            body = json.dumps({"live_fp": live_fp, "live_chain": live_chain, "dirty": live_dirty,
                               "last_save": last_save_ts, "stats": stats,
                               "l2_bytes": l2_total_bytes(), "l2_budget": L2_BUDGET_BYTES,
                               "engine_main": engine_main_metrics(),
                               "pool_workset": pool_workset_metrics(),
                               "pool_budget_mib": POOL_BUDGET_BYTES // 1024**2,
                               "save_reserve_mib": SAVE_RESERVE_MIB,
                               "sched_decode_ms_per_token": DECODE_MS_PER_TOKEN,
                               "memory_pressure": memory_pressure_snapshot(),
                               "sched_policy": {
                                   "live_max_burst": int(os.environ.get("STRATA_SCHED_LIVE_MAX_BURST", "4")),
                                   "live_max_wait_s": float(os.environ.get("STRATA_SCHED_LIVE_MAX_WAIT_S", "10")),
                                   "starve_s": float(os.environ.get("STRATA_SCHED_STARVE_S", "60")),
                                   "decode_est_max_ms": int(os.environ.get("STRATA_SCHED_DECODE_EST_MAX_MS", "180000")),
                                   "completion_history_families": len(SCHED_COMPLETION_HISTORY),
                               },
                               "families": {k: len(v) for k, v in fam_idx.items()},
                               "snapshots": [f for f in os.listdir(SNAP_DIR) if "." != f[0]]},
                               ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True
        if self.path == "/__l2/stats":
            import shutil as _sh
            usage = _sh.disk_usage(HOT_DIR)
            hb = 0
            try: hb = int(open(HOT_DIR + "/.snapshot-heartbeat").read())
            except (OSError, ValueError): pass
            snaps = [f for f in os.listdir(HOT_DIR) if not f.endswith(".tok") and not f.startswith(".")]
            body_out = json.dumps({
                "hot_gib": round(usage.used / 2**30, 1), "hot_pct": round(usage.used * 100 / usage.total, 0),
                "hot_files": len(snaps),
                "snapshot_hb_age_s": int(time.time()) - hb if hb else -1,
                "l2_alarms": os.path.exists(os.path.join(_LOG_DIR, "l2-alarms.log")),
            }).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_out))); self.end_headers()
            self.wfile.write(body_out); return True
        if self.path == "/__gw/save":
            with state_lock:
                fp, ch, hs, dirty = live_fp, live_chain, list(live_hashes), live_dirty
            ok = do_save(fp, ch, hs) if fp and ch else False
            body = json.dumps({"saved": ok, "fp": fp, "chain": ch}).encode()
            self.send_response(200 if ok else 500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True
        return False

    def _fingerprint_of_request(self):
        if self.path.split("?")[0] != "/v1/chat/completions" or self.command != "POST":
            self._body_raw = None
            self._body_obj = None
            return None, None
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            body = json.loads(raw)
            self._body_raw = raw
            self._body_obj = body
            family = fingerprint(body)
            hashes = chain_hashes(body.get("messages"))
            return (family, chain_id(hashes), hashes), body
        except Exception:
            self._body_raw = None
            self._body_obj = None
            return None, None

    def _proxy(self):
        import http.client, uuid
        t_recv = time.time()
        rid = uuid.uuid4().hex[:12]
        # B: experiment 标签 — 受控实验与真实流量分层. 优先 header, 其次 body 字段.
        experiment = (self.headers.get("X-Strata-Experiment") or "").strip()[:64] or None
        ids, fbody = self._fingerprint_of_request()
        if experiment is None and isinstance(fbody, dict):
            _v = fbody.get("strata_experiment")
            experiment = str(_v).strip()[:64] if _v else None
        decision = None
        family = chain = None
        mem0 = eng_mem()   # v2.2.2: 请求级内存采样 (进)
        pressure0 = memory_pressure_snapshot()
        sched_req = None
        if ids:
            family, chain, _hashes = ids
            stats["requests"] += 1
            # [giphoo] issue#3: 入队等放行 (调度器控制次序; restorep 不再被 FIFO 头阻塞)
            sched_req = SchedRequest(family, chain, fbody, _hashes, rid=rid)
            # [giphoo] no pre-header heartbeat: send_response() comes after admission.
            scheduler.submit(sched_req)   # 阻塞等放行
            t_after_sched = time.time()
            rh = hashlib.sha1(getattr(self, "_body_raw", b"") or b"").hexdigest()[:12]
            t0 = time.time()
            decision = intercept(family, chain, _hashes, fbody, rh, rid=rid)
            decision_ms = round((time.time() - t0) * 1000)
            if (decision or {}).get("prefix_hint_injected") and isinstance(fbody, dict):
                # intercept added a server-internal strata_prefix after fingerprinting the
                # client body; forward the enriched JSON with its own content length
                self._body_raw = json.dumps(fbody, ensure_ascii=False,
                                            separators=(",", ":")).encode("utf-8")
            if sched_req and sched_req.sched_action:
                decision["sched_action"] = sched_req.sched_action
                decision["sched_wait_ms"] = round((t_after_sched - t_recv) * 1000)
                decision["queue_wait_ms"] = round((sched_req.admit_ts - sched_req.enqueue_ts) * 1000)
        else:
            rh, decision_ms = "", 0
        fp = family   # 日志字段沿用 fp=族
        t_forward = time.time()
        c = http.client.HTTPConnection(ENGINE[0], ENGINE[1], timeout=600)
        t_first = None; n_deltas = 0; total_bytes = 0; tail = None
        t_resp = None   # v2.2.13: 响应头到达(TTFB) — 非流式 TTFT 等价口径
        # P1 timing semantics: keep the legacy first-SSE timestamp, but also
        # detect the first generated content/reasoning/tool delta.
        t_first_sse = None; t_first_token = None; first_token_kind = None
        first_token_buf = bytearray()
        try:
            hdrs = {k: v for k, v in self.headers.items() if k.lower() not in
                    ("host", "connection", "content-length", "x-strata-request-id", "x-strata-l2-trace",
                     "x-strata-experiment")}
            hdrs["X-Strata-Request-Id"] = rid
            if decision is not None:
                trace = dict(decision)
                trace["decision_ms"] = decision_ms
                if sched_req:
                    trace["scheduler"] = {
                        "action": sched_req.sched_action, "est_cost": round(sched_req.est_cost),
                        "sched_cost": round(sched_req.sched_cost), "est_prompt": sched_req.est_prompt,
                        "cover": sched_req.cover, "prefill_est": round(sched_req.prefill_est),
                        "restore_est": round(sched_req.restore_est), "decode_est": round(sched_req.decode_est),
                        "is_live": sched_req.is_live, "is_starved": sched_req.is_starved,
                        "live_forced": sched_req.live_forced,
                    }
                hdrs["X-Strata-L2-Trace"] = json.dumps(trace, ensure_ascii=True, separators=(",", ":"))
            body = getattr(self, "_body_raw", None)
            if body is None:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else None
            if body is not None:
                hdrs["Content-Length"] = str(len(body))
            c.request(self.command, self.path, body=body, headers=hdrs)
            r = c.getresponse()
            t_resp = time.time()   # 响应头到达; 非流式=引擎整包生成完毕时刻
            # close 分帧: 不转发 CL/TE, 客户端读到连接关闭即结束 (SSE 安全)
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in ("connection", "content-length", "transfer-encoding"):
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            if self.command == "HEAD":
                c.close()
                return
            streamed = "event-stream" in (r.getheader("Content-Type") or "").lower()
            # v2.2.9: 非流式整包缓冲 — tool_calls args 中文转义重写; v2.3: 流式 StreamToolUnescaper 增量转换
            if not streamed and self.command == "POST":
                buf = b""
                while True:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    buf += chunk
                if TOOLJSON_UNESCAPE:
                    buf = rewrite_tool_args(buf)   # tool_calls args \uXXXX -> UTF-8
                total_bytes = len(buf)
                # v2.2.13: 非流式 TTFT 等价 = TTFB(getresponse 响应头到达, 引擎整包生成后一次性返回,
                # 数值≈queue+prefill+generation 全程); 首 token 增量无法观察, 引擎 timings 仍并列落盘。
                n_deltas = buf.count(b'"content"')
                tail = buf[-16384:]
                self.wfile.write(buf)
                self.wfile.flush()
            else:
                unesc = StreamToolUnescaper() if TOOLJSON_UNESCAPE else None
                while True:
                    # read1: 有多少读多少立即转发 (read(n) 会攒满 n 字节才放行 → SSE 变大坨)
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    if t_first is None and b'data:' in chunk:
                        t_first = time.time()          # legacy first SSE frame (may be role-only)
                    if t_first_sse is None and b'data:' in chunk:
                        t_first_sse = t_first
                    if t_first_token is None:
                        first_token_buf.extend(chunk)
                        if len(first_token_buf) > 262144:
                            del first_token_buf[:-131072]
                        first_token_kind = first_model_delta_kind(bytes(first_token_buf))
                        if first_token_kind is not None:
                            t_first_token = time.time()
                            first_token_buf.clear()
                    if b'"content"' in chunk:
                        n_deltas += chunk.count(b'"content"')
                    total_bytes += len(chunk)
                    # v2.2.4: 尾部缓冲 — 抓 usage (flash/gemma 流式末块均携带)
                    tail = (tail + chunk)[-16384:] if tail is not None else chunk[-16384:]
                    self.wfile.write(unesc.feed(chunk) if unesc else chunk)
                    self.wfile.flush()
                end_flush = unesc.finish() if unesc else b""   # flush pending tail
                if end_flush:
                    self.wfile.write(end_flush)
                    self.wfile.flush()
            usage = usage_from_tail(tail or b"")
            engine_timings = timings_from_tail(tail or b"")
            stage_ledger = (engine_timings or {}).get("stage_ledger") or {}
            finish_reason = finish_reason_from_tail(tail or b"")
            if fp:
                on_request_done()
            if sched_req:
                scheduler.done(sched_req)   # [giphoo] issue#3: 通知放行门 — 请求完成
            if fp:
                # v1.8: 全请求时延 timeline 落盘 (v2.1: fp=族 + chain=链)
                t_end = time.time()
                ttfb_ms = round((t_resp - t_forward) * 1000) if t_resp is not None else None
                gateway_wait_ms = round((t_forward - t_recv) * 1000)
                queue_wait_ms = (round((sched_req.admit_ts - sched_req.enqueue_ts) * 1000)
                                 if sched_req and sched_req.admit_ts else 0)
                nonqueue_wait_ms = max(0, gateway_wait_ms - queue_wait_ms)
                wait_stage = classify_gateway_wait(decision, queue_wait_ms, gateway_wait_ms, decision_ms)
                entry = {"rid": rid, "ts": time.strftime("%H:%M:%S"),
                         "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_end)),
                         "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_recv)),
                         "forwarded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_forward)),
                         "gateway_pid": os.getpid(), "fp": fp, "chain": chain,
                         "req_hash": rh, "stream": streamed, "http_status": r.status,
                         "experiment": experiment,
                         "finish_reason": finish_reason, "engine_stop_reason": finish_reason,
                         "client_cancelled": False, "aborted": False,
                         "prompt_tokens": (usage or {}).get("prompt_tokens"),
                         "completion_tokens": (usage or {}).get("completion_tokens"),
                         "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),   # v3.1: 借用可见性 (0+大pt=全量重算告警信号)
                         "wait_ms": gateway_wait_ms,                              # 接收→转发 (含决策)
                         "queue_wait_ms": queue_wait_ms,
                         "nonqueue_wait_ms": nonqueue_wait_ms,
                         "gateway_wait_stage": wait_stage,
                         "decision_ms": decision_ms,
                         "response_ms": round((t_end - t_forward) * 1000),
                         # Legacy names remain for backward compatibility.  New
                         # analysis must prefer the explicit semantics below.
                          "ttft_ms": round((t_first - t_forward) * 1000) if streamed and t_first else ttfb_ms,
                          "first_sse_ms": round((t_first_sse - t_forward) * 1000) if streamed and t_first_sse else ttfb_ms,
                          "true_ttft_ms": round((t_first_token - t_forward) * 1000) if streamed and t_first_token else ttfb_ms,
                          "first_token_kind": first_token_kind,
                          "arrival_to_first_token_ms": round(((t_forward - t_recv) + (t_first_token - t_forward)) * 1000)
                          if t_first_token is not None else (round((t_resp - t_recv) * 1000) if t_resp is not None else None),
                         # [giphoo] arrival→first_sse 细粒度分解（纯观测, 不改调度）。
                         # arrival→first_sse = arrival_to_forward + engine_ttfb + (首帧→first_sse≈0);
                         # engine_ttfb 内部 = prefill_ms + prefill_to_first_frame_ms(黑盒, 待归因)。
                         "arrival_to_forward_ms": round((t_forward - t_recv) * 1000),
                          "engine_ttfb_ms": round((t_first - t_forward) * 1000) if streamed and t_first else ttfb_ms,
                         # 缓存命中时 prompt_ms 只是记账口径(完整 prompt 量), 可能远大于实际
                         # engine_ttfb → 相减为负说明该字段无可归因黑盒, clamp 0 不产负值。
                         "prefill_to_first_frame_ms": (max(0, round((t_first - t_forward) * 1000 - ((engine_timings or {}).get("prompt_ms") or 0)))
                                                       if streamed and t_first and (engine_timings or {}).get("prompt_ms") else None),
                          "prefill_ms": (engine_timings or {}).get("prompt_ms"),
                          "engine_predicted_ms": (engine_timings or {}).get("predicted_ms"),
                          "total_ms": round((t_end - t_recv) * 1000),
                         "decode_ms": round((t_end - t_first) * 1000) if streamed and t_first
                         else (engine_timings or {}).get("predicted_ms"),
                         "post_first_sse_ms": round((t_end - t_first_sse) * 1000) if streamed and t_first_sse else None,
                         "generation_wall_ms": round((t_end - t_first_token) * 1000) if t_first_token is not None else None,
                         "timing_source": "stream" if streamed else "engine" if engine_timings else "unavailable",
                          "timing_semantics": "first_sse_legacy_plus_true_first_generated_delta" if streamed else "engine_timings+nonstream_ttfb",
                          "deltas": n_deltas, "bytes": total_bytes}
                # [giphoo] 引擎侧延迟时间戳 → 相对 arrival 的 ms(观测, 拆 arrival→first_sse 黑盒)
                for _k in ("req_received_ts", "prepare_done_ts", "first_token_ts", "first_sse_sent_ts", "first_model_sse_sent_ts", "pp_first_ts", "pp_last_ts"):
                    _v = (engine_timings or {}).get(_k)
                    if _v is not None:
                        entry["engine_" + _k] = round((_v - t_recv) * 1000)
                # [giphoo] prefill->首token 黑盒段(观测): 末次 PP 到首 token 的间隔与此前最后一类引擎输出
                for _k in ("pp_count", "post_pp_gap_ms"):
                    _v = (engine_timings or {}).get(_k)
                    if _v is not None:
                        entry["engine_" + _k] = _v
                _kind = (engine_timings or {}).get("pre_token_last_kind")
                if _kind:
                    entry["engine_pre_token_last_kind"] = _kind
                if stage_ledger:
                    entry["stage_ledger"] = stage_ledger
                    for key in STAGE_LEDGER_KEYS:
                        if stage_ledger.get(key) is not None:
                            entry[key] = stage_ledger[key]
                    entry["timing_semantics"] += "+engine_stage_ledger"
                if entry["prefill_ms"] and entry["prompt_tokens"]:
                    entry["prefill_tps"] = round(entry["prompt_tokens"] / (entry["prefill_ms"] / 1000), 1)
                if entry["decode_ms"] and entry["completion_tokens"]:
                    entry["decode_tps"] = round(max(entry["completion_tokens"] - 1, 1) / (entry["decode_ms"] / 1000), 1)
                if entry.get("generation_wall_ms") and entry["completion_tokens"]:
                    entry["generation_tps"] = round(max(entry["completion_tokens"] - 1, 1) /
                                                    (entry["generation_wall_ms"] / 1000), 1)
                # P1 scheduler calibration: observed completion feeds the next
                # expected decode cost for this family.
                if entry.get("completion_tokens"):
                    sched_observe_completion(fp, entry["completion_tokens"])
                if decision:
                    entry["action"] = decision.get("action")
                    if "costs" in decision: entry["costs"] = decision["costs"]
                    if "n" in decision: entry["n"] = decision["n"]
                    if "restorep_ms" in decision: entry["restorep_ms"] = decision["restorep_ms"]
                    for key in ("live_lcp", "pool_best", "parked_best", "parked_serial", "parked_hit",
                                "restore_parked", "preload_serial",
                                "l2_est", "probe_ms", "bootstrap_save_ms",
                                "save_action", "save_status", "save_n_saved", "save_n_written", "save_ms",
                                "save_incremental", "save_fallback", "save_n_reused", "save_n_kept",
                                 "save_published_file",
                                 "save_gen", "save_async_pending", "save_commit_state", "save_enqueue_ms",
                                 "save_serial_wait_ms", "save_async_mode",
                                 "save_reserve_mib",
                                "pool_trust_discounted", "sched_action",
                                "l2_reuse", "l2_read", "l2_common_messages", "l2_reuse_source",
                                 "restore_n_read", "restore_n_restored", "restore_engine_ms",
                                  "restore_read_mib_s", "restore_partial", "restore_pending_wait_ms", "pool_workset",
                                  "restore_checkpoint_tokens", "restore_image_bytes",
                                  "restore_payload_read_bytes", "restore_io_read_bytes",
                                  "restore_io_read_mib_s", "restore_archive_format", "restore_fallback",
                                  "restore_shadow", "restore_rule",
                                  "borrow_kind", "borrow_hotspot_hits", "prefix_hint",
                                  "restore_candidates",
                                 "preload_status", "preload_http_status", "preload_error_kind",
                                 "restore_source", "restore_snapshot_bytes", "restore_logical_bytes",
                                 "restore_limit_basis", "restore_limit_bytes",
                                 "restore_guard", "restore_guard_alternative",
                                 "restore_full_fallback_suppressed", "restore_memory_admission",
                                 "save_memory_rejected"):
                        if key in decision:
                            entry[key] = decision[key]
                    if sched_req:
                        entry["scheduler"] = {"est_prompt": sched_req.est_prompt, "cover": sched_req.cover,
                                              "prefill_est": round(sched_req.prefill_est),
                                              "restore_est": round(sched_req.restore_est),
                                               "decode_est": round(sched_req.decode_est),
                                               "est_cost": round(sched_req.est_cost),
                                               "sched_cost": round(sched_req.sched_cost),
                                               "is_live": sched_req.is_live,
                                               "is_starved": sched_req.is_starved,
                                               "sched_action": sched_req.sched_action,
                                               "live_overtakes": sched_req.live_overtakes,
                                               "live_forced": sched_req.live_forced,
                                               "queue_depth_at_enqueue": sched_req.queue_depth_at_enqueue,
                                               "queue_depth_max": sched_req.queue_depth_max,
                                               "engine_busy_at_enqueue": sched_req.engine_busy_at_enqueue,
                                               "blocking_request_ids": sched_req.blocking_request_ids}
                    # [giphoo] issue#4-D: 恢复反馈闭环 — 实际复用率回写, 供 layer_costs 校准
                    try:
                        act = decision.get("action", "")
                        if act and ("restore" in act or act == "engine-native"):
                            n_tok = decision.get("n", 0)
                            cached = entry.get("cached_tokens") or 0
                            with open(os.path.join(_LOG_DIR, "restore-feedback.jsonl"), "a") as rf:
                                rf.write(json.dumps({
                                    "ts": time.strftime("%F %T"), "rid": rid, "chain": entry.get("chain", "")[:12],
                                    "action": act, "n": n_tok, "cached": cached,
                                    "reuse_ratio": round(cached / n_tok, 3) if n_tok else 0,
                                    "ttft_ms": entry.get("ttft_ms"), "prefill_ms": entry.get("prefill_ms"),
                                    "restorep_ms": decision.get("restorep_ms"),
                                    "restore_n_read": decision.get("restore_n_read"),
                                    "restore_engine_ms": decision.get("restore_engine_ms"),
                                    "restore_read_mib_s": decision.get("restore_read_mib_s"),
                                    "parked_best": decision.get("parked_best"),
                                    "parked_hit": decision.get("parked_hit"),
                                    "restore_parked": decision.get("restore_parked"),
                                    "pool_workset": decision.get("pool_workset"),
                                    "pool_best": decision.get("pool_best"),
                                    "l2_reuse": decision.get("l2_reuse"), "l2_read": decision.get("l2_read"),
                                    "l2_reuse_source": decision.get("l2_reuse_source"),
                                    "restore_rule": decision.get("restore_rule")}) + "\n")
                    except OSError:
                        pass
                # v2.2.2: 请求级内存进出快照 — 增长归因到请求 (现场还原)
                try:
                    mem1 = eng_mem()
                    if mem0 and mem1:
                        entry["mem0"] = mem0
                        entry["mem1"] = mem1
                        dm = {k2: round(mem1[k2] - mem0.get(k2, 0), 1) for k2 in mem1}
                        entry["dmem"] = dm
                except Exception:
                    pass
                # P1: service/container/host memory + PSI snapshots.  Host
                # MemAvailable alone proved insufficient on 2026-11-09.
                try:
                    pressure1 = memory_pressure_snapshot()
                    entry["memory_pressure"] = {"start": pressure0, "end": pressure1}
                    reasons = _memory_pressure_reasons(pressure1)
                    if reasons:
                        entry["memory_backpressure"] = {"reasons": reasons}
                        append_event_log(MEMORY_EVENTS_LOG, {
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "request_pressure_observed",
                            "rid": rid, "reasons": reasons, "snapshot": pressure1})
                except Exception:
                    pass
                try:
                    with open(REQUESTS_LOG, "a") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except OSError:
                    pass
        except BrokenPipeError:
            pass
        except Exception as e:
            print(f"[gw] proxy err: {e}", flush=True)
        finally:
            release_preload_pin(decision)   # [giphoo] do not pin an unconsumed preload past its request
            if sched_req:
                scheduler.done(sched_req)   # [giphoo] issue#3: 异常路径也通知放行门 (防死锁)
            c.close()

    def do_GET(self):
        if self._admin():
            return
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_HEAD(self):
        self._proxy()

    def do_OPTIONS(self):
        self._proxy()

# ===================== v2.2.4: 模型 token 用量统计 =====================

def l2_boot_scan():
    """[giphoo] issue#4-B: 启动扫描修复 — 孤儿文件/索引死档/缺失tok/半前滚/代际不一致.
    index.json 是可重建缓存: 扫描结果与索引不一致时以盘为准."""
    repaired = {"orphan": 0, "dead_idx": 0, "missing_tok": 0, "half_roll": 0, "tmp_mini": 0}
    try:
        on_disk = {f for f in os.listdir(ENGINE_SNAP_DIR)
                   if archive_exists(os.path.join(ENGINE_SNAP_DIR, f))}
        indexed = set()
        with fam_idx_lock:
            for fam_name, fam in list(fam_idx.items()):
                for chain, e in list(fam.items()):
                    fn = e.get("file", "")
                    if fn in on_disk:
                        indexed.add(fn)
                    else:
                        fam.pop(chain, None)   # 索引死档
                        repaired["dead_idx"] += 1
        # 孤儿文件 (盘上有但索引无) — 保留但记录 (可能是有价值的旧档, 不盲删)
        orphans = on_disk - indexed - {f for f in on_disk if f.endswith((".stale", ".rescue"))}
        repaired["orphan"] = len(orphans)
        # 缺失 .tok 补写 (从档头读 n_saved 代价高, 用文件 mtime 启发: 标 0 由下次 save 覆盖)
        for f in on_disk:
            if len(f) != 16 or not all(c in "0123456789abcdef" for c in f):
                continue
            if not os.path.exists(tok_path(f)):
                try:
                    with open(tok_path(f), "w") as tf:
                        tf.write("0")
                    repaired["missing_tok"] += 1
                except OSError:
                    pass
        # 半前滚: .rescue/.l2tmp 残留 — 主档在则删残留, 主档丢则 rescue 提正
        for f in list(on_disk):
            if f.endswith(".l2tmp"):
                base = f[:-6]
                if base in on_disk:
                    try:
                        archive_unlink(main_path(f))
                        repaired["half_roll"] += 1
                    except OSError:
                        pass
            elif f.endswith(".rescue"):
                base = f[:-7]
                try:
                    if base in on_disk:
                        archive_unlink(snap_path(f))   # 主档在, rescue 是冗余
                    else:
                        os.replace(snap_path(f), snap_path(base))   # 主档丢, rescue 提正
                    repaired["half_roll"] += 1
                except OSError:
                    pass
        # Obsolete writer-copy minis are redundant once the main generation is present.
        try:
            for f in os.listdir(HOT_DIR):
                if not f.endswith(".l2tmp.state"):
                    continue
                base = f[:-len(".l2tmp.state")]
                if os.path.exists(main_path(base)):
                    try:
                        os.unlink(os.path.join(HOT_DIR, f))
                        repaired["tmp_mini"] += 1
                    except OSError:
                        pass
        except OSError:
            pass
        fam_idx_save()
    except OSError as e:
        print(f"[gw] boot scan: {e}", flush=True)
    if any(repaired.values()):
        print(f"[gw] issue#4-B boot scan repaired: {repaired}", flush=True)

# ============ optional startup manifest (source/argv fingerprint) ============

def _md5_file(path, limit=512 * 1024 * 1024):
    import hashlib
    h = hashlib.md5()
    total = 0
    with open(path, "rb") as f:
        while total < limit:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest()


def write_startup_manifest():
    """Startup fingerprint manifest - binary/config/model/env consistency snapshot.

    用 /proc 扫描 strata 进程(不依赖 pgrep 模式); 大模型文件只记 size+mtime,
    Engine binary is md5'd. Appended to startup-manifest.jsonl; historical
    startup records are kept for post-hoc joins.
    """
    import glob
    man = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "type": "gateway_startup",
           "gateway_pid": os.getpid()}
    _here = os.path.dirname(os.path.abspath(__file__))
    for label, p in (("gateway2_md5", os.path.abspath(__file__)),
                     ("server_py_md5", os.path.join(os.path.dirname(_here), "serve", "server.py"))):
        try:
            man[label] = _md5_file(p)
        except OSError:
            man[label] = None
    # 扫描所有含 strata 的进程 (server.py / strata-vision / strata-host / gateway2)
    procs = []
    for pd in glob.glob("/proc/[0-9]*"):
        try:
            pid = int(pd.rsplit("/", 1)[1])
            if pid == os.getpid():
                continue
            with open(pd + "/cmdline", "rb") as f:
                argv = [a for a in f.read().decode("utf-8", "replace").split("\0") if a]
            if not argv or "strata" not in " ".join(argv).lower():
                continue
            try:
                comm = open(pd + "/comm").read().strip()
            except OSError:
                comm = ""
            procs.append({"pid": pid, "comm": comm, "argv": argv})
        except (OSError, ValueError):
            continue
    man["engine_processes"] = procs
    # 运行中二进制 md5 (attr: argv[0] 可执行)
    bins = {}
    for pr in procs:
        a0 = pr["argv"][0]
        base = os.path.basename(a0)
        if not (os.path.isfile(a0) and os.access(a0, os.X_OK)):
            continue
        if not (base.startswith("strata") or base.startswith("python")):
            continue
        try:
            st = os.stat(a0)
            bins[a0] = {"size": st.st_size, "mtime": int(st.st_mtime), "md5": _md5_file(a0)}
        except OSError:
            bins[a0] = {"path": a0}
    man["engine_binaries"] = bins
    # 模型 (gguf) + 关键引擎参数
    models = []
    keyargs = {}
    KEY = {"--max-context", "--kv", "--kv-resident", "--resident-budget-gib",
           "--conversation-cache-mib", "--conversation-cache-slots", "--spec",
           "--draft-conf", "--memory-limit-mib", "--vram-reserve-mib",
           "--pool-workers", "--prompt-cache", "--mmap-experts", "--model", "--ple-gguf"}
    for pr in procs:
        av = pr["argv"]
        for i, a in enumerate(av):
            if a.lower().endswith(".gguf"):
                try:
                    st = os.stat(a)
                    models.append({"path": a, "size": st.st_size,
                                   "mtime": int(st.st_mtime), "pid": pr["pid"]})
                except OSError:
                    models.append({"path": a, "pid": pr["pid"]})
            if a in KEY and i + 1 < len(av):
                keyargs[a] = av[i + 1]
    man["models"] = models
    man["engine_args"] = keyargs
    man["env"] = {k: v for k, v in os.environ.items() if k.startswith("STRATA_")}
    try:
        with open(os.path.join(_LOG_DIR, "startup-manifest.jsonl"), "a") as f:
            f.write(json.dumps(man, ensure_ascii=False) + "\n")
    except OSError as e:
        print("[gw] manifest write err: " + str(e), flush=True)


if __name__ == "__main__":
    os.makedirs(SNAP_DIR, exist_ok=True)
    os.makedirs(_LOG_DIR, exist_ok=True)
    fam_idx_load()
    normalize_inplace_aliases()
    l2_boot_scan()   # [giphoo] issue#4-B: 启动一致性扫描
    l2_global_gc()   # 启动治理: 清 .stale + 压回预算
    threading.Thread(target=idle_refresher, daemon=True).start()
    os.makedirs(HOT_DIR, exist_ok=True)
    threading.Thread(target=ssd_snapshotter, daemon=True).start()   # L2 2.0
    threading.Thread(target=health_watcher, daemon=True).start()
    threading.Thread(target=scheduler.gate_loop, daemon=True).start()   # [giphoo] issue#3: 放行门
    threading.Thread(target=sched_metrics_loop, daemon=True).start()    # [giphoo] issue#3: 调度指标 30s
    threading.Thread(target=restore_metrics_loop, daemon=True).start()  # [giphoo] issue#3: 恢复成功率 30s
    threading.Thread(target=engine_cgroup_metrics_loop, daemon=True).start()  # P1: cgroup IO/memory
    if STARTUP_MANIFEST:
        threading.Thread(target=write_startup_manifest, daemon=True).start()  # startup fingerprint

    # v2.2.4: 启动恢复当日用量 + 定时落盘

    print(f"[gw] session-gateway v2.1.3 on {LISTEN} → {ENGINE}, snapdir={SNAP_DIR}, "
          f"families={len(fam_idx)}, l2={l2_total_bytes()/1e9:.1f}G/{L2_BUDGET_BYTES/1e9:.0f}G", flush=True)
    import atexit
    atexit.register(_role_unregister)
    _role_register()
    if GW_ROLE_TOKEN:
        print(f"[gw] role: registered with the Control process group (pid {os.getpid()})", flush=True)
    ThreadingHTTPServer(LISTEN, Proxy).serve_forever()
