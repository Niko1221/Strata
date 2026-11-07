#!/usr/bin/env python3
# L2 RAM PARKED control-plane module.
# All gateway-facing cache lifecycle operations live here. The engine remains
# the owner of KV memory; this module only forwards commands and parses
# inventory/pin/preload responses from the line protocol.
import json
import queue
import time

CACHE_KEYS = {"parked_best", "parked_serial", "parked_live", "serial",
              "tokens", "bytes", "ckpts", "pinned", "guarded", "cvec"}


def _error(handler, code, msg):
    handler._json(code, {"error": {"code": code, "message": msg}})


def _engine(service):
    eng = getattr(service, "engine", None)
    if eng is None or getattr(eng, "proc", None) is None:
        return None
    return eng


def _read_line(eng, timeout=10):
    try:
        line = eng.lines.get(timeout=timeout)
    except queue.Empty:
        return None
    return line


def _command(service, command, timeout=10):
    """Send one CACHE command while holding the same FIFO as real requests."""
    eng = _engine(service)
    if eng is None:
        return 503, {"error": {"message": "the engine is not running"}}
    try:
        with service.fifo:
            eng.proc.stdin.write(command + "\n")
            eng.proc.stdin.flush()
            first = _read_line(eng, timeout)
            if first is None:
                return 504, {"error": {"message": "cache command timed out"}}
            parts = first.split()
            if not parts:
                return 502, {"error": {"message": "empty cache response"}}
            head = parts[0]
            if head == "CACHESTAT":
                out = {}
                for item in parts[1:]:
                    key, sep, value = item.partition("=")
                    if sep:
                        try: out[key] = int(value)
                        except ValueError: out[key] = value
                return 200, out
            if head == "CACHELIST":
                count = int(parts[1]) if len(parts) > 1 else 0
                rows = []
                for _ in range(count):
                    line = _read_line(eng, timeout)
                    if line is None:
                        return 504, {"error": {"message": "cache list timed out"}}
                    p = line.split()
                    if not p or p[0] != "CACHEENTRY":
                        return 502, {"error": {"message": "invalid cache entry"}}
                    row = {}
                    for item in p[1:]:
                        key, sep, value = item.partition("=")
                        if not sep: continue
                        if key in ("tokens", "bytes", "ckpts", "pinned", "guarded", "cvec"):
                            try: row[key] = int(value)
                            except ValueError: row[key] = value
                        else: row[key] = value
                    rows.append(row)
                end = _read_line(eng, timeout)
                if end is None or end.split()[:1] != ["CACHEEND"]:
                    return 502, {"error": {"message": "cache list end missing"}}
                return 200, {"entries": rows}
            if head in ("CACHEOK", "CACHEERR"):
                out = {"ok": head == "CACHEOK"}
                for item in parts[1:]:
                    key, sep, value = item.partition("=")
                    if sep:
                        try: out[key] = int(value)
                        except ValueError: out[key] = value
                return (200 if out["ok"] else 404), out
            return 502, {"error": {"message": f"unexpected cache answer: {first[:120]}"}}
    except OSError as e:
        return 503, {"error": {"message": f"cache engine error: {e}"}}
    except (ValueError, IndexError) as e:
        return 502, {"error": {"message": f"cache protocol error: {e}"}}


def cache_stats(service):
    return _command(service, "CACHE STATS")


def cache_list(service):
    return _command(service, "CACHE LIST")


def cache_set_pinned(service, serial, pinned):
    op = "PIN" if pinned else "UNPIN"
    return _command(service, f"CACHE {op} serial={int(serial)}")


def cache_drop(service, serial):
    return _command(service, f"CACHE DROP serial={int(serial)}")


def cache_touch(service, serial):
    return _command(service, f"CACHE TOUCH serial={int(serial)}")


def extend_fields(parts, rec, start=5):
    """Merge key=value fields from a protocol line into a response dict."""
    for item in parts[start:]:
        key, sep, value = item.partition("=")
        if not sep:
            continue
        try:
            rec[key] = int(value)
        except ValueError:
            rec[key] = value
    return rec


def preload(service, body, render, rid="", timeout=180):
    """Read an L2 file as PARKED (RESTOREPP when ids are available), never active."""
    eng = _engine(service)
    if eng is None:
        return 503, {"error": {"message": "the engine is not running"}}
    filename = body.get("filename") if isinstance(body, dict) else None
    if not filename:
        return 400, {"error": {"message": "filename is required"}}
    try:
        messages, tools, kwargs = render(body)
        ids = service.encode_prompt(messages, tools, kwargs, store=False)
    except Exception as e:
        return 400, {"error": {"message": f"preload render failed: {e}"}}
    service._restorepp_ids = ids
    try:
        code, result = service.slot_action("0", "restorepp", filename, rid=rid)
    except Exception as e:
        return 503, {"error": {"message": f"preload engine error: {e}"}}
    if code == 200 and isinstance(result, dict) and body.get("pin", True):
        serial = result.get("serial")
        if serial is not None:
            _pcode, pin_result = cache_set_pinned(service, serial, True)
            result["pin"] = pin_result
    return code, result


def handle_cache_get(handler, service, path):
    if path == "/v1/cache/stats":
        code, body = cache_stats(service)
        handler._json(code, body)
        return True
    if path == "/v1/cache/list":
        code, body = cache_list(service)
        handler._json(code, body)
        return True
    return False


def handle_cache_post(handler, service, path, req, render, rid=""):
    if path in ("/v1/cache/pin", "/v1/cache/unpin", "/v1/cache/drop", "/v1/cache/touch"):
        serial = req.get("serial")
        if isinstance(serial, bool) or not isinstance(serial, int) or serial < 0:
            _error(handler, 400, "serial must be a non-negative integer")
            return True
        if path.endswith("/pin"):
            code, body = cache_set_pinned(service, serial, True)
        elif path.endswith("/unpin"):
            code, body = cache_set_pinned(service, serial, False)
        elif path.endswith("/drop"):
            code, body = cache_drop(service, serial)
        else:
            code, body = cache_touch(service, serial)
        handler._json(code, body)
        return True
    if path == "/v1/cache/preload":
        code, body = preload(service, req, render, rid=rid)
        handler._json(code, body)
        return True
    return False
