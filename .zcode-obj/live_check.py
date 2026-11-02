"""Three probes against a running server: is the constraint actually live right now?

1. literal-forcing GBNF -> the reply must be exactly the forced text
2. the historical tool-call bug shape (nested object arg followed by a required arg)
3. a schema strict mode must refuse -> 400, not a silently unconstrained call
"""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8082/v1"
KEY = "sk-ouzhiyingyyds666!"


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


# ---------------------------------------------------------------- probe 1
print("== 1. literal-forcing grammar ==")
status, body = post("/chat/completions", {
    "model": "swift",
    "messages": [{"role": "user", "content": "Say hello in English."}],
    "max_tokens": 512,
    "temperature": 0,
    "strata_think": False,
    "grammar": 'root ::= "ZZZ-CONSTRAINED-OK"',
})
content = (body.get("choices") or [{}])[0].get("message", {}).get("content")
print("   status", status, "| content", repr(content))
print("   VERDICT:", "CONSTRAINED" if content == "ZZZ-CONSTRAINED-OK" else "NOT CONSTRAINED")

# ---------------------------------------------------------------- probe 2
print("\n== 2. nested object arg + a required arg after it ==")
tools = [{
    "type": "function",
    "function": {
        "name": "patch_file",
        "description": "Replace one line of a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "patch": {
                    "type": "object",
                    "properties": {"old": {"type": "string"}, "new": {"type": "string"}},
                    "required": ["old", "new"],
                    "additionalProperties": False,
                },
                "mode": {"type": "string", "enum": ["replace", "insert"]},
            },
            "required": ["path", "patch", "mode"],
            "additionalProperties": False,
        },
    },
}]
status, body = post("/chat/completions", {
    "model": "swift",
    "messages": [{"role": "user", "content":
                  "In app.conf replace the line timeout=30 with timeout=60. Use patch_file."}],
    "max_tokens": 1024,
    "temperature": 0,
    "strata_think": False,
    "tools": tools,
    "tool_choice": "required",
})
choice = (body.get("choices") or [{}])[0]
msg = choice.get("message") or {}
calls = msg.get("tool_calls") or []
print("   status", status, "| finish_reason", choice.get("finish_reason"), "| calls", len(calls))
if calls:
    fn = calls[0].get("function", {})
    print("   function", fn.get("name"))
    raw = fn.get("arguments")
    print("   arguments(raw)", raw)
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        args = None
        print("   arguments do not parse:", exc)
    if isinstance(args, dict):
        print("   keys     ", sorted(args))
        print("   patch is ", type(args.get("patch")).__name__, "->", repr(args.get("patch")))
        ok = (sorted(args) == ["mode", "patch", "path"]
              and isinstance(args.get("patch"), dict)
              and sorted(args["patch"]) == ["new", "old"]
              and args.get("mode") in ("replace", "insert"))
        print("   VERDICT:", "STRICT AND COMPLETE" if ok else "INCOMPLETE / WRONG SHAPE")

# ---------------------------------------------------------------- probe 3
print("\n== 3. a schema strict mode must refuse (pattern) ==")
bad = [{
    "type": "function",
    "function": {
        "name": "set_code",
        "description": "Set a code.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "pattern": "^[A-Z]{3}$"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}]
status, body = post("/chat/completions", {
    "model": "swift",
    "messages": [{"role": "user", "content": "Set the code to ABC."}],
    "max_tokens": 256,
    "temperature": 0,
    "strata_think": False,
    "tools": bad,
    "tool_choice": "required",
})
err = body.get("error") if isinstance(body, dict) else body
print("   status", status)
print("   error ", json.dumps(err, ensure_ascii=False)[:400] if err else json.dumps(body)[:400])
print("   VERDICT:", "FAILED CLOSED" if status >= 400 else "SILENTLY WIDENED (bad)")
