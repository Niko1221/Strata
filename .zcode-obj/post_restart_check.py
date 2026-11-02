import json, urllib.request

KEY = "sk-ouzhiyingyyds666!"
BASE = "http://127.0.0.1:8080"

def post(body, timeout=180):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")

# 1. literal-forcing grammar: proves the constraint really constrains
s, d = post({"model": "Qwen3.8-Flash-Next",
             "messages": [{"role": "user", "content": "say anything"}],
             "grammar": 'root ::= "ZZZ-CONSTRAINED-OK"',
             "max_tokens": 40, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": False}})
print("1 literal probe:", s, repr(d.get("choices", [{}])[0].get("message", {}).get("content")) if s == 200 else d)

# 2. tools with the historical nested-object bug shape
tools = [{"type": "function", "function": {"name": "write_file",
    "description": "write a file",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["path", "patch", "mode"],
        "properties": {"path": {"type": "string"},
            "patch": {"type": "object", "additionalProperties": False,
                      "required": ["old", "new"],
                      "properties": {"old": {"type": "string"}, "new": {"type": "string"}}},
            "mode": {"type": "string", "enum": ["create", "append", "replace"]}}}}]
s, d = post({"model": "Qwen3.8-Flash-Next",
             "messages": [{"role": "user", "content": "change timeout from 30 to 60 in app.conf"}],
             "tools": tools, "tool_choice": "required",
             "max_tokens": 300, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": False}})
if s == 200:
    msg = d["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    print("2 tool call:", s, d["choices"][0].get("finish_reason"),
          calls[0]["function"]["arguments"] if calls else msg.get("content"))
else:
    print("2 tool call:", s, d)

# 3. pattern schema: strict mode must refuse it
s, d = post({"model": "Qwen3.8-Flash-Next",
             "messages": [{"role": "user", "content": "hi"}],
             "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema":
                 {"type": "object", "additionalProperties": False, "required": ["n"],
                  "properties": {"n": {"type": "string", "pattern": "^[a-z]+$"}}}}},
             "max_tokens": 60, "temperature": 0})
print("3 pattern schema:", s, (d if s != 200 else "UNEXPECTEDLY ACCEPTED"))
