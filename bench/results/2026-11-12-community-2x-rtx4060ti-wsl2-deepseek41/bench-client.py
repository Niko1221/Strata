#!/usr/bin/env python3
"""Standart protokol fixture'larını (bench/prompts/standard-v1) DeepSeek için koşan istemci.

DeepSeek çok düşünür: 512 token cap'i düşünmede biter ve cevap hiç başlamaz. Bu yüzden üç geçiş:
  std      : 512 cap, thinking açık   → standart protokolle kıyaslanabilir (çoğu kesilecek)
  long     : 1024 cap, yalnız short   → kaç token düşünüp ne zaman cevaba geçtiği
  nothink  : 512 cap, thinking kapalı (chat_template_kwargs.enable_thinking=false) → düz decode tabanı
reasoning_content ve content ayrı sayılır; ilk cevap süresi ilk content delta'sıdır.

Kullanım: BASE=http://127.0.0.1:8081 STRATA_KEY=... python3 ds41-bench.py [std|long|nothink|all]
"""
import json, os, pathlib, statistics, sys, time, urllib.error, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8081")
KEY = os.environ.get("STRATA_KEY", "")
FIX = pathlib.Path(os.environ.get("FIX", str(pathlib.Path.home() / "ds41-fixtures" / "requests")))
OUT = pathlib.Path(os.environ.get("OUT", str(pathlib.Path.home() / "ds41-bench-results.jsonl")))
CASES = ["short", "prompt-4k", "prompt-16k"]
REPS = int(os.environ.get("REPS", "3"))


def req(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method="POST" if data else "GET")
    r.add_header("Content-Type", "application/json")
    if KEY:
        r.add_header("Authorization", "Bearer " + KEY)
    return urllib.request.urlopen(r, timeout=3600)


def model_name():
    with req("/v1/models") as f:
        d = json.loads(f.read())
    for key in ("data", "models"):
        items = d.get(key) or []
        if items:
            return items[0].get("id") or items[0].get("name")
    return "local"


def run_one(model, msgs, max_tokens=512, think=True):
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "stream": True,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.monotonic()
    first_reason = first_answer = None
    n_reason = n_answer = 0
    head = []
    finish = None
    usage = {}
    with req("/v1/chat/completions", body) as f:
        for raw in f:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                d = ch.get("delta") or {}
                rc = d.get("reasoning_content") or d.get("reasoning") or ""
                if rc:
                    if first_reason is None:
                        first_reason = time.monotonic() - t0
                    n_reason += len(rc)
                c = d.get("content") or ""
                if c:
                    if first_answer is None:
                        first_answer = time.monotonic() - t0
                        head.append(c[:1])
                    n_answer += len(c)
    total = time.monotonic() - t0
    out_tok = usage.get("completion_tokens")
    # decode hızı: ilk token'dan sonuna kadar üretilen token / geçen süre
    first = first_reason or 1.0
    decode = (out_tok / (total - first)) if (out_tok and total > first) else None
    return {"ttft_s": first_reason, "first_answer_s": first_answer, "total_s": total,
            "in_tokens": usage.get("prompt_tokens"), "out_tokens": out_tok,
            "reason_chars": n_reason, "answer_chars": n_answer,
            "answered": first_answer is not None, "finish": finish,
            "decode_tps": decode}


def load_fixtures():
    files = {}
    for p in sorted(FIX.glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("phase") == "performance" and d.get("case_id") in CASES and d.get("repetition", 1) <= REPS:
            files.setdefault(d["case_id"], []).append(d)
    return files


def do(model, case, d, max_tokens, think, label):
    try:
        r = run_one(model, d["messages"], max_tokens, think)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read()[:200].decode("utf-8", "replace"))
        return None
    except Exception as e:
        print("hata:", e)
        return None
    r.update({"phase": label, "case": case, "rep": d["repetition"],
              "frozen_input_tokens": d.get("input_tokens_frozen"), "cap": max_tokens, "thinking": think})
    with OUT.open("a") as fh:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    dec = f"{r['decode_tps']:.1f}" if r["decode_tps"] else "—"
    fa = f"{r['first_answer_s']:.1f}s" if r["first_answer_s"] else "YOK"
    print(f"{label:8} {case:10} r{r['rep']} · in {r['in_tokens']} · out {r['out_tokens']} · "
          f"düşünme {r['reason_chars']} krkt · cevap {r['answer_chars']} krkt · ilk cevap {fa} · "
          f"decode {dec} tok/s · {r['total_s']:.0f}s · finish {r['finish']}")
    return r


def summary(rows, phase):
    rs = [r for r in rows if r["phase"] == phase]
    print(f"\n=== {phase} özeti")
    for case in CASES:
        sub = [r for r in rs if r["case"] == case]
        if not sub:
            continue
        dec = [r["decode_tps"] for r in sub if r["decode_tps"]]
        ans = [r["first_answer_s"] for r in sub if r["first_answer_s"]]
        print(f"{case:10} n={len(sub)} · decode medyan {statistics.median(dec):.1f} tok/s" if dec else f"{case}: decode yok",
              f"· cevap veren {len(ans)}/{len(sub)}" + (f" · ilk cevap medyan {statistics.median(ans):.1f}s" if ans else ""))


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    model = model_name()
    print(f"model: {model}  base: {BASE}  faz: {phase}", flush=True)
    files = load_fixtures()
    rows = []
    if phase in ("all", "std"):
        warm = files[CASES[0]][0]
        print("warm-up (std)…", flush=True)
        do(model, "short", warm, 512, True, "warmup")
        for case in CASES:
            for d in sorted(files.get(case, []), key=lambda x: x["repetition"]):
                r = do(model, case, d, d.get("max_output_tokens") or 512, True, "std")
                if r:
                    rows.append(r)
        summary(rows, "std")
    if phase in ("all", "long"):
        d = sorted(files["short"], key=lambda x: x["repetition"])[0]
        print("\nlong tur (1024 cap, short)…", flush=True)
        for rep in (1, 2):
            r = do(model, "short", {**d, "repetition": rep}, 1024, True, "long")
            if r:
                rows.append(r)
        summary(rows, "long")
    if phase in ("all", "nothink"):
        d = sorted(files["short"], key=lambda x: x["repetition"])[0]
        print("\nthinking kapalı tur (512 cap, short)…", flush=True)
        for rep in (1, 2):
            r = do(model, "short", {**d, "repetition": rep}, 512, False, "nothink")
            if r:
                rows.append(r)
        summary(rows, "nothink")
    print(f"\nham kayıt: {OUT}")


if __name__ == "__main__":
    main()
