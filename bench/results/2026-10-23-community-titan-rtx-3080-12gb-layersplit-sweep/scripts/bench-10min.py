#!/usr/bin/env python3
"""Z840 Strata 10분 성능시험 — 순차 스트리밍 → 동시 4폭주 → 마무리 (stdlib only)."""
import json, os, sys, time, random, threading, urllib.request, statistics

BASE = os.environ.get("BENCH_URL", "http://127.0.0.1:8000/v1/chat/completions")
_with = os.environ.get("STRATA_BENCH_KEY")
KEY = _with or open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".benchkey")).read().strip()
MODEL = "qwen3.8-flash-next-unsloth-ud-iq4_xs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")

args = sys.argv[1:]
def arg(name, default):
    return float(args[args.index(name) + 1]) if name in args else default
SEQ_S = arg("--seq", 540.0)
BURST_S = arg("--burst", 90.0)
BURST_N = int(arg("--n", 4))

SENT = "서버 랙의 전원 분배 회로를 점검할 때는 부하 중심과 상쇄 선로 저항을 함께 고려해야 하며, 특히 급전 경로가 서로 다른 계통과 만나면 접지 전위차가 흘러 센서 신호에 오프셋이 생긴다. "

def make_messages(kind, r):
    if kind == "short":
        return [{"role": "user", "content": f"{r % 97}과 {r % 89}을 더한 값과 그 곱을 계산하고, 각 결과가 짝수인지 홀수인지 설명해줘."}], 200
    if kind == "medium":
        ctx = f"[문서 {r}] " + SENT * 12
        return [{"role": "user", "content": f"아아 아래 문서의 핵심 주장을 세 줄로 요약해줘.\n\n{ctx}"}], 350
    ctx = f"[로그 {r}] " + SENT * 150
    return [{"role": "user", "content": f"아아 위 로그에서 전원/접지 관련 경고만 골라 번호 목록으로 정리해줘.\n\n{ctx}"}], 250

def stream_chat(messages, max_tokens, collect=None):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.3,
                       "stream": True}).encode()
    req = urllib.request.Request(BASE, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time(); ttft = None; c_chunks = r_chunks = 0; c_chars = r_chars = 0
    err = None
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"): continue
                payload = line[5:].strip()
                if payload == "[DONE]": break
                try:
                    d = json.loads(payload)["choices"][0]["delta"]
                except Exception:
                    continue
                c = d.get("content") or ""
                rc = d.get("reasoning_content") or d.get("reasoning") or ""
                if c or rc:
                    if ttft is None: ttft = time.time() - t0
                    if c: c_chunks += 1; c_chars += len(c)
                    if rc: r_chunks += 1; r_chars += len(rc)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    dur = time.time() - t0
    chunks = c_chunks + r_chunks
    tok = max(chunks, round((c_chars + r_chars) / 3.2))
    gen = dur - (ttft or 0)
    rec = {"t": round(time.time(), 2), "dur": round(dur, 2), "ttft": round(ttft or dur, 2),
           "tok": tok, "chunks": chunks, "content_chunks": c_chunks, "reasoning_chunks": r_chunks,
           "tps": round(tok / gen, 1) if gen > 0.5 and not err else None, "err": err}
    if collect is not None: collect.append(rec)
    return rec

def one(kind, r):
    msgs, mt = make_messages(kind, r)
    rec = stream_chat(msgs, mt)
    rec["kind"] = kind
    rec["prompt_chars"] = sum(len(m["content"]) for m in msgs)
    with open(OUT, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec

def main():
    random.seed(20261023)
    open(OUT, "w").close()
    t0 = time.time(); last_report = 0.0
    print(f"[START] seq={SEQ_S:.0f}s burst={BURST_S:.0f}s n={BURST_N} model={MODEL}", flush=True)

    seq = []
    plan = ["short", "short", "medium", "medium", "long"]
    i = 0
    while time.time() - t0 < SEQ_S:
        kind = plan[i % len(plan)]; i += 1
        rec = one(kind, i * 137 + 11)
        seq.append(rec)
        if time.time() - last_report > 30:
            last_report = time.time()
            ok = [r for r in seq if r.get("tps")]
            el = time.time() - t0
            print(f"PROGRESS seq elapsed={el:.0f}s n={len(seq)} "
                  f"median_tps={statistics.median([r['tps'] for r in ok]) if ok else '-'} "
                  f"ttft_med={statistics.median([r['ttft'] for r in seq]):.1f}s "
                  f"errors={sum(1 for r in seq if r['err'])}", flush=True)

    el = time.time() - t0
    print(f"[PHASE] 순차 종료 elapsed={el:.0f}s n={len(seq)} → 동시 {BURST_N}폭주 {BURST_S:.0f}s", flush=True)

    burst = []
    lock = threading.Lock(); bt0 = time.time()
    def worker(wid):
        j = 0
        while time.time() - bt0 < BURST_S:
            j += 1
            msgs, mt = make_messages("short", (wid + 1) * 1000 + j * 7)
            rec = stream_chat(msgs, 120)
            rec["worker"] = wid
            with lock: burst.append(rec)
    ths = [threading.Thread(target=worker, args=(k,)) for k in range(BURST_N)]
    [t.start() for t in ths]; [t.join() for t in ths]
    bdur = time.time() - bt0
    with open(OUT, "a") as f:
        for r in burst: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tail = one("short", 777)

    ok = [r for r in seq if r.get("tps")]
    def med(xs): return round(statistics.median(xs), 2) if xs else None
    def p90(xs):
        xs = sorted(xs); return round(xs[min(len(xs) - 1, int(len(xs) * 0.9))], 2) if xs else None
    bok = [r for r in burst if r.get("tps")]
    agg = round(sum(r["tok"] for r in bok) / bdur, 1) if bok else None
    summary = {
        "seq": {"n": len(seq), "errors": sum(1 for r in seq if r["err"]),
                "tps_med": med([r["tps"] for r in ok]), "tps_min": min([r["tps"] for r in ok], default=None),
                "tps_max": max([r["tps"] for r in ok], default=None),
                "ttft_med": med([r["ttft"] for r in seq]), "ttft_p90": p90([r["ttft"] for r in seq])},
        "seq_by_kind": {k: {"n": len(v), "tps_med": med([r["tps"] for r in v if r.get("tps")]),
                            "ttft_med": med([r["ttft"] for r in v])}
                        for k, v in {kk: [r for r in seq if r["kind"] == kk] for kk in ("short", "medium", "long")}.items()},
        "burst": {"n": len(burst), "errors": sum(1 for r in burst if r["err"]),
                  "streams": BURST_N, "dur": round(bdur, 1), "agg_tps": agg,
                  "per_stream_tps_med": med([r["tps"] for r in bok])},
        "tail": tail, "total_s": round(time.time() - t0, 1),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)

main()
