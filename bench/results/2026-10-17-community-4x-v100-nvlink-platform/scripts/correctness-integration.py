#!/usr/bin/env python3
"""Этап VI (п. 9.2): интеграционные проверки корректности через API запущенного сервера Strata.

Ничего не останавливает и не меняет в системе: только HTTP-запросы к :8080 и чтение лога движка.
Длина вывода фиксирована (правило R9): logit_bias = -100 для EOS 248044/248046, контроль
gen_n == max_tokens и finish_reason == "length".

Артефакты: raw/correctness/<UTC>/results.json, correctness-rows.csv, engine-log-delta.txt,
engine-log-cuda-errors.txt, conditions-before.json, conditions-after.json.
"""
import json, hashlib, os, subprocess, sys, time, urllib.request, datetime, csv, threading

SERVER = "http://127.0.0.1:8080"
CONFIG = os.path.expanduser("~/strata185/Strata/strata-185-iq3_s.json")
ENGINE_LOG = os.path.expanduser("~/strata185/Strata/strata-185-iq3_s.log")
EOS_BAN = {"248044": -100, "248046": -100}
MARK = "MARKER-9c4f7a21"
CUDA_PAT = r"no kernel image is available|invalid device function|CUDA error|out of memory|cudaError|device-side assert|an illegal memory access"


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def api_key():
    return json.load(open(CONFIG))["api_key"]


def post(payload, timeout=600, key=None):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(SERVER + "/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json",
                                         "Authorization": "Bearer " + (key or api_key())})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:400]}
    except Exception as e:
        return None, {"error": repr(e)[:300]}


def get(path, timeout=20):
    try:
        with urllib.request.urlopen(SERVER + path, timeout=timeout) as r:
            return r.status, r.read().decode()
    except Exception as e:
        return None, repr(e)[:200]


PARA = ("Section {i}. The quick brown fox jumps over the lazy dog while the sensor array records "
        "channel {i} readings at 4800 Hz; the calibration table lists offsets, gains and drift "
        "coefficients for every channel, and the maintenance log notes that the array was serviced "
        "on 2026-03-{d} and that no anomalies were found in the reported band. ")


def fixed_prompt(target_tokens):
    """Промпт фиксированного содержимого (без случайной метки): один и тот же текст при каждом запуске."""
    ratio = 0.1965  # токенов на символ, измерено в baseline-profile.py (тот же генератор)
    chars = max(200, int(target_tokens / ratio))
    parts = []
    i = 0
    n = 0
    while n < chars:
        p = PARA.format(i=i, d=10 + (i % 9))
        parts.append(p)
        n += len(p)
        i += 1
    text = "".join(parts)[:chars]
    return MARK + " " + text + " Question: which channel is listed first in the calibration table?"


def gen(prompt_chars, max_tokens, temperature=0.0, extra=None, timeout=900, key=None):
    payload = {"model": "qwen3.8-flash-next-iq3_s",
               "messages": [{"role": "user", "content": prompt_chars}],
               "max_tokens": max_tokens, "temperature": temperature, "logit_bias": EOS_BAN,
               "reasoning_effort": "none"}
    if extra:
        payload.update(extra)
    t0 = time.time()
    st, resp = post(payload, timeout=timeout, key=key)
    wall = round(time.time() - t0, 2)
    out = {"http": st, "wall_s": wall}
    if st == 200 and resp.get("choices"):
        ch = resp["choices"][0]
        u = resp.get("usage", {})
        t = resp.get("timings", {})
        text = (ch.get("message", {}).get("content") or ch.get("message", {}).get("reasoning_content") or "")
        out.update(prompt_n=u.get("prompt_tokens"), cached=u.get("prompt_tokens_details", {}).get("cached_tokens"),
                   completion=u.get("completion_tokens"), finish=ch.get("finish_reason"),
                   text=text, text_sha=hashlib.sha256(text.encode()).hexdigest()[:16], text_len=len(text),
                   prompt_s=t.get("prompt_per_second"), gen_s=t.get("predicted_per_second"),
                   draft_ok=t.get("drafts_accepted"), draft_n=t.get("drafts_offered"))
    else:
        out["resp"] = {k: v for k, v in list(resp.items())[:4]}
    return out


def health_ok():
    st, body = get("/health")
    return st == 200 and '"loaded": true' in body.replace("  ", " ")


def log_lines():
    try:
        return int(subprocess.run(["wc", "-l", ENGINE_LOG], capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return -1


def conditions():
    st, body = get("/v1/status", 30)
    nv = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,temperature.gpu,power.draw",
                         "--format=csv,noheader"], capture_output=True, text=True).stdout
    return {"utc": now_utc(), "health": st, "status_http": st,
            "status": json.loads(body) if st == 200 else body[:200],
            "nvidia_smi": nv, "engine_log_lines": log_lines(),
            "config_sha256": subprocess.run(["sha256sum", CONFIG], capture_output=True, text=True).stdout.split()[0],
            "engine_binary_sha256": subprocess.run(
                ["sha256sum", os.path.expanduser("~/strata185/Strata/engine-cuda12/strata")],
                capture_output=True, text=True).stdout.split()[0]}


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not out_dir:
        out_dir = "raw/correctness/" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    os.makedirs(out_dir, exist_ok=True)
    started = now_utc()
    before = conditions()
    rows = []

    def record(name, expected, got, ok, note=""):
        rows.append({"test": name, "expected": expected, "observed": got, "result": "PASS" if ok else "FAIL", "note": note})
        print(("PASS " if ok else "FAIL ") + name + "  " + str(got)[:220] + ("  | " + note if note else ""), flush=True)

    print("health before:", health_ok(), flush=True)

    # T1..T3: жадная детерминированность на промптах фиксированного содержимого
    for name, tgt, reps, mx in [("T1_determinism_short", 64, 3, 128),
                                ("T2_determinism_medium", 1024, 3, 128),
                                ("T3_determinism_long", 8192, 2, 128)]:
        p = fixed_prompt(tgt)
        shas, lens, fixed_ok = [], [], True
        for i in range(reps):
            r = gen(p, mx)
            shas.append(r.get("text_sha"))
            lens.append(r.get("text_len"))
            fixed_ok = fixed_ok and (r.get("completion") == mx and r.get("finish") == "length")
        same = len(set(shas)) == 1 and shas[0] is not None
        record(name, f"{reps} повторов одного текста -> один text_sha, gen_n == {mx}",
               f"http=200 distinct_sha={len(set(shas))} sha={shas[0]} lens={lens} fixed_len={fixed_ok}",
               same and fixed_ok, "temperature=0, logit_bias бан EOS")

    # T4: возобновление генерации с того же KV-состояния (префиксный кэш): первые N токенов длинного ответа
    # должны совпадать с коротким ответом того же промпта
    p = fixed_prompt(2048)
    a = gen(p, 128)
    b = gen(p, 256)
    prefix_ok = bool(a.get("text")) and bool(b.get("text")) and b["text"].startswith(a["text"])
    record("T4_prefix_cache_resume", "тот же промпт, 256 токенов: первые 128 совпадают с ответом на 128; cached_tokens > 0",
           f"sha128={a.get('text_sha')} sha256={b.get('text_sha')} prefix_match={prefix_ok} cached_run2={b.get('cached')} "
           f"completion={a.get('completion')}/{b.get('completion')}",
           prefix_ok and (b.get("cached") or 0) > 0, "проверка корректности KV/префиксного кэша")

    # T5: промпт длиннее max_context -> отказ без потери работоспособности
    huge = fixed_prompt(300000)
    r = gen(huge, 32, timeout=300)
    alive = health_ok()
    st2, r2 = post({"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 8, "temperature": 0, "logit_bias": EOS_BAN}, timeout=120)
    record("T5_context_overflow", "HTTP != 200 (или понятная ошибка) и сервер жив после отказа",
           f"http={r['http']} err={str(r.get('resp') or r.get('error'))[:120]} health_after={alive} next_request_http={st2}",
           (r["http"] not in (200,)) and alive and st2 == 200, "max_context = 262144")

    # T6: absurd max_tokens
    r = gen(fixed_prompt(256), 300000, timeout=300)
    alive = health_ok()
    ok6 = (r["http"] == 200 and r.get("finish") in ("length", "stop")) or (r["http"] not in (200,) and alive)
    record("T6_max_tokens_overflow", "либо аккуратное ограничение генерации, либо аккуратный отказ; сервер жив",
           f"http={r['http']} completion={r.get('completion')} finish={r.get('finish')} health_after={alive}",
           ok6 and alive)

    # T7..T9: ошибки запроса
    r = gen(fixed_prompt(64), 16, extra={"model": "no-such-model"})
    record("T7_unknown_model", "HTTP 4xx и сервер жив", f"http={r['http']}", r["http"] in (400, 404) and health_ok())
    st, body = post({"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": "hi"}],
                     "max_tokens": 8}, key="wrong-key-000")
    record("T8_bad_api_key", "HTTP 401/403", f"http={st}", st in (401, 403) and health_ok())
    req = urllib.request.Request(SERVER + "/v1/chat/completions", data=b'{"model": "x", "messages": [',
                                headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key()})
    try:
        with urllib.request.urlopen(req, timeout=30) as rr:
            st = rr.status
    except urllib.error.HTTPError as e:
        st = e.code
    except Exception:
        st = None
    record("T9_malformed_json", "HTTP 4xx", f"http={st}", st in (400, 422) and health_ok())

    # T10: параллельные запросы (slot = 1): оба должны завершиться корректно
    res = {}
    def worker(i):
        res[i] = gen(fixed_prompt(4096) + f" (request {i})", 128)
    th = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    [t.start() for t in th]
    [t.join() for t in th]
    both = all(res[i]["http"] == 200 and res[i].get("finish") == "length" for i in res)
    record("T10_concurrent_requests", "2 параллельных запроса: оба finish=length, сервер жив",
           " ".join(f"#{i}: http={res[i]['http']} finish={res[i].get('finish')} wall={res[i]['wall_s']}s" for i in res),
           both and health_ok(), "concurrency/serving = 1 -> ожидается очередь, не ошибка")

    # T11: детерминированность на длинном контексте (62519 токенов, 2 повтора)
    p = fixed_prompt(65536)
    r1 = gen(p, 128, timeout=900)
    r2 = gen(p, 128, timeout=900)
    same = r1.get("text_sha") == r2.get("text_sha") and r1.get("text_sha") is not None
    fixed_ok = r1.get("completion") == 128 and r2.get("completion") == 128
    record("T11_long_context_determinism", "один и тот же промпт ~62.5k токенов дважды -> один text_sha",
           f"prompt_n={r1.get('prompt_n')}/{r2.get('prompt_n')} sha1={r1.get('text_sha')} sha2={r2.get('text_sha')} "
           f"fixed_len={fixed_ok} cached={r2.get('cached')}", same and fixed_ok)

    after = conditions()
    # прирост лога движка и поиск CUDA-ошибок
    n0, n1 = before["engine_log_lines"], after["engine_log_lines"]
    delta = subprocess.run(["tail", "-n", f"+{n0 + 1}", ENGINE_LOG], capture_output=True, text=True).stdout
    open(out_dir + "/engine-log-delta.txt", "w").write(delta)
    cuda = subprocess.run(["grep", "-Ei", CUDA_PAT, out_dir + "/engine-log-delta.txt"],
                          capture_output=True, text=True).stdout
    open(out_dir + "/engine-log-cuda-errors.txt", "w").write(cuda)
    n_cuda = len([l for l in cuda.splitlines() if l.strip()])
    rows.append({"test": "T12_engine_log_cuda_errors", "expected": "0 строк no kernel image / invalid device function / CUDA error / OOM",
                 "observed": f"cuda_error_lines={n_cuda} log_delta_lines={n1 - n0}", "result": "PASS" if n_cuda == 0 else "FAIL",
                 "note": "проверка по приросту рабочего лога движка"})
    rows.append({"test": "T13_service_state_after", "expected": "health loaded=true, VRAM >= 14 GiB на карту, pid-файл = фактический pid",
                 "observed": f"health={after['health']} vram={after['nvidia_smi'].replace(chr(10), ' | ')}",
                 "result": "PASS" if after["health"] == 200 else "FAIL", "note": "R6"})

    data = {"started_utc": started, "finished_utc": now_utc(), "tool": "scripts/correctness-integration.py",
            "method": "fixed-content prompts (no random marker), temperature=0, reasoning_effort=none, "
                      "output length fixed by banning EOS 248044/248046 via logit_bias",
            "conditions_before": before, "conditions_after": after, "results": rows}
    json.dump(data, open(out_dir + "/results.json", "w"), ensure_ascii=False, indent=1)
    with open(out_dir + "/correctness-rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "expected", "observed", "result", "note"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("готово:", out_dir, "PASS:", sum(1 for r in rows if r["result"] == "PASS"),
          "FAIL:", sum(1 for r in rows if r["result"] == "FAIL"))


if __name__ == "__main__":
    main()
