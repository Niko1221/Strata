#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап IV (п. 7): базовый профиль Strata 0.1.44.1 на 4xV100-SXM2 (<HOSTNAME>).

Методика совпадает с PR #2525 (тот же генератор промптов, тот же стиль PARA_CODE,
temperature=0, reasoning_effort=none, случайная метка в промпте, стриминговый TTFT,
3 повтора на точку, медиана + min/max), но с ОДНИМ изменённым фактором:
ДЛИНА ВЫВОДА ФИКСИРУЕТСЯ.  Движок не имеет ignore_eos/min_tokens для текста, поэтому
EOS-токены (248044 <|endoftext|>, 248046 <|im_end|> — serve/server.py:729, --eos-ids по
умолчанию) запрещаются через logit_bias=-100: генерация идёт ровно до max_tokens,
finish_reason = "length", predicted_n == max_tokens.  Без этого decode-колонка на этой
машине невалидна (модель останавливалась на ~95-110 токенах).

Замер идёт ПРИ ЗАПУЩЕННОМ сервисе: ничего не останавливается, ничего не устанавливается.

Одновременно снимаются:
  - nvidia-smi dmon -s t (rxpci/txpci по 4 картам, 1 с) -> «PCIe-обратный счёт» по окнам
    каждого запроса (NVLink-байтовые счётчики на этом драйвере недоступны);
  - nvidia-smi --query-gpu (util, temp, power, clocks, memory) каждые 2 с;
  - прирост рабочего лога движка (строки «prompt N tokens = ...», hit rate, file tier I/O)
    и поиск CUDA-ошибок в этом приросте.

usage: baseline-profile.py [--out DIR] [--lengths 4k,16k,32k,64k,128k]
                          [--gen-max 1024] [--reps 3] [--no-eos-ban]

Ничего не пишет за пределы --out (кроме stdout).
"""
import argparse
import ast
import datetime
import json
import os
import pathlib
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HOME = pathlib.Path.home()
PROJECT = HOME / "strata185" / "Strata+4xV100+NVlink"
BENCH_SRC = HOME / "strata185" / "bin" / "strata-bench-community.py"   # источник PARA_CODE
ENGINE_LOG = HOME / "strata185" / "Strata" / "strata-185-iq3_s.log"
EOS_IDS = [248044, 248046]          # serve/server.py:729
CUDA_ERR_PAT = re.compile(
    r"no kernel image is available|invalid device function|CUDA error|cudaError|out of memory|"
    r"CUBLAS_STATUS|an illegal memory access|device-side assert", re.I)


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        return ""


# ------------------------------------------------------------------ провенанс

def para_code():
    """Тот же текст промптов, что у PR #2525: берём PARA_CODE из strata-bench-community.py."""
    src = BENCH_SRC.read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PARA_CODE":
            return ast.literal_eval(node.value)
    raise SystemExit("PARA_CODE не найден в " + str(BENCH_SRC))


def find_server():
    for pid in sh(["pgrep", "-f", "serve/server.py"]).split():
        try:
            parts = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            args = [p.decode() for p in parts if p]
        except OSError:
            continue
        cfg, port = None, 8080
        for i, a in enumerate(args):
            if a == "--config" and i + 1 < len(args):
                cfg = args[i + 1]
            if a == "--port" and i + 1 < len(args):
                port = int(args[i + 1])
        if cfg:
            if not os.path.isabs(cfg):
                cfg = os.path.join(os.path.realpath(f"/proc/{pid}/cwd"), cfg)
            return int(pid), cfg, port
    raise SystemExit("Запущенный Strata-сервер (serve/server.py) не найден — замер не делается.")


def http_json(url, key="", data=None, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ------------------------------------------------------------------ фоновые снимки

class GPUSampler(threading.Thread):
    """dmon (rxpci/txpci, 1 с) + query-gpu (util/temp/power/mem, 2 с) — только чтение."""

    FIELDS = ("index,memory.used,utilization.gpu,temperature.gpu,power.draw,"
              "clocks.current.sm,clocks_throttle_reasons.active")

    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop = threading.Event()
        self.dmon_rows = []          # (iso_time, gpu, rx_mb_s, tx_mb_s)
        self.gpu_rows = []           # (iso_time, idx, mem_used_mib, util_pct, temp_c, power_w, sm_mhz, throttle)
        self.errors = []

    def _dmon(self):
        p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "t", "-d", "1"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while not self.stop.is_set():
            line = p.stdout.readline()
            if not line:
                break
            s = line.split()
            if len(s) == 3 and s[0].isdigit():
                self.dmon_rows.append((utc_now(), int(s[0]), float(s[1]), float(s[2])))
        try:
            p.terminate()
        except Exception:                                   # noqa: BLE001
            pass

    def _query(self):
        while not self.stop.is_set():
            out = sh(["nvidia-smi", "--query-gpu=" + self.FIELDS, "--format=csv,noheader,nounits"],
                     timeout=10)
            t0 = utc_now()
            for line in out.splitlines():
                c = [x.strip() for x in line.split(",")]
                if len(c) >= 7:
                    self.gpu_rows.append((t0, int(c[0]), float(c[1]), float(c[2]), float(c[3]),
                                          float(c[4]), float(c[5]), c[6] if len(c) > 7 else ""))
            self.stop.wait(self.interval)

    def run(self):
        t1 = threading.Thread(target=self._dmon, daemon=True)
        t1.start()
        self._query()
        t1.join(timeout=5)

    def halt(self):
        self.stop.set()
        self.join(timeout=15)

    # ---- атрибуция к окну запроса
    def window(self, t0, t1):
        a, b = iso(t0), iso(t1)
        rx = {i: 0.0 for i in range(4)}
        tx = {i: 0.0 for i in range(4)}
        for _, g, r, t in self.dmon_rows:
            if a <= _ <= b:
                rx[g] += r
                tx[g] += t
        peak = {i: {"util_pct": 0.0, "temp_c": 0.0, "power_w": 0.0, "sm_mhz": 0.0,
                    "mem_used_mib": 0.0} for i in range(4)}
        n = 0
        for row in self.gpu_rows:
            _ = row[0]
            if a <= _ <= b:
                n += 1
                g = peak[row[1]]
                g["util_pct"] = max(g["util_pct"], row[3])
                g["temp_c"] = max(g["temp_c"], row[4])
                g["power_w"] = max(g["power_w"], row[5])
                g["sm_mhz"] = max(g["sm_mhz"], row[6])
                g["mem_used_mib"] = max(g["mem_used_mib"], row[2])
        return {"samples_dmon": sum(1 for r in self.dmon_rows if a <= r[0] <= b),
                "samples_query": n,
                "pcie_rx_mb": {g: round(v, 1) for g, v in rx.items()},
                "pcie_tx_mb": {g: round(v, 1) for g, v in tx.items()},
                "peak_per_gpu": peak}


def iso(epoch):
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ запросы

def call_stream(url, key, prompt, max_tokens, ban_eos, want_text=False):
    body = {"model": "strata", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none", "stream": True}
    if ban_eos:
        body["logit_bias"] = {str(i): -100.0 for i in EOS_IDS}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    ttft = None
    content = []
    timings = None
    finish = None
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("timings"):
                timings = d["timings"]
            if d.get("usage"):
                usage = d["usage"]
            ch = (d.get("choices") or [{}])[0]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            delta = (ch.get("delta") or {}).get("content")
            if delta:
                if ttft is None:
                    ttft = time.time() - t0
                content.append(delta)
    wall = time.time() - t0
    if timings is None:
        try:
            st = http_json(url.rsplit("/v1", 1)[0] + "/v1/status", key)
            lt = st.get("last_timings") or {}
            if lt and time.time() - lt.get("at", 0) < 60:
                timings = lt
        except Exception:                                   # noqa: BLE001
            pass
    t = timings or {}
    gen_n = t.get("predicted_n", 0)
    out = "".join(content)
    return {
        "at": iso(t0), "t0": round(t0, 3), "t1": round(wall + t0, 3),
        "prompt_n": t.get("prompt_n", 0), "prompt_ms": t.get("prompt_ms", 0.0),
        "prompt_s": t.get("prompt_per_second", 0.0),
        "gen_n": gen_n, "gen_ms": t.get("predicted_ms", 0.0),
        "gen_s": t.get("predicted_per_second", 0.0),
        "itl_ms": t.get("predicted_per_token_ms", 0.0),
        "draft_n": t.get("draft_n", 0), "draft_ok": t.get("draft_n_accepted", 0),
        "checkpoints": t.get("checkpoints", t.get("n_checkpoints", 0)),
        "cached": (t.get("prompt_tokens_details") or {}).get("cached_tokens", t.get("cache_n", 0)),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "wall_s": round(wall, 2),
        "server_overhead_s": round((ttft - t.get("prompt_ms", 0.0) / 1000.0), 3)
        if (ttft is not None and t) else None,
        "tail_s": round(wall - (ttft + t.get("predicted_ms", 0.0) / 1000.0), 3)
        if (ttft is not None and t) else None,
        "finish": finish,
        "usage_completion": (usage or {}).get("completion_tokens"),
        "text_sha": sha_text(out), "text_len": len(out), "text_head": out[:120],
        "error": None,
    }


def sha_text(s):
    import hashlib
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:16]


def build_prompt(target_tokens, ratio, para, marker=None):
    n = max(1, int(target_tokens / max(1e-6, ratio) / len(para)) + 1)
    mark = f"[метка {marker if marker is not None else random.randint(10**8, 10**9 - 1)}]\n"
    return mark + para * n


def stats(rows, key):
    vals = [x[key] for x in rows if x.get(key) is not None]
    if not vals:
        return None
    s = {"median": round(statistics.median(vals), 2), "min": round(min(vals), 2),
         "max": round(max(vals), 2), "n": len(vals)}
    if len(vals) > 1:
        med = statistics.median(vals)
        s["spread_pct"] = round((max(vals) - min(vals)) / med * 100.0, 2) if med else 0.0
    return s


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--lengths", default="4k,16k,32k,64k,128k")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gen-max", type=int, default=1024)
    ap.add_argument("--warmup-lengths", default="4k,32k",
                    help="прогрев (в замеры не входит): длины промптов, 64 токена каждая")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup-tokens", type=int, default=64)
    ap.add_argument("--no-eos-ban", action="store_true",
                    help="не фиксировать длину вывода (сравнить с методикой PR)")
    args = ap.parse_args()
    ban_eos = not args.no_eos_ban

    out = pathlib.Path(args.out) if args.out else PROJECT / "raw" / "profiles" / utc_stamp()
    out.mkdir(parents=True, exist_ok=False)
    log_lines = [f"[{utc_now()}] baseline-profile.py start", f"out: {out}"]

    def log(s):
        log_lines.append(s)
        print(s, flush=True)

    para = para_code()
    pid, cfg_path, port = find_server()
    cfg = json.loads(pathlib.Path(cfg_path).read_text(encoding="utf-8-sig"))
    key = cfg.get("api_key", "")
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    status = http_json(f"http://127.0.0.1:{port}/v1/status", key)
    log(f"сервер: pid {pid}, порт {port}, конфиг {cfg_path}")
    log(f"движок {status.get('engine')}, контекст {status.get('cache_max_tokens')}, "
        f"uptime {status.get('uptime_s')} с, запросов за сессию {status.get('activity', {}).get('requests')}")

    def parse_len(s):
        s = s.strip().lower()
        return int(float(s[:-1]) * 1024) if s.endswith("k") else int(s)

    lengths = [parse_len(x) for x in args.lengths.split(",") if x.strip()]

    # ---- условия до замера
    pre = {"utc": utc_now(),
           "status": {k: status.get(k) for k in ("engine", "model", "cache_max_tokens", "started",
                                                 "uptime_s", "concurrency", "activity", "machine")},
           "config_sha256": sha_file(cfg_path),
           "config_args": cfg.get("args", []),
           "config_keys_without_secrets": {k: v for k, v in cfg.items() if k != "api_key"},
           "engine_binary_sha256": sha_file(cfg.get("exe", "")),
           "nvidia_smi_before": sh(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,"
                                    "utilization.gpu,temperature.gpu,power.draw,clocks.current.sm",
                                    "--format=csv"]),
           "loadavg": open("/proc/loadavg").read().strip(),
           "uptime": sh(["uptime", "-s"]),
           "engine_log_lines_before": count_lines(ENGINE_LOG)}
    (out / "conditions-before.json").write_text(json.dumps(pre, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    (out / "nvidia-smi-before.csv").write_text(pre["nvidia_smi_before"], encoding="utf-8")
    log("условия до замера собраны")

    # ---- pre-flight: поддерживает ли движок logit_bias (бан EOS) и стриминг
    if ban_eos:
        try:
            probe = call_stream(url, key, "[метка 1]\nThe function parse_record takes a byte slice.",
                                16, True)
            ok = probe["gen_n"] == 16 and probe["finish"] == "length"
            log(f"pre-flight logit_bias(EOS ban): gen_n={probe['gen_n']} finish={probe['finish']} "
                f"-> {'OK' if ok else 'НЕ зафиксирована длина'}")
            if not ok:
                raise SystemExit("бан EOS не дал фиксированной длины — методика не работает")
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            raise SystemExit(f"logit_bias не поддержан движком: {e}")

    # ---- профиль токенизатора (для попадания в целевую длину)
    probe = call_stream(url, key, para * 4 + "[метка 1]", 1, False)
    ratio = probe["prompt_n"] / (len(para) * 4 + 24)   # та же формула, что у PR #2525
    log(f"токенов на символ (PARA_CODE) ≈ {ratio:.4f}")

    warm = [parse_len(x) for x in args.warmup_lengths.split(",") if x.strip()]
    log(f"прогрев {warm} x {args.warmup_tokens} токенов (в замеры не входит; прогревает "
        f"экспертный кэш и графы окон после перезапуска движка)...")
    warmup_rows = []
    for wl in warm:
        t_w = time.time()
        rw = call_stream(url, key, build_prompt(wl, ratio, para), args.warmup_tokens, False)
        rw["case"] = f"warmup-{wl}"
        warmup_rows.append(rw)
        log(f"  warmup {wl}: prompt {rw['prompt_n']} @ {rw['prompt_s']:.1f} ток/с, "
            f"gen {rw['gen_n']} @ {rw['gen_s']:.1f} ток/с, {round(time.time()-t_w,1)} с")

    samp = GPUSampler()
    samp.start()
    time.sleep(2)

    cases = [(f"{n}-prompt", n, args.max_tokens, args.reps) for n in lengths]
    cases.append(("gen-only", 120, args.gen_max, args.reps))

    results = []
    for name, target, gen, reps in cases:
        rows = []
        for i in range(reps):
            prompt = build_prompt(target, ratio, para)
            r = call_stream(url, key, prompt, gen, ban_eos)
            r["case"] = name
            r["target_prompt_tokens"] = target
            r["max_tokens"] = gen
            r["eos_banned"] = ban_eos
            r["prompt_chars"] = len(prompt)
            r["gpu_window"] = samp.window(r["t0"], r["t1"])
            rows.append(r)
            w = r["gpu_window"]
            log(f"  {name} #{i+1}: prompt {r['prompt_n']} @ {r['prompt_s']:.1f} ток/с | "
                f"gen {r['gen_n']}/{gen} @ {r['gen_s']:.1f} ток/с ITL {r['itl_ms']:.2f} мс | "
                f"TTFT {r['ttft_s']} с | wall {r['wall_s']} с | finish {r['finish']} | "
                f"черновики {r['draft_ok']}/{r['draft_n']} | PCIe RX "
                f"{[w['pcie_rx_mb'][g] for g in range(4)]} MB")
        ok_len = all(x["gen_n"] == gen and x["finish"] == "length" for x in rows) if ban_eos else None
        results.append({"case": name, "target_prompt_tokens": target, "max_tokens": gen,
                        "eos_banned": ban_eos, "reps": reps,
                        "fixed_length_ok": ok_len,
                        "prompt_tok_s": stats(rows, "prompt_s"),
                        "prompt_ms": stats(rows, "prompt_ms"),
                        "gen_tok_s": stats(rows, "gen_s"),
                        "itl_ms": stats(rows, "itl_ms"),
                        "ttft": stats(rows, "ttft_s"),
                        "wall": stats(rows, "wall_s"),
                        "server_overhead": stats(rows, "server_overhead_s"),
                        "tail": stats(rows, "tail_s"),
                        "prompt_n": stats(rows, "prompt_n"),
                        "cached": stats(rows, "cached"),
                        "draft_ok": sum(x["draft_ok"] for x in rows),
                        "draft_n": sum(x["draft_n"] for x in rows),
                        "pcie_rx_mb": {g: round(sum(x["gpu_window"]["pcie_rx_mb"][g] for x in rows), 1)
                                       for g in range(4)},
                        "pcie_tx_mb": {g: round(sum(x["gpu_window"]["pcie_tx_mb"][g] for x in rows), 1)
                                       for g in range(4)},
                        "peak_gpu": merge_peaks([x["gpu_window"]["peak_per_gpu"] for x in rows]),
                        "all": rows})

    # ---- проверка корректности: жадная детерминированность + повторное использование префикса
    fixed = "[метка 424242424]\n" + para * 3
    d1 = call_stream(url, key, fixed, 64, False)
    d2 = call_stream(url, key, fixed, 64, False)
    determinism = {"same_text_sha": d1["text_sha"] == d2["text_sha"],
                   "sha_run1": d1["text_sha"], "sha_run2": d2["text_sha"],
                   "cached_run2": d2["cached"], "prompt_n": d2["prompt_n"],
                   "note": "один и тот же промпт дважды, temperature=0"}
    log(f"  determinism: same_text={determinism['same_text_sha']} cached_run2={d2['cached']}")

    samp.halt()

    # ---- условия после + прирост лога движка
    post = {"utc": utc_now(),
            "nvidia_smi_after": sh(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,"
                                    "utilization.gpu,temperature.gpu,power.draw,clocks.current.sm",
                                    "--format=csv"]),
            "engine_log_lines_after": count_lines(ENGINE_LOG),
            "status_after": {k: (lambda s: s.get(k))(http_json(f"http://127.0.0.1:{port}/v1/status", key))
                            for k in ("engine", "uptime_s", "activity", "machine")}}
    delta = read_log_delta(ENGINE_LOG, pre["engine_log_lines_before"], post["engine_log_lines_after"])
    cuda_errors = [l for l in delta if CUDA_ERR_PAT.search(l)]
    summary_lines = [l for l in delta if l.startswith("strata serve: prompt ")
                     or "hit rate" in l or "file tier I/O" in l or "expert tiers" in l
                     or "suffix drafts" in l]
    (out / "conditions-after.json").write_text(json.dumps(post, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    (out / "engine-log-delta.txt").write_text("\n".join(delta), encoding="utf-8")
    (out / "engine-log-summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    (out / "engine-log-cuda-errors.txt").write_text("\n".join(cuda_errors), encoding="utf-8")
    (out / "dmon-rxpci-txpci.csv").write_text(
        "time_utc,gpu,rx_mb_s,tx_mb_s\n" + "\n".join(",".join(map(str, r)) for r in samp.dmon_rows),
        encoding="utf-8")
    (out / "gpu-samples.csv").write_text(
        "time_utc,gpu,mem_used_mib,util_pct,temp_c,power_w,sm_mhz,throttle\n"
        + "\n".join(",".join(str(x) for x in r) for r in samp.gpu_rows), encoding="utf-8")

    method = {
        "prompt_generator": "PARA_CODE из ~/strata185/bin/strata-bench-community.py (тот же текст, "
                            "что у PR #2525), случайная метка в начале -> нет переиспользования "
                            "префиксного кэша",
        "tokens_per_char_measured": round(ratio, 4),
        "sampling": {"temperature": 0, "reasoning_effort": "none", "stream": True,
                     "logit_bias": {str(i): -100.0 for i in EOS_IDS} if ban_eos else None},
        "fixed_output_length": ban_eos,
        "eos_ids_banned": EOS_IDS if ban_eos else None,
        "reps_per_point": args.reps,
        "warmup": f"промпты {warm}, {args.warmup_tokens} токенов каждый, в замеры не входит",
        "warmup_rows": warmup_rows,
        "ttft": "первый непустой streaming-delta (keep-alive игнорируются)",
        "throughput_source": "тайминги движка (prompt_per_second / predicted_per_second) из "
                             "финального chunk",
        "itl": "predicted_per_token_ms из таймингов движка",
        "server_overhead_s": "ttft - prompt_ms/1000 (оценка ожидания/очереди на стороне сервера)",
        "tail_s": "wall - (ttft + predicted_ms/1000) (оценка хвоста запроса)",
        "pcie": "nvidia-smi dmon -s t, 1 с; суммирование rxpci/txpci по окну запроса "
                "(NVLink-байтовые счётчики недоступны: DCGM 4.x без profiling-модуля, "
                "NVML Sampling & Reporting нет)",
        "gpu_sampling": "nvidia-smi --query-gpu каждые 2 с",
        "excluded_runs": "нет (все прогоны записаны; неудачные помечены error/finish)",
    }
    json.dump({"tool": "scripts/baseline-profile.py", "started_utc": pre["utc"],
               "finished_utc": post["utc"], "server_pid": pid, "port": port,
               "engine_version": status.get("engine"), "model": status.get("model"),
               "method": method, "conditions_before": pre, "conditions_after": post,
               "determinism_check": determinism,
               "cuda_error_lines_in_log_delta": len(cuda_errors),
               "results": results},
              open(out / "runs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- обработанная таблица (один ряд на точку)
    csv = ["case,target_prompt_tokens,actual_prompt_n_median,median_reused_tokens,"
           "max_tokens,eos_banned,actual_gen_n_median,reps,"
           "prompt_tok_s_median,prompt_tok_s_min,prompt_tok_s_max,prompt_spread_pct,"
           "decode_tok_s_median,decode_tok_s_min,decode_tok_s_max,decode_spread_pct,"
           "itl_ms_median,itl_ms_min,itl_ms_max,ttft_s_median,ttft_s_min,ttft_s_max,"
           "wall_s_median,server_overhead_s_median,tail_s_median,"
           "draft_accepted,draft_total,draft_accept_pct,"
           "pcie_rx_mb_total_4runs_gpu0,pcie_rx_mb_gpu1,pcie_rx_mb_gpu2,pcie_rx_mb_gpu3,"
           "pcie_tx_mb_gpu0,pcie_tx_mb_gpu1,pcie_tx_mb_gpu2,pcie_tx_mb_gpu3,"
           "peak_vram_mib_gpu0,peak_vram_mib_gpu1,peak_vram_mib_gpu2,peak_vram_mib_gpu3,"
           "peak_util_pct_gpu0,peak_util_pct_gpu1,peak_util_pct_gpu2,peak_util_pct_gpu3,"
           "peak_temp_c_gpu0,peak_power_w_gpu0,peak_sm_mhz_gpu0,fixed_length_ok,"
           "dmon_samples_in_windows,query_samples_in_windows"]
    for r in results:
        def g(s, k):
            return "" if not s else s.get(k, "")
        pk = r["peak_gpu"]
        csv.append(",".join(str(x) for x in [
            r["case"], r["target_prompt_tokens"], g(r["prompt_n"], "median"),
            g(r["cached"], "median"), r["max_tokens"], int(bool(r["eos_banned"])),
            g(stats(r["all"], "gen_n"), "median"), r["reps"],
            g(r["prompt_tok_s"], "median"), g(r["prompt_tok_s"], "min"),
            g(r["prompt_tok_s"], "max"), g(r["prompt_tok_s"], "spread_pct"),
            g(r["gen_tok_s"], "median"), g(r["gen_tok_s"], "min"), g(r["gen_tok_s"], "max"),
            g(r["gen_tok_s"], "spread_pct"),
            g(r["itl_ms"], "median"), g(r["itl_ms"], "min"), g(r["itl_ms"], "max"),
            g(r["ttft"], "median"), g(r["ttft"], "min"), g(r["ttft"], "max"),
            g(r["wall"], "median"), g(r["server_overhead"], "median"), g(r["tail"], "median"),
            r["draft_ok"], r["draft_n"],
            round(r["draft_ok"] / r["draft_n"] * 100, 1) if r["draft_n"] else "",
            *[r["pcie_rx_mb"][i] for i in range(4)], *[r["pcie_tx_mb"][i] for i in range(4)],
            *[pk[i]["mem_used_mib"] for i in range(4)],
            *[pk[i]["util_pct"] for i in range(4)],
            pk[0]["temp_c"], pk[0]["power_w"], pk[0]["sm_mhz"],
            int(bool(r["fixed_length_ok"])),
            sum(x["gpu_window"]["samples_dmon"] for x in r["all"]),
            sum(x["gpu_window"]["samples_query"] for x in r["all"])]))
    (out / "baseline-points.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")

    log(f"CUDA-ошибок в приросте лога движка: {len(cuda_errors)}")
    log("готово: " + str(out))
    (out / "run-notes.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def sha_file(p):
    import hashlib
    if not p or not os.path.exists(p):
        return "not found"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(p):
    try:
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return -1


def read_log_delta(p, a, b):
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(max(0, a)):
                next(f, None)
            return [l.rstrip("\n") for l in f]
    except OSError:
        return []


def merge_peaks(list_of_peaks):
    out = {i: {"util_pct": 0.0, "temp_c": 0.0, "power_w": 0.0, "sm_mhz": 0.0,
               "mem_used_mib": 0.0} for i in range(4)}
    for pk in list_of_peaks:
        for i in range(4):
            for k in out[i]:
                out[i][k] = max(out[i][k], pk.get(i, {}).get(k, 0.0))
    return out


main()
