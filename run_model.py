#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RUN-MODEL - 一条命令启动 Qwen3.8-Flash-Next (IQ3_S) 的推理。

    RUN-MODEL.bat                  网页版，8080 端口，本机实测最快的配置
    RUN-MODEL.bat mi50             同上，并挂上 MI50 (gfx906) 专家层  [实验性：实测更慢]
    RUN-MODEL.bat ask "问题"        命令行问一句，流式打印，问完自动收摊
    RUN-MODEL.bat ask              交互式命令行，模型只加载一次，可以连着问
    RUN-MODEL.bat stop             停掉正在跑的推理服务（连同引擎）
    RUN-MODEL.bat check            只做体检，不启动
    RUN-MODEL.bat port 8081        换端口
    RUN-MODEL.bat ask "问题" tokens 1024   放开生成长度上限（默认 512）

配置直接由 strata-iq3_s.json 派生（模型路径、量化、投机参数全部沿用原值），
只在需要时替换 exe 并追加 --peer-ocl。原文件一个字节都不改。

本机实测（2026-10-06，RTX 5060 Ti 16G + 128G 内存，2749 token 提示词 / 400 token 生成）：
    不开 MI50 层   39.3 tok/s
    开   MI50 层   30.3 tok/s   （专家命中率 65% -> 84%，但每 round 多 12.6 ms）
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
BASE_CFG = ROOT / "strata-iq3_s.json"
SERVER = ROOT / "serve" / "server.py"
STOCK_EXE = ROOT / "engine" / "strata.exe"          # 官方预编译版，不含 MI50 支持
BUILD_EXE = ROOT / "build-cuda" / "strata.exe"      # 本地编译版，含 --peer-ocl (MI50)
PID_FILE = ROOT / ".strata-run.pid"
SERVER_OUT = ROOT / "strata-run-server.out.log"

# 配置里这些参数后面跟的是路径，体检时逐个验它存在
PATH_FLAGS = ("--pack", "--native", "--ple-gguf", "--expert-profile", "--mtp", "--tokenizer")

BAR = "=" * 74


def human(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}T"


def dir_size(p: pathlib.Path) -> int:
    """目录大小，只算一层，够用于体检提示；文件直接取大小。"""
    try:
        if p.is_file():
            return p.stat().st_size
        total = 0
        for c in p.iterdir():
            try:
                total += c.stat().st_size
            except OSError:
                pass
        return total
    except OSError:
        return 0


# ---------------------------------------------------------------- 配置

def load_base() -> dict:
    if not BASE_CFG.exists():
        die(f"找不到配置文件：{BASE_CFG}")
    return json.loads(BASE_CFG.read_text(encoding="utf-8"))


def strip_flag(args: list[str], name: str) -> list[str]:
    """删掉 name 及其后面那个值，避免重复追加时出现两个 --peer-ocl。"""
    out, i = [], 0
    while i < len(args):
        if args[i] == name:
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def arg_paths(args: list[str]) -> list[tuple[str, str]]:
    found = []
    for i, a in enumerate(args[:-1]):
        if a in PATH_FLAGS:
            found.append((a, args[i + 1]))
    return found


# ---------------------------------------------------------------- 体检

def check(cfg: dict, port: int, verbose: bool = True) -> bool:
    """逐项验路径、验二进制、验端口。verbose=False 时只在有问题时才说话。返回能不能跑。"""
    rows: list[tuple[str, str, str, str | None]] = []

    def line(mark: str, what: str, extra: str = "", bad_item: str | None = None) -> None:
        rows.append((mark, what, extra, bad_item))

    # Python
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if py.exists():
        line("ok", "Python 运行环境", ".venv/Scripts/python.exe")
    else:
        line("!!", "Python 运行环境", "缺 .venv，先跑 START-HERE.bat", "python")

    # 引擎
    stock, build = STOCK_EXE.exists(), BUILD_EXE.exists()
    if stock:
        line("ok", "官方引擎 engine/strata.exe", f"{human(dir_size(STOCK_EXE))}（无 MI50 支持）")
    if build:
        line("ok", "自编译引擎 build-cuda/strata.exe", f"{human(dir_size(BUILD_EXE))}（含 --peer-ocl）")
    if not stock and not build:
        line("!!", "引擎二进制", "engine/ 和 build-cuda/ 都没有", "exe")

    # 模型与数据
    for flag, raw in arg_paths(cfg["args"]):
        p = pathlib.Path(raw)
        if p.exists():
            line("ok", f"{flag} {p.name}", human(dir_size(p)))
        else:
            line("!!", f"{flag} {p.name}", str(p), flag)

    # tokenizer
    tok = cfg.get("tokenizer")
    if tok:
        if pathlib.Path(tok).exists():
            line("ok", "tokenizer", pathlib.Path(tok).name)
        else:
            line("!!", "tokenizer", str(tok), "tokenizer")

    # 视觉模块
    vis = cfg.get("vision") or {}
    if vis.get("mmproj"):
        if pathlib.Path(vis["mmproj"]).exists():
            line("ok", "视觉投影 mmproj", pathlib.Path(vis["mmproj"]).name)
        else:
            line("!!", "视觉投影 mmproj", str(vis["mmproj"]), "mmproj")
    if vis.get("exe"):
        if not pathlib.Path(vis["exe"]).exists():
            line("!!", "视觉引擎 strata-vision.exe", str(vis["exe"]), "vision")

    # CUDA 运行库
    for d in cfg.get("lib_dirs") or []:
        p = pathlib.Path(d)
        if p.exists():
            line("ok", "CUDA 运行库目录", p.name + "/")
        else:
            line("!!", "CUDA 运行库目录", str(p), "cuda")

    # OpenCL（MI50 用）：ICD 是注册表登记的，不在 System32 里
    if BUILD_EXE.exists():
        if amd_opencl_icd() is not None:
            line("ok", "AMD OpenCL 运行库", "已注册 ICD，MI50 层可用")
        elif pathlib.Path(r"C:\Windows\System32\OpenCL.dll").exists():
            line("--", "AMD OpenCL 运行库", "只有 ICD 加载器，没登记 AMD 驱动，MI50 层跑不了")
        else:
            line("--", "AMD OpenCL 运行库", "没找到，MI50 层跑不了")

    # 端口
    if port_free(port):
        line("ok", f"端口 {port}", "空闲")
    elif health(port):
        line("--", f"端口 {port}", f"已有一个在跑，网页版会直接打开它")
    else:
        line("!!", f"端口 {port}", "被别的程序占用，加 port 8081 换一个", "port")

    bad = [r[3] for r in rows if r[3]]

    if verbose:
        print(BAR)
        print("  体检")
        print(BAR)
        for mark, what, extra, _ in rows:
            print(f"  [{mark}] {what:<46} {extra}")
        print(BAR)
    elif bad:
        print(BAR)
        for mark, what, extra, _ in rows:
            if mark != "ok":
                print(f"  [{mark}] {what:<46} {extra}")
        print(BAR)

    if bad:
        print(f"  ✗ 有 {len(bad)} 项没过：{'、'.join(bad)}")
        print("    先解决上面标 [!!] 的项再启动。")
        return False
    if verbose:
        print("  ✓ 全部通过")
    return True


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def amd_opencl_icd() -> str | None:
    """AMD 的 OpenCL ICD 只登记在注册表里（文件躺在 DriverStore），所以查注册表而不是查 System32。"""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\OpenCL\Vendors") as k:
            for i in range(winreg.QueryInfoKey(k)[1]):
                name, _, _ = winreg.EnumValue(k, i)
                if "amdocl" in name.lower():
                    return name
    except OSError:
        pass
    return None


def health(port: int) -> bool:
    try:
        with LOCAL.open(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


# 有的机器上配了 http_proxy，直连 127.0.0.1 也会被代理接走，结果拿到 502 而不是「连不上」。
LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------------------------------------------------------- 生成配置

def make_config(cfg: dict, use_mi50: bool, engine: str, port: int) -> tuple[pathlib.Path, pathlib.Path, dict]:
    """从 strata-iq3_s.json 派生一份运行配置，返回 (配置路径, 引擎日志路径, 派生后的配置)。"""
    out = dict(cfg)
    args = strip_flag(list(cfg["args"]), "--peer-ocl")

    if use_mi50:
        exe = BUILD_EXE
        if not exe.exists():
            die("要开 MI50 层就得用 build-cuda/strata.exe，但那个文件不在。")
        # 每层只剩 ~1.4 个 CPU 专家时，默认的 21 worker 纯付屏障开销；
        # 扫描实测 12 worker 最优（36.1 vs 32.2 t/s），见 bench/sweep_workers.out。
        args += ["--peer-ocl", "0", "--pool-workers", "12"]
    else:
        exe = BUILD_EXE if engine == "build" else STOCK_EXE
        if not exe.exists():
            exe = BUILD_EXE if BUILD_EXE.exists() else STOCK_EXE
            if not exe.exists():
                die("找不到任何可用的引擎二进制。")

    tag = tag_of(use_mi50)
    out["exe"] = str(exe)
    out["args"] = args
    out["cwd"] = str(ROOT)
    out["port"] = port
    out["model_name"] = "qwen3.8-flash-next-iq3_s-" + tag
    log = ROOT / f"strata-run-{tag}.log"
    out["log"] = str(log)

    cfgp = ROOT / f"strata-run-{tag}.json"
    cfgp.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return cfgp, log, out


def tag_of(use_mi50: bool) -> str:
    return "mi50" if use_mi50 else "fast"


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def banner(use_mi50: bool, exe: pathlib.Path, port: int, arena: float) -> None:
    print(BAR)
    print("  Strata — Qwen3.8-Flash-Next  IQ3_S")
    print(BAR)
    print(f"  引擎      {rel(exe)}")
    if use_mi50:
        print(f"  专家层    MI50 (gfx906) 已挂上，显存预算 全部空闲（自动，≈31 GiB）")
        print(f"  实测速度  ≈36 tok/s —— 与单卡打平，收益在省下的系统内存带宽上")
    else:
        print(f"  专家层    关（只用 RTX 5060 Ti 的 16G）")
        print(f"  实测速度  ≈37 tok/s")
    print(f"  地址      http://127.0.0.1:{port}")
    print(f"  日志      strata-run-{tag_of(use_mi50)}.log")
    print(BAR)


# ---------------------------------------------------------------- 启动

def server_cmd(py: pathlib.Path, cfgp: pathlib.Path, port: int, open_page: bool) -> list[str]:
    cmd = [str(py), str(SERVER), "--engine", "strata", "--config", str(cfgp), "--port", str(port)]
    if open_page:
        cmd.append("--open")
    return cmd


def server_env() -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def kill_tree(pid: int) -> bool:
    """连同引擎一起停掉：引擎是 server 的子进程，/T 才能保证不留下一个占着 60G 内存的孤儿。"""
    r = subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r == 0


def engine_pids() -> set[int]:
    """当前所有的引擎进程（strata.exe / strata-vision.exe）。"""
    pids: set[int] = set()
    for image in ("strata.exe", "strata-vision.exe"):
        r = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"],
                           capture_output=True, text=True, errors="replace")
        for ln in (r.stdout or "").splitlines():
            ln = ln.strip()
            if not ln.startswith('"'):
                continue
            parts = [x.strip('"') for x in ln.split('","')]
            if len(parts) >= 2 and parts[1].isdigit():
                pids.add(int(parts[1]))
    return pids


def wait_engine_gone(before: set[int], seconds: float) -> set[int]:
    """等引擎真的退出。60G 内存的释放不是瞬间的，不等就立刻重开容易撞上内存/端口。"""
    t0 = time.time()
    while time.time() - t0 < seconds:
        left = engine_pids() - before
        if not left:
            return set()
        time.sleep(0.5)
    return engine_pids() - before


def wait_health(port: int, seconds: float, what: str) -> bool:
    t0, spin = time.time(), "|/-\\"
    i = 0
    while time.time() - t0 < seconds:
        if health(port):
            sys.stdout.write("\r" + " " * 70 + "\r")
            return True
        sys.stdout.write(f"\r  {spin[i % 4]} {what} … {time.time() - t0:.0f}s")
        sys.stdout.flush()
        i += 1
        time.sleep(1.0)
    sys.stdout.write("\r" + " " * 70 + "\r")
    return False


def show_speed(log: pathlib.Path) -> None:
    if not log.exists():
        return
    hits = []
    for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("strata serve:") and re.search(r"\d+ tok/s", ln):
            hits.append(ln.strip())
    if hits:
        print("\n  引擎自报的速度：")
        for ln in hits[-3:]:
            print("    " + ln)


def do_web(py: pathlib.Path, cfg: dict, use_mi50: bool, exe: pathlib.Path, port: int) -> int:
    if not port_free(port) and health(port):
        print(f"  端口 {port} 上已经有一个推理服务在跑了，直接把网页打开。")
        webbrowser_open(f"http://127.0.0.1:{port}")
        return 0
    banner(use_mi50, exe, port, ARENA[0])
    print("  正在加载模型（约 1-2 分钟，第一次会久一点）…\n")
    cfgp, _, _ = make_config(cfg, use_mi50, ENGINE[0], port)
    # STRATA_NO_OPEN=1 时不弹浏览器，方便无头启动或脚本里跑
    p = subprocess.Popen(server_cmd(py, cfgp, port, open_page=os.environ.get("STRATA_NO_OPEN") != "1"),
                         cwd=str(ROOT), env=server_env())
    PID_FILE.write_text(str(p.pid), encoding="ascii")
    try:
        return p.wait()
    except KeyboardInterrupt:
        print("\n  Ctrl+C —— 正在停掉服务 …")
        kill_tree(p.pid)
        return 130
    finally:
        PID_FILE.unlink(missing_ok=True)
        left = wait_engine_gone(set(), 90)
        if left:
            print(f"  !! 还有引擎没退：{sorted(left)}，可以跑 RUN-MODEL.bat stop")


def ask_once(port: int, model: str, question: str, max_tokens: int) -> bool:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = json.dumps({"model": model, "stream": True, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": question}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    print(f"  > {question}\n")
    t0 = time.time()
    got = False
    try:
        with LOCAL.open(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                piece = (j.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                if piece:
                    got = True
                    sys.stdout.write(piece)
                    sys.stdout.flush()
    except Exception as exc:                                  # noqa: BLE001
        print(f"\n  !! 请求失败：{type(exc).__name__}: {exc}")
        return False
    if not got:
        print("  (模型没吐内容 —— 可能撞上了思考长度上限，加 tokens N 放宽试试)")
    print(f"\n\n  用时 {time.time() - t0:.1f} s\n")
    return True


def do_ask(py: pathlib.Path, cfg: dict, use_mi50: bool, exe: pathlib.Path, port: int,
           questions: list[str], max_tokens: int) -> int:
    banner(use_mi50, exe, port, ARENA[0])
    cfgp, log, derived = make_config(cfg, use_mi50, ENGINE[0], port)
    already = engine_pids()

    proc: subprocess.Popen | None = None
    if health(port):
        print("  端口上已有服务在跑，直接用它（用完不会关掉它）。\n")
    else:
        print("  正在加载模型（约 1-2 分钟）…")
        out = open(SERVER_OUT, "ab")
        proc = subprocess.Popen(server_cmd(py, cfgp, port, open_page=False), cwd=str(ROOT),
                                env=server_env(), stdin=subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT)
        PID_FILE.write_text(str(proc.pid), encoding="ascii")
        if not wait_health(port, 900, "模型加载中"):
            print("  !! 模型一直没起来，看看 strata-run-server.out.log")
            kill_tree(proc.pid)
            PID_FILE.unlink(missing_ok=True)
            return 1
        print("  模型就绪。\n")

    try:
        for q in questions:
            ask_once(port, derived["model_name"], q, max_tokens)
        if not questions:
            print("  直接打字提问，空行或 Ctrl+C 退出。\n")
            while True:
                try:
                    q = input("  问> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if q in ("", "exit", "quit", "q"):
                    break
                ask_once(port, derived["model_name"], q, max_tokens)
    finally:
        show_speed(log)
        if proc is not None and proc.poll() is None:
            print("  正在停掉服务（连引擎一起）…")
            kill_tree(proc.pid)
            left = wait_engine_gone(already, 90)
            print("  已全部退出。" if not left else f"  !! 还有引擎没退：{sorted(left)}")
        PID_FILE.unlink(missing_ok=True)
    return 0


def webbrowser_open(url: str) -> None:
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def do_stop() -> int:
    if not PID_FILE.exists():
        print("  没有本脚本启动过的服务记录（PID 文件不在）。")
        print("  如果那个窗口还开着，直接关掉窗口就行。")
        return 0
    pid = PID_FILE.read_text(encoding="ascii").strip()
    PID_FILE.unlink(missing_ok=True)
    ok = kill_tree(int(pid))
    left = wait_engine_gone(set(), 90)
    print(f"  已停掉服务（PID {pid}，连同引擎）。" if ok else f"  PID {pid} 已经不在了。")
    if left:
        print(f"  !! 还有引擎没退：{sorted(left)}，可以再跑一次 stop")
    return 0


def die(msg: str) -> None:
    print(f"  !! {msg}")
    raise SystemExit(1)


# ---------------------------------------------------------------- main

ENGINE: list[str] = ["stock"]
ARENA: list[float] = [13.0]


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("mode", nargs="?", default="web",
                    choices=["web", "mi50", "ask", "stop", "check", "port", "help"])
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()

    mode, rest = a.mode, list(a.rest)
    if mode == "help":
        print(__doc__)
        return 0

    # 抽出 options，剩下的才是问题正文：不能把 "tokens 512" 混进问题里
    def take_opt(name: str, default: int) -> int:
        if name in rest:
            i = rest.index(name)
            if i + 1 < len(rest):
                try:
                    v = int(rest[i + 1])
                except ValueError:
                    v = default
                del rest[i:i + 2]
                return v
        return default

    port = take_opt("port", 8080)
    max_tokens = take_opt("tokens", 512)

    want_mi50 = mode == "mi50" or (mode == "ask" and "mi50" in rest)
    if mode == "ask" and "mi50" in rest:          # ask 也能挂上 MI50：RUN-MODEL.bat ask mi50 "问题"
        rest = [x for x in rest if x != "mi50"]

    # 环境变量在起服务之前设好，server 会把它们传给引擎进程
    os.environ.pop("STRATA_OCL_ARENA_GIB", None)

    cfg = load_base()
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        die("缺 .venv\\Scripts\\python.exe —— 先双击 START-HERE.bat 装一次环境。")

    if mode == "stop":
        return do_stop()
    if mode == "check":
        return 0 if check(cfg, port) else 1

    use_mi50 = want_mi50
    # mi50 模式不设 STRATA_OCL_ARENA_GIB：引擎默认吃"全部空闲显存 - 自身预留"，
    # 即基准验证过的 31.4 GiB / ~16.5k 专家驻留。此前这里写死 13，把 MI50 削到四成，
    # 用户看到的正是"MI50 只占 12.5G"。
    exe = BUILD_EXE if use_mi50 else (STOCK_EXE if STOCK_EXE.exists() else BUILD_EXE)

    if not check(cfg, port, verbose=False):
        return 1
    if use_mi50 and amd_opencl_icd() is None:
        print("  !! 要开 MI50 层，但系统里没有登记 AMD 的 OpenCL 驱动。装 AMD 显卡驱动后再试。")
        return 1
    print()

    if mode == "ask":
        q = " ".join(rest).strip()
        questions = [q] if q else []
        return do_ask(py, cfg, use_mi50, exe, port, questions, max_tokens)

    return do_web(py, cfg, use_mi50, exe, port)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n  已中断。")
        raise SystemExit(130)
