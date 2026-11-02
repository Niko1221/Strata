"""训练路线对比 harness：反传 sft.py vs MeZO sft_mezo.py，外加纯推理基线。

同一份模型 / 同一份数据 / 同一组 LoRA 挂载点，逐变体跑一次独立进程，外部采样
  - CPU 内存：进程树 RSS（psutil）
  - GPU 显存：nvidia-smi 整卡 used（含 CUDA context，绝对值偏大，差值可比）
  - 速度：训练脚本自己打的每步 dt / tok/s
输出 markdown 对比表 + JSON。

用法：
  python3 bench_cmp.py --gguf <shards...> --pack /root/experts_nf4.bin \
      --tokenizer <权重目录>/tokenizer --steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def sample(pid: int, stop: threading.Event, flag: dict, out: list):
    proc = psutil.Process(pid)
    while not stop.is_set():
        try:
            rss = proc.memory_info().rss
            for c in proc.children(recursive=True):
                try:
                    rss += c.memory_info().rss
                except Exception:
                    pass
        except Exception:
            break
        try:
            gpu = float(subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip().split("\n")[0])
        except Exception:
            gpu = float("nan")
        out.append((time.time(), rss / 2**30, gpu / 1024.0, bool(flag.get("train"))))
        time.sleep(0.5)


def run_variant(name: str, argv: list[str], cwd: str) -> dict:
    print(f"\n===== {name} =====", flush=True)
    t0 = time.time()
    p = subprocess.Popen([PY, "-u"] + argv, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    stop, flag, samples = threading.Event(), {}, []
    th = threading.Thread(target=sample, args=(p.pid, stop, flag, samples), daemon=True)
    th.start()

    steps, lines, before, after = [], [], None, None
    for line in p.stdout:
        line = line.rstrip()
        if re.match(r"^step \d+:", line):
            flag["train"] = True
            steps.append(line)
        if line.startswith("before:"):
            before = line
        if line.startswith("after:"):
            after = line
        lines.append(line)
        if any(k in line for k in ("step ", "before:", "after:", "Traceback", "Error", "saved")):
            print("  " + line, flush=True)
    p.wait()
    stop.set()
    th.join(timeout=5)

    wall = time.time() - t0
    load = [s for s in samples if not s[3]]
    train = [s for s in samples if s[3]]
    rss_peak = max((s[1] for s in samples), default=0.0)
    rss_load = max((s[1] for s in load), default=0.0)
    rss_train = max((s[1] for s in train), default=0.0)
    gpu_peak = max((s[2] for s in samples), default=0.0)
    gpu_load = max((s[2] for s in load), default=0.0)
    gpu_train = max((s[2] for s in train), default=0.0)

    dts, tps, losses, pk = [], [], [], []
    for l in steps:
        m = re.search(r"([-\d.]+)s\s+peak", l)
        if m:
            dts.append(float(m.group(1)))
        m = re.search(r"peak ([\d.]+) GiB", l)
        if m:
            pk.append(float(m.group(1)))
        m = re.search(r"tok/s ([-\d.]+)", l)
        if m:
            tps.append(float(m.group(1)))
        m = re.search(r"loss[+-]? ([-\d.]+)", l) or re.search(r"loss ([-\d.]+)", l)
        if m:
            try:
                losses.append(float(m.group(1)))
            except ValueError:      # 纯推理基线打的是占位符 "loss -"
                pass

    def probe(line):
        if not line:
            return None
        m = re.search(r"= ([-\d.]+)", line)
        return float(m.group(1)) if m else None

    def med(xs):
        # 第 1 步是冷启动（专家 streaming 缓存未热、CUDA 未 warmup），中位数不看它
        xs = xs[1:] if len(xs) >= 3 else xs
        return sorted(xs)[len(xs) // 2] if xs else None

    return {
        "variant": name,
        "wall_s": round(wall, 1),
        "steps": len(steps),
        "s_per_step_first": round(dts[0], 3) if dts else None,
        "s_per_step_med": round(med(dts), 3) if dts else None,
        "s_per_step_best": round(min(dts), 3) if dts else None,
        "tok_s_med": round(med(tps), 1) if tps else None,
        "tok_s_best": round(max(tps), 1) if tps else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "probe_before": probe(before),
        "probe_after": probe(after),
        "rss_peak_gib": round(rss_peak, 2),
        "rss_after_load_gib": round(rss_load, 2),
        "rss_train_gib": round(rss_train, 2),
        "gpu_peak_gib": round(gpu_peak, 2),
        "gpu_after_load_gib": round(gpu_load, 2),
        "gpu_train_gib": round(gpu_train, 2),
        "torch_peak_train_gib": round(max(pk), 2) if pk else None,
        "rc": p.returncode,
        "log": lines,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--sft-lr", type=float, default=1e-4)
    ap.add_argument("--mezo-lr", type=float, default=1e-6)
    ap.add_argument("--mezo-eps", type=float, default=1e-3)
    ap.add_argument("--pretrain-data", default="",
                    help="三个变体共用同一份输入块（否则吞吐按各自数据长度不可比）")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0,
                    help="三个变体共用同一条数据顺序（同 seed），否则命中专家数不同，"
                         "tok/s 的差异里混着数据噪声")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="/root/bench_cmp.json")
    args = ap.parse_args()

    from lora import DEFAULT_TARGETS
    targets = ",".join(DEFAULT_TARGETS)
    gg = args.gguf
    base = ["--gguf"] + gg + ["--pack", args.pack, "--tokenizer", args.tokenizer]
    if args.pretrain_data:
        base += ["--pretrain-data", args.pretrain_data, "--seqlen", str(args.seqlen),
                 "--seed", str(args.seed)]

    variants = [
        # 顺序：基线 -> 反传(开 ckpt，长序列的默认可行档) -> MeZO -> 反传(不开 ckpt，显存上限探针，可能 OOM)
        ("infer", [os.path.join(HERE, "bench_infer.py")] + base
         + ["--samples", str(args.steps), "--rank", str(args.rank), "--alpha", str(args.alpha),
            "--targets", targets]),
        ("sft+ckpt", [os.path.join(HERE, "sft.py")] + base
         + ["--steps", str(args.steps), "--accum", "1", "--lr", str(args.sft_lr), "--ckpt",
            "--rank", str(args.rank), "--alpha", str(args.alpha), "--targets", targets,
            "--out", "/root/cmp_sft_ckpt.pt"]),
        ("mezo", [os.path.join(HERE, "sft_mezo.py")] + base
         + ["--steps", str(args.steps), "--lr", str(args.mezo_lr), "--eps", str(args.mezo_eps),
            "--rank", str(args.rank), "--alpha", str(args.alpha), "--targets", targets,
            "--out", "/root/cmp_mezo.pt"]),
        ("sft", [os.path.join(HERE, "sft.py")] + base
         + ["--steps", str(args.steps), "--accum", "1", "--lr", str(args.sft_lr),
            "--rank", str(args.rank), "--alpha", str(args.alpha), "--targets", targets,
            "--out", "/root/cmp_sft.pt"]),
    ]
    if args.only:
        variants = [v for v in variants if v[0] in args.only.split(",")]

    res = []
    for name, argv in variants:
        r = run_variant(name, argv, HERE)
        res.append(r)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    print("\n\n===== 绝对值 =====")
    hdr = ["variant", "s/step", "tok/s", "loss 首→末", "显存训练段 GiB", "显存加载段 GiB",
           "RSS 训练段 GiB", "RSS 加载段 GiB", "probe Δ"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in res:
        d = (r["probe_after"] - r["probe_before"]) if (r["probe_after"] is not None
                                                       and r["probe_before"] is not None) else None
        print("| " + " | ".join([
            r["variant"],
            f"{r['s_per_step_med']} (best {r['s_per_step_best']})"
            if r["s_per_step_med"] is not None else "-",
            f"{r['tok_s_med']} (best {r['tok_s_best']})" if r["tok_s_med"] is not None else "-",
            (f"{r['loss_first']:.2f}→{r['loss_last']:.2f}" if r["loss_first"] is not None else "-"),
            f"{r['torch_peak_train_gib']:.2f}" if r['torch_peak_train_gib'] else "-",
            f"{r['gpu_after_load_gib']:.2f}",
            f"{r['rss_train_gib']:.2f}", f"{r['rss_after_load_gib']:.2f}",
            f"{d:+.2f}" if d is not None else "-",
        ]) + " |")

    base = next((r for r in res if r["variant"] == "infer"), None)
    if base is not None:
        def sub(x, y):
            return None if (x is None or y is None) else x - y

        print("\n===== 相对纯推理基线的「额外」占用 / 减速 =====")
        hdr2 = ["variant", "额外显存 GiB", "额外 RSS GiB", "速度 vs 推理", "每步额外 s"]
        print("| " + " | ".join(hdr2) + " |")
        print("|" + "---|" * len(hdr2))
        for r in res:
            dv = sub(r["torch_peak_train_gib"], base["torch_peak_train_gib"])
            dr = sub(r["rss_train_gib"], base["rss_train_gib"])
            ratio = (r["tok_s_med"] / base["tok_s_med"]) if (r["tok_s_med"] and base["tok_s_med"]) else None
            dts = sub(r["s_per_step_med"], base["s_per_step_med"])
            print("| " + " | ".join([
                r["variant"],
                f"{dv:+.2f}" if dv is not None else "-",
                f"{dr:+.2f}" if dr is not None else "-",
                f"{ratio * 100:.0f}%" if ratio else "-",
                f"{dts:+.2f}" if dts is not None else "-",
            ]) + " |")
        print("\n（额外显存 = 训练段 torch peak − 推理段 torch peak，两者都在训练前 reset_peak 排除了加载期；"
              "\n  RSS 为进程树常驻集，各变体加载同一份 pack，差值即训练态增量）")

    print(f"\njson: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
