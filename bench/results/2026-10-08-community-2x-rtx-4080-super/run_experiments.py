from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def replace_flags(args, values, switches, remove_values, remove_switches):
    value_flags = set(values) | set(remove_values)
    switch_flags = set(switches) | set(remove_switches)
    out = []
    i = 0
    while i < len(args):
        flag = args[i]
        key = flag.split("=", 1)[0]
        if key in value_flags:
            if "=" not in flag:
                if i + 1 == len(args) or args[i + 1].startswith("--"):
                    raise ValueError(f"missing argument for {flag}")
                i += 1
        elif key not in switch_flags:
            out.append(flag)
        i += 1
    for flag, value in values.items():
        out.extend((flag, str(value)))
    return out + list(switches)


def cases(cfg, plan, devices):
    if (len(devices) != plan["required_devices"] or len(set(devices)) != len(devices)
            or any(type(device) is not int or device < 0 for device in devices)):
        raise ValueError("device count must match plan, with distinct nonnegative ordinals")
    if cfg.get("expert_profile_save") or any("--expert-profile-save" in a for a in cfg["args"]):
        raise ValueError("use an immutable initial expert profile, not a learned-profile save path")
    if cfg.get("env"):
        raise ValueError("move intentional engine environment settings into the explicit experiment plan")
    for flag in ("--pack", "--native", "--mtp"):
        if flag not in cfg["args"]:
            raise ValueError(f"base config must include {flag}")
        index = cfg["args"].index(flag)
        if index + 1 == len(cfg["args"]) or cfg["args"][index + 1].startswith("--"):
            raise ValueError(f"base config has no value for {flag}")
    arms = plan["arms"]
    for repetition in range(plan["repetitions"]):
        ordered = arms if repetition % 2 == 0 else list(reversed(arms))
        for arm in ordered:
            values = plan["common_values"] | arm["values"]
            args = replace_flags(cfg["args"], values, plan["common_switches"],
                                 plan["remove_values"], plan["remove_switches"])
            yield {"repetition": repetition, "arm": arm["name"], "args": args,
                   "devices": devices, "environment": plan["environment"] | arm.get("environment", {})}


def prepare(root, tokenizer, plan, target):
    sys.path.insert(0, str(root / "tools"))
    from strata_tokenizer import Tokenizer
    names = ("vocab.json", "merges.txt", "token_type.json")
    raw = {name: (tokenizer / name).read_bytes() for name in names}
    vocab = json.loads(raw["vocab.json"])
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = Tokenizer(tokens, raw["merges.txt"].decode().split("\n"), json.loads(raw["token_type.json"]))
    source = (root / "src/program/generate.cpp").read_bytes()
    body = tok.encode(source.decode(), parse_special=False)
    prefix = tok.encode("<|im_start|>user\n", parse_special=True)
    suffix = tok.encode("\n\nExplain the purpose and structure of this source code in detail.<|im_end|>\n"
                        "<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
    if not body:
        raise ValueError("empty tokenized source")
    prompts = []
    for length in [1024, *plan["prompt_lengths"]]:
        count = length - len(prefix) - len(suffix)
        if count <= 0:
            raise ValueError("prompt length is shorter than its template")
        ids = prefix + (body * ((count + len(body) - 1) // len(body)))[:count] + suffix
        if len(ids) != length:
            raise ValueError("prompt length mismatch")
        prompts.append({"length": length, "ids": ids, "ids_sha256": digest(ids)})
    result = {"kind": "synthetic source-code prompts, not original reporter inputs", "plan_sha256": digest(plan),
              "source_sha256": hashlib.sha256(source).hexdigest(),
              "tokenizer_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
              "prompts": prompts}
    with target.open("x") as f:
        json.dump(result, f)


def verify_inputs(plan, inputs):
    if inputs["plan_sha256"] != digest(plan):
        raise ValueError("prepared prompts belong to a different plan")
    prompts = inputs["prompts"]
    if [p["length"] for p in prompts] != [1024, *plan["prompt_lengths"]]:
        raise ValueError("unexpected prompt lengths/order")
    for prompt in prompts:
        ids = prompt["ids"]
        if len(ids) != prompt["length"] or digest(ids) != prompt["ids_sha256"]:
            raise ValueError("prompt IDs changed after preparation")
        if not all(type(token) is int and token >= 0 for token in ids):
            raise ValueError("invalid token IDs")


def execute(root, cfg, plan, inputs, devices, output, timeout):
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise ValueError("GPU execution prohibited by CUDA_VISIBLE_DEVICES=-1")
    inherited = [key for key in os.environ if key.startswith("STRATA_")]
    if inherited:
        raise ValueError("remove inherited STRATA_* variables; experiment settings must come from the plan")
    verify_inputs(plan, inputs)
    matrix = list(cases(cfg, plan, devices))
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != plan["source_commit"]:
        raise ValueError("checkout does not match the pinned source commit")
    cwd = Path(cfg.get("cwd") or root).resolve()
    exe = Path(cfg["exe"])
    if not exe.is_absolute():
        raise ValueError("config exe must be an absolute path")
    tokenizer = Path(cfg["tokenizer"])
    if not tokenizer.is_absolute():
        tokenizer = cwd / tokenizer
    for name, expected in inputs["tokenizer_sha256"].items():
        if hashlib.sha256((tokenizer / name).read_bytes()).hexdigest() != expected:
            raise ValueError("configured tokenizer differs from prepared inputs")
    engine_hash = hashlib.sha256()
    with exe.open("rb") as binary:
        for block in iter(lambda: binary.read(1 << 20), b""):
            engine_hash.update(block)
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,pci.bus_id,driver_version",
                                        "--format=csv"], text=True)
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("GPU compute processes are already running; no engine was started")
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    from serve.server import StrataEngine, child_env
    manifest = {"plan": plan, "inputs_sha256": digest(inputs), "hardware": hardware, "topology": topology,
                "source_commit": revision,
                "source_diff": subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"], text=True),
                "config_sha256": digest(cfg), "engine_sha256": engine_hash.hexdigest(), "cases": matrix}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    failed = False
    with (output / "requests.jsonl").open("x", buffering=1) as records:
        for case in matrix:
            engine = None
            log = output / f'{case["repetition"]}-{case["arm"]}.log'
            tag = {"repetition": case["repetition"], "arm": case["arm"]}
            try:
                env_cfg = {"gpu": devices, "env": case["environment"], "lib_dirs": cfg.get("lib_dirs", [])}
                engine = StrataEngine(str(exe), case["args"], cwd=str(cwd), log=str(log), env=child_env(env_cfg))
                requests = [(True, inputs["prompts"][0], 0)]
                requests += [(False, p, n) for p in inputs["prompts"][1:] for n in range(plan["requests_per_prompt"])]
                for warmup, prompt, request in requests:
                    row = tag | {"warmup": warmup, "length": prompt["length"], "request": request,
                                 "prompt_sha256": prompt["ids_sha256"]}
                    cancelled = threading.Event()
                    timer = threading.Timer(timeout, cancelled.set)
                    timer.daemon = True
                    started = time.monotonic()
                    timer.start()
                    try:
                        cap = plan["warmup_max_new"] if warmup else plan["max_new"]
                        ids = [token for token in engine.generate(prompt["ids"], cap, plan["sampling"], cancelled)
                               if token is not None]
                        metrics = dict(engine.last)
                        if cancelled.is_set():
                            raise TimeoutError("request reached experiment timeout")
                        if not ids or not all(math.isfinite(float(metrics[k])) and float(metrics[k]) > 0
                                              for k in ("prompt_ms", "decode_ms")):
                            raise ValueError("missing output or invalid engine timings")
                        startup = log.read_text(errors="replace")
                        values = plan["common_values"] | next(a["values"] for a in plan["arms"] if a["name"] == case["arm"])
                        for flag in ("--pipeline-windows", "--adapt-async"):
                            if int(values.get(flag, "0")) > 0 and re.search(re.escape(flag) + r" \d+ is off", startup):
                                raise ValueError(f"requested {flag} disabled by engine; this arm is not a valid comparison")
                        row.update(wall_s=time.monotonic()-started, output_ids=ids, metrics=metrics, info=dict(engine.info))
                    except Exception as exc:
                        row.update(wall_s=time.monotonic()-started, error=repr(exc))
                        records.write(json.dumps(row) + "\n")
                        raise
                    finally:
                        timer.cancel()
                    records.write(json.dumps(row) + "\n")
            except Exception as exc:
                failed = True
                records.write(json.dumps(tag | {"arm_stopped": True, "error": repr(exc)}) + "\n")
            finally:
                if engine is not None:
                    engine.close()
    return int(failed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--prepare", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--request-timeout", type=float, default=1800)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.root.resolve()
    if args.prepare:
        if not args.tokenizer or args.execute:
            ap.error("--prepare requires --tokenizer and cannot execute an engine")
        prepare(root, args.tokenizer.resolve(), plan, args.prepare)
        return 0
    if not args.config:
        ap.error("--config is required for dry-run or execution")
    cfg = json.loads(args.config.read_text())
    devices = [int(value) for value in args.devices.split(",")]
    matrix = list(cases(cfg, plan, devices))
    if not args.execute:
        print(json.dumps({"status": "dry-run, no GPU initialized", "cases": matrix}, indent=2))
        return 0
    if not args.inputs or not args.output or args.request_timeout <= 0:
        ap.error("execution requires --inputs, a new --output directory and a positive timeout")
    return execute(root, cfg, plan, json.loads(args.inputs.read_text()), devices, args.output, args.request_timeout)


if __name__ == "__main__":
    sys.exit(main())
