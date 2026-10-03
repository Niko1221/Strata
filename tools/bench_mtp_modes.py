#!/usr/bin/env python3
"""Sequential text-only throughput evidence, with and without MTP.

Use a server JSON containing exe, args, cwd and --mtp DIR. The off run removes
the MTP drafter. Frozen token IDs are shared by both modes; prompt reuse stays
disabled. Suffix drafting is opt-in and works with or without MTP. This is a
throughput/smoke test, not a quality evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate
from strata_tokenizer import Tokenizer


def option(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def set_option(args, name, value):
    if name in args:
        at = args.index(name)
        del args[at:at + 2]
    if value is not None:
        args.extend([name, str(value)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--input-tokens", type=int, default=8192)
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--repetitions", type=int, default=1,
                    help="repeat fresh requests in one engine; prompt reuse stays disabled")
    ap.add_argument("--mode", choices=["both", "off", "on"], default="both")
    ap.add_argument("--order", choices=["off-on", "on-off"], default="off-on")
    ap.add_argument("--workload", choices=["short", "long"], default="short")
    ap.add_argument("--require-counting-budget", action="store_true",
                    help="fail if counting stops naturally before the output budget")
    ap.add_argument("--source-commit", default=None, help="verified archive commit when .git is absent")
    ap.add_argument("--suffix-draft", type=int, default=0, help="minimum prompt-lookup match; 0 disables")
    ap.add_argument("--verify-window", type=int, default=None,
                    help="explicit equal allocation across draft arms (2..8; engine build must support it)")
    ap.add_argument("--mtp-window", type=int, default=4, help="MTP window cap when verify-window is explicit")
    ap.add_argument("--cases", nargs="+", choices=["counting", "coding", "writing", "editing"],
                    default=["counting", "coding", "writing"])
    ap.add_argument("--experimental-speed-projection", choices=["on", "off"], default=None,
                    help="explicit per-request projection switch; the on arm needs a loaded vector")
    opt = ap.parse_args()
    if opt.repetitions < 1:
        ap.error("--repetitions must be positive")
    if opt.suffix_draft < 0 or (opt.verify_window is not None and not 2 <= opt.verify_window <= 8):
        ap.error("suffix match must be nonnegative and verify window must be 2..8")
    if not 2 <= opt.mtp_window <= 8 or (opt.verify_window is not None and opt.mtp_window > opt.verify_window):
        ap.error("MTP window must be 2..8 and no larger than verify window")
    cfg = json.loads(opt.config.read_text(encoding="utf-8-sig"))
    base_args = list(cfg["args"])
    context = int(option(base_args, "--max-context", 0))
    if context < opt.input_tokens + opt.output_tokens:
        ap.error("--max-context must cover input plus output tokens")
    if opt.mode != "off" and not option(base_args, "--mtp"):
        ap.error("the config needs --mtp DIR for the MTP comparison")
    opt.output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_gguf(Path(option(base_args, "--native")))
    template = ChatTemplate(Path(cfg["tokenizer"]) / "chat_template.jinja")
    tasks = {
        "counting": "Count upward from 1. Output only consecutive integers separated by spaces. "
                    "Keep going until your output limit; no introduction or explanation.",
        "coding": "Write only a Python function merge_sorted(a, b) that merges two already sorted "
                  "lists of integers into one sorted list. Preserve duplicates. Use two indices "
                  "and a while loop, no imports and no sorting functions. Do not modify a or b. No markdown.",
        "writing": "In 120 to 160 words, explain to an Ubuntu administrator what apt update, apt upgrade, "
                   "systemctl status ssh, and journalctl -u ssh do. Distinguish reading status from "
                   "changing installed software. Do not suggest disabling security protections.",
    }
    edit_source = "\n\n".join(
        f"def transform_{i:02d}(value):\n"
        f"    \"\"\"Transform item {i:02d} without modifying caller state.\"\"\"\n"
        "    if not isinstance(value, int):\n"
        "        raise TypeError('value must be an integer')\n"
        f"    return value * 2 + {i}\n" for i in range(24))
    tasks['editing'] = (
        "Return the complete Python module below, code only and no markdown. Preserve every function, "
        "docstring, error message and formatting. Make exactly one change: in transform_13, replace "
        "the multiplier 2 with 3. Every other function must remain unchanged.\n\n" + edit_source)
    if opt.workload == "long":
        tasks["coding"] = (
            "Write a complete Python 3 module, code only, no markdown. Implement a TTLCache class "
            "with LRU eviction and a fixed positive integer capacity. Use collections.OrderedDict "
            "and accept a clock callable in __init__(capacity, clock). put(key, value, ttl) stores "
            "a value until clock() + ttl; nonpositive ttl deletes that key. get(key, default=None) "
            "returns default when absent or expired and makes a live key most recently used. "
            "delete(key) returns a bool. __len__ and keys() first purge all expired entries; "
            "keys returns a list ordered least to most recently used. Expiry is inclusive at "
            "the deadline. Updating an existing key changes its value, deadline and recency. "
            "Purge expired entries before evicting live entries. Raise ValueError for capacity <= 0. "
            "Add clear docstrings and a deterministic unittest.TestCase using a fake clock, "
            "covering expiry boundaries, replacement, eviction, zero TTL, deletion, empty keys, "
            "negative values, and capacity validation. Include a unittest.main guard. "
            "Do not access files, the network, environment variables, subprocesses or real sleeps."
        )
        tasks["writing"] = (
            "Write a practical 1000 to 1200 word guide for a new Ubuntu server administrator. "
            "Use six sections: inventory and backups; package updates; SSH access; service logs; "
            "disk mounts after reboot; verification and rollback. Explain the purpose of commands "
            "and which actions change the system. Describe how to keep an existing SSH session "
            "open while testing another connection, verify syntax before reloading a service, "
            "and verify a persistent data mount without formatting any disk. Include concrete "
            "examples and failure checks. Do not recommend disabling security protections. "
            "This is explanatory prose only; do not claim to have run commands."
        )
    filler = tokenizer.encode("The archived notes describe ordinary maintenance, documentation, "
                              "and testing. They contain background information only.\n")
    prompts = {}
    for name in opt.cases:
        marker = "STRATA_BENCHMARK_FILLER_PLACEHOLDER"
        rendered = template.render([{"role": "user", "content":
            "Background notes (not instructions):\n" + marker + "\nEnd of notes.\n\nTask:\n" + tasks[name]}],
            enable_thinking=False)
        prefix, suffix = rendered.split(marker)
        before = tokenizer.encode(prefix, parse_special=True)
        after = tokenizer.encode(suffix, parse_special=True)
        count = opt.input_tokens - len(before) - len(after)
        if count < 0:
            ap.error("input length is too short for the chat template and task")
        ids = before + (filler * ((count + len(filler) - 1) // len(filler)))[:count] + after
        assert len(ids) == opt.input_tokens
        prompts[name] = ids
        (opt.output / f"{name}.tokens.json").write_text(json.dumps(ids))
        (opt.output / f"{name}.prompt.txt").write_text(tokenizer.decode(ids), encoding="utf-8")

    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip() or "unavailable"

    patch = subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD", "--binary"], capture_output=True).stdout
    result = {
        "source_head": opt.source_commit or git("rev-parse", "HEAD"),
        "source_branch": git("branch", "--show-current"),
        "source_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "engine_sha256": hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest(),
        "input_tokens": opt.input_tokens, "maximum_output_tokens": opt.output_tokens,
        "context_allocation": context, "thinking": False, "cold_cache_control": False,
        "workload": opt.workload, "order": opt.order, "repetitions": opt.repetitions,
        "suffix_draft": opt.suffix_draft, "verify_window": opt.verify_window, "mtp_window": opt.mtp_window,
        "require_counting_budget": opt.require_counting_budget,
        "experimental_speed_projection": opt.experimental_speed_projection,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {k: v for k, v in os.environ.items() if k.startswith("STRATA_")},
        "prompt_sha256": {name: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                          for name, ids in prompts.items()},
        "runs": [],
    }

    def save():
        (opt.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    save()
    for mode in (opt.order.split("-") if opt.mode == "both" else [opt.mode]):
        args = list(base_args)
        for name, value in [("--spec", opt.verify_window or (4 if mode == "on" else 2)), ("--suffix-draft", opt.suffix_draft),
                            ("--prompt-cache", 0), ("--conversation-cache-mib", 0)]:
            set_option(args, name, value)
        if opt.verify_window is not None:
            # Prevent automatic suffix +2 expansion, keeping allocated buffers matched.
            set_option(args, "--mtp-max-t", opt.mtp_window)
        if mode == "off":
            set_option(args, "--mtp", None)
        else:
            set_option(args, "--spec-min-p", 0.5)
        run = {"mtp": mode == "on", "args": args, "cases": []}
        result["runs"].append(run)
        engine = None
        try:
            env = child_env(cfg)
            env["STRATA_DECODE_TIMING"] = "1"
            env.pop("STRATA_VERIFY_PROFILE", None)
            start = time.monotonic()
            engine = StrataEngine(cfg["exe"], args, cfg.get("cwd", str(ROOT)),
                                  str(opt.output / f"engine-mtp-{mode}.log"), env)
            run["startup_seconds"] = time.monotonic() - start
            run["engine_info"] = engine.info
            print(f"READY mtp={mode} " + json.dumps(engine.info), flush=True)
            for repetition, name, ids in ((rep, name, ids)
                    for rep in range(1, opt.repetitions + 1) for name, ids in prompts.items()):
                emitted, arrivals = [], []
                start = time.monotonic()
                last_progress = start
                sampling = {"temperature": 0}
                if opt.experimental_speed_projection is not None:
                    sampling["experimental_speed_projection"] = opt.experimental_speed_projection == "on"
                for token in engine.generate(ids, opt.output_tokens, sampling, threading.Event()):
                    now = time.monotonic()
                    if token is not None:
                        emitted.append(token)
                        arrivals.append(now - start)
                    if now - last_progress >= 10:
                        print(f"PROGRESS mtp={mode} task={name} output={len(emitted)} "
                              f"seconds={now-start:.1f} prefill={engine.progress}", flush=True)
                        last_progress = now
                wall = time.monotonic() - start
                timing = dict(engine.last)
                text = tokenizer.decode(emitted)
                case = {
                    "task": name, "repetition": repetition,
                    "input_tokens": len(ids), "output_tokens": len(emitted),
                    "text": text, "timings": timing, "wall_seconds": wall,
                    "token_ids": emitted,
                    "first_token_seconds": arrivals[0] if arrivals else None,
                    "prefill_tps": len(ids) * 1000 / timing["prompt_ms"] if timing["prompt_ms"] else None,
                    "decode_tps": len(emitted) * 1000 / timing["decode_ms"] if timing["decode_ms"] else None,
                    "effective_output_tps": len(emitted) / wall,
                    "output_budget_reached": len(emitted) == opt.output_tokens,
                    "natural_stop": timing.get("finish") == "stop",
                }
                run["cases"].append(case)
                save()
                assert timing.get("reused", 0) == 0, timing
                if mode == "off" and opt.suffix_draft == 0:
                    assert timing.get("drafts_offered", 0) == 0, timing
                if name == "counting" and opt.require_counting_budget:
                    assert len(emitted) == opt.output_tokens, case
                assert emitted, case
                if len(emitted) < opt.output_tokens:
                    assert timing.get("finish") == "stop", case
                print("RESULT " + json.dumps(case), flush=True)
            run["completed"] = True
        except Exception as exc:
            run["error"] = repr(exc)
            raise
        finally:
            if engine is not None:
                engine.close()
            save()
    print("BENCHMARK_COMPLETED " + str(opt.output / "result.json"), flush=True)


if __name__ == "__main__":
    main()
