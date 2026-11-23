"""Compare greedy token IDs, native reuse counts and wall-clock TTFT on real Inkling weights.

python engine/tests/inkling2/radix_gpu_test.py BUILD/strata-inkling MODEL.strata
python engine/tests/inkling2/radix_gpu_test.py BUILD/strata-inkling MODEL.strata --cross-slot --draft dspark --kv k8v4
"""
import argparse
import json
import random
import subprocess
import time


def request(proc, prompt, slot):
    ids = ",".join(map(str, prompt))
    command = "GEN 12" if slot is None else f"BGEN {slot} 12 slot_read=1"
    started = time.perf_counter()
    proc.stdin.write(f"{command} temperature=0 seed=41 {ids}\n")
    proc.stdin.flush()
    tokens = []
    reused = 0
    ttft = None
    for line in proc.stdout:
        fields = line.split()
        if not fields:
            continue
        tag = fields[0]
        if tag == "ERR":
            raise RuntimeError(line.strip())
        if tag == "RESUME":
            reused = int(fields[1])
        if tag == ("T" if slot is None else "BT"):
            if ttft is None:
                ttft = round(1000 * (time.perf_counter() - started), 2)
            tokens.append(int(fields[-1]))
        if tag == ("DONE" if slot is None else "BDONE"):
            if not tokens:
                raise RuntimeError(f"Request finished without tokens: {line}")
            return dict(tokens=tokens, reused=reused, ttft_ms=ttft)
    raise RuntimeError("Engine exited during a request")


def run(args, cache_mib):
    command = [args.engine, "--native", args.model, "--serve", "--parallel", "2",
               "--max-context", str(args.prefix + 512), "--draft", args.draft, "--kv", args.kv,
               "--radix-cache-mib", str(cache_mib)]
    results = []
    rng = random.Random(4147)
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
        try:
            for line in proc.stdout:
                if line.startswith("READY "):
                    break
            else:
                raise RuntimeError("Engine exited before READY")
            request(proc, list(range(900, 916)), None)
            for i in range(args.rounds):
                shared = [rng.randrange(1000, 30000) for _ in range(args.prefix)]
                if args.cross_slot:
                    # Slot 1 has no drafter. Slot 0 must acquire a complete state
                    # before it can serve as a DSpark donor for subsequent branches.
                    cases = [(1, 23), (0, 47), (1, 73), (0, 47), (1, 23), (0, 73)]
                    for slot, branch in cases:
                        tail = random.Random(branch).choices(range(1000, 20000), k=48)
                        result = request(proc, shared + tail, slot)
                        results.append(dict(iteration=i, slot=slot, branch=branch, **result))
                else:
                    first, second = shared + [47] * 48, shared + [73] * 48
                    for label, prompt in (("initial", first), ("diverge", second), ("return", first),
                                          ("extend", first + [91] * 16)):
                        results.append(dict(iteration=i, case=label, **request(proc, prompt, None)))
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.returncode:
            raise RuntimeError(f"Engine exited with status {proc.returncode}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine")
    parser.add_argument("model")
    parser.add_argument("--prefix", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cache-mib", type=int, default=768)
    parser.add_argument("--draft", choices=("off", "dspark"), default="off")
    parser.add_argument("--kv", choices=("fp16", "fp8", "k8v8", "k8v4"), default="k8v8")
    parser.add_argument("--cross-slot", action="store_true")
    args = parser.parse_args()
    disabled, enabled = (run(args, mib) for mib in (0, args.cache_mib))
    print(json.dumps(dict(disabled=disabled, enabled=enabled), indent=2))
    if [r["tokens"] for r in disabled] != [r["tokens"] for r in enabled]:
        raise SystemExit("Generated token IDs differ")
    case = 2 if args.cross_slot else 1
    if disabled[case]["reused"] or enabled[case]["reused"] != args.prefix // 512 * 512:
        raise SystemExit("Expected divergent branch to reuse the shared checkpoint")
