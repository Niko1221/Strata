import json
import subprocess
import sys
import time
from pathlib import Path

from probe import OUT, engine_pids, generate, http, snapshot

events = json.loads((OUT / "cycles.json").read_text()) if len(sys.argv) > 1 else []
start_pids = engine_pids()


def reserve(label, mib, gpu=None):
    body = {"reserve_mib": mib}
    if gpu is not None:
        body["gpu"] = gpu
    before = snapshot()
    started = time.monotonic()
    response = http("/v1/vram", body)
    event = {"label": label, "request": body, "response": response,
        "elapsed_s": time.monotonic() - started, "before": before,
        "after": snapshot(), "gpu_processes": engine_pids()}
    events.append(event)
    (OUT / "cycles.json").write_text(json.dumps(events, indent=2))
    print("RESERVE", json.dumps(event), flush=True)
    assert response["status"] == "ok", response
    assert engine_pids() == start_pids, "Engine process changed during resize"
    return response


def run():
    if len(sys.argv) > 1:
        reserve("gpu0-full-restore", 0, 0)
        generate("gpu0-restore", "qa_logic")
    else:
        generate("elastic-full", "leads28_en")
        generate("elastic-full", "qa_plan")
    for gpu in ((1,) if len(sys.argv) > 1 else (0, 1)):
        reserve(f"gpu{gpu}-3g", 3072, gpu)
        generate(f"gpu{gpu}-3g", "qa_arithmetic")
        response = reserve(f"gpu{gpu}-5g", 5120, gpu)
        assert response["vram_free_mib"] >= 5120, response
        generate(f"gpu{gpu}-5g", "leads28_en")
        generate(f"gpu{gpu}-5g", "qa_plan")
        response = reserve(f"gpu{gpu}-restore", None, gpu)
        if response["expert_slots"] != response["expert_slots_full"]:
            response = reserve(f"gpu{gpu}-full-restore", 0, gpu)
        part = response["gpus"][0]
        assert part["expert_slots"] == part["expert_slots_full"], part
        generate(f"gpu{gpu}-restore", "qa_logic")
    reserve("both-3g", 3072)
    generate("both-3g", "long32k")
    reserve("both-restore", 0)
    generate("both-restore", "qa_injection")
    reserve("shared-5g", 5632, 1)
    helper = subprocess.Popen(["docker", "run", "--rm", "-i", "--name", "strata-vram-holder-20261024",
        "--gpus", "device=1", "--entrypoint", "/bench/hold-vram", "-v", str(OUT) + ":/bench:ro",
        "strata-local:cuda13", "4096"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        assert helper.stdout.readline().strip() == "READY", helper.stderr.read()
        events.append({"label": "external-4g-held", "after": snapshot(), "gpu_processes": engine_pids()})
        (OUT / "cycles.json").write_text(json.dumps(events, indent=2))
        generate("external-4g-held", "qa_plan")
        response = http("/v1/vram", {"reserve_mib": None, "gpu": 1})
        events.append({"label": "grow-while-occupied", "response": response, "after": snapshot()})
        (OUT / "cycles.json").write_text(json.dumps(events, indent=2))
        assert response["expert_slots"] < response["expert_slots_full"], response
        generate("grow-while-occupied", "qa_logic")
    finally:
        helper.stdin.write("\n")
        helper.stdin.flush()
        helper.wait(timeout=30)
        assert helper.returncode == 0, helper.stderr.read()
    reserve("external-released", 0, 1)
    for repeat in range(3):
        reserve(f"repeat{repeat}-shrink", 5120, repeat % 2)
        reserve(f"repeat{repeat}-grow", 0, repeat % 2)
        generate(f"repeat{repeat}", "qa_negation")
    generate("final-restored", "leads28_en")
    generate("final-restored", "long32k")


if __name__ == "__main__":
    run()
