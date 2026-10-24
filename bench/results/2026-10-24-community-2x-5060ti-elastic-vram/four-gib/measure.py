import json
import os
import time

from probe import OUT, engine_pids, generate, http, snapshot

events = []
pids = engine_pids()
for repeat in range(3):
    os.environ["STRATA_PROMPT_LABEL"] = f"vram4g-pair-{repeat}"
    for phase, gpu in [("full-before", None), ("gpu0-4g", 0), ("gpu1-4g", 1), ("full-after", None)]:
        restored = http("/v1/vram", {"reserve_mib": 0})
        assert all(p["expert_slots"] == p["expert_slots_full"] for p in restored["gpus"])
        before = snapshot()
        response = restored
        elapsed = None
        if gpu is not None:
            free = {int(line.split(",")[0]): int(line.split(",")[2]) for line in before.splitlines()}
            start = time.monotonic()
            response = http("/v1/vram", {"reserve_mib": free[gpu] + 4096, "gpu": gpu})
            elapsed = time.monotonic() - start
        events.append({"repeat": repeat, "phase": phase, "gpu": gpu,
            "before": before, "after": snapshot(), "resize_s": elapsed, "response": response})
        (OUT / "events.json").write_text(json.dumps(events, indent=2))
        print("PHASE", repeat, phase, events[-1]["after"], flush=True)
        for name in ["records32_ru", "leads28_en"]:
            generate(f"r{repeat}-{phase}", name)
        assert engine_pids() == pids, "Engine process changed"
http("/v1/vram", {"reserve_mib": 0})
print("DONE: 24 complete, matched-input measurements", flush=True)
