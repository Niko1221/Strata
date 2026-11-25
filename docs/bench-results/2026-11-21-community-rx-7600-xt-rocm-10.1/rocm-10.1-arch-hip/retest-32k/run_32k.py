"""Run only strata_bench's 32K prompt test (same prompts and settings as `bench.sh --long`) against the live server.

    STRATA_API_KEY=... python run_32k.py [runs]
"""
import os
import sys
import time

sys.path.insert(0, "<strata>/tools")
import strata_bench as sb  # noqa: E402

runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
base = "http://127.0.0.1:8080"
key = os.environ.get("STRATA_API_KEY", "")


def say(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


suite = sb.Suite(base, key, 65536, runs, False, log=say)
say("calibrating (two tiny requests) ...")
suite.calibrate()
suite.run_prompt(32768, "32K")
p = suite.out["prompts"].get("32K", {})
say(f"result: prompt_tps={p.get('prompt_tps')} ttft_s={p.get('ttft_s')} failures={suite.out['failures']}")
