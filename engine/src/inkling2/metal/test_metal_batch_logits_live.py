"""Live test on a Mac: the Metal Inkling engine returns the right logits for every request of a batch (0.2.21 GRID: with --parallel 32
and long prompts, 9 of 32 clients got a wrong first token).

Runs strata-inkling --batch-check: slot 0's logits alone versus inside a batch of K slots whose prompts end in the same scheduler
iteration with more than 64 rows in the batch.  Before the fix the wanted rows beyond the first 64 came back as zeros: the prompt-end
logits had a KL of about 10 against the solo run and the top-1 was token 0.  Rounding-level differences (chunk shape, MPP prefill) are
about 0.01 to 0.1; the test fails above 1.0 and when the first token (step 0) differs.

  python3 test_metal_batch_logits_live.py --engine ~/.../strata-inkling --model .../Inkling-2-Mini-SQ-10.strata --vision .../Inkling-2-Mini-Vision.strata
"""
import argparse
import os
import random
import re
import subprocess
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vision", required=True)
    ap.add_argument("--prompt-len", type=int, default=1500)
    a = ap.parse_args()
    r = random.Random(7)
    ids = [r.randrange(1000, 60000) for _ in range(a.prompt_len)]
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.write(",".join(map(str, ids)))
    f.close()
    bad = []
    try:
        # K slots: per-slot share chunk // K tokens, the last iteration ends the prompts together with more than 64 rows in the batch
        # (K = 2, 4 here). Chunks of 2048 and more are not used: a single slot with such a chunk already answers differently (open item).
        for k, chunk in ((2, 512), (4, 1024)):
            p = subprocess.run([a.engine, "--batch-check", "4", "--parallel", str(k), "--chunk", str(chunk), "--prompt-ids", "@" + f.name,
                                "--native", a.model, "--vision", a.vision, "--ctx", "4096", "--kv", "fp16"],
                               capture_output=True, text=True, timeout=1200)
            out = p.stdout
            m0 = re.search(r"step 0: KL ([0-9.]+) top1 (\w+)", out)
            print("K=%d chunk %d: %s (exit %d)" % (k, chunk, m0.group(0) if m0 else out[-300:], p.returncode))
            if not m0 or m0.group(2) != "same" or float(m0.group(1)) > 1.0 or p.returncode != 0:
                bad.append("K=%d chunk %d: %s" % (k, chunk, m0.group(0) if m0 else "no result"))
    finally:
        os.unlink(f.name)
    if bad:
        print("FAIL")
        for b in bad:
            print("  -", b)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
