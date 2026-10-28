"""agent-and-speed.py - agent-turns.py, then speed-checks.py, against the same server; t630-sweep.py parses both.
Usage: python3 -I agent-and-speed.py [base_url]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for script in ("agent-turns.py", "speed-checks.py"):
    p = subprocess.run([sys.executable, "-I", os.path.join(HERE, script), *sys.argv[1:]], capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
