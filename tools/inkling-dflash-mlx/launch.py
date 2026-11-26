"""Run the experimental greedy SQ-10 target on its native stdin/stdout protocol."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('engine', type=Path)
parser.add_argument('target', type=Path)
parser.add_argument('assets', type=Path)
args = parser.parse_args()
env = dict(os.environ, INK_DFLASH_ASSETS=str(args.assets.resolve()), INK_DFLASH_MODEL=str(args.target.resolve()))
sender, receiver = socket.socketpair()
with sender, receiver, subprocess.Popen([sys.executable, str(Path(__file__).with_name('worker.py')), str(receiver.fileno())], env=env, pass_fds=(receiver.fileno(),), stdout=subprocess.PIPE, text=True) as worker:
    receiver.close()
    if worker.stdout.readline().strip() != 'READY':
        raise SystemExit('Draft worker failed during initialization')
    try:
        result = subprocess.run([str(args.engine.resolve()), '--serve', '--native', str(args.target.resolve()), '--max-context', '16384', '--parallel', '1', '--draft', 'off', '--kv', 'fp8', '--chunk', '512', '--ssd-experts', '--vram-experts-gb', '0.5', '--ram-experts-gb', '0'], pass_fds=(sender.fileno(),), env=dict(env, INK_DFLASH_FD=str(sender.fileno())))
    finally:
        sender.close()
    raise SystemExit(result.returncode)
