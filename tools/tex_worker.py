"""Explicit optional TeX setup. Never invoked by Strata's normal installer or server.

python tools/tex_worker.py build
python tools/tex_worker.py start
Set STRATA_TEX_CONTAINER=strata-math-tex before starting Strata.
"""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "strata-math-tex:comparison"
CONTAINER = "strata-math-tex"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "start", "stop", "status"])
    args = parser.parse_args()
    docker = os.environ.get("STRATA_TEX_DOCKER", "docker")
    if args.action == "build":
        command = ["build", "-t", IMAGE, str(ROOT / "serve/tex_worker")]
    elif args.action == "start":
        # Resolve tag once; the running worker is pinned to the exact image ID.
        image_id = subprocess.check_output([docker, "image", "inspect", "--format", "{{.Id}}", IMAGE], text=True).strip()
        command = ["run", "--detach", "--name", CONTAINER, "--network=none", "--read-only", "--cap-drop=ALL",
                   "--security-opt=no-new-privileges", "--memory=512m", "--memory-swap=512m", "--cpus=1",
                   "--pids-limit=64", "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=128m", image_id]
    elif args.action == "stop":
        command = ["rm", "--force", CONTAINER]
    else:
        command = ["inspect", "--format", "{{.Image}} {{.State.Status}}", CONTAINER]
    subprocess.run([docker, *command], check=True)


if __name__ == "__main__":
    main()
