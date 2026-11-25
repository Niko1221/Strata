#!/usr/bin/env python3
"""Prepare termgrade-train-harbor tasks for local Docker + Codex execution.

Removes `docker_image` from task.toml and writes a per-task `environment/Dockerfile`
that bakes `setup.sh` on top of `codex-termgrade-base:v1`, and switches
`network_mode = "no-network"` to `"public"` so local Docker containers reach the
host OpenEnv capture proxy.
"""
import argparse
import pathlib
import re
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source Harbor tasks directory (e.g. termgrade-train-harbor/tasks)")
    p.add_argument("--dst", required=True, help="Destination directory for local Docker Codex tasks")
    p.add_argument("--image", default="codex-termgrade-base:v1")
    p.add_argument("--limit", type=int, default=0, help="0 = all tasks")
    args = p.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    tasks = sorted([d for d in src.iterdir() if d.is_dir()])
    if args.limit > 0:
        tasks = tasks[:args.limit]

    count = 0
    for t in tasks:
        out_dir = dst / t.name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(t, out_dir)
        toml_path = out_dir / "task.toml"
        if toml_path.exists():
            text = toml_path.read_text(encoding="utf-8")
            text = re.sub(r'^\s*docker_image\s*=\s*"[^"]*"\s*\n?', "", text, flags=re.MULTILINE)
            text = re.sub(r'network_mode\s*=\s*"no-network"', 'network_mode = "public"', text)
            text = re.sub(r'\n?\[environment\.healthcheck\][\s\S]*?(?=\n\[|\Z)', "", text)
            toml_path.write_text(text, encoding="utf-8")

        env_dir = out_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        setup_sh = env_dir / "setup.sh"
        if setup_sh.exists():
            df = (
                f"FROM {args.image}\n"
                "COPY setup.sh /tmp/setup.sh\n"
                "RUN bash /tmp/setup.sh && rm -f /tmp/setup.sh\n"
                "WORKDIR /app\n"
            )
        else:
            df = f"FROM {args.image}\nWORKDIR /app\n"
        (env_dir / "Dockerfile").write_text(df, encoding="utf-8")
        count += 1
    print(f"Prepared {count} TermGrade Codex tasks in {dst}")


if __name__ == "__main__":
    main()
