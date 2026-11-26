"""Local administration of opt-in LAN clusters.

This module is intentionally not wired into `strata`'s main parser yet: it is
groundwork for the scope issue and cannot change the default command surface.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from strata_control.cluster import ClusterError, Node, Registry


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="strata cluster", description="Manage opt-in LAN clusters")
    sub = ap.add_subparsers(dest="action", required=True)

    sub.add_parser("list", help="list saved clusters")

    create = sub.add_parser("create", help="create a cluster file")
    create.add_argument("name")
    create.add_argument("--model", default="", help="model family/pack label, recorded only")
    create.add_argument("--token-env", default="STRATA_CLUSTER_TOKEN",
                        help="name of the environment variable holding the shared secret")

    show = sub.add_parser("show", help="show a cluster and its aggregated capacity")
    show.add_argument("name")

    add = sub.add_parser("add", help="register a node")
    add.add_argument("name")
    add.add_argument("--id", required=True)
    add.add_argument("--address", required=True)
    add.add_argument("--port", type=int, default=8791)
    add.add_argument("--backend", default="cuda")
    add.add_argument("--gpu", default="")
    add.add_argument("--vram-gib", type=float, default=0)
    add.add_argument("--ram-gib", type=float, default=0)
    add.add_argument("--expert-slots", type=int, default=0)
    add.add_argument("--role", default="worker", choices=("head", "worker"))
    add.add_argument("--token-env", default="STRATA_CLUSTER_TOKEN")

    remove = sub.add_parser("remove", help="unregister a node")
    remove.add_argument("name")
    remove.add_argument("node_id")

    delete = sub.add_parser("delete", help="delete a cluster file")
    delete.add_argument("name")
    return ap


def main(argv: Sequence[str] | None = None, *, registry: Registry | None = None) -> int:
    args = parser().parse_args(argv)
    reg = registry or Registry()
    try:
        if args.action == "list":
            payload = reg.names()
        elif args.action == "create":
            cluster = reg.create(args.name, model=args.model, token_env=args.token_env)
            payload = cluster.to_dict()
        elif args.action == "show":
            cluster = reg.load(args.name)
            payload = {**cluster.to_dict(), "capacities": cluster.capacities()}
        elif args.action == "add":
            node = Node(id=args.id, address=args.address, port=args.port, backend=args.backend,
                        gpu=args.gpu, vram_gib=args.vram_gib, ram_gib=args.ram_gib,
                        expert_slots=args.expert_slots, role=args.role, token_env=args.token_env)
            payload = reg.add_node(args.name, node).to_dict()
        elif args.action == "remove":
            payload = reg.remove_node(args.name, args.node_id).to_dict()
        else:
            reg.delete(args.name)
            payload = {"deleted": args.name}
    except ClusterError as exc:
        print("strata cluster: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
