"""Persistent registry of opt-in LAN clusters.

A cluster file holds node metadata only. Credentials are never written to disk:
a node stores the *name* of an environment variable, and ``Registry.token()``
reads the value from the environment. The default single-machine path never
touches this module.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

BACKENDS = ("cuda", "hip", "sycl", "metal", "cpu")
ROLES = ("head", "worker")
_STATUSES = ("healthy", "degraded", "offline", "unknown")
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_HOST = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")


class ClusterError(ValueError):
    """Raised for an invalid cluster name, node, or stored file."""


def clusters_dir(env: Mapping[str, str] | None = None) -> Path:
    """Where cluster files live: ``<STRATA_HOME>/clusters`` (default ``~/.strata/clusters``)."""
    environ = os.environ if env is None else env
    base = environ.get("STRATA_HOME") or str(Path.home() / ".strata")
    return Path(base) / "clusters"


def _name(value: str, what: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ClusterError(f"{what} must be lowercase letters, digits, '-' or '_' and start with a letter")
    return value


def _env_name(value: str, field_name: str = "token_env") -> str:
    if not isinstance(value, str) or not _ENV_NAME.fullmatch(value):
        raise ClusterError(f"{field_name} must be an uppercase environment variable name")
    return value


def _address(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 253:
        raise ClusterError("address must be a host name or IP address")
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        if not _HOST.fullmatch(value) or ".." in value:
            raise ClusterError("address must be a host name or IP address") from None
    return value


def _port(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ClusterError("port must be an integer between 1 and 65535")
    return value


def _number(value: float | int, field_name: str, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ClusterError(f"{field_name} must be a number >= {minimum}")
    return float(value)


@dataclass
class Node:
    """One registered machine. Expert ownership is assigned by the coordinator, not here."""

    id: str
    address: str
    port: int = 8791
    backend: str = "cuda"
    gpu: str = ""
    vram_gib: float = 0
    ram_gib: float = 0
    expert_slots: int = 0
    role: str = "worker"
    token_env: str = "STRATA_CLUSTER_TOKEN"
    status: str = "unknown"
    last_seen: str = ""

    def __post_init__(self) -> None:
        _name(self.id, "node id")
        self.address = _address(self.address)
        self.port = _port(self.port)
        if self.backend not in BACKENDS:
            raise ClusterError(f"backend must be one of {', '.join(BACKENDS)}")
        if self.role not in ROLES:
            raise ClusterError(f"role must be one of {', '.join(ROLES)}")
        if self.status not in _STATUSES:
            raise ClusterError(f"status must be one of {', '.join(_STATUSES)}")
        if isinstance(self.expert_slots, bool) or not isinstance(self.expert_slots, int) or self.expert_slots < 0:
            raise ClusterError("expert_slots must be a non-negative integer")
        _number(self.vram_gib, "vram_gib")
        _number(self.ram_gib, "ram_gib")
        _env_name(self.token_env)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Node":
        if not isinstance(raw, Mapping):
            raise ClusterError("node must be an object")
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(raw) - known
        if unknown:
            raise ClusterError(f"unknown node fields: {', '.join(sorted(unknown))}")
        return cls(**dict(raw))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Cluster:
    """A named, opt-in group of nodes. Secrets are represented by environment variable names."""

    name: str
    model: str = ""
    token_env: str = "STRATA_CLUSTER_TOKEN"
    tls: bool = True
    nodes: dict[str, Node] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _name(self.name, "cluster name")
        _env_name(self.token_env)
        if not isinstance(self.tls, bool):
            raise ClusterError("tls must be a boolean")
        if not isinstance(self.nodes, dict):
            raise ClusterError("nodes must be an object")
        for key, value in self.nodes.items():
            if key != getattr(value, "id", None):
                raise ClusterError("node keys must match their ids")
            if not isinstance(value, Node):
                raise ClusterError("nodes must contain Node objects")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Cluster":
        if not isinstance(raw, Mapping):
            raise ClusterError("cluster must be an object")
        known = {"name", "model", "token_env", "tls", "nodes"}
        unknown = set(raw) - known
        if unknown:
            raise ClusterError(f"unknown cluster fields: {', '.join(sorted(unknown))}")
        nodes = {k: Node.from_dict(v) for k, v in (raw.get("nodes") or {}).items()}
        return cls(name=raw.get("name", ""), model=raw.get("model", ""),
                   token_env=raw.get("token_env", "STRATA_CLUSTER_TOKEN"),
                   tls=raw.get("tls", True), nodes=nodes)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model, "token_env": self.token_env,
                "tls": self.tls, "nodes": {k: v.to_dict() for k, v in self.nodes.items()}}

    def add(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ClusterError(f"node {node.id} is already registered")
        self.nodes[node.id] = node

    def remove(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise ClusterError(f"node {node_id} is not registered")
        del self.nodes[node_id]

    def token(self, env: Mapping[str, str] | None = None) -> str:
        """Read the shared secret from the environment. Never stored in the registry."""
        environ = os.environ if env is None else env
        value = environ.get(self.token_env, "")
        if not value:
            raise ClusterError(f"set {self.token_env} before using cluster {self.name}")
        return value

    def capacities(self) -> dict[str, float]:
        """Optimistic totals for the dashboard; the coordinator reserves headroom separately."""
        return {
            "nodes": len(self.nodes),
            "vram_gib": round(sum(n.vram_gib for n in self.nodes.values()), 2),
            "ram_gib": round(sum(n.ram_gib for n in self.nodes.values()), 2),
            "expert_slots": sum(n.expert_slots for n in self.nodes.values()),
            "healthy": sum(n.status == "healthy" for n in self.nodes.values()),
        }


class Registry:
    """Read/write cluster files atomically, with no implicit defaults changed."""

    VERSION = 1

    def __init__(self, root: Path | str | None = None, env: Mapping[str, str] | None = None):
        self.root = Path(root) if root is not None else clusters_dir(env)
        self._env = env

    def path(self, name: str) -> Path:
        return self.root / f"{_name(name, 'cluster name')}.json"

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.json") if _NAME.fullmatch(p.stem))

    def create(self, name: str, *, model: str = "", token_env: str = "STRATA_CLUSTER_TOKEN",
               tls: bool = True) -> Cluster:
        path = self.path(name)
        if path.exists():
            raise ClusterError(f"cluster {name} already exists")
        cluster = Cluster(name=name, model=model, token_env=token_env, tls=tls)
        self.save(cluster)
        return cluster

    def load(self, name: str) -> Cluster:
        path = self.path(name)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ClusterError(f"cluster {name} does not exist") from None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ClusterError(f"cannot read cluster {name}: {exc}") from None
        if not isinstance(raw, Mapping) or raw.get("version") != self.VERSION:
            raise ClusterError(f"unsupported cluster file: {name}")
        data = dict(raw)
        data.pop("version", None)
        return Cluster.from_dict(data)

    def save(self, cluster: Cluster) -> None:
        data = {"version": self.VERSION, **cluster.to_dict()}
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{cluster.name}-", suffix=".json", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            try:
                os.chmod(temporary, 0o600)
            except OSError:  # Windows: best effort
                pass
            os.replace(temporary, self.path(cluster.name))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def add_node(self, name: str, node: Node) -> Cluster:
        cluster = self.load(name)
        cluster.add(node)
        self.save(cluster)
        return cluster

    def remove_node(self, name: str, node_id: str) -> Cluster:
        cluster = self.load(name)
        cluster.remove(node_id)
        self.save(cluster)
        return cluster

    def delete(self, name: str) -> None:
        path = self.path(name)
        try:
            path.unlink()
        except FileNotFoundError:
            raise ClusterError(f"cluster {name} does not exist") from None
