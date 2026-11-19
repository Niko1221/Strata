#!/usr/bin/env python3
"""One-way upstream -> fork merge with full custom-feature preservation.

The upstream v0.2.16 restructure renames ~7050 of ~7750 changed paths, so a
plain ``git merge`` cannot express the intent.  This tool works in the
*final upstream path space*: every fork path is normalised through the
rename map, then the three blobs (base/fork/upstream) are compared at the
correct old/new paths.

Policy
------
* upstream-only change          -> adopt upstream bytes at its final path
* fork-only change              -> retain fork bytes, relocated to the new layout
* both changed, non-owned       -> three-way diff3; conflicts keep fork bytes
                                   and write a marked review copy
* fork-owned engine/feature     -> never overwritten; relocated to a dedicated
                                   area; stale API references are rewritten
* upstream delete / fork change -> conflict, fork retained (never silently lost)
* fork delete / upstream change -> upstream adopted

The tool is non-destructive by default (``--mode dry-run``).  ``--mode apply``
writes only into the isolated worktree given by ``--repo``; it never pushes
or moves any other branch.  Fork-only edits in the primary checkout are
untouched because the tool refuses a dirty or mismatched worktree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # optional progress bar; the tool stays dependency-free without it
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:  # type: ignore
        return iterable

LOG = logging.getLogger("strata.oneway_merge")
LOG_FMT = "%(asctime)s [%(levelname)s] [strata.oneway_merge] %(message)s"

# Fork property: upstream bytes never replace these.  Keys are OLD paths.
OWNED_PREFIXES: Tuple[str, ...] = (
    "src/qwen38/", "include/strata/qwen38/",
    "src/dsv41/", "include/strata/dsv41/",
    "src/core/native_cache_", "include/strata/core/native_cache_",
    "src/core/spin8_attention",
    "spin8/", "tqcodec/", "adapters/upstream/",
    "apps/host/src/components/BtwReplies",
    "apps/host/src/components/HakuaLlamaChat",
    "apps/host/src/components/RefusalProjectionStatus",
    "apps/host/src/components/SkillImportDialog",
    "apps/host/src/components/WebSearchDialog",
    "apps/host/src/components/WorkflowLearning",
    "apps/host/src/lib/build-identity", "apps/host/src/lib/deepseek-",
    "apps/host/src/lib/refusal-projection", "apps/host/src/lib/run-selection",
    "apps/host/src/lib/strata-engine-backend", "apps/host/src/lib/use-request-owner",
    "apps/host/server/q38-dense-contract", "apps/host/server/local-backends",
    "apps/host/server/spin8-backend", "apps/host/server/memory",
    "apps/host/server/web-search-rustxng", "apps/host/server/rustxng-runtime",
    "apps/host/server/skill-learning", "apps/host/server/skill-import",
    "strata_control/backends/", "strata_control/qwen38_dense",
    "strata_control/dsv41_", "strata_control/model_bundle",
    "strata_control/model_registry", "strata_control/model_memory",
    "strata_control/qualification", "strata_control/extensions",
    "strata_control/steering",
    "serve/native_family", "serve/memory_harness", "serve/llama_memory",
    "packaging/hakua", "packaging/full_",
    "schemas/fork-extension.schema.json", "data/custom-models",
    "third_party/antirez-ds4", "third_party/project-maya",
    "_docs/", ".superpowers/",
)

# OLD -> NEW prefix.  Derived from git rename evidence (7052 renames).
RELOCATE_PREFIXES: Tuple[Tuple[str, str], ...] = (
    # dedicated areas for fork-only engines (hybrid layout decision)
    ("src/qwen38/", "engine/q38/src/"),
    ("include/strata/qwen38/", "engine/q38/include/strata/qwen38/"),
    ("src/dsv41/", "engine/dsv41/src/"),
    ("include/strata/dsv41/", "engine/dsv41/include/strata/dsv41/"),
    # common code follows upstream
    ("src/kernels/cuda/", "engine/src/kernels/cuda/"),
    ("src/qwen35/", "engine/src/qwen35/"),
    ("src/prefill/", "engine/src/prefill/"),
    ("src/", "engine/src/"),
    ("include/", "engine/include/"),
    ("cmake/", "engine/cmake/"),
    ("metal/", "engine/metal/"),
    ("sycl/", "engine/sycl/"),
    ("third_party/", "engine/third_party/"),
    ("apps/host/", "app/host/"),
    ("apps/desktop/", "app/desktop/"),
    ("serve/", "terminal/serve/"),
    ("strata_cli/", "terminal/strata_cli/"),
    ("strata_control/", "terminal/strata_control/"),
    ("tests/terminal/", "terminal/tests/"),
    ("tests/snapshot/", "terminal/tests/snapshot/"),
    ("tests/data/", "terminal/tests/data/"),
    ("tests/", "terminal/tests/"),
    ("bench/", "docs/bench-results/"),
    ("history/", "docs/history/"),
    ("packaging/", "tools/packaging/"),
    ("docker/", "tools/docker/"),
    ("research/", "tools/research/"),
    ("data/", "terminal/data/"),
    ("integrations/", "terminal/integrations/"),
    ("bin/", "terminal/bin/"),
    ("ref/", "terminal/ref/"),
    ("spin8/", "engine/q38/src/spin8/"),
    ("tqcodec/", "engine/q38/src/tqcodec/"),
    ("adapters/upstream/", "engine/q38/src/adapters/upstream/"),
    ("tools/", "tools/"),
    ("docs/", "docs/"),
    ("schemas/", "schemas/"),
    ("_docs/", "_docs/"),
    (".superpowers/", ".superpowers/"),
)

# Root files that upstream moved individually.
ROOT_RELOCATE: Dict[str, str] = {
    "CMakeLists.txt": "engine/CMakeLists.txt",
    "CMakePresets.json": "engine/CMakePresets.json",
    "setup.py": "terminal/setup.py",
    "conftest.py": "terminal/conftest.py",
    "strata.py": "terminal/strata.py",
    "requirements.txt": "terminal/requirements.txt",
    "SECURITY.md": ".github/SECURITY.md",
    "CONTRIBUTING.md": ".github/CONTRIBUTING.md",
    "Dockerfile": "tools/docker/Dockerfile",
    "flake.nix": "tools/nix/flake.nix",
    "flake.lock": "tools/nix/flake.lock",
}

# Mechanical reference following in fork-owned source (references only).
API_REWRITES: Tuple[Tuple[bytes, bytes], ...] = (
    (b'from src.qwen38', b'from engine.q38.src'),
    (b'from src.dsv41', b'from engine.dsv41.src'),
    (b'from src.kernels', b'from engine.src.kernels'),
    (b'from src.core', b'from engine.src.core'),
    (b'from strata_cli', b'from terminal.strata_cli'),
    (b'from strata_control', b'from terminal.strata_control'),
    (b'from serve.', b'from terminal.serve.'),
    (b'include "src/', b'include "engine/src/'),
    (b'#include "src/', b'#include "engine/src/'),
    (b'../apps/host/', b'../app/host/'),
    (b'../../apps/host/', b'../../app/host/'),
    (b'../../strata_control/', b'../../terminal/strata_control/'),
)

ADOPT, RETAIN, MERGE, BINARY, DELETE, DIVERGE = (
    "adopt_upstream", "retain_fork", "three_way_merge",
    "binary_conflict", "delete", "add_delete_conflict",
)

# Build-system foundations of the NEW upstream layout.  When both sides
# changed these, the official version must win or the hybrid tree cannot be
# built; the fork version is preserved as a review copy instead of discarded.
FOUNDATION_TARGETS: Tuple[str, ...] = (
    "engine/CMakeLists.txt",
    "engine/CMakePresets.json",
    "engine/cmake/",
)


@dataclass
class Record:
    old: str            # path in base / fork space
    new: str            # path in upstream final space
    fork_status: str = ""
    up_status: str = ""


@dataclass
class Decision:
    path: str           # old path (for the report)
    target: str         # final path
    kind: str
    source: str
    conflicts: int = 0
    note: str = ""
    review_copy: Optional[str] = None
    sha256: str = ""
    applied: bool = False
    data: bytes = field(default=b"", repr=False, compare=False)

    def public(self) -> Dict[str, Any]:
        row = asdict(self)
        row.pop("data", None)
        return row


@dataclass
class MergeReport:
    generated_at: str
    repo: str
    base: str
    fork: str
    upstream: str
    mode: str
    rename_rules: int = 0
    paths: int = 0
    decisions: List[Decision] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    port_checklist: List[str] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for row in self.decisions:
            out[row.kind] = out.get(row.kind, 0) + 1
        return out

    def to_json(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at, "repo": self.repo,
            "base": self.base, "fork": self.fork, "upstream": self.upstream,
            "mode": self.mode, "rename_rules": self.rename_rules,
            "paths": self.paths, "counts": self.counts(),
            "decisions": [d.public() for d in self.decisions],
            "errors": self.errors, "port_checklist": self.port_checklist,
        }


def git(repo: Path, *args: str, check: bool = False, timeout: int = 300) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(repo), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result


def resolve(repo: Path, ref: str) -> str:
    proc = git(repo, "rev-parse", "--verify", ref)
    if proc.returncode != 0:
        raise RuntimeError(f"cannot resolve ref {ref!r}")
    return proc.stdout.decode().strip()


class BlobStore:
    """One persistent `git cat-file --batch` process for every blob read.

    Windows process startup dominates `git show` for thousands of files; a
    single batch process makes the classification O(1) process launches.
    """

    def __init__(self, repo: Path) -> None:
        self._proc = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        self._cache: Dict[Tuple[str, str], Optional[bytes]] = {}

    def get(self, ref: str, path: Optional[str]) -> Optional[bytes]:
        if not path:
            return None
        key = (ref, path)
        if key in self._cache:
            return self._cache[key]
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(f"{ref}:{path}\n".encode("utf-8"))
        self._proc.stdin.flush()
        header = self._proc.stdout.readline().decode("utf-8", "replace").strip()
        if not header or header.endswith(" missing"):
            self._cache[key] = None
            return None
        parts = header.split()
        if len(parts) < 3 or parts[1] != "blob":
            self._cache[key] = None
            return None
        data = self._proc.stdout.read(int(parts[2]))
        self._proc.stdout.read(1)  # trailing newline
        self._cache[key] = data
        return data

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=30)
        except Exception:  # pragma: no cover - best effort shutdown
            self._proc.terminate()


def safe_target(path: str) -> Path:
    rel = PurePosixPath(path.replace("\\", "/"))
    if rel.is_absolute() or any(p in ("..", ".") for p in rel.parts) or ":" in path:
        raise ValueError(f"unsafe repository-relative path: {path!r}")
    return Path(rel)


def parse_status_lines(proc: subprocess.CompletedProcess) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        old = parts[1]
        new = parts[2] if len(parts) > 2 and status[:1] in ("R", "C") else old
        rows.append((status, old, new))
    return rows


def collect_records(repo: Path, base: str, fork: str, upstream: str) -> Tuple[Dict[str, Record], Dict[str, str]]:
    """Unify base/fork/upstream deltas into the final upstream path space."""
    up_rows = parse_status_lines(git(repo, "-c", "diff.renameLimit=30000", "diff",
                                     "--name-status", "-M", "-C", base, upstream))
    rename_map = {old: new for status, old, new in up_rows if status[:1] in ("R", "C")}
    records: Dict[str, Record] = {}

    for status, old, new in up_rows:
        rec = records.setdefault(new, Record(old=old, new=new))
        rec.up_status = status[:1]

    fork_rows = parse_status_lines(git(repo, "diff", "--name-status", "-M", base, fork))
    for status, old, _new in fork_rows:
        final = rename_map.get(old, old)
        rec = records.setdefault(final, Record(old=old, new=final))
        rec.fork_status = status[:1]
        if final == old and not rec.old:
            rec.old = old
    return records, rename_map


def blob(repo: Path, ref: str, path: Optional[str]) -> Optional[bytes]:
    if not path:
        return None
    proc = git(repo, "show", f"{ref}:{path}")
    return proc.stdout if proc.returncode == 0 else None


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8000]


def target_for(old: str, new: str, owned: bool) -> str:
    if owned and old.startswith("src/qwen38/"):
        return "engine/q38/src/" + old[len("src/qwen38/"):]
    if owned and old.startswith("include/strata/qwen38/"):
        return "engine/q38/include/strata/qwen38/" + old[len("include/strata/qwen38/"):]
    if owned and old.startswith("src/dsv41/"):
        return "engine/dsv41/src/" + old[len("src/dsv41/"):]
    if owned and old.startswith("include/strata/dsv41/"):
        return "engine/dsv41/include/strata/dsv41/" + old[len("include/strata/dsv41/"):]
    if old in ROOT_RELOCATE:
        return ROOT_RELOCATE[old]
    for prefix, replacement in RELOCATE_PREFIXES:
        if old.startswith(prefix):
            return replacement + old[len(prefix):]
    return new


def api_follow(data: bytes) -> Tuple[bytes, List[str]]:
    applied: List[str] = []
    for old, new in API_REWRITES:
        if old in data:
            data = data.replace(old, new)
            applied.append(f"{old.decode('ascii', 'replace').strip()} -> {new.decode('ascii', 'replace').strip()}")
    return data, applied


def three_way(repo: Path, fork: bytes, base: bytes, upstream: bytes, folder: Path) -> Tuple[bytes, int]:
    paths = [folder / name for name in ("fork", "base", "upstream")]
    for target, data in zip(paths, (fork, base, upstream)):
        target.write_bytes(data.replace(b"\r\n", b"\n"))
    result = git(repo, "merge-file", "-p", "--diff3",
                 "-L", "fork", "-L", "base", "-L", "upstream", *map(str, paths))
    if not 0 <= result.returncode <= 127:
        raise RuntimeError(f"git merge-file failed ({result.returncode})")
    return result.stdout, result.returncode


def decide(repo: Path, rec: Record, *, base: str, fork: str, upstream: str,
           evidence: Path, store: "BlobStore") -> Decision:
    b = store.get(base, rec.old)
    f = store.get(fork, rec.old)
    u = store.get(upstream, rec.new)
    owned = any(rec.old.startswith(p) for p in OWNED_PREFIXES)
    target = target_for(rec.old, rec.new, owned)
    safe_target(target)

    if owned:
        data = f if f is not None else (b or b"")
        if data and not is_binary(data):
            data, applied = api_follow(data)
            if applied:
                LOG.info("API-follow %s: %s", rec.old, applied)
        note = "fork-owned; upstream bytes never applied"
        if u is not None and b is not None and u != b:
            note += "; upstream changed the same path - review API drift"
            review = evidence / "review" / (rec.new.replace("/", "__") + ".upstream")
            review.parent.mkdir(parents=True, exist_ok=True)
            review.write_bytes(u)
        return Decision(rec.old, target, RETAIN, fork, note=note,
                        sha256=hashlib.sha256(data).hexdigest(), data=data)

    if f == b:
        if u is None:
            # A deletion must act on the path that actually exists in the
            # fork worktree, not on a relocated destination that was never
            # written.
            return Decision(rec.old, rec.old, DELETE, "none",
                            note="upstream deleted; fork untouched")
        return Decision(rec.old, target, ADOPT, upstream,
                        sha256=hashlib.sha256(u).hexdigest(), data=u)

    if u == b or rec.up_status not in ("M", "A", "R", "C"):
        if rec.up_status == "D":
            return Decision(rec.old, target, DIVERGE, fork,
                            note="upstream deleted but fork changed it; fork retained",
                            sha256=hashlib.sha256(f or b"").hexdigest(), data=f or b"")
        return Decision(rec.old, target, RETAIN, fork,
                        sha256=hashlib.sha256(f or b"").hexdigest(), data=f or b"")

    if rec.fork_status == "D":
        return Decision(rec.old, target, ADOPT, upstream,
                        note="fork deleted, upstream kept",
                        sha256=hashlib.sha256(u or b"").hexdigest(), data=u or b"")

    if f is None or u is None:
        data = f if f is not None else (u or b"")
        return Decision(rec.old, target, DIVERGE, fork if f is not None else upstream,
                        note="add/delete divergence; semantic review required",
                        sha256=hashlib.sha256(data).hexdigest(), data=data)

    if is_binary(f) or is_binary(u):
        return Decision(rec.old, target, BINARY, fork,
                        note="binary changed on both sides; fork bytes retained",
                        sha256=hashlib.sha256(f).hexdigest(), data=f)

    review = evidence / "review" / rec.new.replace("/", "__")
    review.parent.mkdir(parents=True, exist_ok=True)
    merged, conflicts = three_way(repo, f, b or b"", u, evidence / "tmp")
    review.write_bytes(merged)
    if conflicts and any(target == t or target.startswith(t) for t in FOUNDATION_TARGETS):
        fork_review = evidence / "review" / (rec.new.replace("/", "__") + ".fork-retained")
        fork_review.write_bytes(f)
        return Decision(rec.old, target, ADOPT, upstream,
                        note="official build foundation adopted; fork version saved as review copy",
                        review_copy=str(fork_review),
                        sha256=hashlib.sha256(u).hexdigest(), data=u)
    working = f if conflicts else merged
    return Decision(
        rec.old, target, MERGE, fork if conflicts else "three-way",
        conflicts=conflicts, review_copy=str(review),
        note="fork bytes retained pending resolution" if conflicts else "clean three-way merge",
        sha256=hashlib.sha256(working).hexdigest(), data=working,
    )


def run(args: argparse.Namespace) -> MergeReport:
    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        raise RuntimeError(f"not a git worktree: {repo}")
    base, fork, upstream = (resolve(repo, r) for r in (args.base, args.fork, args.upstream))
    if resolve(repo, "HEAD") != fork:
        raise RuntimeError(f"worktree HEAD != fork ref {fork}")
    if git(repo, "status", "--porcelain").stdout.decode().strip():
        raise RuntimeError("worktree is dirty; refusing to merge")

    evidence = Path(args.evidence).resolve()
    (evidence / "review").mkdir(parents=True, exist_ok=True)
    (evidence / "tmp").mkdir(parents=True, exist_ok=True)
    report = MergeReport(datetime.now(timezone.utc).isoformat(), str(repo),
                         base, fork, upstream, args.mode)

    records, renames = collect_records(repo, base, fork, upstream)
    report.rename_rules, report.paths = len(renames), len(records)
    LOG.info("classifying %d unified paths (%d rename rules)", len(records), len(renames))

    store = BlobStore(repo)
    try:
        for rec in tqdm(sorted(records.values(), key=lambda r: (r.new, r.old)),
                        desc="one-way merge", unit="file"):
            try:
                report.decisions.append(decide(repo, rec, base=base, fork=fork,
                                               upstream=upstream, evidence=evidence,
                                               store=store))
            except Exception as exc:
                report.errors.append(f"{rec.old} -> {rec.new}: {exc}")
    finally:
        store.close()

    report.port_checklist = [
        "engine/include/strata/artifact/model_family.hpp: add Qwen38 family to ModelFamily/model_family_from_arch/model_family_runs",
        "engine/include/strata/core/family_layout.hpp: add qwen38 geometry + tensor layout table",
        "engine/CMakeLists.txt: add_subdirectory(q38) for the dedicated engine area",
        "engine/q38: implement --serve stdin/stdout protocol with 'prompt N tokens = ...' progress line",
        "engine-component.json schema 2 + BUILD.json for the signed qwen38 component",
        "terminal/strata_control/engine_component.py: register qwen38 backend/platform",
    ]

    if args.mode == "apply":
        # De-duplicate by target.  A DELETE for a relocated source must not
        # remove a file another decision has written to the same target.
        by_target: Dict[str, Decision] = {}
        for decision in report.decisions:
            if decision.kind == DELETE:
                continue
            current = by_target.get(decision.target)
            if current is None or (decision.path == decision.target and current.path != current.target):
                by_target[decision.target] = decision
        write_targets = set(by_target)
        for target, decision in by_target.items():
            decision.applied = True
            path = repo / safe_target(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(decision.data)  # empty files must be created too
        removed = 0
        for decision in report.decisions:
            if decision.kind == DELETE and decision.path not in write_targets:
                victim = repo / safe_target(decision.path)
                if victim.exists():
                    victim.unlink()
                    removed += 1
        # Relocation cleanup: remove the old path once its content has been
        # written to the new path and nothing else targets the old path.
        for decision in by_target.values():
            if decision.target != decision.path and decision.path not in write_targets:
                old = repo / safe_target(decision.path)
                if old.exists():
                    old.unlink()
                    removed += 1
        LOG.info("applied %d file writes, removed %d stale paths into %s",
                 len(by_target), removed, repo)
    else:
        LOG.info("dry-run: %d decisions staged in memory only", len(report.decisions))

    (evidence / "manifest.json").write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--fork", default="HEAD")
    parser.add_argument("--upstream", default="upstream/main")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--mode", choices=("dry-run", "apply"), default="dry-run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FMT, stream=sys.stderr)
    try:
        report = run(args)
    except Exception as exc:
        LOG.error("merge failed: %s", exc)
        return 2
    print(json.dumps({
        "counts": report.counts(), "paths": report.paths,
        "rename_rules": report.rename_rules, "errors": len(report.errors),
        "port_checklist": len(report.port_checklist), "evidence": args.evidence,
    }, indent=2))
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
