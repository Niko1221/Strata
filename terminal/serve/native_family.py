"""One adapter/contract admission invariant, without granting engine capability."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from terminal.strata_control import user_args


def valid_q36_memory_capability(cap: object, digest: str | None = None) -> bool:
    """Validate only the memory schema; a valid expectation is not engine proof."""
    keys = {
        "schema",
        "family",
        "modes",
        "flag",
        "requiresExpertStore",
        "requiresExplicitSlots",
        "scope",
        "totalProcessBound",
        "qualification",
    }
    if digest is not None:
        keys.add("binarySha256")
    return (
        isinstance(cap, dict)
        and set(cap) == keys
        and type(cap["schema"]) is int
        and cap["schema"] == 1
        and cap["family"] == "qwen35moe"
        and cap["modes"] in ([], ["bounded-experts-v1"])
        and cap["flag"] == "--q36-memory"
        and cap["requiresExpertStore"] == "mapped-pack"
        and cap["requiresExplicitSlots"] is True
        and cap["scope"] == "expert-working-storage"
        and cap["totalProcessBound"] is False
        and cap["qualification"] == "experimental-unqualified"
        and (digest is None or (_valid_sha(digest) and cap["binarySha256"] == digest))
    )


def _argument_value(args: object, flag: str) -> str | None:
    args = [] if args is None else args
    if not isinstance(args, (list, tuple)) or any(not isinstance(a, str) for a in args):
        raise ValueError("Q36 memory arguments must be a list of strings")
    found = [a for a in args if a == flag or a.startswith(flag + "=")]
    if not found:
        return None
    if len(found) != 1 or found[0] != flag:
        raise ValueError(f"Q36 memory: {flag} is malformed or repeated")
    at = args.index(flag)
    if at + 1 >= len(args) or args[at + 1] == "" or args[at + 1].startswith("--"):
        raise ValueError(f"Q36 memory: {flag} requires a value")
    return args[at + 1]


def q36_memory_mode(args: object, requested: object = None) -> str:
    """Reject conflicting representations rather than silently dropping one."""
    values = [
        v
        for v in (_argument_value(args, "--q36-memory"), requested)
        if v is not None and v != ""
    ]
    if any(v not in ("legacy", "bounded-experts-v1") for v in values):
        raise ValueError("Q36 memory mode is unsupported")
    if len(set(values)) > 1:
        raise ValueError("Q36 memory mode conflicts with runtime arguments")
    return values[0] if values else "legacy"


def native_merged_args(cfg: Mapping[str, object]) -> list[str]:
    """The runtime's argument precedence, also used by early admission."""
    args, mine = cfg.get("args") or [], cfg.get("user_args") or []
    # A malformed explicit memory token is not a default, even if another layer
    # could otherwise mask it. Top-level empty/null still means clear the setting.
    _argument_value(args, "--q36-memory")
    _argument_value(mine, "--q36-memory")
    return user_args.merge(args, mine) if mine else list(args)


def validate_q36_memory_args(args: object, mode: str) -> None:
    """Bounded storage requires actual mapped-pack and explicit positive slots."""
    if mode != "bounded-experts-v1":
        return
    if _argument_value(args, "--q36-expert-store") != "mapped-pack":
        raise ValueError("Q36 memory requires --q36-expert-store mapped-pack")
    slots = _argument_value(args, "--expert-slots")
    if (
        not slots
        or re.fullmatch(r"[0-9]+", slots) is None
        or not 1 <= int(slots) <= 2**31 - 1
    ):
        raise ValueError("Q36 memory requires positive explicit --expert-slots")


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _private_cache_requirement(args: object) -> int:
    if args is None:
        return 0
    if not isinstance(args, (list, tuple)) or any(not isinstance(a, str) for a in args):
        raise ValueError("Q36 cache arguments must be a list of strings")
    if "--private-conversation-cache" in args:
        return 2
    return int(
        any(flag in args for flag in ("--prompt-cache", "--conversation-cache-mib"))
    )


@dataclass(frozen=True)
class NativeAdmission:
    """Validated requirements for one start, never proof supplied by an engine."""

    q36: bool
    private_cache: int = 0
    expected_kv_sha256: str | None = None
    generation: str | None = None
    memory_mode: str = "legacy"
    expected_memory_sha256: str | None = None
    memory_capability_json: str | None = None

    def __post_init__(self) -> None:
        if type(self.q36) is not bool or type(self.private_cache) is not int:
            raise ValueError("Q36 admission requirements have invalid types")
        if self.private_cache not in (0, 1, 2):
            raise ValueError("Q36 private-cache requirement must be 0, 1 or 2")
        if not self.q36 and (
            self.private_cache
            or self.expected_kv_sha256 is not None
            or self.generation is not None
            or self.memory_mode != "legacy"
            or self.expected_memory_sha256 is not None
            or self.memory_capability_json is not None
        ):
            raise ValueError("Native adapter conflicts with retained Q36 requirements")
        if self.expected_kv_sha256 is not None and not _valid_sha(
            self.expected_kv_sha256
        ):
            raise ValueError(
                "Q36 binary expectation must be a lowercase SHA-256 digest"
            )
        if self.generation is not None and (
            not isinstance(self.generation, str)
            or re.fullmatch(r"[a-zA-Z0-9-]{1,80}", self.generation) is None
        ):
            raise ValueError("Q36 launch generation is invalid")
        if self.memory_mode not in ("legacy", "bounded-experts-v1"):
            raise ValueError("Q36 memory mode is invalid")
        if self.memory_capability_json is not None:
            try:
                cap = json.loads(self.memory_capability_json)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Q36 memory capability expectation is invalid"
                ) from exc
            if (
                not self.expected_memory_sha256
                or not valid_q36_memory_capability(cap, self.expected_memory_sha256)
                or self.memory_mode not in cap["modes"]
            ):
                raise ValueError(
                    "Q36 memory capability/digest expectation is inconsistent"
                )
        elif self.expected_memory_sha256 is not None:
            raise ValueError("Q36 memory digest requires a capability expectation")
        if self.memory_mode == "legacy" and self.memory_capability_json is not None:
            raise ValueError("Q36 memory expectation cannot be downgraded to legacy")


def native_admission(cfg: Mapping[str, object]) -> NativeAdmission:
    """Resolve explicit/legacy identity and validate all consumed Q36 metadata.

    A retained contract may require Q36 verification when old configs omit arch.
    It cannot override a conflicting explicit adapter or establish capability:
    the actual binary and READY receipts still have to satisfy these requirements.
    """
    arch = cfg.get("arch")
    explicit = arch is not None and arch != ""
    if explicit:
        q36 = arch in ("qwen35moe", "qwen36")
    else:
        q36 = "strata-q36" in str(cfg.get("exe", "")) or bool(
            re.search(r"qwen3\.6|qwen36|35b-a3b", str(cfg.get("model_name", "")), re.I)
        )
    expected = generation = memory_digest = memory_json = None
    combined = native_merged_args(cfg)
    mode = q36_memory_mode(combined, cfg.get("q36_memory"))
    validate_q36_memory_args(combined, mode)
    run = cfg.get("q36_run")
    if run is not None:
        if not isinstance(run, Mapping):
            raise ValueError("q36_run must be a Q36 contract object or null")
        backend = run.get("backend")
        if not isinstance(backend, Mapping) or backend.get("family") != "qwen36":
            raise ValueError("q36_run.backend.family must identify qwen36")
        if explicit and not q36:
            raise ValueError(
                "arch conflicts with q36_run.backend.family; "
                "correct the adapter or clear the retained contract"
            )
        if type(run.get("schemaVersion")) is not int or run["schemaVersion"] != 1:
            raise ValueError("q36_run requires schemaVersion 1")
        q36 = True
        generation = run.get("generation")
        digest, claim = backend.get("binarySha256"), backend.get("kvCapability")
        if digest is not None and not _valid_sha(digest):
            raise ValueError("q36_run.backend.binarySha256 is invalid")
        if claim is not None:
            if (
                not isinstance(claim, Mapping)
                or claim.get("family") != "qwen35moe"
                or not _valid_sha(digest)
                or claim.get("binarySha256") != digest
            ):
                raise ValueError(
                    "q36_run KV expectation has inconsistent family/digest identity"
                )
            expected = digest
        memory_claim = backend.get("memoryCapability")
        sections = [run.get(key) for key in ("requested", "planned", "effective")]
        retained = [
            s.get("q36_memory")
            for s in sections
            if isinstance(s, Mapping) and "q36_memory" in s
        ]
        memory_contract = memory_claim is not None or any(
            v != "legacy" for v in retained
        )
        if memory_contract:
            if (
                len(retained) != 3
                or any(v != mode for v in retained)
                or mode != "bounded-experts-v1"
                or not valid_q36_memory_capability(memory_claim, digest)
                or memory_claim.get("binarySha256") != digest
                or mode not in memory_claim["modes"]
            ):
                raise ValueError("q36_run memory expectations are inconsistent")
            if (
                q36_memory_mode(combined) != mode
                or q36_memory_mode(run.get("argv")) != mode
            ):
                raise ValueError("q36_run memory mode was lost from runtime argv")
            validate_q36_memory_args(run.get("argv"), mode)
            memory_digest, memory_json = digest, json.dumps(
                memory_claim, sort_keys=True
            )
        elif mode != "legacy" or any(v != mode for v in retained):
            raise ValueError("q36_run is missing the bounded memory expectation")
    if not q36 and (
        cfg.get("q36_memory") is not None
        or _argument_value(combined, "--q36-memory") is not None
    ):
        raise ValueError("Q36 memory setting conflicts with the native adapter")
    # engine_args merges the user layer before starting. Neither layer may lose
    # its acknowledgement requirement while projecting the config to env.
    private = 0
    if q36:
        private = max(
            _private_cache_requirement(cfg.get("args")),
            _private_cache_requirement(cfg.get("user_args")),
        )
    return NativeAdmission(
        q36, private, expected, generation, mode, memory_digest, memory_json
    )


def is_q36_config(cfg: Mapping[str, object]) -> bool:
    """Return the validated identity; conflicting retained contracts are errors."""
    return native_admission(cfg).q36


def native_admission_from_env(
    env: Mapping[str, object], args: object = None
) -> NativeAdmission:
    """Validate the same invariant again at the constructor's captured boundary."""
    private = env.get("STRATA_REQUIRE_Q36_PRIVATE_CACHE", "0")
    if private not in ("0", "1", "2"):
        raise ValueError("Q36 private-cache environment requirement is invalid")
    if "STRATA_REQUIRE_Q36_KV" in env:
        flag = env["STRATA_REQUIRE_Q36_KV"]
        if flag not in ("0", "1"):
            raise ValueError("Q36 KV environment requirement is invalid")
        q36 = flag == "1"
    else:
        # Older callers supplied only the private-cache requirement. Keep that
        # check, without manufacturing typed-KV capability for a legacy binary.
        q36 = private != "0"
    admission = NativeAdmission(
        q36,
        int(private),
        env.get("STRATA_Q36_KV_HOST_SHA256"),
        env.get("STRATA_Q36_LAUNCH_GENERATION"),
        env.get("STRATA_Q36_MEMORY_MODE", "legacy"),
        env.get("STRATA_Q36_MEMORY_HOST_SHA256"),
        env.get("STRATA_Q36_MEMORY_CAPABILITY"),
    )
    if q36_memory_mode(args) != admission.memory_mode:
        raise ValueError("Q36 memory environment and actual argv disagree")
    if not admission.q36 and _argument_value(args, "--q36-memory") is not None:
        raise ValueError("Q36 memory arguments conflict with the native adapter")
    validate_q36_memory_args(args, admission.memory_mode)
    if admission.q36 and admission.private_cache < _private_cache_requirement(args):
        raise ValueError(
            "Q36 private-cache requirement was lost from the runtime arguments"
        )
    return admission
