from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from . import GateError
from .schema import validate_schema


DECISIONS = {"ours", "upstream", "delta", "hybrid", "defer", "drop"}


UPSTREAM_STATUSES = {"full", "partial", "none"}


REWORK_EFFECTS = {"better", "same", "worse", "unknown"}


EVIDENCE_STATUSES = {"passed", "failed", "not-run", "not-applicable"}


BASE_REQUIRED_EVIDENCE = (
    "build",
    "tests",
    "correctness",
    "quality",
    "route",
    "fallback",
    "backends",
    "architectureReview",
)


OPTIONAL_EVIDENCE = (
    "benchmarksResources",
    "implementationCombinations",
    "targetCoverage",
    "regression",
    "bisect",
    "rollback",
)


REQUIRED_EVIDENCE = BASE_REQUIRED_EVIDENCE + OPTIONAL_EVIDENCE
EVIDENCE_CATEGORIES = set(REQUIRED_EVIDENCE)


HYBRID_CONTRACT_KEYS = (
    "boundary",
    "inputs",
    "outputs",
    "ownership",
    "streamOrder",
    "fallback",
)


DELTA_CONTRACT_KEYS = (
    "upstreamBaseAnchor",
    "minimalInsertion",
    "insertionPoint",
    "boundary",
    "inputs",
    "outputs",
    "ownership",
    "streamOrder",
    "fallback",
    "invariants",
    "validation",
    "rollback",
)


DEFAULT_THRESHOLDS = {
    "maxChangedFiles": 60,
    "maxChangedLines": 3000,
    "maxIfdefAdded": 25,
    "maxInvariantsChanged": 1,
}


def require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise GateError(f"{name} должно быть непустой строкой без NUL")
    return value.strip()


def require_choice(value: Any, choices: set[str], name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise GateError(f"{name} недопустим")
    return value


def normalize_relative_path(value: Any, name: str, allow_dot: bool = False) -> str:
    text = require_text(value, name).replace("\\", "/")
    if text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        raise GateError(f"{name} должен оставаться внутри репозитория")
    parts = [part for part in text.split("/") if part]
    if not parts or ".." in parts or (text == "." and not allow_dot):
        raise GateError(f"{name} должен быть относительным путём внутри репозитория")
    return "/".join(parts)


def validate_argv(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise GateError(f"{name} должен быть непустым argv-списком")
    result: list[str] = []
    for index, arg in enumerate(value):
        if not isinstance(arg, str) or not arg or "\0" in arg:
            raise GateError(f"{name}[{index}] должен быть непустой строкой без NUL")
        result.append(arg)
    return result


def validate_anchor_method(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise GateError(f"{name} должен быть объектом")
    allowed = {"method", "command", "checkedRefs", "notes"}
    unknown = set(value) - allowed
    if unknown:
        raise GateError(f"{name} содержит неизвестные поля: {', '.join(sorted(unknown))}")
    require_choice(
        value.get("method"),
        {"symbol-search", "source-review", "call-trace", "runtime-route", "binary-symbol-check", "manual-review", "other"},
        f"{name}.method",
    )
    command = value.get("command")
    if isinstance(command, str):
        require_text(command, f"{name}.command")
    else:
        validate_argv(command, f"{name}.command")
    if "checkedRefs" in value:
        refs = value["checkedRefs"]
        if not isinstance(refs, list) or not refs:
            raise GateError(f"{name}.checkedRefs должен быть непустым списком")
        seen: set[str] = set()
        for index, ref in enumerate(refs):
            text = require_text(ref, f"{name}.checkedRefs[{index}]")
            if text in seen:
                raise GateError(f"{name}.checkedRefs не должен содержать дубликаты")
            seen.add(text)
    if "notes" in value:
        require_text(value["notes"], f"{name}.notes")


def validate_date(value: Any, name: str) -> str:
    text = require_text(value, name)
    try:
        dt.date.fromisoformat(text)
    except ValueError as exc:
        raise GateError(f"{name} должен быть датой YYYY-MM-DD") from exc
    return text


def looks_like_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return "/" in value or "\\" in value or bool(Path(value).suffix) or value.startswith(".")


def validate_evidence_reference(value: Any, name: str) -> None:
    if isinstance(value, str):
        text = require_text(value, name)
        if looks_like_path(text):
            normalize_relative_path(text, name)
        return
    if not isinstance(value, dict):
        raise GateError(f"{name} должен быть путём или объектом evidence")
    allowed = {"category", "artifact", "status", "revision"}
    unknown = set(value) - allowed
    if unknown:
        raise GateError(f"{name} содержит неизвестные поля: {', '.join(sorted(unknown))}")
    require_choice(value.get("category"), EVIDENCE_CATEGORIES, f"{name}.category")
    if "artifact" in value:
        normalize_relative_path(value["artifact"], f"{name}.artifact")
    if "status" in value:
        require_choice(value["status"], EVIDENCE_STATUSES, f"{name}.status")
    if "revision" in value:
        require_text(value["revision"], f"{name}.revision")


def validate_evidence_item(item: Any, name: str) -> None:
    if not isinstance(item, dict):
        raise GateError(f"{name} должен быть объектом")
    require_choice(item.get("status"), EVIDENCE_STATUSES, f"{name}.status")
    if "artifact" in item:
        artifact = require_text(item["artifact"], f"{name}.artifact")
        if looks_like_path(artifact):
            normalize_relative_path(artifact, f"{name}.artifact")
    if "artifacts" in item:
        artifacts = item["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise GateError(f"{name}.artifacts должен быть непустым списком")
        for index, artifact in enumerate(artifacts):
            text = require_text(artifact, f"{name}.artifacts[{index}]")
            if looks_like_path(text):
                normalize_relative_path(text, f"{name}.artifacts[{index}]")
    if "command" in item:
        command = item["command"]
        if isinstance(command, str):
            require_text(command, f"{name}.command")
        else:
            validate_argv(command, f"{name}.command")
    if "commands" in item:
        commands = item["commands"]
        if not isinstance(commands, list) or not commands:
            raise GateError(f"{name}.commands должен быть непустым списком")
        for index, command in enumerate(commands):
            validate_argv(command, f"{name}.commands[{index}]")
    for key in ("revision", "reason", "details", "reviewer"):
        if key in item:
            require_text(item[key], f"{name}.{key}")
    if item["status"] == "not-applicable":
        require_text(item.get("reason"), f"{name}.reason")
    if "date" in item:
        validate_date(item["date"], f"{name}.date")
    if "manualVerdict" in item:
        require_choice(item["manualVerdict"], {"accepted", "rejected"}, f"{name}.manualVerdict")
    if "combinations" in item:
        combinations = item["combinations"]
        if not isinstance(combinations, list) or not combinations:
            raise GateError(f"{name}.combinations должен быть непустым списком")
        for index, combination in enumerate(combinations):
            prefix = f"{name}.combinations[{index}]"
            if not isinstance(combination, dict):
                raise GateError(f"{prefix} должен быть объектом")
            require_choice(
                combination.get("implementation"),
                {"upstream-only", "ours-only", "hybrid", "delta"},
                f"{prefix}.implementation",
            )
            combination_status = require_choice(
                combination.get("status"),
                {"passed", "failed", "not-run"},
                f"{prefix}.status",
            )
            if item["status"] == "passed" and combination_status != "passed":
                raise GateError(f"{prefix}.status должен быть passed при passed evidence")
            require_text(combination.get("revision"), f"{prefix}.revision")
            refs = combination.get("evidence")
            if not isinstance(refs, list) or not refs:
                raise GateError(f"{prefix}.evidence должен быть непустым списком")
            for ref_index, ref in enumerate(refs):
                validate_evidence_reference(ref, f"{prefix}.evidence[{ref_index}]")


def is_legacy_plan(plan: dict[str, Any]) -> bool:
    if any(key in plan for key in ("author", "architectureReview", "acceptance")):
        return False
    for key in BASE_REQUIRED_EVIDENCE:
        item = plan["evidence"][key]
        if "revision" in item:
            return False
        artifact = item.get("artifact")
        if artifact is not None and looks_like_path(artifact):
            return False
    return True


def validate_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan.get("version"), int) or isinstance(plan.get("version"), bool) or plan["version"] != 1:
        raise GateError("поддерживается только version=1")
    for key in ("oursRef", "upstreamRef", "candidateRef", "decision", "rationale"):
        require_text(plan.get(key), key)
    require_choice(plan["decision"], DECISIONS, "decision")
    if "author" in plan:
        require_text(plan["author"], "author")

    matrix = plan.get("matrix")
    if not isinstance(matrix, list) or not matrix:
        raise GateError("matrix должна быть непустым списком")
    ids: set[str] = set()
    hybrid_needed = plan["decision"] == "hybrid"
    delta_needed = plan["decision"] == "delta"
    defer_needed = plan["decision"] == "defer"
    for index, item in enumerate(matrix):
        if not isinstance(item, dict):
            raise GateError(f"matrix[{index}] должен быть объектом")
        item_id = require_text(item.get("id"), f"matrix[{index}].id")
        if item_id in ids:
            raise GateError(f"дублирующийся id в matrix: {item_id}")
        ids.add(item_id)
        require_text(item.get("oursAnchor"), f"matrix[{index}].oursAnchor")
        anchor_status = require_choice(
            item.get("anchorStatus"),
            {"preserved", "adapted", "removed", "missing"},
            f"matrix[{index}].anchorStatus",
        )
        if anchor_status == "missing":
            raise GateError(f"matrix[{index}].anchorStatus=missing блокирует план")
        validate_anchor_method(item.get("anchorMethod"), f"matrix[{index}].anchorMethod")
        anchor_evidence = item.get("anchorEvidence")
        if not isinstance(anchor_evidence, list) or not anchor_evidence:
            raise GateError(f"matrix[{index}].anchorEvidence должен быть непустым списком")
        for ref_index, ref in enumerate(anchor_evidence):
            validate_evidence_reference(ref, f"matrix[{index}].anchorEvidence[{ref_index}]")
        require_choice(item.get("upstreamStatus"), UPSTREAM_STATUSES, f"matrix[{index}].upstreamStatus")
        require_choice(item.get("reworkEffect"), REWORK_EFFECTS, f"matrix[{index}].reworkEffect")
        decision = require_choice(item.get("decision"), DECISIONS, f"matrix[{index}].decision")
        if anchor_status == "removed" and (decision != "drop" or item.get("upstreamStatus") != "full"):
            raise GateError(f"matrix[{index}].anchorStatus=removed допускается только для drop с полной upstream-заменой")
        require_text(item.get("reason"), f"matrix[{index}].reason")
        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            raise GateError(f"matrix[{index}].evidence должен быть списком")
        if item["decision"] != "defer" and not evidence:
            raise GateError(f"matrix[{index}] требует evidence")
        for ref_index, ref in enumerate(evidence):
            validate_evidence_reference(ref, f"matrix[{index}].evidence[{ref_index}]")
        if item["upstreamStatus"] in {"full", "partial"}:
            require_text(item.get("upstreamAlgorithm"), f"matrix[{index}].upstreamAlgorithm")
        if "upstreamAnchor" in item and item["upstreamAnchor"] is not None:
            require_text(item["upstreamAnchor"], f"matrix[{index}].upstreamAnchor")
        if "candidateAnchor" in item:
            require_text(item["candidateAnchor"], f"matrix[{index}].candidateAnchor")
        if item["decision"] == "hybrid":
            hybrid_needed = True
        if item["decision"] == "delta":
            delta_needed = True
        if item["decision"] == "defer":
            deferred = item if any(key in item for key in ("experimentRef", "missingEvidence", "unblockConditions", "reviewer")) else plan.get("defer")
            if not isinstance(deferred, dict):
                raise GateError(f"matrix[{index}].defer требует experimentRef, reviewer, missingEvidence и unblockConditions")
            require_text(deferred.get("experimentRef"), f"matrix[{index}].defer.experimentRef")
            require_text(deferred.get("reviewer"), f"matrix[{index}].defer.reviewer")
            for key in ("missingEvidence", "unblockConditions"):
                value = deferred.get(key)
                if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() and "\0" not in x for x in value):
                    raise GateError(f"matrix[{index}].defer.{key} должен быть непустым списком строк")
        if item["decision"] == "drop":
            if item["upstreamStatus"] != "full":
                raise GateError(f"matrix[{index}].drop требует upstreamStatus=full")
            require_text(item.get("replacementUpstreamAnchor"), f"matrix[{index}].replacementUpstreamAnchor")
            proofs = item.get("fullReplacementEvidence")
            if not isinstance(proofs, list) or not proofs:
                raise GateError(f"matrix[{index}].drop требует fullReplacementEvidence полной замены")
            for proof_index, proof in enumerate(proofs):
                validate_evidence_reference(proof, f"matrix[{index}].fullReplacementEvidence[{proof_index}]")

    if hybrid_needed:
        contract = plan.get("hybridContract")
        if not isinstance(contract, dict):
            raise GateError("для hybrid требуется hybridContract")
        for key in HYBRID_CONTRACT_KEYS:
            require_text(contract.get(key), f"hybridContract.{key}")
    if delta_needed:
        contract = plan.get("deltaContract")
        if not isinstance(contract, dict):
            raise GateError("для delta требуется deltaContract")
        for key in DELTA_CONTRACT_KEYS:
            value = contract.get(key)
            if isinstance(value, list):
                if not value or not all(isinstance(x, str) and x.strip() and "\0" not in x for x in value):
                    raise GateError(f"deltaContract.{key} должен быть непустым списком строк")
            else:
                require_text(value, f"deltaContract.{key}")

    if defer_needed:
        deferred = plan.get("defer")
        if not isinstance(deferred, dict):
            raise GateError("для defer требуется объект defer")
        require_text(deferred.get("experimentRef"), "defer.experimentRef")
        require_text(deferred.get("reviewer"), "defer.reviewer")
        for key in ("missingEvidence", "unblockConditions"):
            value = deferred.get(key)
            if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() and "\0" not in x for x in value):
                raise GateError(f"defer.{key} должен быть непустым списком строк")

    invariants = plan.get("invariantsChanged", [])
    if not isinstance(invariants, list) or not all(isinstance(x, str) and x.strip() and "\0" not in x for x in invariants):
        raise GateError("invariantsChanged должен быть списком непустых строк")
    if len(set(invariants)) != len(invariants):
        raise GateError("invariantsChanged не должен содержать дубликатов")

    evidence = plan.get("evidence")
    if not isinstance(evidence, dict):
        raise GateError("evidence должен быть объектом")
    for key in BASE_REQUIRED_EVIDENCE:
        if key not in evidence:
            raise GateError(f"evidence.{key} отсутствует")
    for key in EVIDENCE_CATEGORIES:
        if key in evidence:
            validate_evidence_item(evidence[key], f"evidence.{key}")
    if evidence["architectureReview"].get("status") == "passed":
        require_text(evidence["architectureReview"].get("reviewer"), "evidence.architectureReview.reviewer")
        require_text(evidence["architectureReview"].get("artifact"), "evidence.architectureReview.artifact")

    legacy = is_legacy_plan(plan)
    if not legacy:
        for key in OPTIONAL_EVIDENCE:
            if key not in evidence:
                raise GateError(f"evidence.{key} отсутствует")
    if not legacy and (hybrid_needed or delta_needed):
        combinations = evidence.get("implementationCombinations")
        if not isinstance(combinations, dict) or combinations.get("status") != "passed":
            raise GateError("для hybrid/delta требуется passed implementationCombinations")
        runs = combinations.get("combinations")
        if not isinstance(runs, list) or not runs:
            raise GateError("implementationCombinations.combinations должен быть непустым списком")
        implementations = {run.get("implementation") for run in runs if isinstance(run, dict)}
        required = set()
        if hybrid_needed:
            required.update({"upstream-only", "ours-only", "hybrid"})
        if delta_needed:
            required.update({"upstream-only", "delta"})
        missing = required - implementations
        if missing:
            raise GateError("implementationCombinations не содержит: " + ", ".join(sorted(missing)))

    checks = plan.get("checks", [])
    if not isinstance(checks, list):
        raise GateError("checks должен быть списком")
    check_names: set[str] = set()
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise GateError(f"checks[{index}] должен быть объектом")
        name = require_text(check.get("name"), f"checks[{index}].name")
        if name in check_names:
            raise GateError(f"дублирующееся имя проверки: {name}")
        check_names.add(name)
        validate_argv(check.get("command"), f"checks[{index}].command")
        normalize_relative_path(check.get("cwd", "."), f"checks[{index}].cwd", allow_dot=True)
        timeout = check.get("timeoutMs", 120000)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            raise GateError(f"checks[{index}].timeoutMs должен быть положительным целым")

    acknowledgments = plan.get("warningAcknowledgments", [])
    if not isinstance(acknowledgments, list):
        raise GateError("warningAcknowledgments должен быть списком")
    ack_codes: set[str] = set()
    for index, item in enumerate(acknowledgments):
        if not isinstance(item, dict):
            raise GateError(f"warningAcknowledgments[{index}] должен быть объектом")
        code = require_text(item.get("code"), f"warningAcknowledgments[{index}].code")
        if code in ack_codes:
            raise GateError(f"дублирующийся warning acknowledgment: {code}")
        ack_codes.add(code)
        require_text(item.get("reason"), f"warningAcknowledgments[{index}].reason")
        require_text(item.get("reviewer"), f"warningAcknowledgments[{index}].reviewer")
        if "scope" in item:
            scope = item["scope"]
            if not isinstance(scope, list) or not scope or not all(isinstance(x, str) and x.strip() for x in scope):
                raise GateError(f"warningAcknowledgments[{index}].scope должен быть непустым списком строк")
        if "mitigation" in item:
            require_text(item["mitigation"], f"warningAcknowledgments[{index}].mitigation")
        if "evidence" in item:
            refs = item["evidence"]
            if not isinstance(refs, list) or not refs:
                raise GateError(f"warningAcknowledgments[{index}].evidence должен быть непустым списком")
            for ref_index, ref in enumerate(refs):
                validate_evidence_reference(ref, f"warningAcknowledgments[{index}].evidence[{ref_index}]")
        if "date" in item:
            validate_date(item["date"], f"warningAcknowledgments[{index}].date")
        if "outcome" in item:
            require_choice(item["outcome"], {"accepted", "mitigated", "deferred", "rejected"}, f"warningAcknowledgments[{index}].outcome")
        if not legacy:
            for key in ("scope", "mitigation", "evidence", "date", "outcome"):
                if key not in item:
                    raise GateError(f"warningAcknowledgments[{index}].{key} отсутствует")

    thresholds = dict(DEFAULT_THRESHOLDS)
    supplied = plan.get("thresholds", {})
    if not isinstance(supplied, dict):
        raise GateError("thresholds должен быть объектом")
    unknown = set(supplied) - set(thresholds)
    if unknown:
        raise GateError(f"неизвестные thresholds: {', '.join(sorted(unknown))}")
    for key, default in thresholds.items():
        if key in supplied:
            value = supplied[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise GateError(f"thresholds.{key} должен быть неотрицательным целым")
            if value > default:
                raise GateError(f"thresholds.{key} не может быть выше встроенного {default}")
            thresholds[key] = value
    plan["thresholds"] = thresholds
    plan["thresholdRequests"] = dict(supplied)

    if "architectureReview" in plan:
        review = plan["architectureReview"]
        if not isinstance(review, dict):
            raise GateError("architectureReview должен быть объектом")
        if "status" in review:
            require_choice(review["status"], {"pending", "completed"}, "architectureReview.status")
        if "verdict" in review:
            require_choice(review["verdict"], {"pending", "approved", "changes-requested", "rejected"}, "architectureReview.verdict")
        for key in ("author", "reviewer", "signature"):
            if key in review:
                require_text(review[key], f"architectureReview.{key}")
        if "independent" in review and not isinstance(review["independent"], bool):
            raise GateError("architectureReview.independent должен быть boolean")
        if "date" in review:
            validate_date(review["date"], "architectureReview.date")
        if "artifact" in review:
            normalize_relative_path(review["artifact"], "architectureReview.artifact")
        if "scope" in review:
            scope = review["scope"]
            if not isinstance(scope, list) or not scope or not all(isinstance(x, str) and x.strip() for x in scope):
                raise GateError("architectureReview.scope должен быть непустым списком строк")


def load_plan(path: Path, schema_path: Path | None = None) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise GateError(f"план не является корректным UTF-8: {exc}") from exc
    except OSError as exc:
        raise GateError(f"план не читается: {exc}") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"некорректное число {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"дублирующийся ключ {key}")
            result[key] = value
        return result

    try:
        plan = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise GateError(f"план не является корректным JSON: {exc}") from exc
    if not isinstance(plan, dict):
        raise GateError("план должен быть объектом JSON")
    if schema_path is not None:
        try:
            schema_raw = schema_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise GateError(f"schema не читается: {exc}") from exc
        try:
            schema = json.loads(schema_raw)
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise GateError(f"schema не является корректным JSON: {exc}") from exc
        validate_schema(plan, schema)
    validate_plan(plan)
    return plan


def clone_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(plan, ensure_ascii=False))
