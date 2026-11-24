from __future__ import annotations

import json
import sys
from typing import Any

from .evidence import evidence_commands


def format_command(command: Any) -> str:
    if isinstance(command, list):
        return json.dumps(command, ensure_ascii=False)
    return str(command)


def human_report(report: dict[str, Any]) -> str:
    labels = {
        "pass": "PASS",
        "blocked": "BLOCKED",
        "review": "REVIEW",
        "invalid": "INVALID",
    }
    lines = [f"ИТОГ: {labels.get(report['result'], report['result'])}"]
    if report.get("error"):
        lines.append(f"Ошибка: {report['error']}")
    if "repo" in report:
        lines.append(f"Репозиторий: {report['repo']}")
        lines.append(f"Решение: {report.get('decision')} — {report.get('rationale')}")
        facts = report["facts"]
        lines.append(
            "Refs: "
            f"ours={facts['ours'][:12]} upstream={facts['upstream'][:12]} "
            f"candidate={facts['candidate'][:12]} base={facts['mergeBase'][:12]} HEAD={facts['head'][:12]}"
        )
        lines.append(
            f"Ahead/behind: ours-only={facts['aheadBehind']['oursOnly']}, "
            f"upstream-only={facts['aheadBehind']['upstreamOnly']}"
        )
        lines.append(
            f"Diff: файлов={facts['changedFiles']}, +{facts['addedLines']}/-{facts['deletedLines']}"
        )
        lines.append("Матрица решений:")
        for item in report.get("matrix", []):
            lines.append(
                f"  - {item['id']}: {item['decision']} ({item['upstreamStatus']}, "
                f"rework={item['reworkEffect']}) — {item['reason']}"
            )
        for contract_key, title in (("hybridContract", "Hybrid contract"), ("deltaContract", "Delta contract")):
            contract = report.get(contract_key)
            if contract:
                lines.append(f"{title}:")
                for key, value in contract.items():
                    lines.append(f"  - {key}: {format_command(value)}")
        lines.append("Evidence:")
        for key, item in report.get("evidence", {}).items():
            artifacts = ([item["artifact"]] if "artifact" in item else []) + item.get("artifacts", [])
            commands = evidence_commands(item)
            lines.append(
                f"  - {key}: {item['status']}; revision={item.get('effectiveRevision', item.get('revision', 'не указан'))}; "
                f"artifacts={format_command(artifacts)}; commands={format_command(commands)}"
            )
            if item.get("reason"):
                lines.append(f"    причина: {item['reason']}")
            if item.get("details"):
                lines.append(f"    подробности: {item['details']}")
        review = report.get("architectureReview", {})
        lines.append(
            f"Architecture review: reviewer={review.get('reviewer', 'не указан')}, "
            f"independent={review.get('independent', False)}, verdict={review.get('verdict', 'legacy')}"
        )
        lines.append(f"Инварианты изменены: {report.get('invariantsChanged', [])}")
        lines.append(f"Thresholds: {report.get('thresholds', {})}")
    if report.get("blockers"):
        lines.append("Блокеры:")
        lines.extend(f"  - {item}" for item in report["blockers"])
    if report.get("warnings"):
        lines.append("Предупреждения:")
        for item in report["warnings"]:
            suffix = " (подтверждено)" if item["acknowledged"] else ""
            lines.append(f"  - {item['code']}{suffix}")
            if item.get("reason"):
                lines.append(f"    причина: {item['reason']}; reviewer={item.get('reviewer')}")
            if item.get("rejectedReason"):
                lines.append(f"    acknowledgment отклонён: {item['rejectedReason']}")
    if report.get("notRun"):
        lines.append("Не запущено:")
        for name in report["notRun"]:
            lines.append(f"  - {name}: {report.get('notRunReasons', {}).get(name, 'не указано')}")
    if report.get("checks"):
        lines.append("Проверки:")
        for item in report["checks"]:
            status = "ok" if item.get("ok") else "FAIL"
            lines.append(f"  - {item['name']}: {status}; command={format_command(item.get('command'))}")
    return "\n".join(lines)


def emit(report: dict[str, Any], output_format: str) -> None:
    if output_format in {"json", "both"}:
        sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if output_format in {"human", "both"}:
        stream = sys.stderr if output_format == "both" else sys.stdout
        stream.write(human_report(report) + "\n")
