from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def unquote_git_path(value: str) -> str:
    if not value.startswith('"'):
        return value
    data = value.encode("utf-8")
    result = bytearray()
    index = 1
    escapes = {
        ord("a"): 7,
        ord("b"): 8,
        ord("t"): 9,
        ord("n"): 10,
        ord("v"): 11,
        ord("f"): 12,
        ord("r"): 13,
        ord('"'): ord('"'),
        ord("\\"): ord("\\"),
    }
    while index < len(data):
        byte = data[index]
        index += 1
        if byte == ord('"'):
            break
        if byte != ord("\\"):
            result.append(byte)
            continue
        if index >= len(data):
            result.append(ord("\\"))
            break
        byte = data[index]
        index += 1
        if ord("0") <= byte <= ord("7"):
            digits = bytes([byte])
            while index < len(data) and len(digits) < 3 and ord("0") <= data[index] <= ord("7"):
                digits += bytes([data[index]])
                index += 1
            result.append(int(digits, 8))
        else:
            result.append(escapes.get(byte, byte))
    return result.decode("utf-8", errors="replace")


def code_without_noise(line: str) -> str:
    value = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    value = re.sub(r"'(?:\\.|[^'\\])*'", "''", value)
    value = re.sub(r"/\*.*?\*/", "", value)
    value = re.sub(r"//.*$", "", value)
    return value


def brace_delta(line: str) -> int:
    value = code_without_noise(line)
    return value.count("{") - value.count("}")


def added_code_lines(diff_text: str) -> dict[str, list[tuple[str, int]]]:
    result: dict[str, list[tuple[str, int]]] = {}
    current: str | None = None
    depth = 0
    for raw_line in diff_text.splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("diff --git "):
            current = None
            depth = 0
            continue
        if line.startswith("+++ "):
            value = line[4:]
            current = unquote_git_path(value)
            if current.startswith("b/"):
                current = current[2:]
            result.setdefault(current, [])
            continue
        if line.startswith("+") and not line.startswith("+++") and current is not None:
            result[current].append((line[1:], depth))
            depth = max(0, depth + brace_delta(line[1:]))
        elif line.startswith(" ") and current is not None:
            depth = max(0, depth + brace_delta(line[1:]))
    return result


def added_lines_by_file(diff_text: str) -> dict[str, list[str]]:
    return {path: [line for line, _depth in entries] for path, entries in added_code_lines(diff_text).items()}


def duplicate_windows(lines: list[str], window: int = 8) -> int:
    normalized: list[str] = []
    boilerplate = {"{", "}", "};", "});", "break;", "continue;", "return;", "default:", "else", "else {"}
    for line in lines:
        value = re.sub(r"\s+", " ", line.strip())
        if not value or value in boilerplate:
            continue
        if value.startswith(("#", "//", "/*", "*", "case ", "default:")):
            continue
        normalized.append(value)
    seen: dict[tuple[str, ...], int] = {}
    duplicates = 0
    for index in range(max(0, len(normalized) - window + 1)):
        block = tuple(normalized[index:index + window])
        previous = seen.get(block)
        if previous is not None and index >= previous + window:
            duplicates += 1
        else:
            seen[block] = index
    return duplicates


def structural_scan(diff_text: str) -> dict[str, Any]:
    by_file = added_code_lines(diff_text)
    added = [line for entries in by_file.values() for line, _depth in entries]
    ifdefs = sum(bool(re.match(r"^\s*#\s*(if|ifdef|ifndef)\b", line)) for line in added)
    globals_found: list[str] = []
    for entries in by_file.values():
        for line, depth in entries:
            value = line.strip()
            if depth != 0 or not value or value.startswith(("#", "//", "/*", "*")):
                continue
            if re.match(r"^(?:namespace|class|struct|union|enum|typedef|using|template|extern\s+\")\b", value):
                continue
            typed_global = re.match(
                r"^(?:(?:static|thread_local|extern|const|volatile|inline|constexpr|register)\s+)*"
                r"(?:std::)?[A-Za-z_]\w*(?:::\w+)*(?:<[^;{}]+>)?[\s*&]+"
                r"[A-Za-z_]\w*(?:\s*\[[^\]]*\])?\s*(?:=[^;]+)?;$",
                value,
            )
            resource_global = re.match(
                r"^(?:(?:static|thread_local|extern|const|volatile)\s+)*"
                r"(?:std::)?(?:mutex|atomic|recursive_mutex|shared_mutex|condition_variable|cudaEvent_t|cudaStream_t)\b.*;$",
                value,
            )
            if typed_global or resource_global:
                globals_found.append(line)
    offsets = [
        line for line in added
        if re.search(r"\boffsetof\s*\(|\breinterpret_cast\s*<[^>]*(?:char|byte|uint8_t|int8_t)", line)
        or re.search(
            r"\b(?:base|buffer|ptr|pointer|data|address|kv|cache)\w*\s*\+\s*(?:0x[0-9A-Fa-f]+|\d+)\s*(?:;|\)|,|$)",
            line,
        )
    ]
    copies = sum(duplicate_windows([line for line, _depth in entries]) for entries in by_file.values())
    return {
        "conflictMarkers": [],
        "conflictMarkerFragments": [],
        "ifdefAdded": ifdefs,
        "globalState": globals_found[:20],
        "offsetCoupling": offsets[:20],
        "copyPasteWindows": copies,
    }


def scan_candidate_conflicts(repo: Path, candidate: str) -> tuple[list[str], list[str]]:
    from .gitops import git_stdout, run_git

    pattern = r"^(<{7,}|={7,}|>{7,}|\|{7,})([[:space:]]|$)"
    result = run_git(
        repo,
        ["grep", "-z", "-I", "-l", "-E", "-e", pattern, candidate, "--"],
        timeout=120,
    )
    if result.returncode == 1:
        return [], []
    if result.returncode != 0:
        raise GateError((result.stderr or result.stdout).strip()[:2000])
    conflicts: list[str] = []
    fragments: list[str] = []
    prefix = candidate + ":"
    for entry in result.stdout.split("\0"):
        if not entry:
            continue
        path = entry[len(prefix):] if entry.startswith(prefix) else entry
        text = git_stdout(repo, ["cat-file", "blob", f"{candidate}:{path}"], timeout=120)
        markers: list[tuple[int, str, str]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            match = re.match(r"^([<>=|])\1{6,}(?:[ \t].*)?$", line)
            if match:
                markers.append((line_number, match.group(1), line))
        used: set[int] = set()
        for start, (_line, kind, _text) in enumerate(markers):
            if kind != "<" or start in used:
                continue
            separator = next((i for i in range(start + 1, len(markers)) if markers[i][1] == "="), None)
            closer = next((i for i in range(separator + 1, len(markers)) if markers[i][1] == ">"), None) if separator is not None else None
            if closer is None:
                continue
            used.update(range(start, closer + 1))
            for line_number, _kind, text_line in markers[start:closer + 1]:
                conflicts.append(f"{path}:{line_number}:{text_line}")
        for index, (line_number, kind, text_line) in enumerate(markers):
            if index not in used and kind != "=":
                fragments.append(f"{path}:{line_number}:{text_line}")
    return conflicts[:100], fragments[:100]
