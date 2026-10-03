#!/usr/bin/env python3
"""Rewrite CUDA launch syntax and shared/constant declarations so icpx can
compile Strata's .cu files as SYCL C++. Kernel bodies stay as written; the
XPU compatibility headers supply threadIdx, shuffles and the runtime.

Comments and string literals are left alone (several comments contain <<<>>>).
"""
from __future__ import annotations

import pathlib
import sys


def skip_comment_or_string(text: str, i: int) -> int | None:
    n = len(text)
    if text.startswith("//", i):
        j = text.find("\n", i)
        return n if j < 0 else j
    if text.startswith("/*", i):
        j = text.find("*/", i + 2)
        return n if j < 0 else j + 2
    if text[i] in "\"'":
        q = text[i]
        j = i + 1
        while j < n:
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == q:
                return j + 1
            j += 1
        return n
    return None


def match_span(text: str, i: int, open_ch: str, close_ch: str) -> int:
    """i points at open_ch. Return index just past the matching close, skipping comments."""
    depth = 0
    n = len(text)
    while i < n:
        skipped = skip_comment_or_string(text, i)
        if skipped is not None:
            i = skipped
            continue
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError(f"unbalanced {open_ch}{close_ch} at scan")


def callee_start(text: str, launch_at: int) -> int:
    i = launch_at - 1
    while i >= 0 and text[i] in " \t\r\n":
        i -= 1
    if i >= 0 and text[i] == ">":
        depth = 0
        while i >= 0:
            skipped = None
            # comments are rare walking backward; handle strings crudely
            if text[i] == ">":
                depth += 1
            elif text[i] == "<":
                depth -= 1
                if depth == 0:
                    i -= 1
                    break
            i -= 1
        while i >= 0 and text[i] in " \t\r\n":
            i -= 1
    while i >= 0 and (text[i].isalnum() or text[i] in "_:"):
        i -= 1
    return i + 1


def split_config(config: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    i = 0
    while i < len(config):
        skipped = skip_comment_or_string(config, i)
        if skipped is not None:
            buf.append(config[i:skipped])
            i = skipped
            continue
        c = config[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(c)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def rewrite_launches(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        skipped = skip_comment_or_string(text, i)
        if skipped is not None:
            out.append(text[i:skipped])
            i = skipped
            continue
        if text.startswith("<<<", i):
            start = callee_start(text, i)
            # drop the callee already copied into out
            # out currently includes text up to i, which includes the callee.
            # Recompute: we appended everything before i, so the callee is the
            # suffix of out that corresponds to text[start:i].
            already = "".join(out)
            # safer: rebuild from text[0:start] + replacement, but out is the
            # rewritten prefix of text[0:i]. The callee in text[start:i] was
            # copied verbatim (no launches inside a callee). Trim it.
            callee = text[start:i]
            prefix_len = len(already) - len(callee)
            if prefix_len < 0 or already[prefix_len:] != callee:
                raise ValueError(f"callee trim failed near {text[max(0,i-40):i+20]!r}")
            out = [already[:prefix_len]]
            cfg_end = match_span(text, i + 2, "<", ">")  # '<<<' so i+2 is the third '<'? 
            # '<<<' is three chars. match_span expects to start ON an open '<'.
            # We want to match the outer <<< >>> which is not a single char pair.
            # Do it manually.
            raise RuntimeError("unreachable")
        out.append(text[i])
        i += 1
    return "".join(out)


def find_launch_end(text: str, i: int) -> tuple[int, str, str]:
    """i at '<<<'. Return (index past argument ')', config, args-without-parens)."""
    if not text.startswith("<<<", i):
        raise ValueError("not a launch")
    j = i + 3
    n = len(text)
    cfg_start = j
    paren = 0
    while j < n:
        skipped = skip_comment_or_string(text, j)
        if skipped is not None:
            j = skipped
            continue
        c = text[j]
        if c in "([{":
            paren += 1
        elif c in ")]}":
            paren -= 1
        elif paren == 0 and text.startswith(">>>", j):
            j += 3
            break
        j += 1
    else:
        raise ValueError("unclosed <<<")
    config = text[cfg_start:j - 3]
    # j is just past >>>
    while j < n and text[j] in " \t\r\n":
        j += 1
    if j >= n or text[j] != "(":
        raise ValueError("launch missing call paren")
    arg_end = match_span(text, j, "(", ")")
    args = text[j + 1:arg_end - 1]
    return arg_end, config, args


def rewrite_launches(text: str) -> str:  # noqa: F811
    out: list[str] = []
    i = 0
    n = len(text)
    count = 0
    while i < n:
        skipped = skip_comment_or_string(text, i)
        if skipped is not None:
            out.append(text[i:skipped])
            i = skipped
            continue
        if text.startswith("<<<", i):
            start = callee_start(text, i)
            already = "".join(out)
            callee = text[start:i]
            if not already.endswith(callee):
                raise ValueError(f"callee mismatch near byte {i}: {callee!r}")
            out = [already[: len(already) - len(callee)]]
            end, config, args = find_launch_end(text, i)
            parts = split_config(config)
            if not parts:
                raise ValueError(f"empty launch config near byte {i}")
            grid = parts[0]
            block = parts[1] if len(parts) > 1 else "1"
            smem = parts[2] if len(parts) > 2 else "0"
            stream = parts[3] if len(parts) > 3 else "0"
            arg_list = args.strip()
            parts = split_config(arg_list) if arg_list else []
            # Hoist arguments on the host so the device lambda does not capture `this`
            # (a member launch such as poison_kernel(base_ + ...) otherwise does).
            hoists = " ".join(f"auto strata_xpu_a{i} = {part};" for i, part in enumerate(parts))
            names = ", ".join(f"strata_xpu_a{i}" for i in range(len(parts)))
            repl = (
                f"[&]() {{ {hoists} ::strata::xpu::launch({grid}, {block}, {smem}, {stream}, \"{callee}\", "
                f"[=]() {{ {callee}({names}); }}); }}()"
            )
            out.append(repl)
            i = end
            count += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out), count


def split_decls(rest: str) -> list[str]:
    decls = []
    buf = []
    depth = 0
    for ch in rest:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            decls.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        decls.append(tail)
    return decls


def emit_smem(indent: str, decl: str, ty: str) -> tuple[str, str]:
    if "[" in decl:
        name = decl.split("[", 1)[0].split()[-1]
        dims = decl[decl.find("["):]
        prefix = decl[: decl.rfind(name)].strip()
        if prefix:
            ty = prefix
        shaped = f"{ty}{dims}"
    else:
        parts = decl.split()
        name = parts[-1]
        if len(parts) > 1:
            ty = " ".join(parts[:-1])
        shaped = ty
    alias = f"{name}_smem_t"
    code = (
        f"{indent}using {alias} = {shaped};\n"
        f"{indent}auto& {name} = *sycl::ext::oneapi::group_local_memory_for_overwrite<"
        f"{alias}>(sycl::ext::oneapi::this_work_item::get_work_group<3>());\n"
    )
    return code, ty


def rewrite_shared(text: str) -> tuple[str, int]:
    """Single-line __shared__ / extern __shared__ declarations, including comma lists."""
    lines = text.splitlines(keepends=True)
    n = 0
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("__shared__") or stripped.startswith("extern __shared__"):
            code = stripped.split("//", 1)[0]
            if ";" not in code:
                out.append(line)
                continue
            indent = line[: len(line) - len(stripped)]
            nl = "\n" if line.endswith("\n") else ""
            body = code.strip().rstrip(";").strip()
            comment = ""
            if "//" in stripped:
                comment = "  //" + stripped.split("//", 1)[1].rstrip("\n")
            if body.startswith("extern __shared__"):
                rest = body[len("extern __shared__"):].strip()
                rest = rest.replace("__align__(16)", "").replace("__align__(8)", "").strip()
                if not rest.endswith("[]"):
                    out.append(line)
                    continue
                decl = rest[:-2].strip()
                name = decl.split()[-1]
                ty = decl[: decl.rfind(name)].strip()
                out.append(f"{indent}{ty}* {name} = ::strata::xpu::dynamic_smem<{ty}>();{comment}{nl}")
                n += 1
                continue
            rest = body[len("__shared__"):].strip()
            rest = rest.replace("__align__(16)", "").replace("__align__(8)", "").strip()
            ty = ""
            for decl in split_decls(rest):
                emitted, ty = emit_smem(indent, decl, ty)
                out.append(emitted)
                n += 1
            if comment:
                out.append(f"{indent}{comment}{nl}")
            continue
        out.append(line)
    return "".join(out), n


def rewrite_constant(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    n = 0
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("__constant__"):
            block = line
            while ";" not in block.split("//", 1)[0] and i + 1 < len(lines):
                i += 1
                block += lines[i]
            indent = line[: len(line) - len(stripped)]
            nl = "\n" if block.endswith("\n") else ""
            body = block.split("//", 1)[0].strip().rstrip(";").strip()
            # drop newlines inside the declaration so the split below is simple
            body = " ".join(body.split())
            rest = body[len("__constant__"):].strip()
            if "=" in rest:
                decl, init = rest.split("=", 1)
                decl = decl.strip()
                init = init.strip()
                name = decl.split("[", 1)[0].split()[-1]
                ty = decl[: decl.rfind(name)].strip()
                dims = decl[decl.find("["):] if "[" in decl else ""
                fn = f"strata_xpu_const_{name}"
                out.append(
                    f"{indent}constexpr {ty} {fn}{dims} = {init};\n"
                    f"{indent}#define {name} {fn}{nl}"
                )
            else:
                decl = rest
                name = decl.split("[", 1)[0].split()[-1]
                ty = decl[: decl.rfind(name)].strip()
                dims = decl[decl.find("["):] if "[" in decl else ""
                fn = f"strata_xpu_sym_{name}"
                out.append(
                    f"{indent}inline {ty} (&{fn}()){dims} {{\n"
                    f"{indent}    using Arr = {ty}{dims};\n"
                    f"{indent}    static Arr* p = nullptr;\n"
                    f"{indent}    if (!p) p = static_cast<Arr*>(::strata::xpu::shared_symbol(sizeof(Arr)));\n"
                    f"{indent}    return *p;\n"
                    f"{indent}}}\n"
                    f"{indent}#define {name} ({fn}()){nl}"
                )
            n += 1
            i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out), n


def replace_call(text: str, name: str, repl: str) -> str:
    """Replace a call name (including the '(') only at an identifier boundary."""
    out: list[str] = []
    i = 0
    n = len(text)
    L = len(name)
    while i < n:
        skipped = skip_comment_or_string(text, i)
        if skipped is not None:
            out.append(text[i:skipped])
            i = skipped
            continue
        if text.startswith(name, i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            out.append(repl)
            i += L
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def replace_builtin(text: str, name: str, repl: str) -> str:
    """Replace a CUDA builtin identifier, but not a member access (.name)."""
    out: list[str] = []
    i = 0
    n = len(text)
    L = len(name)
    while i < n:
        skipped = skip_comment_or_string(text, i)
        if skipped is not None:
            out.append(text[i:skipped])
            i = skipped
            continue
        if (text.startswith(name, i)
                and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"))
                and (i + L == n or not (text[i + L].isalnum() or text[i + L] == "_"))):
            j = i - 1
            while j >= 0 and text[j] in " \t":
                j -= 1
            if j >= 0 and (text[j] == "." or (text[j] == ">" and j > 0 and text[j - 1] == "-")):
                out.append(name)
            else:
                out.append(repl)
            i += L
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def rewrite(text: str) -> tuple[str, dict]:
    text, n_const = rewrite_constant(text)
    text, n_shared = rewrite_shared(text)
    text, n_launch = rewrite_launches(text)
    for name, repl in (
        ("threadIdx", "(::strata::xpu::thread_idx())"),
        ("blockIdx", "(::strata::xpu::block_idx())"),
        ("blockDim", "(::strata::xpu::block_dim())"),
        ("gridDim", "(::strata::xpu::grid_dim())"),
        ("warpSize", "32"),
    ):
        text = replace_builtin(text, name, repl)
    # libc declares these; a macro of the same name breaks <math.h>. Substitute calls only,
    # and only as whole identifiers (ldexpf/frexpf contain "expf").
    for old, new in (
        ("__expf(", "sycl::exp("),
        ("expf(", "sycl::exp("),
        ("rsqrtf(", "sycl::rsqrt("),
        ("sqrtf(", "sycl::sqrt("),
        ("fabsf(", "sycl::fabs("),
        ("fmaxf(", "sycl::fmax("),
        ("fminf(", "sycl::fmin("),
        ("fmaf(", "sycl::fma("),
        ("roundf(", "sycl::round("),
        ("log1pf(", "sycl::log1p("),
        ("isnan(", "sycl::isnan("),
        ("__isnanf(", "sycl::isnan("),
        ("__isinf(", "sycl::isinf("),
    ):
        text = replace_call(text, old, new)
    return text, {"const": n_const, "shared": n_shared, "launch": n_launch}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: rewrite_cuda.py INPUT OUTPUT", file=sys.stderr)
        return 2
    src = pathlib.Path(sys.argv[1]).read_text(errors="replace")
    out, stats = rewrite(src)
    dest = pathlib.Path(sys.argv[2])
    dest.parent.mkdir(parents=True, exist_ok=True)
    banner = (
        "// Rewritten for the Intel XPU backend by tools/xpu/rewrite_cuda.py.\n"
        f"// launches={stats['launch']} shared={stats['shared']} constant={stats['const']}\n"
    )
    dest.write_text(banner + out)
    print(f"{sys.argv[1]} -> {sys.argv[2]} {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
