"""One JSON request on stdin, one JSON response on stdout. Runs only in the container.

No HTTP listener, host files, network, or shell escape. The host serializes docker exec calls.
"""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

LIMIT = 16384
SVG_LIMIT = 2 * 1024 * 1024
PREAMBLE = r"""\documentclass[border=0pt]{standalone}
\usepackage{amsmath,amssymb,mathtools,unicode-math}
\setmathfont{Latin Modern Math}
\pagestyle{empty}
\begin{document}
"""


def limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    os.setsid()


def command(args, directory, deadline):
    proc = subprocess.Popen(args, cwd=directory, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=limits)
    try:
        proc.wait(timeout=max(.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
        proc.wait()
        raise
    return proc.returncode == 0


def render(req):
    if not isinstance(req, dict):
        return {"ok": False, "reason": "invalid"}
    source = req.get("source")
    display = req.get("display")
    if not isinstance(source, str) or not isinstance(display, bool):
        return {"ok": False, "reason": "invalid"}
    if len(source.encode()) > LIMIT:
        return {"ok": False, "reason": "limit"}
    deadline = time.monotonic() + 5
    with tempfile.TemporaryDirectory(prefix="equation-", dir="/tmp") as tmp:
        directory = Path(tmp)
        # A real TeX compiler; packages are fixed by the image, not generated source.
        math = "$" + (r"\displaystyle " if display else "") + source + "$"
        (directory / "equation.tex").write_text(PREAMBLE + math + "\n\\end{document}\n")
        env_before = dict(os.environ)
        os.environ.update({"openin_any": "p", "openout_any": "p", "shell_escape": "f",
                           "TEXMFOUTPUT": tmp, "HOME": tmp})
        try:
            # XeTeX supports Unicode math without exposing a Lua interpreter to model output.
            if not command(["xelatex", "--no-shell-escape", "--interaction=batchmode", "--halt-on-error",
                            "--jobname=equation", "equation.tex"], tmp, deadline):
                return {"ok": False, "reason": "invalid"}
            if not command(["dvisvgm", "--pdf", "--no-fonts", "--bbox=min", "--page=1",
                            "--output=equation.svg", "equation.pdf"], tmp, deadline):
                return {"ok": False, "reason": "invalid"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "limit"}
        finally:
            os.environ.clear(); os.environ.update(env_before)
        path = directory / "equation.svg"
        if not path.is_file() or path.stat().st_size > SVG_LIMIT:
            return {"ok": False, "reason": "limit"}
        root = ET.fromstring(path.read_text())
        # Keep only vector geometry. TeX specials cannot introduce scripts or external resources.
        allowed = {"svg", "g", "defs", "path", "use", "rect", "circle", "ellipse", "line", "polyline", "polygon", "clipPath"}
        attributes = {"id", "d", "transform", "fill", "stroke", "stroke-width", "fill-rule", "clip-rule",
                      "clip-path", "opacity", "x", "y", "x1", "y1", "x2", "y2", "width", "height", "viewBox",
                      "cx", "cy", "r", "rx", "ry", "points", "version", "href"}
        for parent in list(root.iter()):
            for child in list(parent):
                if child.tag.split("}")[-1] not in allowed:
                    parent.remove(child)
            for key, value in list(parent.attrib.items()):
                local = key.split("}")[-1]
                if local not in attributes or local == "href" and not value.startswith("#") or "url(" in value and not value.startswith("url(#"):
                    del parent.attrib[key]
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
        # SVG dimensions are TeX points. Give the browser an em size at 10pt TeX base.
        width, height = (float(x.rstrip("pt")) / 10 for x in (root.get("width", "0"), root.get("height", "0")))
        if not 0 < width <= 200 or not 0 < height <= 200:
            return {"ok": False, "reason": "limit"}
        return {"ok": True, "svg": ET.tostring(root, encoding="unicode"), "width": width, "height": height}


if __name__ == "__main__":
    try:
        data = sys.stdin.buffer.read(65537)
        result = render(json.loads(data)) if len(data) <= 65536 else {"ok": False, "reason": "limit"}
    except (ValueError, OSError, KeyError, TypeError, ET.ParseError):
        result = {"ok": False, "reason": "invalid"}
    print(json.dumps(result))
