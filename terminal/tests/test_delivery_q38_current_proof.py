"""Local actual-byte delivery regressions; no model execution or build work."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "_docs/stable-060-20261116"
RETAINED = Path("H:/Strata/releases/stable-060-20261116-codex")
CONTROL_PYTHON = ROOT / "apps/desktop/src-tauri/resources/control/python/python.exe"


def acceptance() -> Any:
    """Load the actual release adapter without adding it to global import paths."""
    spec = importlib.util.spec_from_file_location("current_q38_acceptance", HERE / "acceptance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not (RETAINED / "native-input-r3/receipt.json").is_file(), reason="Local frozen evidence required"
)
@pytest.mark.parametrize("change", ("none", "repinned-source"))
def test_retained_actual_three_binary_binding_uses_real_q38_verifier(tmp_path: Path, change: str) -> None:
    a = acceptance()
    receipt = RETAINED / "native-input-r3/receipt.json"
    built = a.release.load(receipt)
    binding = a.release.load(receipt.parent / "native-binding.json")
    component = a.release.load(RETAINED / "component-r3/receipt.json")
    signed = Path(component["signedComponent"])
    legacy = {name: a.release.sha(signed / name) for name in ("strata.exe", "strata-q36.exe")}
    if change == "repinned-source":
        # Re-pin both copies and their hashes: the real semantic source verifier
        # must reject even internally consistent, newly bound malformed provenance.
        receipt = tmp_path / "native/receipt.json"
        unsigned = receipt.parent / "unsigned" / a.release.NAME
        changed = tmp_path / "signed"
        unsigned.mkdir(parents=True)
        changed.mkdir()
        for folder in (unsigned, changed):
            shutil.copyfile(signed / "strata-q38.exe", folder / "strata-q38.exe")
        shutil.copyfile(signed / "engine-component.json", changed / "engine-component.json")
        record = a.release.load(signed / "q38-build.json")
        record["sources"][next(iter(record["sources"]))] = "f" * 64
        for folder in (unsigned, changed):
            (folder / "q38-build.json").write_text(json.dumps(record), encoding="utf-8")
        binding["q38BuildSHA256"] = a.release.sha(changed / "q38-build.json")
        built["files"]["q38-build.json"] = binding["q38BuildSHA256"]
        built["nativeComponent"] = str(unsigned)
        with pytest.raises(RuntimeError, match="Q38 compiled source bytes changed"):
            a.q38_identity(changed, receipt, built, binding, legacy)
        return
    identity = a.q38_identity(signed, receipt, built, binding, legacy)
    assert set(binding["binaries"]) == {*legacy, "strata-q38.exe"}
    assert identity["compiledSourceCommit"] == a.native.Q38.COMMIT
    assert identity["executable"]["sha256"] == a.native.Q38.EXECUTABLE_SHA256
    historical = a.release.load(
        Path(
            a.release.load(RETAINED / "acceptance-final-native-r1/pending.json")["inputs"]["gpu_binding"][
                "path"
            ]
        )
    )
    assert {Path(name).name for name in historical["before"]} == set(legacy)


@pytest.mark.skipif(not CONTROL_PYTHON.is_file(), reason="Local bundled Python required")
def test_actual_release_runtime_probe_never_writes_bytecode(tmp_path: Path) -> None:
    a = acceptance()
    source = (ROOT / "apps/desktop/scripts/verify-release.mjs").read_text(encoding="utf-8")
    probe = next(line for line in source.splitlines() if line.startswith("const probe=spawnSync(py,"))
    # Evaluate the production argv with a capture-only spawn; do not run the release script.
    capture = "const py='';const spawnSync=(_py,args)=>console.log(JSON.stringify(args));" + probe
    result = subprocess.run(
        [str(a.release.NODE), "--input-type=module", "-e", capture],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    argv = json.loads(result.stdout)
    (tmp_path / "fresh_bytecode_probe.py").write_text("VALUE = 7\n", encoding="utf-8")
    argv[argv.index("-c") + 1] = (
        "import sys;sys.path.insert(0," + repr(str(tmp_path)) + ");"
        "import fresh_bytecode_probe;assert fresh_bytecode_probe.VALUE == 7;"
        + argv[argv.index("-c") + 1]
        + ";assert sys.dont_write_bytecode"
    )
    subprocess.run([str(CONTROL_PYTHON), *argv], capture_output=True, text=True, check=True, timeout=30)
    assert not list(tmp_path.rglob("*.pyc"))
    assert not list(tmp_path.rglob("__pycache__"))
