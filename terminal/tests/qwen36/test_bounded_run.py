"""CPU fake processes only; no native engine, model, GPU or service queries."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.qwen35 import q36_bounded_run as runner

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Job integration")


@pytest.fixture(autouse=True)
def fixture_volume(tmp_path, monkeypatch):
    # These CPU processes exercise the real Windows Job guard in pytest's
    # private scratch volume. Production keeps the explicit H: admission rule.
    monkeypatch.setattr(runner, "OUTPUT_VOLUME", tmp_path.resolve().drive.upper())
    from perf131k import windows_job

    monkeypatch.setattr(
        windows_job, "BOUNDED_STDIN_VOLUME", tmp_path.resolve().drive.upper()
    )


def test_owned_stdin_volume_policy_is_still_enforced(tmp_path, monkeypatch):
    from perf131k import windows_job

    monkeypatch.setattr(
        windows_job,
        "BOUNDED_STDIN_VOLUME",
        "Z:" if tmp_path.drive.upper() != "Z:" else "Y:",
    )
    with pytest.raises(
        ValueError, match="bounded stdin file inside owned H run directory required"
    ):
        runner.run_bounded(spec_for(tmp_path))


def test_production_volume_policy_is_still_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "OUTPUT_VOLUME", "Z:" if tmp_path.drive.upper() != "Z:" else "Y:"
    )
    with pytest.raises(ValueError, match="existing_guard_requires_H_volume_output"):
        runner.validate_spec(spec_for(tmp_path))


def spec_for(tmp_path, body=None):
    model = tmp_path / "fake-model.bin"
    model.write_bytes(b"CPU fixture; not model weights")
    fixture = tmp_path / "prompts.json"
    fixture.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "prompt_id": "p0",
                        "input_ids": [1, 2, 3],
                        "max_tokens": 2,
                        "output_reservation": 2,
                        "mtp_reservation": 0,
                    }
                ]
            }
        )
    )
    if body is None:
        body = """import sys
print('Q36_EFFECTIVE schema=1 generation=fake context=4096 prefill=256 '
      'parallel=1 kv_k=f32 kv_v=f32 mtp=0', file=sys.stderr, flush=True)
print('READY', flush=True)
for line in sys.stdin:
    if line.startswith('STOP'): break
    print('PP 1 2', flush=True)
    print('T 7', flush=True)
    print('T 8', flush=True)
    print('DONE', flush=True)
"""
    return {
        "schema_version": 1,
        "run_id": "fake",
        "engine_path": sys.executable,
        "engine_sha256": runner.digest(Path(sys.executable)),
        "argv": ["-u", "-c", body],
        "model_path": str(model),
        "model_sha256": runner.digest(model),
        "prompt_fixture_path": str(fixture),
        "prompt_fixture_sha256": runner.digest(fixture),
        "toolchain_sha256": "a" * 64,
        "ready_timeout_s": 2,
        "total_timeout_s": 5,
        "min_available_ram_bytes": 3 * runner.GIB,
        "output_root": str(tmp_path),
        "output_dir": str(tmp_path / "run"),
        "job_commit_bytes": 256 * 1024**2,
        "requested_settings": dict(
            context=4096,
            prefill=256,
            parallel=1,
            kv_k="f32",
            kv_v="f32",
            mtp=0,
            router_top_k=8,
            expert_quantization="fixture",
        ),
        "experiment": {"dimensions": [], "max_relative_regression": 0.05},
    }


def test_complete_raw_timing_and_unknown_consumption(tmp_path):
    result = runner.run_bounded(spec_for(tmp_path))
    assert result["status"] == "PASS", result
    assert result["guard"]["contained_before_resume"]
    assert result["guard"]["termination_confirmed"]
    case = result["cases"][0]
    assert case["tokens"] == [7, 8]
    assert case["ttft_s"] == case["token_times_s"][0]
    assert case["wall_s"] >= case["token_times_s"][-1]
    assert case["consumed_input"] is None
    assert case["effective_max_tokens"] is None
    assert result["qualification"] == "NOT_RUN"
    assert (tmp_path / "run/stdout.log").read_text().startswith("READY")
    assert "Q36_EFFECTIVE" in (tmp_path / "run/stderr.log").read_text()
    assert (tmp_path / "run/memory.jsonl").stat().st_size > 0


@pytest.mark.parametrize("hotspot", [None, "0", "1"])
def test_opt_in_phase_artifacts_from_owned_process(tmp_path, monkeypatch, hotspot):
    if hotspot is None:
        monkeypatch.delenv("STRATA_Q36_HOTSPOT_PROFILE", raising=False)
    else:
        monkeypatch.setenv("STRATA_Q36_HOTSPOT_PROFILE", hotspot)
    spec = spec_for(tmp_path)
    spec["observation"] = dict(
        schema_version=1,
        family="qwen36",
        source_sha="a" * 40,
        binary_source_sha=None,
        environment_manifest=None,
        identity_files={},
    )
    result = runner.run_bounded(spec)
    assert result["status"] == "PASS", result
    assert result["instrumentation"]["mode"] == (
        "host_counters" if hotspot is None else "diagnostic_sync"
    )
    assert 0 <= result["engine_start_to_ready_s"] < result["ready_s"]
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert manifest["status"] == "PASS" and manifest["performance_eligible"] is False
    assert manifest["identities"]["engine_sha256"] == spec["engine_sha256"]
    assert (tmp_path / "run/report.md").exists()
    rows = [
        json.loads(line)
        for line in (tmp_path / "run/phase_metrics.jsonl").read_text().splitlines()
    ]
    assert any(
        row["phase"] == "request" and row["committed_tokens"] == 2 for row in rows
    )
    assert any(row["phase"] == "unload" and row["value"] is None for row in rows)
    native = result["native_process_memory"]
    assert native["samples"] > 0
    assert native["status"] in {"AVAILABLE", "PARTIAL"}
    assert native["sampled_max_private_bytes"] > 0
    assert native["sampled_max_working_set_bytes"] > 0
    samples = [
        json.loads(line)
        for line in (tmp_path / "run/native-process-memory.jsonl")
        .read_text()
        .splitlines()
    ]
    assert samples and all(
        row["engine_pid"] == result["owned_engine_pid"] for row in samples
    )
    assert all(row["engine_elapsed_s"] >= 0 for row in samples)


def test_preflight_hashing_does_not_spend_engine_ready_deadline(tmp_path, monkeypatch):
    spec = spec_for(tmp_path)
    spec.update(ready_timeout_s=1, total_timeout_s=8)
    original = runner.digest

    def delayed(path, check=lambda: None):
        if Path(path) == Path(spec["model_path"]):
            time.sleep(1.5)
        return original(path, check)

    monkeypatch.setattr(runner, "digest", delayed)
    result = runner.run_bounded(spec)
    assert result["status"] == "PASS", result
    assert result["preflight_s"] >= 1.5
    assert result["engine_start_to_ready_s"] < 1


def test_auto_prefill_is_recorded_without_overriding_engine_choice(tmp_path):
    spec = spec_for(tmp_path)
    spec["requested_settings"]["prefill"] = "auto"
    result = runner.run_bounded(spec)
    assert result["status"] == "PASS", result
    assert result["requested_settings"]["prefill"] == "auto"
    assert result["effective_settings"]["prefill"] == 256


def test_observation_artifacts_survive_preflight_pressure(tmp_path, monkeypatch):
    spec = spec_for(tmp_path)
    spec["observation"] = dict(
        schema_version=1,
        family="qwen36",
        source_sha="a" * 40,
        binary_source_sha=None,
        environment_manifest=None,
        identity_files={},
    )
    monkeypatch.setattr(
        runner,
        "available_memory",
        lambda: dict(
            available_physical_bytes=2 * runner.GIB,
            available_commit_bytes=8 * runner.GIB,
        ),
    )
    result = runner.run_bounded(spec)
    assert (
        result["status"] == "ABORTED" and result["reason"] == "physical_memory_pressure"
    )
    assert "owned_engine_pid" not in result
    for name in ("run_manifest.json", "phase_metrics.jsonl", "report.md"):
        assert (tmp_path / "run" / name).exists()


@pytest.mark.parametrize(
    "body,reason",
    [
        ("import time; time.sleep(30)", "ready_deadline"),
        ("import time; print('READY',flush=True); time.sleep(30)", "total_deadline"),
        ("print('READY',flush=True); print('ERR fixture',flush=True)", "engine_error"),
        ("print('READY',flush=True); print('DONE',flush=True)", "empty_output"),
        ("raise RuntimeError('synthetic')", "child_eof"),
    ],
)
def test_failures_are_bounded(tmp_path, body, reason):
    spec = spec_for(tmp_path, body)
    spec.update(ready_timeout_s=2, total_timeout_s=3)
    result = runner.run_bounded(spec)
    assert result["status"] in ("ABORTED", "FAILED")
    assert result["reason"] == reason, result
    assert result["guard"]["termination_confirmed"]
    assert result["elapsed_s"] < 7


def test_later_request_stalls_and_remaining_not_run(tmp_path):
    spec = spec_for(
        tmp_path,
        """import sys,time
print('READY',flush=True)
for i,line in enumerate(sys.stdin):
    if i: time.sleep(30)
    print('T 9',flush=True); print('DONE',flush=True)
""",
    )
    path = Path(spec["prompt_fixture_path"])
    cases = json.loads(path.read_text())["cases"]
    cases += [dict(cases[0], prompt_id="p1"), dict(cases[0], prompt_id="p2")]
    path.write_text(json.dumps({"cases": cases}))
    spec.update(
        prompt_fixture_sha256=runner.digest(path), total_timeout_s=3, ready_timeout_s=2
    )
    result = runner.run_bounded(spec)
    assert [c["status"] for c in result["cases"]] == ["PASS", "ABORTED", "NOT_RUN"]
    assert result["reason"] == "total_deadline"


@pytest.mark.parametrize(
    "key,value",
    [
        ("min_available_ram_bytes", 3 * runner.GIB - 1),
        ("min_commit_headroom_bytes", 4 * runner.GIB - 1),
        ("total_timeout_s", float("nan")),
        ("argv", "shell command"),
        ("engine_sha256", None),
    ],
)
def test_invalid_specs_never_start(tmp_path, key, value):
    spec = spec_for(tmp_path)
    spec[key] = value
    with pytest.raises(ValueError):
        runner.run_bounded(spec)
    assert not (tmp_path / "run").exists()


def test_capacity_and_overwrite(tmp_path):
    spec = spec_for(tmp_path)
    spec["requested_settings"]["context"] = 4
    with pytest.raises(ValueError, match="input_plus"):
        runner.run_bounded(spec)
    spec["requested_settings"]["context"] = 4096
    Path(spec["output_dir"]).mkdir()
    with pytest.raises(ValueError, match="new_contained"):
        runner.run_bounded(spec)


def test_resource_boundary_and_observer_failure(tmp_path, monkeypatch):
    spec = spec_for(tmp_path, "import time; time.sleep(30)")
    healthy = {
        "available_physical_bytes": 3 * runner.GIB,
        "available_commit_bytes": 4 * runner.GIB,
    }
    monkeypatch.setattr(runner, "available_memory", lambda: healthy)
    # Equality is admitted; a later low observation aborts without weakening floors.
    calls = 0

    def memory():
        nonlocal calls
        calls += 1
        if calls > 10:
            return dict(healthy, available_physical_bytes=3 * runner.GIB - 1)
        return healthy

    monkeypatch.setattr(runner, "available_memory", memory)
    result = runner.run_bounded(spec)
    assert result["reason"] == "physical_memory_pressure"
    assert result["guard"]["termination_confirmed"]
    last = json.loads((tmp_path / "run/memory.jsonl").read_text().splitlines()[-1])
    assert last["available_physical_bytes"] == 3 * runner.GIB - 1


def test_output_flood_is_bounded(tmp_path):
    result = runner.run_bounded(
        spec_for(
            tmp_path,
            "import sys,time; print('READY',flush=True); sys.stdout.write('x'*200000); "
            "sys.stdout.flush(); time.sleep(30)",
        )
    )
    assert result["reason"] in ("line_limit", "queue_limit", "output_limit")
    assert (tmp_path / "run/stdout.log").stat().st_size <= runner.MAX_LOG
    assert result["guard"]["termination_confirmed"]


def open_process(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x100000, False, pid)
    return kernel, handle


def wait_dead(pid):
    kernel, handle = open_process(pid)
    if not handle:
        return
    try:
        assert kernel.WaitForSingleObject(handle, 5000) == 0
    finally:
        kernel.CloseHandle(handle)


@pytest.mark.parametrize("parent_exit", [False, True])
def test_owned_tree_cleanup_and_unrelated_sentinel(tmp_path, parent_exit):
    pidfile = tmp_path / "descendant.pid"
    body = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(20)"
    )
    spec = spec_for(tmp_path, body)
    spec.update(ready_timeout_s=2, total_timeout_s=4)
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(20)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    owner = None
    try:
        if parent_exit:
            specpath = tmp_path / "input.json"
            specpath.write_text(json.dumps(spec))
            # This interpreter needs the same explicit CPU-fixture volume as
            # the pytest process; the production CLI correctly keeps H:.
            launcher = (
                "import json,sys; from pathlib import Path; "
                "sys.path.insert(0,sys.argv[1]); "
                "from tools.qwen35 import q36_bounded_run as r; "
                "from perf131k import windows_job as w; "
                "s=json.loads(Path(sys.argv[2]).read_text()); "
                "v=Path(s['output_root']).resolve().drive.upper(); "
                "r.OUTPUT_VOLUME=v; w.BOUNDED_STDIN_VOLUME=v; r.run_bounded(s)"
            )
            owner = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    launcher,
                    str(ROOT),
                    str(specpath),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            until = time.monotonic() + 5
            while not pidfile.exists() and time.monotonic() < until:
                time.sleep(0.02)
            assert pidfile.exists()
            owner.terminate()
            owner.wait(timeout=5)
        else:
            result = runner.run_bounded(spec)
            assert result["reason"] == "ready_deadline"
            assert result["guard"]["job_active_after"] == 0
        wait_dead(int(pidfile.read_text()))
        assert sentinel.poll() is None
    finally:
        if owner is not None and owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=5)
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_reused_guard_default_and_callback_failure(tmp_path):
    from tools.perf131k.windows_job import run_owned

    budget = SimpleNamespace(
        timeout_s=2,
        job_commit_bytes=128 * 1024**2,
        max_processes=4,
        output_bytes=1024,
        min_available_ram=3 * runner.GIB,
        min_commit_headroom=4 * runner.GIB,
        min_H_free=runner.GIB,
    )

    def sample(_):
        value = runner.available_memory()
        return dict(
            available_ram=value["available_physical_bytes"],
            commit_headroom=value["available_commit_bytes"],
            H_free=10 * runner.GIB,
        )

    result = run_owned(
        [sys.executable, "-c", "print('default')"],
        tmp_path,
        dict(os.environ),
        tmp_path / "default.log",
        budget,
        sample,
    )
    assert result["status"] == "FINISHED" and result["native_exit_code"] == 0
    assert (tmp_path / "default.log").read_bytes().strip() == b"default"

    def fail(_):
        raise RuntimeError("synthetic callback failure")

    result = run_owned(
        [sys.executable, "-u", "-c", "import time; print('x'); time.sleep(20)"],
        tmp_path,
        dict(os.environ),
        tmp_path / "callback.log",
        budget,
        sample,
        on_output=fail,
    )
    assert result["status"] == "CONTROL_ERROR"
    assert result["termination_confirmed"] and result["job_active_after"] == 0


def test_uncapped_job_still_owns_and_cleans(tmp_path):
    spec = spec_for(tmp_path)
    spec["job_commit_bytes"] = None
    result = runner.run_bounded(spec)
    assert result["status"] == "PASS"
    assert result["guard"]["configured_job_commit_bytes"] is None
    assert (
        result["guard"]["job_commit_limit_reason"]
        == "no diagnostic process cap configured"
    )
    assert result["guard"]["contained_before_resume"]
    assert result["guard"]["termination_confirmed"]
    assert result["guard"]["peak_job_commit_bytes"] >= 0


@pytest.mark.parametrize("failure", ["commit", "disk", "observer"])
def test_runtime_resource_failures_are_preserved(tmp_path, monkeypatch, failure):
    spec = spec_for(tmp_path, "import time; print('READY',flush=True); time.sleep(20)")
    healthy = {
        "available_physical_bytes": 8 * runner.GIB,
        "available_commit_bytes": 8 * runner.GIB,
    }
    count = 0

    def memory():
        nonlocal count
        count += 1
        if count > 12:
            if failure == "observer":
                raise OSError("synthetic")
            if failure == "commit":
                return dict(healthy, available_commit_bytes=4 * runner.GIB - 1)
        return healthy

    monkeypatch.setattr(runner, "available_memory", memory)
    original_disk = runner.shutil.disk_usage

    def disk(path):
        if failure == "disk" and count > 12:
            return SimpleNamespace(free=runner.GIB - 1)
        return original_disk(path)

    monkeypatch.setattr(runner.shutil, "disk_usage", disk)
    result = runner.run_bounded(spec)
    assert (
        result["reason"]
        == {
            "commit": "commit_memory_pressure",
            "disk": "disk_pressure",
            "observer": "resource_observer_failure",
        }[failure]
    )
    assert result["guard"]["termination_confirmed"]


def test_guard_import_is_cpu_only_and_digest_contract(tmp_path):
    # -S prevents site .pth imports; the guard itself is strictly stdlib.
    code = (
        f"import sys; sys.path.insert(0,{str(ROOT / 'tools')!r}); "
        "import native_model_resource_guard as g; "
        "assert 'torch' not in sys.modules; "
        "assert 'iq_pack' not in sys.modules; "
        "from pathlib import Path; import hashlib; "
        "assert g.digest(Path(sys.executable)) == hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        capture_output=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == b""


def test_preflight_pressure_retains_not_run_receipt(tmp_path, monkeypatch):
    spec = spec_for(tmp_path)
    monkeypatch.setattr(
        runner,
        "available_memory",
        lambda: {
            "available_physical_bytes": 3 * runner.GIB - 1,
            "available_commit_bytes": 4 * runner.GIB,
        },
    )
    result = runner.run_bounded(spec)
    assert result["status"] == "ABORTED"
    assert result["reason"] == "physical_memory_pressure"
    assert result["requests_status"] == "NOT_RUN"
    assert not (tmp_path / "run/stdout.log").exists()


@pytest.mark.parametrize("cause", ["physical_memory_pressure", "total_deadline"])
def test_hash_preflight_abort_after_healthy_admission(tmp_path, monkeypatch, cause):
    from perf131k import windows_job

    spec = spec_for(tmp_path)
    observations = 0

    def memory():
        nonlocal observations
        observations += 1
        return {
            "available_physical_bytes": (
                3 * runner.GIB - 1
                if observations > 1 and cause == "physical_memory_pressure"
                else 8 * runner.GIB
            ),
            "available_commit_bytes": 8 * runner.GIB,
        }

    def forbidden(*args, **kwargs):
        pytest.fail("artifact preflight abort must not create a child")

    monkeypatch.setattr(runner, "available_memory", memory)
    monkeypatch.setattr(
        runner,
        "time",
        SimpleNamespace(
            monotonic=lambda: (
                10 if observations > 1 and cause == "total_deadline" else 0
            )
        ),
    )
    monkeypatch.setattr(windows_job, "run_owned", forbidden)
    result = runner.run_bounded(spec)
    assert observations == 2
    assert result["status"] == "ABORTED"
    assert result["reason"] == cause
    assert result["last_stage"] == "preflight"
    assert result["requests_status"] == "NOT_RUN"
    assert result["child_exit_code"] is None
    sample = result["initial_resources"]
    assert sample["available_physical_bytes"] == (
        3 * runner.GIB - 1 if cause == "physical_memory_pressure" else 8 * runner.GIB
    )
    assert sample["available_commit_bytes"] == 8 * runner.GIB
    assert sample["disk_free_bytes"] >= runner.GIB
    assert json.loads((tmp_path / "run/result.json").read_text()) == result
    assert not (tmp_path / "run/events.jsonl").exists()


def test_missing_artifact_retains_failure_without_launch(tmp_path, monkeypatch):
    from perf131k import windows_job

    spec = spec_for(tmp_path)
    spec["observation"] = dict(
        schema_version=1,
        family="qwen36",
        source_sha="a" * 40,
        binary_source_sha=None,
        environment_manifest=None,
        identity_files={},
    )
    Path(spec["model_path"]).unlink()

    def must_not_launch(*args, **kwargs):
        pytest.fail("missing artifacts must not start a child")

    monkeypatch.setattr(windows_job, "run_owned", must_not_launch)
    result = runner.run_bounded(spec)
    assert result["status"] == "FAILED"
    assert result["reason"] == "artifact_unavailable"
    assert result["requests_status"] == "NOT_RUN"
    assert result["child_exit_code"] is None
    assert result["qualification"] == "NOT_RUN"
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["identities"] == {}
    assert not manifest["performance_eligible"]


def test_preflight_digest_mismatch_remains_distinct(tmp_path, monkeypatch):
    from perf131k import windows_job

    spec = spec_for(tmp_path)
    spec["engine_sha256"] = "0" * 64

    def forbidden(*args, **kwargs):
        pytest.fail("digest mismatch must not create a child")

    monkeypatch.setattr(windows_job, "run_owned", forbidden)
    with pytest.raises(ValueError, match="engine_digest_mismatch"):
        runner.run_bounded(spec)
    assert not (tmp_path / "run").exists()
