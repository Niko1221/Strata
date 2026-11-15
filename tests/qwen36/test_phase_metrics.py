"""Synthetic PR00 receipts only; no model, GPU probe or speed/quality claim."""

from copy import deepcopy
import json

import pytest

from tools.qwen35.q36_phase_metrics import (
    COUNTS,
    TIMES,
    build_artifacts,
    parse_phase_line,
    protocol_metrics,
    validate_observation_spec,
    verify_observation_inputs,
    write_artifacts,
)


def observation():
    return dict(
        schema_version=1,
        family="qwen36",
        source_sha="a" * 40,
        binary_source_sha=None,
        environment_manifest=None,
        identity_files={},
    )


def native_line(**changes):
    fields = dict(
        schema=1,
        generation="fixture",
        request=1,
        phase="prefill",
        state="complete",
        layer=-1,
        instrumentation="host_counters",
        gpu_time="unavailable",
        disk_physical_bytes="unavailable",
    )
    fields.update({key: 0 for key in (*COUNTS, *TIMES)})
    fields.update(input_tokens=10, reused_tokens=2, processed_tokens=8)
    fields.update(changes)
    return "Q36_PHASE " + " ".join(f"{k}={v}" for k, v in fields.items())


def case():
    return dict(
        request_id="fixture:0",
        prompt_id="public0",
        status="PASS",
        tokens=[7, 8, 9],
        token_times_s=[2.0, 2.2, 2.8],
        ttft_s=2.0,
        wall_s=3.0,
        consumed_input=None,
    )


def receipt():
    return dict(
        run_id="fixture",
        status="PASS",
        reason=None,
        startup_generation="fixture",
        instrumentation=dict(
            mode="host_counters",
            phase_counters=True,
            hotspot_sync=False,
            startup_detail=False,
        ),
        cases=[case()],
        requested_settings={"context": 65536},
        effective_settings={"context": 65536},
    )


def test_codec_or_unobserved_numbers_are_never_inferred():
    manifest, metrics, report = build_artifacts(
        receipt(),
        {"observation": observation(), "requested_settings": {}},
        [native_line()],
    )
    assert manifest["requested"]["context"] == 65536
    assert manifest["observed"]["gpu_memory"] is None
    assert manifest["identity_files"]["chat_template"]["sha256"] is None
    assert manifest["performance_eligible"] is False
    assert manifest["releaseQualified"] is False
    for phase in ("snapshot", "cancel", "unload"):
        row = next(r for r in metrics if r["phase"] == phase)
        assert row["value"] is None and row["status"] == "NOT_OBSERVED"
    assert "Qualification: NOT_RUN" in report
    row = next(r for r in metrics if r["phase"] == "prefill")
    assert row["gpu_time"] is None and row["disk_physical_bytes"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"schema": 2},
        {"layer": 65},
        {"wall_ms": "nan"},
        {"cpu_compute_ms": -1},
        {"read_requested_bytes": 2, "read_completed_bytes": 3},
        {"read_failed_calls": 1},
        {"committed_tokens": -1},
        {"input_tokens": 1},
        {"state": "success-ish"},
        {"instrumentation": "gpu_events"},
        {"committed_tokens": 2**64},
    ],
)
def test_malformed_native_record_is_rejected(change):
    with pytest.raises(ValueError):
        parse_phase_line(native_line(**change))


def test_duplicate_fields_unknown_schema_and_foreign_generation_are_rejected():
    with pytest.raises(ValueError):
        parse_phase_line(native_line() + " schema=1")
    _, metrics, _ = build_artifacts(
        receipt(),
        {"observation": observation(), "requested_settings": {}},
        [native_line(generation="other"), native_line(), native_line()],
    )
    assert len([r for r in metrics if r.get("layer") == -1]) == 1
    manifest, _, _ = build_artifacts(
        receipt(),
        {"observation": observation(), "requested_settings": {}},
        [native_line(generation="other")],
    )
    assert manifest["telemetry_errors"] == ["invalid_native_phase_record"]


def test_confirmed_tokens_and_ttft_samples_are_separate():
    row = protocol_metrics(case())
    assert row["committed_tokens"] == 3 and row["interval_samples"] == 2
    assert row["committed_after_first_per_s"] == pytest.approx(2 / 0.8)
    assert row["ttft_s"] == 2.0
    assert row["token_interval_p95_s"] == pytest.approx(0.6)
    single = dict(case(), tokens=[7], token_times_s=[2.0])
    assert protocol_metrics(single)["committed_after_first_per_s"] is None
    with pytest.raises(ValueError):
        protocol_metrics(dict(case(), token_times_s=[2, 1, 3]))


def test_failed_and_aborted_cases_remain_in_report_without_raw_errors():
    run = receipt()
    run.update(status="FAILED", reason="private/secret.txt")
    run["cases"] += [
        dict(
            case(),
            prompt_id="public1",
            status="ABORTED",
            tokens=[],
            token_times_s=[],
            ttft_s=None,
        )
    ]
    manifest, _, report = build_artifacts(
        run, {"observation": observation(), "requested_settings": {}}, []
    )
    assert [row["status"] for row in manifest["case_statuses"]] == ["PASS", "ABORTED"]
    assert "ABORTED" in report and "secret.txt" not in report
    assert manifest["status"] == "FAILED"


def test_spec_identity_validation_and_digest_binding(tmp_path):
    obs = observation()
    validate_observation_spec(obs)
    obs["identity_files"]["tokenizer"] = {
        "path": str(tmp_path / "tokens.json"),
        "sha256": "a" * 64,
    }
    verify_observation_inputs(obs, lambda _: "a" * 64)
    with pytest.raises(ValueError, match="digest_mismatch"):
        verify_observation_inputs(obs, lambda _: "b" * 64)
    for changes in (
        {"family": "generic"},
        {"source_sha": "main"},
        {"schema_version": True},
        {"identity_files": {"api_key": None}},
    ):
        with pytest.raises(ValueError):
            validate_observation_spec(dict(obs, **changes))


def test_artifacts_bind_raw_files_and_refuse_overwrite(tmp_path):
    (tmp_path / "stderr.log").write_text(native_line() + "\n")
    (tmp_path / "result.json").write_text('{"synthetic":true}')
    spec = {
        "observation": observation(),
        "requested_settings": {},
        "engine_sha256": "a" * 64,
    }
    write_artifacts(tmp_path, receipt(), spec)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert len(manifest["artifact_sha256"]["stderr.log"]) == 64
    assert manifest["declared_identities"]["engine_sha256"] == "a" * 64
    assert manifest["identities"] == {}  # Expected bytes are not observed verification.
    assert (tmp_path / "phase_metrics.jsonl").read_text()
    with pytest.raises(FileExistsError):
        write_artifacts(tmp_path, receipt(), spec)


def test_profile_sync_cannot_become_performance_evidence():
    run = deepcopy(receipt())
    run["instrumentation"].update(mode="diagnostic_sync", hotspot_sync=True)
    manifest, _, _ = build_artifacts(
        run, {"observation": observation(), "requested_settings": {}}, [native_line()]
    )
    assert manifest["performance_eligible"] is False
