"""Synthetic receipts only: these do not qualify native input integrity or A/A."""

from copy import deepcopy
from itertools import count

import pytest

from tools.qwen35.q36_bounded_run import token_hash
from tools.qwen35.q36_compare_runs import compare, validate_pair

RUN_IDS = count()


def receipt():
    settings = dict(
        context=4096,
        prefill=256,
        parallel=1,
        kv_k="f32",
        kv_v="f32",
        mtp=0,
        router_top_k=8,
        expert_quantization="fixture",
    )
    return {
        "schema_version": 1,
        "run_id": f"synthetic-{next(RUN_IDS)}",
        "status": "PASS",
        "child_exit_code": 0,
        "startup_generation": "generation-A",
        "instrumentation": dict(
            mode="off",
            hotspot_sync=False,
            phase_counters=False,
            startup_detail=False,
            kernel_verify=False,
        ),
        "guard": dict(
            contained_before_resume=True, termination_confirmed=True, exit_code=0
        ),
        "identities": {
            k + "_sha256": "a" * 64
            for k in (
                "engine",
                "model",
                "prompt_fixture",
                "toolchain",
                "harness",
                "guard",
                "resource_guard",
                "worker_interpreter",
            )
        },
        "execution_argv": ["--serve"],
        "expected_case_ids": ["p0"],
        "requested_settings": settings,
        "effective_settings": deepcopy(settings),
        "experiment": {"dimensions": [], "max_relative_regression": 0.05},
        "cases": [
            {
                "prompt_id": "p0",
                "request_id": "request-A",
                "startup_generation": "generation-A",
                "status": "PASS",
                "submitted_input_count": 3,
                "submitted_input_sha256": token_hash([1, 2, 3]),
                "requested_max_tokens": 2,
                "effective_max_tokens": 2,
                "output_reservation": 2,
                "mtp_reservation": 0,
                "tokens": [4, 5],
                "token_times_s": [0.1, 0.2],
                "ttft_s": 0.1,
                "wall_s": 0.3,
                "consumed_input": dict(
                    ids=[1, 2, 3],
                    count=3,
                    sha256=token_hash([1, 2, 3]),
                    initial_position=0,
                    final_position=3,
                    request_id="request-A",
                    startup_generation="generation-A",
                    truncated=False,
                ),
            }
        ],
    }


def test_synthetic_identical_and_independent_request_bindings():
    a, b = receipt(), receipt()
    b["run_id"] = "synthetic-B"
    b["startup_generation"] = "generation-B"
    b["cases"][0].update(startup_generation="generation-B", request_id="request-B")
    b["cases"][0]["consumed_input"].update(
        startup_generation="generation-B", request_id="request-B"
    )
    assert validate_pair(a, b) == []
    result = compare(a, b)
    assert result["cases"][0]["wall_delta_s"] == 0
    assert result["qualification"] == "NOT_RUN"


def test_sync_profile_and_instrumentation_mismatch_block_comparison():
    a, b = receipt(), receipt()
    a["instrumentation"].update(mode="host_counters", phase_counters=True)
    assert any("instrumentation" in why for why in validate_pair(a, b))
    a["instrumentation"].update(mode="diagnostic_sync", hotspot_sync=True)
    b["instrumentation"] = deepcopy(a["instrumentation"])
    assert any("diagnostic only" in why for why in validate_pair(a, b))


def test_auto_prefill_still_requires_same_observed_numeric_chunk():
    a, b = receipt(), receipt()
    a["requested_settings"]["prefill"] = b["requested_settings"]["prefill"] = "auto"
    assert validate_pair(a, b) == []
    b["effective_settings"]["prefill"] = 512
    assert any(
        "effective setting differs: prefill" in why for why in validate_pair(a, b)
    )
    b["effective_settings"]["prefill"] = "auto"
    assert any("observed auto prefill" in why for why in validate_pair(a, b))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b.update(cases=[]),
        lambda b: b["cases"].append(deepcopy(b["cases"][0])),
        lambda b: b["cases"].append(dict(deepcopy(b["cases"][0]), prompt_id="extra")),
        lambda b: b["cases"][0].update(tokens=[]),
        lambda b: b["cases"][0].update(prompt_id="different"),
        lambda b: b["cases"][0].update(consumed_input=None),
        lambda b: b["cases"][0]["consumed_input"].update(truncated=True),
        lambda b: b["cases"][0]["consumed_input"].update(ids=[1, 2, 9]),
        lambda b: b["cases"][0]["consumed_input"].update(request_id="foreign"),
        lambda b: b["cases"][0]["consumed_input"].update(startup_generation="stale"),
        lambda b: b["cases"][0].update(requested_max_tokens=3),
        lambda b: b["cases"][0].update(effective_max_tokens=1),
        lambda b: b["cases"][0].update(submitted_input_sha256=None),
        lambda b: b.update(effective_settings={}),
        lambda b: b["effective_settings"].update(context=8192),
        lambda b: b["identities"].update(model_sha256=None),
        lambda b: b["identities"].update(engine_sha256="b" * 64),
        lambda b: b["experiment"].update(dimensions=["context"]),
        lambda b: b["experiment"].update(max_relative_regression=float("nan")),
        lambda b: b["cases"][0].update(token_times_s=[0.2, 0.1]),
        lambda b: b["cases"][0].update(output_reservation=4096),
        lambda b: b["guard"].update(termination_confirmed=False),
    ],
)
def test_incomplete_or_changed_evidence_cannot_compare(mutation):
    a, b = receipt(), receipt()
    mutation(b)
    assert validate_pair(a, b)
    assert compare(a, b)["status"] == "INCOMPARABLE"


def test_missing_extra_cases_both_directions():
    a, b = receipt(), receipt()
    extra = deepcopy(b["cases"][0])
    extra.update(prompt_id="p1", request_id="r1")
    extra["consumed_input"]["request_id"] = "r1"
    b["cases"].append(extra)
    b["expected_case_ids"].append("p1")
    assert any("case set" in r for r in validate_pair(a, b))
    assert any("case set" in r for r in validate_pair(b, a))


@pytest.mark.parametrize(
    "axis,field", [("kv_storage", "kv_k"), ("engine_toolchain", None)]
)
def test_only_explicit_narrow_axes_are_allowed(axis, field):
    a, b = receipt(), receipt()
    a["experiment"]["dimensions"] = [axis]
    b["experiment"]["dimensions"] = [axis]
    b["identities"]["engine_sha256"] = "b" * 64
    if field:
        b["requested_settings"][field] = "f16"
        b["effective_settings"][field] = "f16"
    assert validate_pair(a, b) == []
    for forbidden, value in (("router_top_k", 4), ("context", 8192), ("mtp", 1)):
        c = deepcopy(b)
        c["requested_settings"][forbidden] = value
        c["effective_settings"][forbidden] = value
        assert validate_pair(a, c)


@pytest.mark.parametrize(
    "value", [None, [], {}, {"cases": None}, {"schema_version": 1}]
)
def test_malformed_receipts_are_reasons_not_exceptions(value):
    assert validate_pair(receipt(), value)


def test_self_comparison_is_not_independent_aa():
    a = receipt()
    assert "independent run identities required" in validate_pair(a, deepcopy(a))


def test_missing_same_case_in_both_does_not_pass():
    a, b = receipt(), receipt()
    a["expected_case_ids"].append("p1")
    b["expected_case_ids"].append("p1")
    assert validate_pair(a, b)


def test_argument_changes_not_covered_by_axis_are_rejected():
    a, b = receipt(), receipt()
    a["experiment"]["dimensions"] = b["experiment"]["dimensions"] = ["kv_storage"]
    b["execution_argv"] = ["--serve", "--top-k", "4"]
    assert validate_pair(a, b)
