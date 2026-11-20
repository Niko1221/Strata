"""Machine-bound evidence tests are explicit opt-ins, not portable CI prerequisites.

The unittest resource-gated runners are unaffected. Portable trace-accounting
tests still collect normally; local integration fixtures require their frozen
binary, model metadata and H-drive receipts, which are deliberately not in Git.
"""
import os

if os.environ.get("STRATA_LOCAL_PERF131K_TESTS") != "1":
    collect_ignore = [
        "test_collector_output.py", "test_e2e_consumer.py", "test_job_evidence.py",
        "test_native_pilot.py", "test_pilot.py", "test_service_pilot.py",
        "test_blob_failure.py", "test_diagnostics.py", "test_exit_status.py",
    ]
