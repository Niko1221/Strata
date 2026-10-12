"""Record actual unittest IDs/results for a small ticket, including red runs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


class RecordedResult(unittest.TextTestResult):
    def startTest(self, test):
        self.executed.append(test.id())
        super().startTest(test)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed = []


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ('src', 'tests', 'scripts', 'native')
            for p in (ROOT / directory).rglob('*')
            if p.is_file() and p.suffix in ('.py', '.cpp', '.h')}


def execution_profile():
    name = os.environ.get('CTT_PROFILE')
    if name is None:
        return None
    path = Path(name).resolve()
    try:
        raw = path.read_bytes()
        profile = json.loads(raw)
    except (OSError, ValueError) as error:
        return dict(path=str(path), error=repr(error))
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                execution_profile_id=profile.get('execution_profile_id'),
                model_contract_id=profile.get('model_contract_id'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticket')
    parser.add_argument('--pattern', default='test_continuation.py')
    parser.add_argument('--gate',default='G1',choices=['G1','G2','G3','G4','G5'])
    args = parser.parse_args()
    started = time.time()
    source = source_hashes()
    profile = execution_profile()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests/gates'/args.gate), pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    integrity_errors = []
    if profile is not None and 'error' in profile:
        integrity_errors.append('Execution profile could not be read before tests')
    if source_hashes() != source:
        integrity_errors.append('Source files changed during test execution')
    if execution_profile() != profile:
        integrity_errors.append('Execution profile changed during test execution')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    good = (result.wasSuccessful() and result.testsRun > 0 and not result.skipped
            and not result.expectedFailures and not integrity_errors)
    report = dict(ticket=args.ticket, status='PASS' if good else 'FAIL',
        scope='ticket only; does not certify a cumulative gate',
        executed_ids=result.executed, test_count=result.testsRun,
        failures=[dict(id=t.id(), traceback=tb) for t,tb in result.failures],
        errors=[dict(id=t.id(), traceback=tb) for t,tb in result.errors],
        skipped=[dict(id=t.id(), reason=r) for t,r in result.skipped],
        expected_failures=[t.id() for t,_ in result.expectedFailures],
        platform=platform.platform(), python=sys.version, source_sha256=source,
        command=sys.argv, started=started, finished=time.time(),
        execution_profile=profile, evidence_integrity_errors=integrity_errors)
    destination = ROOT/'evidence/runs'/f'{stamp}-{args.ticket}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(destination)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
