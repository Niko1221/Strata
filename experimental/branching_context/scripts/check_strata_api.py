"""Record real API integration regressions independently of gate qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class Results(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed = []

    def startTest(self, test):
        self.executed.append(test.id())
        super().startTest(test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('tests', nargs='*')
    args = parser.parse_args()
    os.environ['CTT_PROFILE'] = str(args.profile.resolve())
    os.environ['CTT_NATIVE_TRACE'] = str(args.output.with_suffix('.native.jsonl').resolve())
    os.environ['CTT_ORACLE_TRACE'] = str(args.output.with_suffix('.oracle.jsonl').resolve())
    os.environ['CTT_API_OBSERVATIONS'] = str(args.output.with_suffix('.observations.jsonl').resolve())
    with zipfile.ZipFile(args.output.with_suffix('.source.zip'), 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ('src', 'tests', 'scripts', 'spec', 'docs', 'plans'):
            for path in sorted((ROOT/directory).rglob('*')):
                if path.is_file() and path.suffix in ('.py', '.md', '.json'):
                    archive.write(path, path.relative_to(ROOT))
    start = time.monotonic()
    with args.output.with_suffix('.log').open('x') as log:
        if args.tests:
            sys.path.insert(0, str(ROOT/'tests/native'))
            suite = unittest.defaultTestLoader.loadTestsFromNames(args.tests)
        else:
            suite = unittest.defaultTestLoader.discover(str(ROOT/'tests/native'), pattern='test_strata_api.py')
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Results).run(suite)
    good = result.wasSuccessful() and result.testsRun > 0 and not result.skipped and not result.expectedFailures
    report = dict(status='PASS' if good else 'FAIL', scope='Native API regressions; no gate qualification',
        executed_ids=result.executed, test_count=result.testsRun, failures=[(t.id(), tb) for t, tb in result.failures],
        errors=[(t.id(), tb) for t, tb in result.errors], skipped=[(t.id(), msg) for t, msg in result.skipped],
        expected_failures=[t.id() for t, _ in result.expectedFailures], seconds=time.monotonic()-start,
        profile=json.loads(args.profile.read_text()), profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in ('src', 'tests/native', 'scripts') for p in (ROOT/d).rglob('*.py')}, command=sys.argv)
    with args.output.with_suffix('.json').open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(args.output.with_suffix('.log').read_text(), flush=True)
    print(json.dumps(dict(status=report['status'], test_count=result.testsRun, seconds=report['seconds'],
                         report=str(args.output.with_suffix('.json')))), flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
