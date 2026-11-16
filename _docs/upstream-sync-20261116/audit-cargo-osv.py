"""Query public locked crate names/versions only; retain every OSV response."""
from __future__ import annotations
import hashlib
import argparse
import json
from pathlib import Path
import tomllib
import re
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
API = 'https://api.osv.dev/v1/'


def fetch(endpoint, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(API + endpoint, data=data,
                      headers={'Content-Type': 'application/json', 'User-Agent': 'Strata-lock-audit/0.6.0'})
    with urlopen(request, timeout=30) as response:
        raw = response.read(32 * 1024**2 + 1)
    if len(raw) > 32 * 1024**2:
        raise ValueError('OSV response too large')
    return raw, json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name', default='cargo-osv-v1')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9-]+', args.output_name):
        parser.error('contained evidence directory name required')
    OUT = Path('H:/Strata/evidence/upstream-sync-20261116') / args.output_name
    OUT.mkdir()  # Never overwrite an earlier audit or interrupted response.
    lock = ROOT / 'apps/desktop/src-tauri/Cargo.lock'
    raw = lock.read_bytes()
    packages = tomllib.loads(raw.decode())['package']
    registry = sorted({(p['name'], p['version']) for p in packages
                       if p.get('source', '').startswith('registry+')})
    skipped = [{'name': p['name'], 'version': p['version'], 'source': p.get('source', 'local')}
               for p in packages if not p.get('source', '').startswith('registry+')]
    results = {}
    sequence = 0
    for start in range(0, len(registry), 80):
        pending = [{'package': {'ecosystem': 'crates.io', 'name': name}, 'version': version}
                   for name, version in registry[start:start+80]]
        while pending:
            response_raw, response = fetch('querybatch', {'queries': pending})
            (OUT / f'batch-{sequence:03}.json').write_bytes(response_raw)
            sequence += 1
            rows = response.get('results')
            if not isinstance(rows, list) or len(rows) != len(pending):
                raise ValueError('OSV response cardinality mismatch')
            more = []
            for query, row in zip(pending, rows):
                key = (query['package']['name'], query['version'])
                if 'error' in row:
                    raise ValueError(f'OSV query error for {key}')
                results.setdefault(key, set()).update(v['id'] for v in row.get('vulns', []))
                if row.get('next_page_token'):
                    more.append({**query, 'page_token': row['next_page_token']})
            pending = more
    ids = sorted({identity for matches in results.values() for identity in matches})
    for identity in ids:
        if not all(c.isalnum() or c in '-_.' for c in identity):
            raise ValueError('Unexpected advisory identity')
        advisory_raw, _ = fetch('vulns/' + identity)
        (OUT / (identity + '.json')).write_bytes(advisory_raw)
    report = {'lock': str(lock), 'lockSHA256': hashlib.sha256(raw).hexdigest(),
              'endpoint': API, 'registryPackagesQueried': len(registry),
              'skippedNonRegistry': skipped, 'advisories': ids,
              'matches': [{'name': name, 'version': version, 'ids': sorted(matches)}
                          for (name, version), matches in results.items() if matches],
              'scope': 'All locked registry crates, including non-Windows targets; no exploitability claim.'}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
