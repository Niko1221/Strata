"""Offline consistency checks for this report; does not run inference."""
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
data = json.loads((root / 'results.json').read_text())
assert data['max_context'] == 131072 and data['kv'] == 'int8'
assert sum(map(len, data['runs'].values())) == 18
for name, rows in data['runs'].items():
    for row in rows:
        t, u = row['timings'], row['usage']
        assert t['cache_n'] == u['prompt_tokens_details']['cached_tokens'] == 0
        assert t['prompt_n'] == u['prompt_tokens']
        assert t['predicted_n'] == u['completion_tokens'] == 8
        assert abs(1000 * t['prompt_n'] / t['prompt_ms'] - t['prompt_per_second']) < 0.2
    for target, median in data['medians_tok_s'][name].items():
        assert statistics.median(r['timings']['prompt_per_second'] for r in rows if r['target'] == int(target)) == median
for size, expected in [('4096', 458.5), ('16384', 389.1)]:
    gain = (data['medians_tok_s']['resident14'][size] / data['medians_tok_s']['baseline'][size] - 1) * 100
    assert round(gain, 1) == expected
baseline = json.loads((root / 'config-baseline.json').read_text())
resident = json.loads((root / 'config-resident14.json').read_text())
assert resident['args'] == baseline['args'] + ['--resident-budget-gib', '14']
assert resident['vision'] == baseline['vision']
readme = (root / 'README.md').read_text()
for path in re.findall(r'\]\(([^)]+)\)', readme):
    if not path.startswith('https://'):
        assert (root / path).is_file(), path
for path in root.iterdir():
    if path.is_file():
        text = path.read_text()
        assert not re.search(r'[A-Z]:[\\/]|gh[pousr]_[A-Za-z0-9]{20,}', text), path.name
print('PASS: 18 requests, zero reused tokens, timing arithmetic, medians, gains, config delta, local links and path/token scan')
