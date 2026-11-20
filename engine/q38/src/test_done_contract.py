"""Exercise the production C++ formatter through the real Python consumer."""
import pathlib
import subprocess
import sys

fixture = pathlib.Path(sys.argv[1])
consumer_root = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(consumer_root))
from terminal.serve.server import StrataEngine

proc = subprocess.run([str(fixture), '--emit-done-fixtures'], capture_output=True,
                      text=True, encoding='utf-8', timeout=20, check=True)
lines = proc.stdout.splitlines()
assert len(lines) == 3, lines
engine = object.__new__(StrataEngine)
cases = [
    {'generated': 3, 'prompt_tokens': 17, 'prompt_ms': 12.5, 'decode_ms': 6.2,
     'finish': 'eos', 'prompt_read': 17},
    {'generated': 1, 'prompt_tokens': 128, 'prompt_ms': 20.0, 'decode_ms': 3.5,
     'finish': 'cancel', 'prompt_read': 64},
    {'generated': 0, 'prompt_tokens': 8, 'prompt_ms': 0.0, 'decode_ms': 0.0,
     'finish': 'length', 'prompt_read': 8},
]
for line, expected in zip(lines, cases):
    actual = engine._parse_done(line)
    assert engine.last == actual
    for name, value in expected.items():
        assert actual[name] == value, (line, name, actual, value)
    for name in ('drafts_accepted', 'drafts_offered', 'reused', 'hits', 'lookups',
                 'ram_blobs', 'file_blobs', 'offloaded'):
        assert actual[name] == 0, (line, name, actual)
    assert actual['file_mb'] == 0.0, (line, actual)
print('C++ DONE formatter -> actual StrataEngine._parse_done: 3 cases PASS')
