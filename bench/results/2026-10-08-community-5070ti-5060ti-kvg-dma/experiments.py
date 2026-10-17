"""Bounded generated-code validation used by the isolated harness."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validators import validate_code

parser = argparse.ArgumentParser()
parser.add_argument('--validate-code', type=Path, required=True)
args = parser.parse_args()
try:
    result = validate_code(args.validate_code.read_text(encoding='utf-8'))
except Exception as exc:
    result = dict(passed=False, error=f'{type(exc).__name__}: {exc}')
print(json.dumps(result, ensure_ascii=False))
