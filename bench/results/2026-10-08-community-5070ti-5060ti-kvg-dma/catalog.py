"""Portable constants and validation for this synthetic two-GPU report only."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT
PYTHON = Path(sys.executable)
PORT = 18080
URL = f'http://127.0.0.1:{PORT}'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def arg(args, flag, value=None):
    result = list(args)
    if flag in result:
        i = result.index(flag)
        del result[i:i + (1 if value is None else 2)]
    result.append(flag)
    if value is not None: result.append(value)
    return result

def validate_config(cfg):
    args = cfg['args']
    assert cfg['host'] == '127.0.0.1' and cfg['port'] == PORT
    assert cfg['gpu'] == [0, 1] and cfg['layer_split'] == '32' and cfg['parallel'] == 1
    assert args[args.index('--max-context') + 1] == '262144'
    assert args[args.index('--kv') + 1] == 'int8'
    assert args[args.index('--pipeline-windows') + 1] == '2' and '--kv-grow' in args
    assert not any(cfg.get(k) for k in ('api_key', 'api_key_file', 'mcp_servers'))
    assert not any(flag in args for flag in ('--batch', '--batch-mtp', '--peer-device', '--resident-experts'))
