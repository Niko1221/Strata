#!/usr/bin/env python3
"""Render one recorded configuration into a new run folder. Does not start a model."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=['DUAL-BASE', 'SINGLE-BASE', 'DUAL-TUNED', 'SINGLE-TUNED'])
    for key in ('strata', 'engine', 'cuda', 'model', 'pack', 'mtp', 'expert-profile', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((Path(__file__).parent / 'configs' / (args.arm + '.json')).read_text())
    paths = {key: val.expanduser().resolve() for key, val in vars(args).items() if isinstance(val, Path)}
    for key, path in paths.items():
        if key != 'output' and not path.exists():
            parser.error(f'Missing {key}: {path}')
    required = [paths['strata'] / 'serve/server.py', paths['pack'] / 'tokenizer', paths['pack'] / 'index.txt',
                paths['cuda'] / 'lib64']
    required += [paths['model'] / f'Qwen3.8-Flash-Next-UD-Q4_K_XL-{i:05d}-of-00004.gguf' for i in range(1, 5)]
    for path in required:
        if not path.exists():
            parser.error(f'Missing required asset: {path}')
    paths['output'].mkdir(parents=True, exist_ok=False)
    def replace(value):
        if isinstance(value, str):
            for key, path in paths.items():
                value = value.replace('@' + key.upper() + '@', str(path))
            return value
        if isinstance(value, list):
            return [replace(v) for v in value]
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        return value
    rendered = replace(config)
    serialized = json.dumps(rendered, indent=2)
    if '@' in serialized:
        raise RuntimeError('An unresolved configuration marker remains')
    target = paths['output'] / 'config.json'
    target.write_text(serialized + '\n')
    print(target)

if __name__ == '__main__':
    main()
