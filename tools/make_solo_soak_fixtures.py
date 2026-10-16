"""Make the three synthetic archives for soak_solo_priority.py.

Uses only Python's standard library. Token lengths depend on the server's
tokenizer/template; the measured IQ3_S model reads about 64K, 96K and 128K.
No user documents or credentials are included.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    source = (Path(__file__).resolve().parents[1] / 'bench/results/'
              '2026-10-16-solo-priority-rtx3090/make_fixture.py')
    spec = importlib.util.spec_from_file_location('solo_archive', source)
    archive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archive)
    args.out.mkdir(parents=True, exist_ok=False)
    for size, seed, rows in ((64000, 117, 391), (96000, 291, 590), (128000, 803, 789)):
        data = archive.fixture(seed, rows)
        data['messages'][0]['content'] = archive.SYSTEM
        data.update(target=size, seed=seed, rows=rows)
        path = args.out / f'fixture-{size}-{seed}.json'
        path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        print(path)


if __name__ == '__main__':
    main()
