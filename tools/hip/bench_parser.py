#!/usr/bin/env python3
"""Compare an archived frontend.py with the current parser; no model/GPU needed."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--candidate', type=Path, default=Path('serve/frontend.py'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=5)
    a = p.parse_args()
    modules = {name: load('benchmark_' + name, path) for name, path in
               [('control', a.control), ('candidate', a.candidate)]}
    schema = [{'name': 'write', 'parameters': {'properties': {'content': {'type': 'string'}}}}]
    results = []
    for stream in (False, True):
        for size in (65536, 131072, 262144):
            text = '<tool_call><function=write><parameter=content>' + 'a' * size + '</parameter></function></tool_call>'
            for name, module in modules.items():
                timings = []
                for _ in range(a.repetitions):
                    parser = module.OutputParser(thinking=False, tools=schema, stream_tools=stream)
                    events = []
                    start = time.perf_counter()
                    for at in range(0, len(text), 8):
                        events.extend(parser.feed(text[at:at + 8]))
                    events.extend(parser.finish())
                    timings.append(time.perf_counter() - start)
                    call = next(e.call for e in events if e.kind == 'tool_call')
                    assert call.arguments['content'] == 'a' * size
                    if stream:
                        assert json.loads(''.join(e.text for e in events if e.kind == 'tool_args')) == call.arguments
                results.append(dict(label=name, stream_tools=stream, bytes=size, delta_chars=8,
                                    seconds=timings, median_s=statistics.median(timings)))
                a.output.write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps(results[-1]), flush=True)


if __name__ == '__main__':
    main()
