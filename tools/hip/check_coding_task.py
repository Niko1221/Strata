#!/usr/bin/env python3
"""Complete a small coding task and check its answer against an independent oracle.

Only function definitions are accepted; generated code runs with a restricted
set of pure builtins. This is a smoke test, not a model quality benchmark.
"""
import argparse
import ast
import json
from pathlib import Path
import random
import re
import time
import urllib.request

PROMPT = ('Return only a Python code block defining merge_intervals(intervals). '
          'Input is a list of integer pairs [start,end] with start <= end. '
          'Return sorted disjoint intervals as lists, merging overlapping or touching endpoints. '
          'Handle empty input and negative endpoints, and do not mutate input. No imports or explanations.')


def oracle(intervals):
    endpoints = sorted(set(x for pair in intervals for x in pair))
    if not endpoints:
        return []
    covered = [any(a <= x <= b for a, b in intervals) for x in endpoints]
    links = [any(a <= x and y <= b for a, b in intervals) for x, y in zip(endpoints, endpoints[1:])]
    result = []
    for i, x in enumerate(endpoints):
        if covered[i] and (i == 0 or not links[i - 1]):
            result.append([x, x])
        if covered[i]:
            result[-1][1] = x
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:19931')
    p.add_argument('--model', default='strata-discover')
    p.add_argument('--label', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    body = dict(model=a.model, messages=[{'role': 'user', 'content': PROMPT}], max_tokens=512,
                temperature=0, top_k=1, top_p=1, min_p=0, seed=42, reasoning_effort='none')
    start = time.monotonic()
    req = urllib.request.Request(a.url.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        reply = json.load(response)
    record = dict(label=a.label, prompt=PROMPT, wall_s=time.monotonic() - start, reply=reply)
    try:
        choice = reply['choices'][0]
        if choice['finish_reason'] != 'stop':
            raise ValueError('task did not complete before output cap')
        text = choice['message']['content']
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.S)
        source = blocks[0] if blocks else text
        tree = ast.parse(source)
        if not tree.body or any(not isinstance(n, ast.FunctionDef) for n in tree.body):
            raise ValueError('expected only function definitions')
        if any(isinstance(n, ast.Name) and n.id.startswith('__') or
               isinstance(n, ast.Attribute) and n.attr.startswith('__') for n in ast.walk(tree)):
            raise ValueError('dunder access rejected')
        namespace = {'__builtins__': {k: v for k, v in dict(sorted=sorted, list=list, len=len,
            range=range, min=min, max=max, enumerate=enumerate, zip=zip, tuple=tuple).items()}}
        exec(compile(tree, '<model-task>', 'exec'), namespace)
        function = namespace['merge_intervals']
        rng = random.Random(91)
        cases = [[], [[1, 3], [3, 5]], [[-5, -3], [-1, 2]], [[0, 0]], [[1, 8], [2, 3]]]
        for _ in range(200):
            cases.append([sorted([rng.randint(-30, 30), rng.randint(-30, 30)])
                          for _ in range(rng.randint(0, 20))])
        for case in cases:
            before = [x[:] for x in case]
            actual = function(case)
            if case != before or actual != oracle(before):
                raise AssertionError(f'case={before} got={actual} expected={oracle(before)} input={case}')
        record.update(verdict='PASS', cases=len(cases))
    except Exception as error:
        record.update(verdict='FAIL', error=str(error))
    a.output.write_text(json.dumps(record, indent=2) + '\n')
    print(record['verdict'], record.get('error', f"{record.get('cases')} cases"), flush=True)
    return record['verdict'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
