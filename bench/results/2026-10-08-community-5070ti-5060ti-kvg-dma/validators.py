"""Bounded structural/fact checks; code validation runs in a separate timed process."""
from __future__ import annotations
import ast
import builtins
import copy
import json



def unfence(text: str) -> str:
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        return text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return text


def validate_code(text: str) -> dict:
    tree = ast.parse(unfence(text))
    if not tree.body or any(not isinstance(n, ast.FunctionDef) for n in tree.body):
        raise ValueError('Only function definitions are permitted.')
    methods = {'get', 'items', 'values', 'keys', 'copy', 'append', 'setdefault', 'sort'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal)):
            raise ValueError('Unsupported code construct.')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('Private runtime names are not permitted.')
        if isinstance(node, ast.Attribute) and node.attr not in methods:
            raise ValueError('Unsupported method.')
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.args.defaults or
                                                any(x is not None for x in node.args.kw_defaults)):
            raise ValueError('Decorators/default expressions are not permitted.')
    safe = {k: getattr(builtins, k) for k in ('sorted', 'len', 'dict', 'list', 'set', 'tuple', 'enumerate',
                                           'range', 'sum', 'min', 'max', 'int', 'str', 'bool', 'isinstance')}
    scope = {'__builtins__': safe}
    exec(compile(tree, '<generated-code>', 'exec'), scope)
    function = scope.get('summarize_events')
    if not callable(function):
        raise ValueError('summarize_events is missing.')
    cases = [[], [{'id': 'x', 'timestamp': 2, 'category': 'A', 'amount': 4}],
             [{'id': 'x', 'timestamp': 2, 'category': 'A', 'amount': 4},
              {'id': 'x', 'timestamp': 1, 'category': 'B', 'amount': 99},
              {'id': 'y', 'timestamp': 2, 'category': 'B', 'amount': -3}],
             [{'id': 'z', 'timestamp': -2, 'category': 'A', 'amount': 7},
              {'id': 'a', 'timestamp': 1, 'category': 'B', 'amount': 3},
              {'id': 'z', 'timestamp': -2, 'category': 'B', 'amount': 6},
              {'id': 'b', 'timestamp': 1, 'category': 'B', 'amount': 2}]]
    for events in cases:
        original = copy.deepcopy(events)
        latest = {}
        for event in events:
            if event['id'] not in latest or event['timestamp'] >= latest[event['id']]['timestamp']:
                latest[event['id']] = event
        records = sorted(latest.values(), key=lambda x: (x['timestamp'], x['id']))
        totals, counts = {}, {}
        for event in records:
            totals[event['category']] = totals.get(event['category'], 0) + event['amount']
            counts[event['category']] = counts.get(event['category'], 0) + 1
        if function(events) != dict(latest=records, totals=totals, counts=counts) or events != original:
            raise ValueError('Functional case or input preservation failed.')
    return dict(passed=True, check='four functional cases and input preservation', cases=4)
