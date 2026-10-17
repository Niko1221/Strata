"""Check requested pure algorithms; reject imports, I/O and reflective attributes."""
import ast
import copy
import itertools
import json
import random
import re
import sys

BUILTINS = {
    'len': len, 'sorted': sorted, 'min': min, 'max': max, 'sum': sum,
    'enumerate': enumerate, 'range': range, 'zip': zip, 'list': list, 'dict': dict,
    'set': set, 'tuple': tuple, 'abs': abs, 'int': int, 'str': str, 'bool': bool,
    'reversed': reversed, 'any': any, 'all': all, 'ValueError': ValueError,
}
METHODS = {'append', 'extend', 'insert', 'pop', 'sort', 'reverse', 'get', 'items', 'keys',
           'values', 'setdefault', 'add', 'discard', 'remove', 'copy', 'clear',
           'index', 'count'}
NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AnnAssign, ast.AugAssign, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod,
    ast.Expr, ast.If, ast.IfExp, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.UnaryOp, ast.USub,
    ast.UAdd, ast.Not, ast.BoolOp, ast.And, ast.Or, ast.For, ast.While,
    ast.Break, ast.Continue, ast.Pass, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.Subscript, ast.Slice, ast.Call, ast.Attribute, ast.keyword,
    ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.comprehension,
    ast.Raise, ast.Lambda, ast.JoinedStr, ast.FormattedValue,
)


def reference_topo(nodes, edges):
    graph = {node: [] for node in nodes}
    degrees = {node: 0 for node in nodes}
    for left, right in set(map(tuple, edges)):
        graph[left].append(right)
        degrees[right] += 1
    ready = sorted(node for node in nodes if not degrees[node])
    answer = []
    while ready:
        node = ready.pop(0)
        answer.append(node)
        for neighbor in graph[node]:
            degrees[neighbor] -= 1
            if degrees[neighbor] == 0:
                ready.append(neighbor)
                ready.sort()
    if len(answer) != len(nodes):
        raise ValueError('cycle')
    return answer


def reference_edit(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def reference_profit(jobs):
    best = 0
    for bits in itertools.product((False, True), repeat=len(jobs)):
        chosen = sorted(job for job, take in zip(jobs, bits) if take)
        if all(chosen[i][1] <= chosen[i + 1][0] for i in range(len(chosen) - 1)):
            best = max(best, sum(job[2] for job in chosen))
    return best


def reference_bfs(graph, start, target):
    from collections import deque
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []


def fixtures(case):
    rng = random.Random(906061)
    if case == 'toposort':
        name = 'toposort'
        inputs = [([], []), ([4, 2, 1, 3], []),
                  ([0, 1, 2, 3], [(0, 2), (1, 2), (2, 3)]),
                  ([0, 1, 2], [(0, 1), (0, 1), (1, 2)]),
                  ([0, 1], [(0, 1), (1, 0)]), ([0], [(0, 0)])]
        for _ in range(20):
            nodes = list(range(rng.randrange(2, 10)))
            edges = [(a, b) for a in nodes for b in nodes if a < b and rng.random() < .25]
            rng.shuffle(nodes)
            inputs.append((nodes, edges))
        oracle = reference_topo
    elif case == 'edit_distance':
        name = 'edit_distance'
        inputs = [('', ''), ('', 'abc'), ('abc', ''), ('kitten', 'sitting'),
                  ('你好代码', '您好代码'), ('a' * 100, 'b' * 100),
                  ('same', 'same'), ('ab', 'ba')]
        for _ in range(30):
            inputs.append(tuple(''.join(rng.choice('abc你') for _ in range(rng.randrange(12)))
                                for _ in range(2)))
        oracle = reference_edit
    elif case == 'schedule_profit':
        name = 'schedule_profit'
        inputs = [([],), ([(0, 1, 5)],), ([(0, 1, -5)],),
                  ([(0, 2, 5), (2, 4, 6), (0, 4, 10)],),
                  ([(0, 1, 5), (0, 1, 7), (1, 2, 2)],)]
        for _ in range(25):
            jobs = []
            for _ in range(rng.randrange(1, 10)):
                start = rng.randrange(10)
                jobs.append((start, start + rng.randrange(1, 5), rng.randrange(-3, 15)))
            rng.shuffle(jobs)
            inputs.append((jobs,))
        oracle = reference_profit
    elif case == 'bfs_path':
        name = 'bfs_path'
        inputs = [({}, 0, 0), ({}, 0, 1), ({0: [1], 1: [2]}, 0, 2),
                  ({0: [0, 1, 1], 1: [0, 2], 2: [2]}, 0, 2),
                  ({0: [1, 2], 1: [3], 2: [3]}, 0, 3),
                  ({0: [2, 1], 1: [3], 2: [3]}, 0, 3)]
        for _ in range(35):
            size = rng.randrange(2, 15)
            graph = {i: [j for j in range(size) if rng.random() < .17] for i in range(size)}
            inputs.append((graph, rng.randrange(size), rng.randrange(size)))
        oracle = reference_bfs
    else:
        raise ValueError('Unknown oracle case')
    return name, inputs, oracle


def check(case, answer):
    name, inputs, oracle = fixtures(case)
    # Consume complete fences before selecting Python. The old optional-language
    # regex could treat a JSON block's closing fence as an unlabelled opener.
    fences = re.findall(r'^[ \t]*```([^\r\n`]*)\r?\n(.*?)^[ \t]*```[ \t]*\r?$',
                        answer, re.M | re.S)
    blocks = [body for language, body in fences
              if language.strip().lower() in ('', 'python', 'py')]
    source = '\n'.join(blocks) if blocks else answer.strip()
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != name:
        raise ValueError('Expected exactly the requested function')
    for node in ast.walk(tree):
        if not isinstance(node, NODES):
            raise ValueError(f'Unapproved syntax: {type(node).__name__}')
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise ValueError('Decorators are not approved')
        if isinstance(node, (ast.Name, ast.arg)):
            text = node.id if isinstance(node, ast.Name) else node.arg
            if '__' in text:
                raise ValueError('Reflective identifiers are not approved')
        if isinstance(node, ast.Attribute) and node.attr not in METHODS:
            # A standard builtin constructor, used to initialize indegrees.
            # Admit this exact receiver only; keep other fromkeys/attributes denied.
            builtin_fromkeys = (node.attr == 'fromkeys' and isinstance(node.value, ast.Name)
                                and node.value.id == 'dict')
            if not builtin_fromkeys:
                raise ValueError(f'Unapproved attribute: {node.attr}')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in BUILTINS and node.func.id != name:
                    raise ValueError(f'Unapproved call: {node.func.id}')
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError('Indirect calls are not approved')
    scope = {'__builtins__': BUILTINS}
    exec(compile(tree, '<pure-algorithm>', 'exec'), scope)
    results = []
    for arguments in inputs:
        # The tested implementation receives its own copy; mutation cannot change the oracle input.
        copied = copy.deepcopy(arguments)
        try:
            expected = oracle(*arguments)
            error_expected = False
        except ValueError:
            expected, error_expected = 'ValueError', True
        try:
            actual = scope[name](*copied)
            passed = not error_expected and type(actual) is type(expected) and actual == expected
            if isinstance(expected, list) and isinstance(actual, list):
                passed = passed and all(type(left) is type(right) for left, right in zip(actual, expected))
        except ValueError:
            actual, passed = 'ValueError', error_expected
        except Exception as error:
            actual, passed = f'{type(error).__name__}: {error}', False
        if copied != arguments:
            passed = False
        results.append({'Input': arguments, 'Expected': expected, 'Actual': actual, 'Passed': passed})
    return {'Passed': all(item['Passed'] for item in results),
            'PassedCount': sum(item['Passed'] for item in results), 'Count': len(results),
            'Tests': results, 'CheckedCode': source}


if __name__ == '__main__':
    try:
        request = json.load(sys.stdin)
        print(json.dumps(check(request['Case'], request['Answer']), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({'Passed': False, 'Reason': f'{type(error).__name__}: {error}'}, ensure_ascii=False))
