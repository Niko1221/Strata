#!/usr/bin/env python3
"""Run the small, pure-function quality fixture in a bounded child process."""
import ast
import json
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
    source = json.load(sys.stdin)["code"]
    tree = ast.parse(source)
    allowed = {ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AnnAssign,
               ast.For, ast.If, ast.Compare, ast.Name, ast.Load, ast.Store, ast.List, ast.Set, ast.Dict,
               ast.Tuple, ast.Constant, ast.Expr, ast.Call, ast.Attribute, ast.UnaryOp, ast.USub,
               ast.Not, ast.In, ast.NotIn, ast.Eq, ast.NotEq, ast.Subscript, ast.ListComp,
               ast.comprehension, ast.IfExp, ast.BoolOp, ast.And, ast.Or, ast.keyword}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "dedupe_keep_order" or len(tree.body) != 1:
        raise ValueError("expected exactly the requested function")
    builtins = {"len": len, "set": set, "list": list, "dict": dict, "range": range, "int": int}
    for node in ast.walk(tree):
        if type(node) not in allowed or (isinstance(node, ast.Name) and node.id.startswith("__")):
            raise ValueError(f"unsupported construct {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr not in {"add", "append", "fromkeys"}:
            raise ValueError("unsupported attribute")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in builtins:
            raise ValueError("unsupported call")
    namespace = {"__builtins__": builtins}
    exec(compile(tree, "<quality-fixture>", "exec"), namespace)
    cases = [([], []), ([3, 1, 3, 2, 1], [3, 1, 2]), ([-1, 0, -1, 2, 0], [-1, 0, 2]),
             ([7, 7, 7], [7]), (list(range(100)) * 2, list(range(100)))]
    for values, expected in cases:
        original = list(values)
        if namespace["dedupe_keep_order"](values) != expected or values != original:
            raise ValueError("function output or input preservation failed")
    print(json.dumps({"passed": True, "cases": len(cases)}))


if __name__ == "__main__":
    main()
