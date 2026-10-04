"""Label free features for each generation. Nothing here may read y."""

import ast

import numpy as np

BLOCKS = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try,
          ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _nesting(node, depth=0):
    """Deepest stack of nested blocks (loops, ifs, functions, ...) under node."""
    best = depth
    for child in ast.iter_child_nodes(node):
        best = max(best, _nesting(child, depth + isinstance(child, BLOCKS)))
    return best


def code_structure(code):
    """AST counts. For empty or unparsable code, counts are NaN and the flags say why."""
    out = {"empty_code": code.strip() == "", "syntax_valid": False, "nesting": np.nan, "branches": np.nan,
           "loops": np.nan, "functions": np.nan, "imports": np.nan}
    if out["empty_code"]:
        return out
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return out
    nodes = list(ast.walk(tree))
    out.update(syntax_valid=True, nesting=_nesting(tree),
               branches=sum(isinstance(n, (ast.If, ast.IfExp, ast.Match)) for n in nodes),
               loops=sum(isinstance(n, (ast.For, ast.AsyncFor, ast.While, ast.comprehension)) for n in nodes),
               functions=sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for n in nodes),
               imports=sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes))
    return out
