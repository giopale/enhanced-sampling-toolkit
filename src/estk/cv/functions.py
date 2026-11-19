"""PLUMED's ``COMBINE`` and ``CUSTOM``: functions of other printed values.

``COMBINE ARG=a,b [COEFFICIENTS=..] [PARAMETERS=..] [POWERS=..] [NORMALIZE]`` is
``sum_i c_i (a_i - p_i)^n_i`` (defaults c = 1, p = 0, n = 1; NORMALIZE divides the coefficients
by their sum). ``CUSTOM ARG=a,b [VAR=x,y] FUNC=...`` evaluates a Lepton-style expression: numbers,
the variables (default x, y, z, t for the arguments in order), ``+ - * / ^``, parentheses, ``pi``
and the functions below. The expression is parsed and checked once; nothing else is evaluated.
``PERIODIC`` is accepted and ignored (values are not wrapped).
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path

from estk.errors import EstError
from estk.io.plumed import Action

_FUNCTIONS = {
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "erf": math.erf, "erfc": math.erfc, "abs": abs, "min": min, "max": max,
    "floor": math.floor, "ceil": math.ceil, "square": lambda v: v * v,
    "cube": lambda v: v * v * v, "recip": lambda v: 1.0 / v,
    "step": lambda v: 1.0 if v >= 0 else 0.0,
    "delta": lambda v: 1.0 if v == 0 else 0.0,
    "select": lambda c, a, b: a if c != 0 else b,
    "sec": lambda v: 1.0 / math.cos(v), "csc": lambda v: 1.0 / math.sin(v),
    "cot": lambda v: 1.0 / math.tan(v),
}  # fmt: skip
_CONSTANTS = {"pi": math.pi}
_OPERATORS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a**b,
}  # fmt: skip


@dataclass
class Function:
    """A value computed from other values (``args``, by name)."""

    args: list[str]
    kind: str  # "combine" or "custom"
    coefficients: list[float] = field(default_factory=list)
    parameters: list[float] = field(default_factory=list)
    powers: list[float] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    tree: ast.Expression | None = None

    def value(self, values: dict[str, float]) -> float:
        x = [values[a] for a in self.args]
        if self.kind == "combine":
            terms = zip(x, self.coefficients, self.parameters, self.powers, strict=True)
            return float(sum(c * (v - p) ** n for v, c, p, n in terms))
        names = dict(zip(self.variables, x, strict=True))
        try:
            return float(_evaluate(self.tree.body, names))  # type: ignore[union-attr]
        except (ValueError, ZeroDivisionError, OverflowError):
            return math.nan


def function_from_action(action: Action, path: Path) -> Function:
    where = f"{path}: {action.name} {action.label}"
    args = action.get_list("ARG")
    if not args:
        raise EstError(f"{where}: needs ARG")
    if action.name == "COMBINE":

        def numbers(key: str, default: float) -> list[float]:
            raw = action.get_list(key)
            if not raw:
                return [default] * len(args)
            if len(raw) != len(args):
                raise EstError(f"{where}: {key} needs one value per ARG ({len(args)})")
            try:
                return [float(v) for v in raw]
            except ValueError:
                raise EstError(f"{where}: {key}={','.join(raw)}: not numbers") from None

        coefficients = numbers("COEFFICIENTS", 1.0)
        if "NORMALIZE" in action.flags:
            total = sum(coefficients)
            coefficients = [c / total for c in coefficients]
        return Function(
            args,
            "combine",
            coefficients=coefficients,
            parameters=numbers("PARAMETERS", 0.0),
            powers=numbers("POWERS", 1.0),
        )
    # CUSTOM
    variables = action.get_list("VAR")
    if not variables:
        if len(args) > 4:
            raise EstError(f"{where}: more than 4 ARG: name them with VAR=")
        variables = ["x", "y", "z", "t"][: len(args)]
    if len(variables) != len(args):
        raise EstError(f"{where}: VAR needs one name per ARG ({len(args)})")
    func = action.get("FUNC")
    if not func:
        raise EstError(f"{where}: needs FUNC")
    try:
        tree = ast.parse(func.replace("^", "**"), mode="eval")
    except SyntaxError:
        raise EstError(f"{where}: FUNC={func}: can't parse it") from None
    _check(tree.body, set(variables), where, func)
    return Function(args, "custom", variables=variables, tree=tree)


def _check(node: ast.AST, variables: set[str], where: str, func: str) -> None:
    """Allow only numbers, the variables, constants, arithmetic and the known functions."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return
    if isinstance(node, ast.Name):
        if node.id in variables or node.id in _CONSTANTS:
            return
        raise EstError(f"{where}: FUNC={func}: unknown name '{node.id}' "
                       f"(variables: {', '.join(sorted(variables))})")  # fmt: skip
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        _check(node.left, variables, where, func)
        _check(node.right, variables, where, func)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        _check(node.operand, variables, where, func)
        return
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):  # fmt: skip
        for a in node.args:
            _check(a, variables, where, func)
        return
    raise EstError(f"{where}: FUNC={func}: only numbers, variables, + - * / ^ and functions "
                   f"({', '.join(sorted(_FUNCTIONS))}) are supported")  # fmt: skip


def _evaluate(node: ast.AST, names: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return names[node.id] if node.id in names else _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp):
        return _OPERATORS[type(node.op)](_evaluate(node.left, names), _evaluate(node.right, names))
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, names)
        return -value if isinstance(node.op, ast.USub) else value
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    return _FUNCTIONS[node.func.id](*(_evaluate(a, names) for a in node.args))
