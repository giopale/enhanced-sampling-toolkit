"""Frame selections: conditions on CVs that restrict an analysis to some frames.

A selection is an expression such as ``(cn - 1.5) * indirect_delta_B < 0.05`` or
``1.8 < d0 < 2.3 and cn > 1.5``: numbers, CV names, ``+ - * / ^`` (or ``**``), comparisons (also
chained), ``and``/``or``/``not``, parentheses, ``pi``, and the functions ``abs sqrt exp log sin
cos tan min max``, and ``isnan(x)`` / ``defined(x)`` (true where x is undefined, e.g. a
proton-transfer CV without a chain, or defined). A frame is selected where the expression is
true (or non-zero).

CV names: as they are if they are Python-like names (``cn``, ``d0``), ``dOOliq.min`` as written,
names with other characters with ``_`` in their place (``indirect-delta-B`` as
``indirect_delta_B``) or in braces (``{indirect-delta-B}``). The expression is parsed and
checked against a whitelist; nothing else is evaluated. Frames where a CV in the expression
is NaN are not selected, unless the expression asks for them with ``isnan``.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

import numpy as np

from estk.errors import EstError

_FUNCTIONS = {
    "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "sin": np.sin,
    "cos": np.cos, "tan": np.tan, "min": np.minimum, "max": np.maximum,
    "isnan": np.isnan, "defined": np.isfinite,
}  # fmt: skip
_CONSTANTS = {"pi": np.pi}
_BINARY = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
    ast.Pow: np.power, ast.Mod: np.mod,
}  # fmt: skip
_COMPARE = {
    ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
    ast.Eq: np.equal, ast.NotEq: np.not_equal,
}  # fmt: skip


def _key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


@dataclass
class Selection:
    """A parsed selection; ``names`` are the CV names it uses (after ``resolve``)."""

    text: str
    label: str | None = None  # its name in estk.yaml, if it has one
    _tree: ast.Expression | None = field(default=None, repr=False)
    _variables: dict[str, str] = field(default_factory=dict, repr=False)  # in tree -> CV name

    @classmethod
    def parse(cls, text: str, label: str | None = None) -> Selection:
        braced: dict[str, str] = {}

        def placeholder(match: re.Match[str]) -> str:
            name = f"__cv{len(braced)}"
            braced[name] = match.group(1).strip()
            return name

        source = re.sub(r"\{([^{}]+)\}", placeholder, text).replace("^", "**")
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except SyntaxError:
            raise EstError(f"selection {text!r}: can't parse it") from None
        variables: dict[str, str] = {}
        _check(tree.body, text, variables)
        for name, original in braced.items():
            if name in variables:
                variables[name] = original
        return cls(text=text, label=label, _tree=tree, _variables=variables)

    @property
    def names(self) -> list[str]:
        return list(dict.fromkeys(self._variables.values()))

    def resolve(self, available: Collection[str]) -> Selection:
        """Map ``_``-names to the CVs they stand for (``indirect_delta_B`` ->
        ``indirect-delta-B``) when that CV is in ``available`` and the name itself is not."""
        by_key: dict[str, list[str]] = {}
        for name in available:
            by_key.setdefault(_key(name), []).append(name)
        for var, name in self._variables.items():
            if name in available or name not in by_key:
                continue
            matches = by_key[name]
            if len(matches) > 1:
                raise EstError(
                    f"selection {self.text!r}: {name} could be any of {', '.join(matches)}; "
                    f"write it in braces, e.g. {{{matches[0]}}}"
                )
            self._variables[var] = matches[0]
        return self

    def mask(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        """True for the selected frames; ``columns`` holds every CV in ``names``."""
        missing = [n for n in self.names if n not in columns]
        if missing:
            raise EstError(f"selection {self.text!r}: {', '.join(missing)} not loaded")
        values = {var: np.asarray(columns[name], dtype=float)
                  for var, name in self._variables.items()}  # fmt: skip
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            result = _evaluate(self._tree.body, values)  # type: ignore[union-attr]
        result = np.asarray(result)
        if result.dtype != bool:
            result = np.nan_to_num(result.astype(float), nan=0.0) != 0
        n = len(next(iter(values.values()))) if values else 1
        result = np.broadcast_to(result, (n,)).copy()
        if not self.asks_for_undefined:  # an undefined CV never selects a frame (not even
            for v in values.values():  # through "not"), unless asked for with isnan/defined
                result &= np.isfinite(v)
        return result

    @property
    def asks_for_undefined(self) -> bool:
        """Whether the expression tests for undefined values (``isnan`` or ``defined``)."""
        return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in ("isnan", "defined")
                   for node in ast.walk(self._tree))  # type: ignore[arg-type]  # fmt: skip

    def describe(self) -> str:
        return f"{self.label}: {self.text}" if self.label else self.text


def _dotted(node: ast.AST) -> str | None:
    """``dOOliq.min`` (an attribute chain of names) -> "dOOliq.min"."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _check(node: ast.AST, text: str, variables: dict[str, str]) -> None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):  # fmt: skip
        return
    if isinstance(node, (ast.Name, ast.Attribute)):
        name = _dotted(node)
        if name is None:
            raise EstError(f"selection {text!r}: not understood near {ast.unparse(node)!r}")
        if name not in _CONSTANTS:
            variables[ast.unparse(node)] = name
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        _check(node.left, text, variables)
        _check(node.right, text, variables)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        _check(node.operand, text, variables)
        return
    if isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
        for part in (node.left, *node.comparators):
            _check(part, text, variables)
        return
    if isinstance(node, ast.BoolOp):
        for part in node.values:
            _check(part, text, variables)
        return
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):  # fmt: skip
        for part in node.args:
            _check(part, text, variables)
        return
    raise EstError(
        f"selection {text!r}: only numbers, CV names, + - * / ^, comparisons, and/or/not and "
        f"the functions {', '.join(_FUNCTIONS)} are allowed"
    )


def _evaluate(node: ast.AST, values: Mapping[str, np.ndarray]) -> np.ndarray | float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, (ast.Name, ast.Attribute)):
        key = ast.unparse(node)
        return values[key] if key in values else _CONSTANTS[_dotted(node)]  # type: ignore[index]
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](_evaluate(node.left, values), _evaluate(node.right, values))
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand, values)
        if isinstance(node.op, ast.Not):
            return np.logical_not(_truth(operand))
        return np.negative(operand) if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.Compare):
        result: np.ndarray | bool = True
        left = _evaluate(node.left, values)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate(comparator, values)
            result = np.logical_and(result, _COMPARE[type(op)](left, right))
            left = right
        return result  # type: ignore[return-value]
    if isinstance(node, ast.BoolOp):
        parts = [_truth(_evaluate(v, values)) for v in node.values]
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        result = parts[0]
        for part in parts[1:]:
            result = combine(result, part)
        return result
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    return _FUNCTIONS[node.func.id](*(_evaluate(a, values) for a in node.args))


def _truth(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype == bool:
        return value
    return np.nan_to_num(value.astype(float), nan=0.0) != 0


def selection_from(text: str, named: Mapping[str, str] | None = None) -> Selection:
    """``text`` is a name in ``named`` (``selections:`` of estk.yaml) or an expression."""
    if named and text in named:
        return Selection.parse(named[text], label=text)
    return Selection.parse(text)
