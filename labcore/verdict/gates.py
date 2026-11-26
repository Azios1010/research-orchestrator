"""A tiny, safe parser for gate expressions. Nothing here uses eval or exec.

Grammar::

    expr       := and_expr ("or" and_expr)*
    and_expr   := not_expr ("and" not_expr)*
    not_expr   := "not" not_expr | comparison
    comparison := operand (("<" | "<=" | ">" | ">=" | "==" | "!=") operand)?
    operand    := NUMBER | "-" NUMBER | FIELD | "(" expr ")"
    FIELD      := name | name "(" arm ")" | name "(" arm "-" arm ")"

A FIELD such as ``ci_lo(raw-maps)`` is ONE literal scorecard key. It is not a function call:
the text is looked up as-is in the scorecard.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_LENGTH = 500
MAX_DEPTH = 40
KEYWORDS = ("and", "or", "not")
COMPARATORS = ("<=", ">=", "==", "!=", "<", ">")

_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<keyword>(?:and|or|not)(?![A-Za-z0-9_]))
  | (?P<field>[A-Za-z_][A-Za-z0-9_]*(?:\([A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)?\))?)
  | (?P<cmp><=|>=|==|!=|<|>)
  | (?P<lparen>\()
  | (?P<rparen>\))
  | (?P<minus>-)
    """,
    re.VERBOSE,
)


class GateError(ValueError):
    """The gate expression is invalid, or it cannot be evaluated on this scorecard."""


@dataclass(frozen=True)
class Node:
    kind: str  # "or" | "and" | "not" | "cmp" | "num" | "field"
    value: Any = None
    children: tuple[Node, ...] = ()


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if not match:
            raise GateError(f"unexpected character {text[position]!r} at position {position}")
        kind = match.lastgroup or ""
        if kind != "space":
            tokens.append((kind, match.group()))
        position = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.position = 0
        self.depth = 0

    def peek(self) -> tuple[str, str]:
        return self.tokens[self.position] if self.position < len(self.tokens) else ("end", "")

    def take(self) -> tuple[str, str]:
        token = self.peek()
        self.position += 1
        return token

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise GateError("expression is nested too deeply")

    def expr(self) -> Node:
        self.enter()
        parts = [self.and_expr()]
        while self.peek() == ("keyword", "or"):
            self.take()
            parts.append(self.and_expr())
        self.depth -= 1
        return parts[0] if len(parts) == 1 else Node("or", children=tuple(parts))

    def and_expr(self) -> Node:
        parts = [self.not_expr()]
        while self.peek() == ("keyword", "and"):
            self.take()
            parts.append(self.not_expr())
        return parts[0] if len(parts) == 1 else Node("and", children=tuple(parts))

    def not_expr(self) -> Node:
        if self.peek() == ("keyword", "not"):
            self.take()
            self.enter()
            child = self.not_expr()
            self.depth -= 1
            return Node("not", children=(child,))
        return self.comparison()

    def comparison(self) -> Node:
        left = self.operand()
        if self.peek()[0] != "cmp":
            return left
        operator = self.take()[1]
        right = self.operand()
        if self.peek()[0] == "cmp":
            raise GateError("chained comparisons are not allowed; join them with 'and'")
        return Node("cmp", operator, (left, right))

    def operand(self) -> Node:
        kind, text = self.take()
        if kind == "number":
            return Node("num", float(text))
        if kind == "minus":
            kind, text = self.take()
            if kind != "number":
                raise GateError("'-' is only allowed directly before a number")
            return Node("num", -float(text))
        if kind == "field":
            if self.peek()[0] == "lparen":
                raise GateError(f"{text!r} cannot be called: gates may only name scorecard fields")
            return Node("field", text)
        if kind == "lparen":
            inner = self.expr()
            if self.take()[0] != "rparen":
                raise GateError("missing ')'")
            return inner
        raise GateError(f"expected a number, a field or '(', got {text or 'end of expression'!r}")


def parse(text: str) -> Node:
    if not isinstance(text, str) or not text.strip():
        raise GateError("gate expression is empty")
    if len(text) > MAX_LENGTH:
        raise GateError(f"gate expression is longer than {MAX_LENGTH} characters")
    parser = _Parser(_tokenize(text))
    node = parser.expr()
    if parser.peek()[0] != "end":
        raise GateError(f"unexpected {parser.peek()[1]!r} after the expression")
    if node.kind in ("num",):
        raise GateError("a gate must be a boolean expression, not a number")
    return node


def fields(node: Node) -> set[str]:
    if node.kind == "field":
        return {node.value}
    found: set[str] = set()
    for child in node.children:
        found |= fields(child)
    return found


def _boolean(node: Node, values: Mapping[str, Any]) -> bool:
    if node.kind == "or":
        return any([_boolean(child, values) for child in node.children])
    if node.kind == "and":
        return all([_boolean(child, values) for child in node.children])
    if node.kind == "not":
        return not _boolean(node.children[0], values)
    if node.kind == "cmp":
        left, right = (_number(child, values) for child in node.children)
        return {
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
            "==": left == right,
            "!=": left != right,
        }[node.value]
    if node.kind == "field" and isinstance(values[node.value], bool):
        return values[node.value]
    raise GateError(f"expected a true/false value, got {node.value!r}")


def _number(node: Node, values: Mapping[str, Any]) -> float:
    if node.kind == "num":
        return node.value
    if node.kind == "field":
        value = values[node.value]
        if isinstance(value, bool):
            raise GateError(f"{node.value} is true/false and cannot be compared with a number")
        return float(value)
    raise GateError("comparisons need numbers or numeric scorecard fields on both sides")


def evaluate(expression: str | Node, values: Mapping[str, Any]) -> bool:
    """Evaluate a gate. Every named field must exist and be finite, even in unused branches."""
    node = parse(expression) if isinstance(expression, str) else expression
    for name in sorted(fields(node)):
        if name not in values:
            raise GateError(f"scorecard has no field {name!r}")
        value = values[name]
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise GateError(f"scorecard field {name!r} is not a finite number: {value!r}")
    return _boolean(node, values)
