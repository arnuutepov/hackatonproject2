"""Safe calculator: decimal arithmetic only. No names, calls, attributes, power or strings."""

from __future__ import annotations

import ast
import re
from decimal import ROUND_HALF_EVEN, Context, Decimal, DivisionByZero, InvalidOperation, Overflow, localcontext

MAX_LEN = 200
MAX_NODES = 120
MAX_LITERAL_DIGITS = 30
PRECISION = 60  # significant digits; division results are rounded half-even at this precision
_ALLOWED_CHARS = re.compile(r"[0-9+\-*/(). ]+")  # used with fullmatch: no trailing newline slips through
_OPS = {ast.Add: Decimal.__add__, ast.Sub: Decimal.__sub__, ast.Mult: Decimal.__mul__,
        ast.Div: Decimal.__truediv__}


class CalcError(ValueError):
    pass


def evaluate(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise CalcError("expression must be a non-empty string")
    if len(expression) > MAX_LEN:
        raise CalcError(f"expression longer than {MAX_LEN} characters")
    if not _ALLOWED_CHARS.fullmatch(expression):
        raise CalcError("only digits, '.', '+', '-', '*', '/', '(', ')' and spaces are allowed")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise CalcError("invalid arithmetic expression") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise CalcError("expression too complex")

    def ev(node) -> Decimal:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression, node)
            if literal is None or len(literal.replace(".", "")) > MAX_LITERAL_DIGITS:
                raise CalcError("number literal too long")
            return Decimal(literal)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            value = _OPS[type(node.op)](ev(node.left), ev(node.right))
            if value and value.adjusted() >= PRECISION:
                # More integer digits than the precision: the value could only be returned rounded
                # in its integer part, i.e. wrong. A bounded error instead (hard-jury audit, item 7).
                raise CalcError(f"result too large: more than {PRECISION} integer digits")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = ev(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        raise CalcError("unsupported syntax")

    # One explicit context for evaluation AND formatting: the result never depends on the caller's
    # decimal context (audit F11). 60 significant digits keep products of two 30-digit literals exact.
    with localcontext(Context(prec=PRECISION, rounding=ROUND_HALF_EVEN,
                              traps=[DivisionByZero, InvalidOperation, Overflow])):
        try:
            result = ev(tree)
            text = format(result.normalize(), "f")
        except (DivisionByZero, InvalidOperation, Overflow):
            raise CalcError("division by zero or invalid operation") from None
    return "0" if text in ("-0", "") else text
