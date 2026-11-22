import pytest

from auditenv.env.calculator import CalcError, evaluate


@pytest.mark.parametrize("expr,expected", [
    ("1+2", "3"), ("(1200.50*3)/7", "514.5"), ("-5 + 2", "-3"), ("0.1+0.2", "0.3"),
    ("123456789012345678 * 10", "1234567890123456780"), ("10/4", "2.5"), ("2-2", "0"), ("(" * 50 + "1" + ")" * 50, "1"),
])
def test_arithmetic_is_exact_decimal(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os')", "open('x')", "a+1", "2**10", "7//2", "7%2", "1e5", "'1'+'2'", "[1,2]",
    "", "   ", "1/0", "1" * 201, "1+" * 70 + "1", "lambda: 1", "1 if 1 else 2",
])
def test_rejects_code_files_and_abuse(expr):
    with pytest.raises(CalcError):
        evaluate(expr)


def test_rejects_non_string():
    with pytest.raises(CalcError):
        evaluate(None)
