import builtins
import math

import pytest

from labcore.verdict.gates import GateError, evaluate, fields, parse
from labcore.verdict.verdict import compute_verdict

GO = "mean(raw-maps) > 0.02 and ci_lo(raw-N1) > 0 and mde149(raw-maps) <= 0.03"
PARK = "ci_hi(raw-maps) < 0 or (mean(raw-maps) <= 0 and mde149(raw-maps) > 0.03)"


def card(mean=0.03, hi=0.06, lo_n1=0.01, mde=0.02):
    return {"mean(raw-maps)": mean, "ci_hi(raw-maps)": hi, "ci_lo(raw-N1)": lo_n1, "mde149(raw-maps)": mde}


def test_contrast_names_are_single_fields_not_calls():
    assert fields(parse(GO)) == {"mean(raw-maps)", "ci_lo(raw-N1)", "mde149(raw-maps)"}


@pytest.mark.parametrize(
    "values, verdict",
    [
        (card(), "GO"),
        (card(mean=0.02), "REVISE"),  # exactly 0.02 is not "> 0.02"
        (card(mde=0.03), "GO"),  # exactly 0.03 is allowed
        (card(mde=0.0301), "REVISE"),
        (card(lo_n1=0.0), "REVISE"),
        (card(mean=-0.04, hi=-0.01, mde=0.02), "PARK"),  # maps strictly better
        (card(mean=-0.04, hi=-0.01, mde=0.05), "PARK"),
        (card(mean=-0.01, hi=0.03, mde=0.05), "PARK"),  # not ahead, and too noisy
        (card(mean=0.0, hi=0.03, mde=0.05), "PARK"),
        (card(mean=0.0, hi=0.0, mde=0.02), "REVISE"),  # an upper bound of exactly 0 is not "< 0"
        (card(mean=-0.005, hi=0.02, mde=0.02), "REVISE"),  # not ahead, but precise
        (card(mean=0.01, hi=0.05, mde=0.05), "REVISE"),
    ],
)
def test_isles_gates(values, verdict):
    result = compute_verdict({"GO": GO, "PARK": PARK}, {"pipeline_ok": True, "metrics": values})
    assert result.verdict == verdict
    assert result.notes == []  # GO and PARK never both hold here


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("a > 1", True),
        ("a >= 2 and a <= 2", True),
        ("a != 2 or b == -0.5", True),
        ("not a < 1", True),
        ("not (a > 1 and b > 0)", True),
        ("a > 3 or a > 1 and b < 0", True),
        ("(a > 3 or a > 1) and b > 0", False),
        ("flag and a > 1e-3", True),
        ("not flag", False),
    ],
)
def test_boolean_logic(expression, expected):
    assert evaluate(expression, {"a": 2.0, "b": -0.5, "flag": True}) is expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned') > 0",
        "__import__(os) > 0 or exec(x)",
        "open('/etc/passwd').read()",
        "a.__class__ > 0",
        "a > 0; import os",
        "lambda: 1",
        "a if a else b",
        "[a for a in b]",
        "a[0] > 1",
        "ci_lo(raw-maps)(1) > 0",
        "max(a, b) > 0",
        "exit() > 0",
        "a + 1 > 0",
        "a * 2 > 0",
        "a > 0 > b",
        "-a > 0",
        "a = 1",
        "a > '0'",
        "a >",
        "(a > 1",
        "a > 1)",
        "5",
        "",
        "   ",
        "a > 1 and " * 100 + "a > 1",
        "(" * 100 + "a > 1" + ")" * 100,
        "not " * 100 + "flag",
    ],
)
def test_rejects_anything_that_is_not_a_gate(expression, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the gate parser must never evaluate Python")

    node = None
    with monkeypatch.context() as patched:
        patched.setattr(builtins, "eval", forbidden)
        patched.setattr(builtins, "exec", forbidden)
        try:
            node = parse(expression)
        except GateError:
            return
    # The few that parse (a bare or unknown name) must still fail on evaluation.
    with pytest.raises(GateError):
        evaluate(node, {"a": 1.0, "b": 1.0, "flag": True})


def test_every_named_field_is_required_even_in_a_branch_that_is_not_needed():
    with pytest.raises(GateError, match="no field 'missing'"):
        evaluate("a < 0 and missing > 0", {"a": 1.0})


@pytest.mark.parametrize("bad", [math.nan, math.inf, None, "0.5"])
def test_non_finite_values_never_produce_a_result(bad):
    with pytest.raises(GateError):
        evaluate("a > 0 or b > 0", {"a": 1.0, "b": bad})


def test_types_are_not_mixed():
    with pytest.raises(GateError):
        evaluate("flag > 0", {"flag": True})
    with pytest.raises(GateError):
        evaluate("a and flag", {"a": 1.0, "flag": True})
