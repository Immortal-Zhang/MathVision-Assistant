from __future__ import annotations

from mathvision.evaluation.metrics import (
    exact_match,
    keyword_coverage,
    numeric_match,
)


def test_exact_match_allows_structured_answer() -> None:
    assert exact_match("答案：2\n依据：line slope", "2") == 1.0
    assert exact_match("positive correlation", "positive correlation") == 1.0
    assert exact_match("negative correlation", "positive correlation") == 0.0


def test_numeric_match_supports_decimal_fraction_and_percent() -> None:
    assert numeric_match("the value is 70 degrees", "70") == 1.0
    assert numeric_match("simplified result is 3/4", "0.75") == 1.0
    assert numeric_match("share is 50%", "0.5") == 1.0
    assert numeric_match("answer is 2.001", "2.0005", tol=1e-2) == 1.0


def test_keyword_coverage() -> None:
    pred = "The chart shows positive correlation."
    assert keyword_coverage(pred, ["positive", "correlation"]) == 1.0
    assert keyword_coverage(pred, ["positive", "slope"]) == 0.5
    assert keyword_coverage(pred, []) == 1.0
