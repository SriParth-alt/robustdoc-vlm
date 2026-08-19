"""Schema-free value scoring — a diagnostic companion to field-level F1.

Field F1 in `metrics.py` compares `(dotted.path, value)` pairs, so it answers
"did the model produce the target schema, correctly filled in?". That is the
right primary metric, and it stays the headline. But it collapses two very
different failure modes onto the same number:

    cannot read the receipt at all          -> field F1 0.0
    reads it perfectly, names keys its way  -> field F1 0.0

For a fine-tuned model that collapse is harmless: once the schema is learned,
field F1 is measuring reading ability. For an un-fine-tuned model it is total.
Measured on CORD test samples, base Qwen2-VL-2B emits well-formed JSON with
correct values under its own key names — `items[].name` where CORD wants
`menu[].nm` — and scores exactly 0.000. Every corruption condition then also
scores 0.000, and a robustness sweep whose baseline is pinned at the floor
cannot show degradation at all.

So this module ignores keys entirely and asks only: **which of the target's
values did the model recover, anywhere in its output?** Values are compared as a
multiset, so emitting a value twice does not earn double credit, and a model that
dumps many spurious values is caught by precision rather than flattered by recall.

This is deliberately NOT the headline metric. It cannot tell a correctly
structured record from a bag of right-looking strings, which is exactly why
`metrics.py` exists. Reported alongside, never instead. See DECISIONS.md #14 for
why it was added and what that costs in credibility.

Numeric normalisation is delegated to `metrics._canonical_number` rather than
reimplemented, so the Indonesian-locale handling has one definition and cannot
drift between the two metrics (see DECISIONS.md #5).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from .metrics import _canonical_number, flatten, parse_prediction

_WS = re.compile(r"\s+")
# Only canonicalise as a number when the whole string is numeric. Without this
# guard a product code like "A3" would have its letters stripped and be compared
# as the number 3, inventing matches that are not there.
_PURE_NUMERIC = re.compile(r"^-?[\d.,]+$")
_CURRENCY_WORDS = ("rp.", "rp", "idr", "$")


def normalise_value(value: str) -> str:
    """Key-agnostic value normalisation.

    Applied identically to prediction and target, so the comparison is internally
    consistent even where it differs from `metrics.normalise_value` (which can
    consult the key to decide whether a field is numeric, and which this cannot).
    """
    s = _WS.sub(" ", value.strip()).lower()
    core = s
    for w in _CURRENCY_WORDS:
        if core.startswith(w):
            core = core[len(w):].strip()
    core = core.replace(" ", "")
    if _PURE_NUMERIC.match(core):
        canon = _canonical_number(core)
        if canon is not None:
            return canon
    return s


def value_multiset(obj: Any) -> Counter:
    """Every non-empty leaf value in the record, keys discarded."""
    return Counter(
        normalise_value(v) for _, v in flatten(obj) if v.strip() != ""
    )


@dataclass
class ValueScores:
    n: int
    value_precision: float
    value_recall: float
    value_f1: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def value_score(predictions: List[str], targets: List[Any]) -> ValueScores:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must be the same length")

    tp = fp = fn = 0
    for pred_text, target in zip(predictions, targets):
        pred_obj, ok = parse_prediction(pred_text)
        gold = value_multiset(target)
        got = value_multiset(pred_obj) if ok else Counter()
        overlap = sum((gold & got).values())
        tp += overlap
        fp += sum(got.values()) - overlap
        fn += sum(gold.values()) - overlap

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ValueScores(
        n=len(targets),
        value_precision=round(precision, 4),
        value_recall=round(recall, 4),
        value_f1=round(f1, 4),
    )


def _self_test() -> None:
    from .metrics import score

    # Normalisation: numbers canonicalise across locales, non-numeric strings do
    # not get mangled into numbers.
    assert normalise_value("Rp 15.000") == "15000"
    assert normalise_value("15,000") == "15000"
    assert normalise_value("1.234,56") == "1234.56"
    assert normalise_value("A3") == "a3", normalise_value("A3")
    assert normalise_value("2016-01-16") == "2016-01-16"
    assert normalise_value("  KOPI  susu ") == "kopi susu"

    gold = {
        "menu": [{"nm": "Kopi", "price": "Rp 15.000"}],
        "total": {"total_price": "25,000"},
    }

    # The case this metric exists for: right values, entirely different schema.
    # Field F1 must be 0.0 while value recall is 1.0 — if this ever stops being
    # true the two metrics are no longer measuring different things.
    off_schema = json.dumps(
        {"items": [{"name": "Kopi", "cost": "15000"}], "sum": "25000"}
    )
    assert score([off_schema], [gold]).field_f1 == 0.0
    vs = value_score([off_schema], [gold])
    assert vs.value_recall == 1.0, vs.as_dict()
    assert vs.value_precision == 1.0, vs.as_dict()

    # Partial recovery: two of three values present.
    partial = json.dumps({"a": "Kopi", "b": "15000"})
    assert value_score([partial], [gold]).value_recall == round(2 / 3, 4)

    # Spurious extra values cost precision, not recall.
    noisy = json.dumps({"a": "Kopi", "b": "15000", "c": "25000",
                        "d": "999", "e": "888"})
    n = value_score([noisy], [gold])
    assert n.value_recall == 1.0 and n.value_precision == 0.6, n.as_dict()

    # Multiset semantics: repeating a value does not earn double credit.
    dup = json.dumps({"a": "Kopi", "b": "Kopi", "c": "Kopi"})
    assert value_score([dup], [gold]).value_recall == round(1 / 3, 4)

    # Unparseable output scores zero here too, same as field F1.
    assert value_score(["not json at all {"], [gold]).value_recall == 0.0

    print("value_metrics self-test passed")


if __name__ == "__main__":
    _self_test()
