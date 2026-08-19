"""Field-level scoring for structured document extraction.

Why not string similarity on the raw JSON: a model that emits every field
correctly but orders the keys differently, or wraps prices in quotes, would score
badly under BLEU or edit distance while being perfectly usable downstream. And a
model that emits a syntactically pretty object with the wrong total would score
well. Both are the wrong signal.

So the unit of measurement here is the (flattened key, normalised value) pair.
Predictions and targets are flattened to dotted paths, normalised, and compared
as multisets to get precision, recall, and F1 over fields. Exact-match over the
whole record is reported alongside as the strict view.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

_WS = re.compile(r"\s+")
_CURRENCY = re.compile(r"[^\d.,\-]")


def flatten(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """Flatten nested dicts/lists into (dotted_path, leaf_value) pairs.

    List indices are included in the path. That is a deliberate strictness: for
    receipts, line-item order is meaningful, and a model that returns the right
    items in the wrong order has made a real error.
    """
    out: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(flatten(v, f"{prefix}[{i}]"))
    else:
        out.append((prefix, "" if obj is None else str(obj)))
    return out


_NUMERIC_LEAVES = ("price", "total", "cnt", "num", "amount", "cash", "change")


def _canonical_number(raw: str) -> str | None:
    """Parse a human-formatted number without assuming a locale.

    CORD is Indonesian, where `15.000` is fifteen thousand — the dot is a
    thousands separator, not a decimal point. Hardcoding either convention
    silently corrupts one of them, so the separator role is inferred from the
    digit grouping instead:

      - both `.` and `,` present  -> whichever comes last is the decimal mark
      - one separator, repeated   -> thousands grouping
      - one separator, once, with exactly 3 digits after -> thousands grouping
      - otherwise                 -> decimal mark

    `1.234,56` and `1,234.56` both land on `1234.56`; `15.000` and `15,000`
    both land on `15000`. The one genuinely ambiguous case, a single separator
    with three trailing digits, resolves to thousands — correct for this corpus
    and documented here so the assumption is visible rather than buried.
    """
    s = _CURRENCY.sub("", raw)
    if s in ("", "-", ".", ","):
        return None

    neg = s.startswith("-")
    s = s.lstrip("-")
    has_dot, has_comma = "." in s, "," in s

    if has_dot and has_comma:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        s = s.replace("," if dec == "." else ".", "").replace(dec, ".")
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        tail = s.rsplit(sep, 1)[1]
        if s.count(sep) > 1 or len(tail) == 3:
            s = s.replace(sep, "")          # thousands grouping
        else:
            s = s.replace(sep, ".")         # decimal mark

    try:
        f = float(s)
    except ValueError:
        return None
    if neg:
        f = -f
    return str(int(f)) if f.is_integer() else str(f)


def normalise_value(key: str, value: str) -> str:
    """Collapse formatting differences that do not change meaning.

    Numeric fields are canonicalised so that `Rp 25.000`, `25,000` and `25000`
    compare equal. Penalising the model for presentation would be measuring the
    tokenizer, not the extraction.
    """
    v = _WS.sub(" ", value.strip()).lower()
    leaf = key.rsplit(".", 1)[-1]
    if any(t in leaf for t in _NUMERIC_LEAVES):
        canon = _canonical_number(v)
        if canon is not None:
            return canon
    return v


def normalise_record(obj: Any) -> Counter:
    return Counter(
        (k, normalise_value(k, v)) for k, v in flatten(obj) if v.strip() != ""
    )


def parse_prediction(text: str) -> Tuple[Any, bool]:
    """Recover a JSON object from raw model output.

    Returns (object, parsed_ok). A model that emits unparseable output scores
    zero on every field rather than crashing the eval — malformed JSON is a real
    failure mode and gets counted as one, not silently dropped from the average.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1]), True
        except json.JSONDecodeError:
            pass
    return {}, False


@dataclass
class Scores:
    n: int
    field_precision: float
    field_recall: float
    field_f1: float
    exact_match: float
    parse_rate: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def score(predictions: List[str], targets: List[Any]) -> Scores:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must be the same length")

    tp = fp = fn = 0
    exact = parsed = 0

    for pred_text, target in zip(predictions, targets):
        pred_obj, ok = parse_prediction(pred_text)
        parsed += int(ok)

        gold = normalise_record(target)
        got = normalise_record(pred_obj) if ok else Counter()

        overlap = sum((gold & got).values())
        tp += overlap
        fp += sum(got.values()) - overlap
        fn += sum(gold.values()) - overlap
        exact += int(ok and got == gold)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = len(targets)

    return Scores(
        n=n,
        field_precision=round(precision, 4),
        field_recall=round(recall, 4),
        field_f1=round(f1, 4),
        exact_match=round(exact / n, 4) if n else 0.0,
        parse_rate=round(parsed / n, 4) if n else 0.0,
    )


def _self_test() -> None:
    cases = {
        "Rp 15.000": "15000",   # Indonesian thousands
        "25,000": "25000",      # English thousands
        "1.234,56": "1234.56",  # Indonesian decimal
        "1,234.56": "1234.56",  # English decimal
        "15.5": "15.5",         # genuine decimal, 1 trailing digit
        "-2.000": "-2000",
        "3": "3",
    }
    for raw, want in cases.items():
        got = normalise_value("total.total_price", raw)
        assert got == want, f"{raw!r} -> {got!r}, wanted {want!r}"

    gold = {"total": {"total_price": "25,000"}, "menu": [{"nm": "Kopi", "price": "Rp 15.000"}]}
    perfect = json.dumps({"total": {"total_price": "25000"}, "menu": [{"nm": "kopi", "price": "15000"}]})
    s = score([perfect], [gold])
    assert s.field_f1 == 1.0 and s.exact_match == 1.0, s.as_dict()

    half = json.dumps({"total": {"total_price": "25000"}, "menu": [{"nm": "teh", "price": "15000"}]})
    assert 0.0 < score([half], [gold]).field_f1 < 1.0

    broken = score(["here is the json: {oops"], [gold])
    assert broken.parse_rate == 0.0 and broken.field_f1 == 0.0

    print("metrics self-test passed")


if __name__ == "__main__":
    _self_test()
