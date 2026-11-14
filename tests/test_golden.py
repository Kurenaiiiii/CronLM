import json
import os

from cronlm.cron_spec import is_canonical, is_valid_cron
from cronlm.tokenizer import ALLOWED_CHARS

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.jsonl")


def test_golden_set_shape_and_validity():
    assert os.path.exists(GOLDEN)
    rows = [json.loads(l) for l in open(GOLDEN, encoding="utf-8") if l.strip()]
    assert len(rows) == 200

    seen = set()
    allowed = set(ALLOWED_CHARS)
    for row in rows:
        phrase, cron = row["phrase"], row["cron"]
        assert phrase not in seen, f"duplicate golden phrase: {phrase!r}"
        seen.add(phrase)
        assert is_valid_cron(cron), f"invalid golden cron: {phrase!r} -> {cron!r}"
        assert is_canonical(cron), f"non-canonical golden cron: {phrase!r} -> {cron!r}"
        assert all(c in allowed for c in phrase)
        assert phrase == phrase.lower()
