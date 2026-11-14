import pytest
from datetime import datetime

from cronlm.cron_spec import (
    CronSpec, CronError, canonical_cron, differing_fields, is_canonical,
    is_valid_cron, parse_cron, semantic_equal, next_runs,
)


# ---------- formatting / canonicalization

@pytest.mark.parametrize("values,expected", [
    ((5,), "5"),
    ((0, 6), "0,6"),
    ((1, 2), "1,2"),
    ((1, 2, 3), "1-3"),
    ((1, 2, 3, 5), "1-3,5"),
    ((1, 3, 5), "1,3,5"),
    ((0,), "0"),
])
def test_format_field_simple(values, expected):
    from cronlm.cron_spec import format_field
    assert format_field(values, 0, 59) == expected


def test_full_set_is_star():
    assert canonical_cron("* * * * *") == "* * * * *"
    assert canonical_cron("0-59 0-23 1-31 1-12 0-6") == "* * * * *"


def test_step_collapse():
    assert CronSpec(minute=tuple(range(0, 60, 15)), hour=tuple(range(24)),
                    dom=tuple(range(1, 32)), month=tuple(range(1, 13)),
                    dow=tuple(range(7))).to_cron() == "*/15 * * * *"
    assert CronSpec(minute=tuple(range(0, 60, 5)),
                    hour=tuple(range(24)), dom=tuple(range(1, 32)),
                    month=tuple(range(1, 13)), dow=tuple(range(7))).to_cron() == "*/5 * * * *"


def test_weekend_is_list_not_step():
    # dow only allows lists/ranges; 0,6 must stay "0,6" (never "*/6")
    spec = CronSpec(minute=(0,), hour=(9,), dom=tuple(range(1, 32)),
                    month=tuple(range(1, 13)), dow=(0, 6))
    assert spec.to_cron() == "0 9 * * 0,6"


def test_dom_never_steps():
    spec = CronSpec(minute=(0,), hour=(9,), dom=(1, 15),
                    month=tuple(range(1, 13)), dow=tuple(range(7)))
    assert spec.to_cron() == "0 9 1,15 * *"


def test_range_run_rule():
    assert canonical_cron("* * * * 1,2") == "* * * * 1,2"   # run of 2 stays a list
    assert canonical_cron("* * * * 1-3") == "* * * * 1-3"   # run of 3 becomes a range
    assert canonical_cron("1,2,4,5,6,7 * * * *") == "1,2,4-7 * * * *"
    assert canonical_cron("1,2,3,5,6,9 * * * *") == "1-3,5,6,9 * * * *"


def test_canonical_roundtrip_normalizes():
    assert canonical_cron("0-59 0-23 1-31 1-12 0-6") == "* * * * *"
    assert canonical_cron("*/1 * * * *") == "* * * * *"
    assert canonical_cron("1,2,3,4,5 * * * *") == "1-5 * * * *"


# ---------- parsing / validation

@pytest.mark.parametrize("cron", [
    "* * * * *",
    "0 9 * * *",
    "*/15 * * * *",
    "30 8 * * 1-5",
    "0 9 1 * *",
    "0 0 1 */3 *",
    "0 9 * 3,6 *",
    "59 23 * * *",
    "0 0 25 12 *",
    "0 9 * * 0,6",
])
def test_valid_crons(cron):
    assert is_valid_cron(cron)
    assert is_canonical(cron)


@pytest.mark.parametrize("cron", [
    "",
    "0 9 * *",               # 4 fields
    "0 9 * * * *",           # 6 fields
    "60 * * * *",            # minute out of range
    "* 24 * * *",            # hour out of range
    "* * 32 * *",            # dom out of range
    "* * * 13 *",            # month out of range
    "* * * * 7",             # dow out of range
    "0 9 * * MON",           # names forbidden
    "*/0 * * * *",           # zero step
    "5/10 * * * *",          # a/N forbidden
    "9-17/2 * * * *",        # a-b/N forbidden
    "5-3 * * * *",           # descending range
    "09 * * * *",            # leading zero
    "* * *,5 * *",           # star inside list
    "* * 1, * *",            # empty element
    "* * * * *\n",           # trailing junk (newline not allowed in a field)
])
def test_invalid_crons(cron):
    assert not is_valid_cron(cron)


def test_empty_and_nonstring():
    with pytest.raises(CronError):
        parse_cron(5)  # type: ignore


def test_leading_zero_rejected_but_zero_ok():
    assert not is_valid_cron("00 9 * * *")
    assert is_valid_cron("0 9 * * *")


# ---------- semantics

def test_semantic_equal():
    assert semantic_equal("1-5 * * * *", "1,2,3,4,5 * * * *")
    assert semantic_equal("* * * * *", "* * * * *")
    assert not semantic_equal("0 9 * * *", "0 10 * * *")
    # dom star-ness is part of semantic equality
    assert not semantic_equal("* * * * *", "* * 1-31 * *")
    assert not semantic_equal("* * * * *", "* * * * 0-6")


def test_semantic_equal_invalid_inputs():
    assert not semantic_equal("garbage", "garbage")


def test_differing_fields():
    assert differing_fields("0 9 * * *", "0 9 * * *") == []
    assert set(differing_fields("0 9 * * *", "30 9 * * *")) == {"minute"}
    assert set(differing_fields("0 9 * * *", "0 9 * * 1-5")) == {"dow"}
    assert set(differing_fields("0 9 * * *", "nonsense")) == {"minute", "hour", "dom", "month", "dow"}


# ---------- next runs

def test_next_runs_basic():
    after = datetime(2026, 11, 14, 9, 29)
    runs = next_runs("30 9 * * *", after, count=2)
    assert [r.strftime("%H:%M") for r in runs] == ["09:30", "09:30"]
    assert runs[0].date() == datetime(2026, 11, 14).date()
    assert runs[1].date() == datetime(2026, 11, 15).date()


def test_next_runs_first_of_month():
    after = datetime(2026, 11, 14, 0, 0)
    runs = next_runs("0 0 1 * *", after, count=2)
    assert [r.strftime("%Y-%m-%d") for r in runs] == ["2026-12-01", "2027-01-01"]


def test_next_runs_weekend():
    # 2026-11-14 is a Saturday; weekends are dow 0 and 6
    after = datetime(2026, 11, 13, 12, 0)  # friday
    runs = next_runs("0 8 * * 0,6", after, count=2)
    assert [r.strftime("%Y-%m-%d %H:%M") for r in runs] == ["2026-11-14 08:00", "2026-11-15 08:00"]


def test_next_runs_dom_dow_or_semantics():
    # dom 13 OR dow Friday (day 5): whichever comes first wins
    # 2026-11-20 is a Friday; 2026-11-13 is the 13th (already past from this 'after')
    after = datetime(2026, 11, 14, 0, 0)
    runs = next_runs("0 0 13 * 5", after, count=1)
    assert runs[0].strftime("%Y-%m-%d") == "2026-11-20"


def test_next_runs_step():
    after = datetime(2026, 11, 14, 9, 7)
    runs = next_runs("*/15 * * * *", after, count=3)
    assert [r.strftime("%H:%M") for r in runs] == ["09:15", "09:30", "09:45"]


def test_next_runs_invalid_cron():
    with pytest.raises(CronError):
        next_runs("not a cron", datetime(2026, 11, 14))
