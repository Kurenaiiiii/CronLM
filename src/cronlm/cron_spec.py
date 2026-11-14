"""Structured representation of a 5-field cron expression, plus helpers.

This module is the single source of truth for the *canonical output format*
described in Section 3 of the PRD:

- Numbers only, no names, no leading zeros.
- Lists are ascending, comma separated, no spaces.
- Runs of 3+ consecutive values become a range ``a-b``; runs of 1-2 stay as
  list items.
- Steps use ``*/N`` only, never ``a/N`` or ``a-b/N``.
- Fields are joined by exactly one space.

It also provides the validator, the set expander (used by the semantic-match
metric), and a small next-run scheduler used by ``predict``. No third-party
cron library is used anywhere in the project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

# (name, lo, hi) for the five fields, in canonical order.
FIELD_SPECS = [
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("dom", 1, 31),     # day of month
    ("month", 1, 12),
    ("dow", 0, 6),      # day of week, 0 = Sunday
]

FIELD_NAMES = [name for name, _, _ in FIELD_SPECS]

# Steps (*/N) are only allowed in these fields (Section 3.1, rule 5).
STEP_FIELDS = {"minute", "hour", "month"}

# Canonical step values allowed per field (Section 3.1, rule 5). The generator
# only ever emits these; the validator accepts any positive step but the
# canonical formatter only produces sets that collapse into ``*/N``.
ALLOWED_MINUTE_STEPS = (2, 3, 4, 5, 6, 10, 12, 15, 20, 30)
ALLOWED_HOUR_STEPS = (2, 3, 4, 6, 8, 12)
ALLOWED_MONTH_STEPS = (2, 3, 4, 6)

_FULL_RANGES = {name: set(range(lo, hi + 1)) for name, lo, hi in FIELD_SPECS}

_INT_RE = re.compile(r"[0-9]+")


class CronError(ValueError):
    """Raised when a cron string cannot be parsed."""


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_field(values, lo, hi, allow_step=True):
    """Render a set of field values to the canonical field string.

    Steps (``*/N``) are only legal in minute, hour and month (Section 3.1,
    rule 5), so the step-collapse branch is disabled for dom and dow.
    ``values`` is an iterable of ints; they are deduplicated and sorted.
    Raises CronError on out-of-range values or an empty set.
    """
    vs = sorted(set(values))
    if not vs:
        raise CronError("empty field")
    if vs[0] < lo or vs[-1] > hi:
        raise CronError(f"field value {vs[0] if vs[0] < lo else vs[-1]} out of range [{lo},{hi}]")

    # Rule: the full set is written as "*".
    if len(vs) == hi - lo + 1:
        return "*"

    # Rule: an arithmetic progression lo, lo+N, ..., <= hi that exactly covers
    # the set is written as "*/N" (N >= 2; N == 1 would be "*"). Only in minute,
    # hour and month.
    if allow_step and len(vs) >= 2:
        step = vs[1] - vs[0]
        if step >= 2 and vs == list(range(lo, hi + 1, step)):
            return f"*/{step}"

    # Walk the sorted values, grouping consecutive runs. Rules 3 and 4:
    # runs of >= 3 become "a-b", runs of 1-2 stay as individual numbers,
    # everything is comma-joined, ascending, no spaces.
    items = []
    i = 0
    while i < len(vs):
        j = i
        while j + 1 < len(vs) and vs[j + 1] == vs[j] + 1:
            j += 1
        run_len = j - i + 1
        if run_len >= 3:
            items.append(f"{vs[i]}-{vs[j]}")
        else:
            items.extend(str(v) for v in vs[i:j + 1])
        i = j + 1
    return ",".join(items)


# ---------------------------------------------------------------------------
# CronSpec: the structured object the generator builds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CronSpec:
    """The five fields as sorted tuples of concrete values.

    A field holding every value in its range is rendered as ``*`` by
    :meth:`to_cron`. Labels are correct by construction: the data generator
    builds one of these and renders it via :meth:`to_cron`, never from the
    phrase text.
    """

    minute: tuple
    hour: tuple
    dom: tuple
    month: tuple
    dow: tuple

    def __post_init__(self):
        for name, values in (("minute", self.minute), ("hour", self.hour),
                             ("dom", self.dom), ("month", self.month), ("dow", self.dow)):
            lo, hi = dict((n, (a, b)) for n, a, b in FIELD_SPECS)[name]
            t = tuple(sorted(set(values)))
            if not t:
                raise CronError(f"empty {name} field")
            if t[0] < lo or t[-1] > hi:
                raise CronError(f"{name} out of range [{lo},{hi}]")
            object.__setattr__(self, name, t)

    def to_cron(self):
        return " ".join(
            format_field(values, lo, hi, allow_step=(name in STEP_FIELDS))
            for (name, lo, hi), values in zip(FIELD_SPECS,
                                              (self.minute, self.hour, self.dom, self.month, self.dow))
        )


# ---------------------------------------------------------------------------
# Parsing / validation / expansion
# ---------------------------------------------------------------------------

def _parse_uint(token, field_name):
    """Parse a canonical (or at least plain) unsigned integer token."""
    if not _INT_RE.fullmatch(token):
        raise CronError(f"{field_name}: '{token}' is not a number")
    if len(token) > 1 and token[0] == "0":
        # Leading zeros are not canonical ("09"); a lone "0" is fine.
        raise CronError(f"{field_name}: '{token}' has a leading zero")
    return int(token)


def _expand_field(token, name, lo, hi):
    """Expand one field token to a (set, star) pair.

    Validity rules (deliberate, documented):
    - ``*`` means the full range, ``*/N`` means range(lo, hi+1, N), N >= 1.
    - ``a-b`` is an inclusive range, ``a`` a single value, comma separated.
    - ``a/N`` and ``a-b/N`` are rejected (the canonical format forbids them).
    - Names, wildcards inside lists, empty elements, descending ranges and
      out-of-range values are rejected.
    - ``*/1`` is accepted syntactically (it equals ``*``) but never emitted.
    - List order and duplicates are *not* policed here; cron is a set
      language for our purposes. Canonical strictness lives in
      :func:`is_canonical`.
    """
    if token == "":
        raise CronError(f"{name}: empty field")
    if token == "*":
        return frozenset(range(lo, hi + 1)), True
    values = set()
    star = False
    for part in token.split(","):
        if part == "":
            raise CronError(f"{name}: empty element in '{token}'")
        if part.startswith("/"):
            raise CronError(f"{name}: malformed step in '{part}'")
        if "/" in part:
            if not part.startswith("*/"):
                raise CronError(f"{name}: '{part}' — steps must look like */N, not a/N or a-b/N")
            n = _parse_uint(part[2:], name)
            if n < 1:
                raise CronError(f"{name}: step in '{part}' must be >= 1")
            values |= set(range(lo, hi + 1, n))
            continue
        if "-" in part:
            if part.count("-") != 1:
                raise CronError(f"{name}: malformed range '{part}'")
            a_tok, b_tok = part.split("-")
            a = _parse_uint(a_tok, name)
            b = _parse_uint(b_tok, name)
            if a > b:
                raise CronError(f"{name}: descending range '{part}'")
            if a < lo or b > hi:
                raise CronError(f"{name}: range '{part}' outside [{lo},{hi}]")
            values |= set(range(a, b + 1))
            continue
        v = _parse_uint(part, name)
        if v < lo or v > hi:
            raise CronError(f"{name}: value {v} outside [{lo},{hi}]")
        values.add(v)
    if not values:
        raise CronError(f"{name}: empty expansion")
    if token == "*":
        star = True
    return frozenset(values), star


@dataclass(frozen=True)
class ParsedCron:
    """Result of parsing a cron string."""

    sets: tuple        # 5 frozensets, in FIELD_SPECS order
    dom_star: bool
    dow_star: bool
    canonical_fields: tuple


def parse_cron(cron):
    """Parse and validate ``cron``. Raises CronError on any problem."""
    if not isinstance(cron, str):
        raise CronError("cron expression must be a string")
    fields = cron.split(" ")
    if len(fields) != 5:
        raise CronError(f"expected 5 space-separated fields, got {len(fields)} in {cron!r}")
    sets = []
    stars = []
    for (name, lo, hi), token in zip(FIELD_SPECS, fields):
        expanded, star = _expand_field(token, name, lo, hi)
        sets.append(expanded)
        stars.append(star)
    return ParsedCron(
        sets=tuple(sets),
        dom_star=stars[2],
        dow_star=stars[4],
        canonical_fields=tuple(
            format_field(s, lo, hi, allow_step=(name in STEP_FIELDS))
            for (name, lo, hi), s in zip(FIELD_SPECS, sets)
        ),
    )


def is_valid_cron(cron):
    """True if ``cron`` parses under the rules in this module."""
    try:
        parse_cron(cron)
        return True
    except CronError:
        return False


def canonical_cron(cron):
    """Return the canonical rendering of a parseable cron string.

    ``is_canonical`` compares the input with this rendering, so ``1-31``
    becomes ``*`` and ``1,2,3,4,5`` becomes ``1-5``.
    """
    parsed = parse_cron(cron)
    return " ".join(parsed.canonical_fields)


def is_canonical(cron):
    try:
        return canonical_cron(cron) == cron
    except CronError:
        return False


def semantic_equal(a, b):
    """Semantic match (Section 9): same expanded sets and same ``*``-ness of
    the day-of-month and day-of-week fields. ``1,2,3,4,5`` == ``1-5``."""
    try:
        pa = parse_cron(a)
        pb = parse_cron(b)
    except CronError:
        return False
    return (pa.sets == pb.sets
            and pa.dom_star == pb.dom_star
            and pa.dow_star == pb.dow_star)


def field_names():
    return list(FIELD_NAMES)


def differing_fields(expected, predicted):
    """Return the list of field names whose canonical rendering differs.

    A predicted string that does not parse counts as all five fields wrong.
    """
    try:
        pe = parse_cron(expected)
    except CronError:  # labels should always parse; if not, flag everything
        return list(FIELD_NAMES)
    try:
        pp = parse_cron(predicted)
    except CronError:
        return list(FIELD_NAMES)
    return [name for name, fe, fp in zip(FIELD_NAMES, pe.canonical_fields, pp.canonical_fields) if fe != fp]


# ---------------------------------------------------------------------------
# Next-run scheduler (used by predict + tests). Not a dependency, just this.
# ---------------------------------------------------------------------------

def _cron_dow(d):
    """datetime.date.weekday() is Mon=0..Sun=6; cron dow is Sun=0..Sat=6."""
    return (d.weekday() + 1) % 7


def _day_matches(t, parsed):
    dom_set, dow_set = parsed.sets[2], parsed.sets[4]
    dom_star, dow_star = parsed.dom_star, parsed.dow_star
    dom_ok = t.day in dom_set
    dow_ok = _cron_dow(t.date()) in dow_set
    if dom_star and dow_star:
        return True
    if dom_star:
        return dow_ok
    if dow_star:
        return dom_ok
    # Standard cron: when both are restricted, a day matches if either does.
    return dom_ok or dow_ok


def _first_of_next_month(t):
    if t.month == 12:
        return datetime(t.year + 1, 1, 1)
    return datetime(t.year, t.month + 1, 1)


def next_runs(cron, after, count=5):
    """Return the next ``count`` run times strictly after ``after``.

    ``after`` is any datetime; matching is at minute precision. Implements
    standard cron dom/dow OR semantics. Raises CronError on a bad cron
    string or RuntimeError if nothing matches within ~8 years (which cannot
    happen for a parseable cron).
    """
    parsed = parse_cron(cron)
    minutes, hours, months = parsed.sets[0], parsed.sets[1], parsed.sets[3]

    t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    results = []
    guard = 0
    limit = 8 * 366 * 24 * 60  # ~8 years of minutes
    while len(results) < count:
        guard += 1
        if guard > limit:
            raise RuntimeError(f"no occurrence of {cron!r} within 8 years")
        if t.month not in months:
            # Skip whole months when the month field does not match.
            t = _first_of_next_month(t)
            continue
        if not _day_matches(t, parsed):
            # Skip to 00:00 of the next day.
            t = (t + timedelta(days=1)).replace(hour=0, minute=0)
            continue
        if t.hour not in hours:
            # Skip to the top of the next hour.
            t = t.replace(minute=0) + timedelta(hours=1)
            continue
        if t.minute not in minutes:
            t += timedelta(minutes=1)
            continue
        results.append(t)
        t += timedelta(minutes=1)
    return results
