"""Grammar-based data generator: structured spec in, phrase out.

Labels are correct by construction: we never parse a phrase to get a cron.
Instead we sample a structured spec (which field gets which values) and render
both the English phrase and the canonical cron string from it.

Each ``clause kind`` has a table of surface-variant templates with stable
``template_id`` tags (e.g. ``time.exact.3``). Two templates per clause kind
are *reserved*: training never samples them, and they are used only for the
paraphrase-holdout split (Section 5.1). The reserved set is spelled out in
``RESERVED_TEMPLATES`` below and must stay in sync with the tables; the test
suite asserts every reserved id exists in the tables.

Level distribution (Section 4.5): L1 15%, L2 30%, L3 25%, L4 20%, L5 10%.

Reserved combo holdout (Section 5.1): 5% of all (hour, minute) pairs are
reserved by hash and never appear in training. This reservation applies to
*exact-time* examples, the only ones that pin a single (hour, minute) pair.
"""

import hashlib
import random

from .cron_spec import CronSpec
from .tokenizer import BLOCK_SIZE

# ---------------------------------------------------------------------------
# Vocabulary pieces
# ---------------------------------------------------------------------------

DAYS = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
DAY_ABBR = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
MON_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul",
            "aug", "sep", "oct", "nov", "dec"]

ALLOWED_PHRASE_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789 : , . - '")

MINUTE_STEPS = (2, 3, 4, 5, 6, 10, 12, 15, 20, 30)
HOUR_STEPS = (2, 3, 4, 6, 8, 12)
MONTH_STEPS = (2, 3, 4, 6)
STEP_WORDS = {2: "two", 3: "three", 4: "four", 6: "six"}

# Hash bucket thresholds over md5(phrase) % 1000 (Section 5.1).
VAL_BUCKETS = 40          # buckets [0, 40) -> val
TEST_BUCKETS = 240        # buckets [40, 240) -> test
# everything >= 240 is train
COMBO_RESERVED = 50       # md5("hour:minute") % 1000 < this -> reserved combo


def ordinal(n):
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def join_en(items):
    """[a] -> 'a'; [a,b] -> 'a and b'; [a,b,c] -> 'a, b and c'."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def join_oxford(items):
    items = list(items)
    if len(items) <= 2:
        return join_en(items)
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def join_comma(items):
    return ", ".join(items)


def _h12(hour):
    return hour % 12 or 12


def _ampm(hour):
    return "am" if hour < 12 else "pm"


def _12h(hour, minute):
    h, ap = _h12(hour), _ampm(hour)
    return f"{h}{ap}" if minute == 0 else f"{h}:{minute:02d}{ap}"


# ---------------------------------------------------------------------------
# Time clause renderers. Each returns (template_id, phrase_text).
# ---------------------------------------------------------------------------

def _time_exact(minute, hour, rng):
    h12, ap = _h12(hour), _ampm(hour)
    m = f"{minute:02d}"
    if minute == 0:
        opts = [
            ("time.exact.0", f"at {h12}{ap}"),
            ("time.exact.1", f"at {h12} {ap}"),
            ("time.exact.2", f"at {hour} o'clock"),
            ("time.exact.3", f"{h12}{ap}"),
            ("time.exact.4", f"at {hour}:00"),
            ("time.exact.5", f"at {hour:02d}:00"),
            ("time.exact.6", f"at {h12}:00{ap}"),
            ("time.exact.7", f"at exactly {h12}{ap}"),
            ("time.exact.8", f"at {h12}{ap} sharp"),
            ("time.exact.13", f"at precisely {h12}{ap}"),
            ("time.exact.14", f"at {h12} in the morning" if hour < 12 else f"at {h12} in the evening"),
            ("time.exact.15", f"at {h12}{ap} exactly"),
            ("time.exact.16", f"at {hour:02d}00 hours"),
        ]
        if hour == 0:
            opts += [("time.exact.10", "at midnight"), ("time.exact.11", "at 12am"),
                     ("time.exact.12", "at 0:00")]
        elif hour == 12:
            opts += [("time.exact.10", "at noon"), ("time.exact.11", "at 12pm"),
                     ("time.exact.12", "at 12:00")]
    else:
        opts = [
            ("time.exact.0", f"at {h12}:{m}{ap}"),
            ("time.exact.1", f"at {h12}:{m} {ap}"),
            ("time.exact.2", f"at {hour}:{m}"),
            ("time.exact.3", f"at {hour:02d}:{m}"),
            ("time.exact.4", f"{h12}:{m}{ap}"),
            ("time.exact.5", f"at {minute} minutes past {h12}{ap}"),
            ("time.exact.6", f"at {minute} past {h12}{ap}"),
            ("time.exact.7", f"at {minute} minutes after {h12}{ap}"),
            ("time.exact.8", f"at {minute} after {h12}{ap}"),
            ("time.exact.9", f"at {h12}:{m}{ap} sharp"),
            ("time.exact.13", f"at precisely {h12}:{m}{ap}"),
            ("time.exact.14", f"at {h12}:{m} in the morning" if hour < 12 else f"at {h12}:{m} in the evening"),
            ("time.exact.15", f"at {h12}:{m}{ap} exactly"),
        ]
    return rng.choice(opts)


def _time_verbose(minute, hour, rng):
    if minute == 30:
        h12, ap = _h12(hour), _ampm(hour)
        opts = [
            ("time.verbose.0", f"at half past {h12}{ap}"),
            ("time.verbose.1", f"half past {h12}{ap}"),
            ("time.verbose.2", f"at half past {h12} {ap}"),
            ("time.verbose.3", f"at thirty minutes past {h12}{ap}"),
            ("time.verbose.4", f"at 30 minutes past {h12}{ap}"),
            ("time.verbose.5", f"at half {h12}{ap}"),
            ("time.verbose.6", f"at half past the hour of {hour}"),
            ("time.verbose.7", f"at {hour}:30"),
            ("time.verbose.8", f"at {hour:02d}:30"),
        ]
    elif minute == 15:
        h12, ap = _h12(hour), _ampm(hour)
        opts = [
            ("time.verbose.9", f"at quarter past {h12}{ap}"),
            ("time.verbose.10", f"quarter past {h12}{ap}"),
            ("time.verbose.11", f"at a quarter past {h12}{ap}"),
            ("time.verbose.12", f"at quarter after {h12}{ap}"),
            ("time.verbose.13", f"at fifteen past {h12}{ap}"),
            ("time.verbose.14", f"at 15 minutes past {h12}{ap}"),
            ("time.verbose.15", f"at fifteen minutes after {h12}{ap}"),
            ("time.verbose.16", f"at {hour}:15"),
            ("time.verbose.17", f"at {hour:02d}:15"),
        ]
    else:  # minute == 45, rendered against the *next* hour
        nxt = (hour + 1) % 24
        h12, ap = _h12(nxt), _ampm(nxt)
        opts = [
            ("time.verbose.18", f"at quarter to {h12}{ap}"),
            ("time.verbose.19", f"quarter to {h12}{ap}"),
            ("time.verbose.20", f"at a quarter to {h12}{ap}"),
            ("time.verbose.21", f"at quarter of {h12}{ap}"),
            ("time.verbose.22", f"at fifteen to {h12}{ap}"),
            ("time.verbose.23", f"at 45 minutes before {h12}{ap}"),
            ("time.verbose.24", f"at forty five minutes before {h12}{ap}"),
            ("time.verbose.25", f"at {hour}:45"),
            ("time.verbose.26", f"at {hour:02d}:45"),
        ]
    return rng.choice(opts)


def _time_every_minute(minute, hour, rng):
    opts = [
        ("time.every_minute.0", "every minute"),
        ("time.every_minute.1", "each minute"),
        ("time.every_minute.2", "every minute of the day"),
        ("time.every_minute.3", "every sixty seconds"),
        ("time.every_minute.4", "every minute of every day"),
        ("time.every_minute.5", "every minute, all day"),
        ("time.every_minute.6", "minutely"),
        ("time.every_minute.7", "every minute, all day long"),
        ("time.every_minute.8", "all day, every minute"),
        ("time.every_minute.9", "every minute of the hour"),
        ("time.every_minute.10", "every single minute"),
        ("time.every_minute.11", "each and every minute"),
        ("time.every_minute.12", "once a minute"),
    ]
    return rng.choice(opts)


def _time_minute_step(step, hour, rng):
    n = step
    opts = [
        ("time.minute_step.0", f"every {n} minutes"),
        ("time.minute_step.1", f"each {n} minutes"),
        ("time.minute_step.2", f"every {n} mins"),
        ("time.minute_step.3", f"every {n} min"),
        ("time.minute_step.4", f"all day every {n} minutes"),
        ("time.minute_step.5", f"each {n} mins"),
        ("time.minute_step.6", f"once every {n} minutes"),
        ("time.minute_step.7", f"every {n} minutes exactly"),
        ("time.minute_step.8", f"each {n} min"),
        ("time.minute_step.9", f"every {n} minutes, every hour"),
        ("time.minute_step.10", f"each {n} minutes, every hour"),
        ("time.minute_step.11", f"every {n} minutes, around the clock"),
    ]
    return rng.choice(opts)


def _time_hourly(minute, hour, rng):
    opts = [
        ("time.hourly.0", "every hour"),
        ("time.hourly.1", "hourly"),
        ("time.hourly.2", "each hour"),
        ("time.hourly.3", "every hour on the hour"),
        ("time.hourly.4", "once an hour"),
        ("time.hourly.5", "once per hour"),
        ("time.hourly.6", "every 60 minutes"),
        ("time.hourly.7", "on the hour, every hour"),
        ("time.hourly.8", "every single hour"),
        ("time.hourly.9", "once an hour, on the hour"),
        ("time.hourly.10", "every hour, sharp"),
        ("time.hourly.11", "each and every hour"),
    ]
    return rng.choice(opts)


def _time_hour_step(minute, step, rng):
    n = step
    opts = [
        ("time.hour_step.0", f"every {n} hours"),
        ("time.hour_step.1", f"each {n} hours"),
        ("time.hour_step.2", f"every {n} hrs"),
        ("time.hour_step.3", f"every {n} hour"),
        ("time.hour_step.4", f"once every {n} hours"),
        ("time.hour_step.5", f"every {n} hours on the hour"),
        ("time.hour_step.6", f"every {n} hours, on the hour"),
        ("time.hour_step.7", f"each {n} hrs"),
        ("time.hour_step.8", f"every {n}h"),
        ("time.hour_step.9", f"every {n} hours from midnight"),
        ("time.hour_step.10", f"each and every {n} hours"),
        ("time.hour_step.11", f"once every {n} hours, on the hour"),
    ]
    return rng.choice(opts)


def _time_hourly_at_minute(minute, hour, rng):
    m = minute
    opts = [
        ("time.hourly_at_minute.0", f"at {m} minutes past every hour"),
        ("time.hourly_at_minute.1", f"at minute {m} of every hour"),
        ("time.hourly_at_minute.2", f"at {m} minutes past each hour"),
        ("time.hourly_at_minute.3", f"at {m} mins past every hour"),
        ("time.hourly_at_minute.4", f"at {m} minutes past the hour, every hour"),
        ("time.hourly_at_minute.5", f"every hour at {m} minutes past"),
        ("time.hourly_at_minute.6", f"at {m} past every hour"),
        ("time.hourly_at_minute.7", f"every hour at {m} past"),
        ("time.hourly_at_minute.8", f"at {m} minutes after every hour"),
        ("time.hourly_at_minute.9", f"at minute {m} past the hour"),
        ("time.hourly_at_minute.10", f"at {m} minutes after the hour"),
        ("time.hourly_at_minute.11", f"every hour, {m} minutes past"),
    ]
    return rng.choice(opts)


def _time_window(step, hour_pair, rng):
    n = step
    h1, h2 = hour_pair
    a1, ap1 = _h12(h1), _ampm(h1)
    a2, ap2 = _h12(h2), _ampm(h2)
    opts = [
        ("time.window.0", f"every {n} minutes between {a1}{ap1} and {a2}{ap2}"),
        ("time.window.1", f"every {n} minutes from {a1}{ap1} to {a2}{ap2}"),
        ("time.window.2", f"every {n} mins between {a1}{ap1} and {a2}{ap2}"),
        ("time.window.3", f"each {n} minutes between {a1}{ap1} and {a2}{ap2}"),
        ("time.window.4", f"every {n} minutes starting at {a1}{ap1} until {a2}{ap2}"),
        ("time.window.5", f"every {n} minutes from {a1}{ap1} until {a2}{ap2}"),
        ("time.window.6", f"every {n} minutes between {h1} and {h2} hours"),
        ("time.window.7", f"every {n} minutes in the hours from {a1}{ap1} to {a2}{ap2}"),
        ("time.window.8", f"every {n} minutes during {a1}{ap1} to {a2}{ap2}"),
        ("time.window.9", f"every {n} minutes between {h1:02d} and {h2:02d}"),
    ]
    return rng.choice(opts)


def _time_same_minute(minute, hours, rng):
    m = minute
    times = [_12h(h, m) for h in hours]
    en = join_en(times)
    ox = join_oxford(times)
    opts = [
        ("time.same_minute.0", f"at {en}"),
        ("time.same_minute.1", f"at {ox}"),
        ("time.same_minute.2", f"{en}"),
        ("time.same_minute.3", f"at {join_comma(times)}"),
        ("time.same_minute.4", f"at {en}, daily"),
        ("time.same_minute.5", f"every day at {en}"),
        ("time.same_minute.6", f"at {en} each day"),
        ("time.same_minute.7", f"at {en} exactly"),
        ("time.same_minute.8", f"at {en} on the dot"),
    ]
    return rng.choice(opts)


# ---------------------------------------------------------------------------
# Day clause renderers. Return (template_id, text, field_kind, values).
# ---------------------------------------------------------------------------

def _day_weekdays(rng):
    opts = [
        ("day.weekdays.0", "every weekday"),
        ("day.weekdays.1", "on weekdays"),
        ("day.weekdays.2", "weekdays"),
        ("day.weekdays.3", "monday to friday"),
        ("day.weekdays.4", "monday through friday"),
        ("day.weekdays.5", "mon to fri"),
        ("day.weekdays.6", "mon-fri"),
        ("day.weekdays.7", "mondays to fridays"),
        ("day.weekdays.8", "from monday to friday"),
        ("day.weekdays.9", "each weekday"),
    ]
    tid, text = rng.choice(opts)
    return tid, text, "dow", (1, 2, 3, 4, 5)


def _day_weekends(rng):
    opts = [
        ("day.weekends.0", "on weekends"),
        ("day.weekends.1", "every weekend"),
        ("day.weekends.2", "weekends"),
        ("day.weekends.3", "saturday and sunday"),
        ("day.weekends.4", "sat and sun"),
        ("day.weekends.5", "sat-sun"),
        ("day.weekends.6", "on saturday and sunday"),
        ("day.weekends.7", "every saturday and sunday"),
        ("day.weekends.8", "each weekend"),
        ("day.weekends.9", "on weekends at the weekend"),
    ]
    tid, text = rng.choice(opts)
    return tid, text, "dow", (0, 6)


def _day_named(rng):
    days = sorted(rng.sample(range(7), k=rng.choice([1, 1, 2, 2, 3])))
    full = [DAYS[d] for d in days]
    abbr = [DAY_ABBR[d] for d in days]
    plural = [DAYS[d] + "s" for d in days]
    en_f, en_a = join_en(full), join_en(abbr)
    ox_a = join_oxford(abbr)
    opts = [
        ("day.named_days.0", f"every {en_f}"),
        ("day.named_days.1", f"on {en_f}"),
        ("day.named_days.2", f"every {en_a}"),
        ("day.named_days.3", f"on {en_a}"),
        ("day.named_days.4", f"each {en_f}"),
        ("day.named_days.5", f"on {join_en(plural)}"),
        ("day.named_days.6", f"every {ox_a}"),
        ("day.named_days.7", f"on {ox_a}"),
        ("day.named_days.8", f"every single {en_f}"),
    ]
    tid, text = rng.choice(opts)
    return tid, text, "dow", tuple(days)


def _day_range(rng):
    a = rng.randrange(0, 6)
    span = rng.choice([2, 3, 3, 4])
    b = min(6, a + span)
    if a == b:
        a, b = 0, 2
    full_a, full_b = DAYS[a], DAYS[b]
    ab_a, ab_b = DAY_ABBR[a], DAY_ABBR[b]
    opts = [
        ("day.day_range.0", f"{full_a} through {full_b}"),
        ("day.day_range.1", f"{full_a} to {full_b}"),
        ("day.day_range.2", f"from {full_a} through {full_b}"),
        ("day.day_range.3", f"between {full_a} and {full_b}"),
        ("day.day_range.4", f"{ab_a}-{ab_b}"),
        ("day.day_range.5", f"{ab_a} to {ab_b}"),
        ("day.day_range.6", f"every day from {full_a} to {full_b}"),
        ("day.day_range.7", f"{full_a} thru {full_b}"),
        ("day.day_range.8", f"from {full_a} to {full_b}"),
    ]
    tid, text = rng.choice(opts)
    return tid, text, "dow", tuple(range(a, b + 1))


def _day_dom(rng):
    days = sorted(rng.sample(range(1, 32), k=rng.choice([1, 1, 2, 2, 3])))
    ords = [ordinal(d) for d in days]
    en, ox = join_en(ords), join_oxford(ords)
    opts = [
        ("day.dom.0", f"on the {en}"),
        ("day.dom.1", f"on the {ox}"),
        ("day.dom.2", (f"on day {days[0]} of the month" if len(days) == 1
                       else f"on days {join_en([str(d) for d in days])} of the month")),
        ("day.dom.3", f"on the {en} of the month"),
        ("day.dom.4", f"on {en}"),
        ("day.dom.5", f"on the {en} of each month"),
        ("day.dom.6", f"each month on the {en}"),
        ("day.dom.7", f"on the {en} of every month"),
        ("day.dom.8", f"on the {ox} day of the month"),
    ]
    if days == [1]:
        opts += [("day.dom.10", "first of every month"),
                 ("day.dom.11", "first of the month"),
                 ("day.dom.12", "the 1st of every month")]
    tid, text = rng.choice(opts)
    return tid, text, "dom", tuple(days)


# ---------------------------------------------------------------------------
# Month clause renderers. Return
# (template_id, text, field_kind, values, dom_override, month_step).
# dom_override pins the day-of-month field (used by month.step -> 1 and by
# month.with_dom -> the day from the phrase).
# ---------------------------------------------------------------------------

def _month_single(rng):
    m = rng.randrange(12)
    name, abbr = MONTHS[m], MON_ABBR[m]
    opts = [
        ("month.single.0", f"in {name}"),
        ("month.single.1", f"every {name}"),
        ("month.single.2", f"each {name}"),
        ("month.single.3", f"in the month of {name}"),
        ("month.single.4", f"during {name}"),
        ("month.single.5", f"in {abbr}"),
        ("month.single.6", f"every {abbr}"),
        ("month.single.7", f"during the month of {name}"),
        ("month.single.8", f"in {name} every year"),
    ]
    tid, text = rng.choice(opts)
    return tid, text, "months", (m + 1,), None


def _month_list(rng):
    ms = sorted(rng.sample(range(12), k=rng.choice([2, 2, 3])))
    names = [MONTHS[m] for m in ms]
    abbrs = [MON_ABBR[m] for m in ms]
    en_n, en_a = join_en(names), join_en(abbrs)
    ox_n, ox_a = join_oxford(names), join_oxford(abbrs)
    opts = [
        ("month.list.0", f"in {en_n}"),
        ("month.list.1", f"in {en_a}"),
        ("month.list.2", f"in {ox_n}"),
        ("month.list.3", f"in {join_comma(names)}"),
        ("month.list.4", f"during {en_n}"),
        ("month.list.5", f"every {en_n}"),
        ("month.list.6", f"in {ox_a}"),
        ("month.list.7", f"in the months of {en_n}"),
        ("month.list.8", f"during the months of {en_n}"),
    ]
    tid, text = rng.choice(opts)
    return tid, text, "months", tuple(m + 1 for m in ms), None


def _month_range(rng):
    a = rng.randrange(0, 11)
    b = min(11, a + rng.choice([1, 2, 2, 3, 4]))
    if a == b:
        a = max(0, a - 1)
    name_a, name_b = MONTHS[a], MONTHS[b]
    ab_a, ab_b = MON_ABBR[a], MON_ABBR[b]
    opts = [
        ("month.range.0", f"from {name_a} to {name_b}"),
        ("month.range.1", f"{name_a} through {name_b}"),
        ("month.range.2", f"{name_a} to {name_b}"),
        ("month.range.3", f"{ab_a} to {ab_b}"),
        ("month.range.4", f"between {name_a} and {name_b}"),
        ("month.range.5", f"from {name_a} until {name_b}"),
        ("month.range.6", f"{ab_a} through {ab_b}"),
        ("month.range.7", f"{ab_a}-{ab_b}"),
        ("month.range.8", f"from the month of {name_a} to the month of {name_b}"),
    ]
    tid, text = rng.choice(opts)
    return tid, text, "months", tuple(range(a + 1, b + 2)), None


def _month_step(rng):
    n = rng.choice(MONTH_STEPS)
    word = STEP_WORDS[n]
    if n == 3:
        special, special2, special3 = "quarterly", "once a quarter", "every quarter"
    elif n == 2:
        special, special2, special3 = "every other month", "every two months", "one month on, one month off"
    elif n == 4:
        special, special2, special3 = "every trimester", "every four months", "once every trimester"
    else:  # 6
        special, special2, special3 = "twice a year", "every half year", "semiannually"
    opts = [
        ("month.step.0", f"every {n} months"),
        ("month.step.1", f"every {word} months"),
        ("month.step.2", f"once every {n} months"),
        ("month.step.3", f"each {n} months"),
        ("month.step.4", f"every {word}-month period"),
        ("month.step.5", special),
        ("month.step.6", special2),
        ("month.step.7", special3),
        ("month.step.8", f"every {n} months on the 1st"),
    ]
    tid, text = rng.choice(opts)
    # "quarterly" / friends: dom defaults to 1 (Section 4.3).
    return tid, text, "months", tuple(range(1, 13, n)), (1,)


def _month_with_dom(rng):
    m = rng.randrange(12)
    d = rng.randrange(1, 32)
    name, abbr = MONTHS[m], MON_ABBR[m]
    od = ordinal(d)
    opts = [
        ("month.with_dom.0", f"on {name} {od}"),
        ("month.with_dom.1", f"on {abbr} {od}"),
        ("month.with_dom.2", f"on the {od} of {name}"),
        ("month.with_dom.3", f"on the {od} of {abbr}"),
        ("month.with_dom.4", f"{name} {od}"),
        ("month.with_dom.5", f"{abbr} {od}"),
        ("month.with_dom.6", f"on {name} the {od}"),
        ("month.with_dom.7", f"every {name} {od}"),
        ("month.with_dom.8", f"each {name} {od}"),
    ]
    tid, text = rng.choice(opts)
    return tid, text, "months", (m + 1,), (d,)


def _sample_month(rng, level):
    kind = rng.choices(
        ["single", "list", "range", "step", "with_dom"],
        weights=[30, 18, 16, 20, 16])[0]
    fn = {"single": _month_single, "list": _month_list, "range": _month_range,
          "step": _month_step, "with_dom": _month_with_dom}[kind]
    return fn(rng)


# ---------------------------------------------------------------------------
# Reserved templates for the paraphrase holdout (2 per clause kind), Section
# 5.1. Training never sees these; the holdout samples only these.
# ---------------------------------------------------------------------------

RESERVED_TEMPLATES = {
    # time clauses
    "time.exact.7", "time.exact.8",
    "time.verbose.25", "time.verbose.26",  # 24h rendering of :45
    "time.every_minute.6", "time.every_minute.7",
    "time.minute_step.6", "time.minute_step.7",
    "time.hourly.6", "time.hourly.7",
    "time.hour_step.6", "time.hour_step.7",
    "time.hourly_at_minute.6", "time.hourly_at_minute.7",
    "time.window.8", "time.window.9",
    "time.same_minute.7", "time.same_minute.8",
    # day clauses
    "day.weekdays.8", "day.weekdays.9",
    "day.weekends.8", "day.weekends.9",
    "day.named_days.7", "day.named_days.8",
    "day.day_range.7", "day.day_range.8",
    "day.dom.7", "day.dom.8",
    # month clauses
    "month.single.7", "month.single.8",
    "month.list.7", "month.list.8",
    "month.range.7", "month.range.8",
    "month.step.6", "month.step.7",
    "month.with_dom.6", "month.with_dom.7",
}


def all_template_ids():
    """Enumerate every template id the tables can produce (for tests)."""
    ids = set()
    rng = random.Random(0)
    probes = []
    # Time clauses: sweep every minute/hour combo so rare conditional variants
    # (noon/midnight) are covered too.
    for m in (0, 15, 30, 45, 5):
        for h in range(24):
            probes.append(lambda r, m=m, h=h: _time_exact(m, h, r))
    for subtype in (15, 30, 45):
        for h in range(24):
            probes.append(lambda r, s=subtype, h=h: _time_verbose(s, h, r))
    probes += [
        lambda r: _time_every_minute(None, None, r),
        lambda r: _time_minute_step(5, None, r),
        lambda r: _time_hourly(None, None, r),
        lambda r: _time_hour_step(4, None, r),
        lambda r: _time_hourly_at_minute(30, None, r),
        lambda r: _time_window(15, (9, 17), r),
        lambda r: _time_same_minute(30, (9, 17), r),
        lambda r: _day_weekdays(r),
        lambda r: _day_weekends(r),
        lambda r: _day_named(r)[:2],
        lambda r: _day_range(r)[:2],
        lambda r: _day_dom(r)[:2],
        lambda r: _month_single(r)[:2],
        lambda r: _month_list(r)[:2],
        lambda r: _month_range(r)[:2],
        lambda r: _month_step(r)[:2],
        lambda r: _month_with_dom(r)[:2],
    ]
    for p in probes:
        for _ in range(200):
            out = p(rng)
            ids.add(out[0])
    return ids


# ---------------------------------------------------------------------------
# Split helpers (hash-based, Section 5.1)
# ---------------------------------------------------------------------------

def split_bucket(phrase):
    """Stable bucket in [0, 1000) for a phrase string."""
    return int(hashlib.md5(phrase.encode("utf-8")).hexdigest(), 16) % 1000


def is_reserved_combo(hour, minute):
    """True if this (hour, minute) pair is reserved for the combo holdout."""
    return int(hashlib.md5(f"{hour}:{minute}".encode("utf-8")).hexdigest(), 16) % 1000 < COMBO_RESERVED


def assign_split(phrase, template_ids, hour=None, minute=None):
    """Which split a generated example belongs to.

    Priority: combo holdout (reserved exact hour:minute pairs) > paraphrase
    holdout (reserved templates) > hash bucket (val / test / train).
    A phrase therefore lands in exactly one split, always, regardless of which
    template produced it.
    """
    if hour is not None and minute is not None and is_reserved_combo(hour, minute):
        return "combo"
    tids = [template_ids] if isinstance(template_ids, str) else list(template_ids)
    if any(t in RESERVED_TEMPLATES for t in tids):
        return "paraphrase"
    h = split_bucket(phrase)
    if h < VAL_BUCKETS:
        return "val"
    if h < TEST_BUCKETS:
        return "test"
    return "train"


def is_train_example(phrase, template_ids, hour=None, minute=None):
    return assign_split(phrase, template_ids, hour, minute) == "train"


# ---------------------------------------------------------------------------
# Composition (levels L1..L5, Section 4.5)
# ---------------------------------------------------------------------------

FRAME_WORDS = ["run", "schedule", "trigger", "fire", "execute",
               "do it", "please run this",
               "please", "could you", "can you", "would you",
               "kindly", "note", "remember", "be sure to",
               "make sure to", "don't forget to"]

LEVEL_DISTRIBUTION = ((1, 0.15), (2, 0.30), (3, 0.25), (4, 0.20), (5, 0.10))


def _pick_level(rng):
    r = rng.random()
    acc = 0.0
    for level, share in LEVEL_DISTRIBUTION:
        acc += share
        if r < acc:
            return level
    return 5


def _sample_time(rng, level):
    """Returns (kind, payload, template_id, text)."""
    if level == 1:
        kind = rng.choices(
            ["exact", "every_minute", "minute_step", "hourly", "hour_step", "hourly_at_minute"],
            weights=[32, 8, 18, 12, 14, 16])[0]
    elif level in (2, 3):
        kind = rng.choices(
            ["exact", "every_minute", "minute_step", "hourly", "hour_step", "hourly_at_minute"],
            weights=[50, 4, 10, 8, 12, 16])[0]
    elif level == 4:
        kind = rng.choices(
            ["window", "same_minute", "verbose", "exact", "minute_step"],
            weights=[34, 30, 24, 6, 6])[0]
    else:  # 5
        kind = rng.choices(
            ["exact", "verbose", "window", "same_minute", "minute_step"],
            weights=[34, 24, 16, 16, 10])[0]

    if kind == "exact":
        minute = rng.randrange(60)
        hour = rng.randrange(24)
        tid, text = _time_exact(minute, hour, rng)
        return "exact", (minute, hour), tid, text, (hour, minute)
    if kind == "verbose":
        subtype = rng.choice([30, 15, 45])
        hour = rng.randrange(24)
        tid, text = _time_verbose(subtype, hour, rng)
        return "exact", (subtype, hour), tid, text, (hour, subtype)
    if kind == "every_minute":
        tid, text = _time_every_minute(None, None, rng)
        return "every_minute", None, tid, text, None
    if kind == "minute_step":
        n = rng.choice(MINUTE_STEPS)
        tid, text = _time_minute_step(n, None, rng)
        return "minute_step", n, tid, text, None
    if kind == "hourly":
        tid, text = _time_hourly(None, None, rng)
        return "hourly", None, tid, text, None
    if kind == "hour_step":
        n = rng.choice(HOUR_STEPS)
        tid, text = _time_hour_step(None, n, rng)
        return "hour_step", n, tid, text, None
    if kind == "hourly_at_minute":
        m = rng.randrange(1, 60)
        tid, text = _time_hourly_at_minute(m, None, rng)
        return "hourly_at_minute", m, tid, text, None
    if kind == "window":
        n = rng.choice(MINUTE_STEPS)
        h1 = rng.randrange(0, 22)
        span = rng.choice([2, 3, 4, 4, 5, 6, 8, 9])
        h2 = min(23, h1 + span)
        tid, text = _time_window(n, (h1, h2), rng)
        return "window", (n, h1, h2), tid, text, None
    if kind == "same_minute":
        m = rng.randrange(60)
        hours = tuple(sorted(rng.sample(range(24), k=rng.choice([2, 2, 3]))))
        tid, text = _time_same_minute(m, hours, rng)
        return "same_minute", (m, hours), tid, text, None
    raise AssertionError(f"unknown time kind {kind}")


def _time_fields(time_res):
    """Map a time sample to (minute values, hour values)."""
    kind, payload = time_res[0], time_res[1]
    full_minute = tuple(range(60))
    full_hour = tuple(range(24))
    if kind == "exact":
        minute, hour = payload
        return (minute,), (hour,)
    if kind == "every_minute":
        return full_minute, full_hour
    if kind == "minute_step":
        return tuple(range(0, 60, payload)), full_hour
    if kind == "hourly":
        return (0,), full_hour
    if kind == "hour_step":
        return (0,), tuple(range(0, 24, payload))
    if kind == "hourly_at_minute":
        return (payload,), full_hour
    if kind == "window":
        n, h1, h2 = payload
        return tuple(range(0, 60, n)), tuple(range(h1, h2 + 1))
    if kind == "same_minute":
        m, hours = payload
        return (m,), tuple(hours)
    raise AssertionError(kind)


def sample_pair(rng, level=None):
    """Sample one (phrase, cron, level, template_id) dict.

    Exact-time examples also carry hour/minute metadata used by the combo
    holdout logic. The caller owns ``rng`` so everything is reproducible.
    """
    for _ in range(100):
        lvl = level or _pick_level(rng)
        time_res = _sample_time(rng, lvl)
        tids = [time_res[2]]
        texts = [time_res[3]]
        exact_pair = time_res[4]  # (hour, minute) for exact times only

        # -- day clause
        day_res = None
        if lvl == 2:
            day_res = _sample_day(rng)
        elif lvl >= 3:
            if lvl == 3:
                if rng.random() < 0.72:
                    day_res = _sample_day(rng)
            else:  # 4, 5
                if rng.random() < 0.60:
                    day_res = _sample_day(rng)
        if lvl == 5 and day_res is None:
            day_res = _sample_day(rng)

        # -- month clause
        month_res = None
        if lvl >= 3:
            if lvl == 3:
                month_res = _sample_month(rng, lvl)
            elif rng.random() < 0.5:
                month_res = _sample_month(rng, lvl)
        if lvl == 5 and month_res is None:
            month_res = _sample_month(rng, lvl)

        # -- month clauses that pin the dom field force dom; when that happens
        # the day clause must be dropped entirely (Section 3.3: never both a
        # dom restriction and a dow restriction; and two competing dom pins
        # cannot both hold).
        if month_res is not None and month_res[4] is not None:
            day_res = None

        if day_res is not None:
            tid, text, kind, values = day_res
            tids.append(tid)
            texts.append(text)
            if kind == "dow":
                dow_vals = tuple(values)
                dom_vals = None
            else:
                dom_vals = tuple(values)
                dow_vals = None
        else:
            dom_vals = dow_vals = None

        if month_res is not None:
            tid, text, kind, values, dom_override = month_res
            tids.append(tid)
            texts.append(text)
            month_vals = tuple(values)
            if dom_override is not None:
                dom_vals = tuple(dom_override)
        else:
            month_vals = None

        if month_vals is None:
            month_vals = tuple(range(1, 13))
        if dom_vals is None:
            dom_vals = tuple(range(1, 32))
        if dow_vals is None:
            dow_vals = tuple(range(0, 7))

        minute_vals, hour_vals = _time_fields(time_res)
        spec = CronSpec(minute=minute_vals, hour=hour_vals, dom=dom_vals,
                        month=month_vals, dow=dow_vals)
        cron = spec.to_cron()

        # -- assemble the phrase with surface variation
        order = list(range(len(texts)))
        rng.shuffle(order)
        phrase = " ".join(texts[i] for i in order)

        if rng.random() < 0.35:
            phrase = f"{rng.choice(FRAME_WORDS)} {phrase}"
        r = rng.random()
        if r < 0.12:
            phrase += "."
        elif r < 0.18:
            phrase += ","

        phrase = " ".join(phrase.split()).strip()

        if len(phrase) + len(cron) + 4 > BLOCK_SIZE:
            continue
        if any(c not in ALLOWED_PHRASE_CHARS for c in phrase):
            raise ValueError(f"generator produced out-of-alphabet phrase: {phrase!r}")

        return {"phrase": phrase, "cron": cron, "level": lvl,
                "template_id": "+".join(tids),
                "hour": exact_pair[0] if exact_pair else None,
                "minute": exact_pair[1] if exact_pair else None}

    raise RuntimeError("could not sample a valid pair after 100 attempts")


def _sample_day(rng):
    kind = rng.choices(
        ["weekdays", "weekends", "named_days", "day_range", "dom"],
        weights=[18, 16, 30, 18, 18])[0]
    fn = {"weekdays": _day_weekdays, "weekends": _day_weekends,
          "named_days": _day_named, "day_range": _day_range, "dom": _day_dom}[kind]
    return fn(rng)


def sample_training_example(rng):
    """Sample an example guaranteed to belong to the train split (online mode)."""
    while True:
        cand = sample_pair(rng)
        tids = cand["template_id"].split("+")
        if assign_split(cand["phrase"], tids, cand["hour"], cand["minute"]) == "train":
            return cand


def sample_paraphrase_example(rng):
    """Sample an example whose template set intersects the reserved templates."""
    while True:
        cand = sample_pair(rng)
        tids = cand["template_id"].split("+")
        if any(t in RESERVED_TEMPLATES for t in tids):
            return cand
