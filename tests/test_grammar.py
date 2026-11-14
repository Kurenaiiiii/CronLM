import random

from cronlm import grammar
from cronlm.cron_spec import parse_cron
from cronlm.tokenizer import BLOCK_SIZE


def test_50k_pairs_all_valid():
    rng = random.Random(20260514)
    seen = {}
    bad = []
    both_restricted = []
    for _ in range(50_000):
        cand = grammar.sample_pair(rng)
        cron = cand["cron"]
        # parses + in-range
        parsed = parse_cron(cron)
        # label conflict check
        prev = seen.get(cand["phrase"])
        if prev is not None and prev != cron:
            bad.append((cand["phrase"], prev, cron))
        else:
            seen[cand["phrase"]] = cron
        # never both dom and dow restricted
        dom_star = parsed.dom_star
        dow_star = parsed.dow_star
        if not dom_star and not dow_star:
            both_restricted.append(cand["phrase"])
        # sequence fits block size
        assert len(cand["phrase"]) + len(cron) + 4 <= BLOCK_SIZE
        # phrase alphabet
        assert all(c in grammar.ALLOWED_PHRASE_CHARS for c in cand["phrase"])
        # tags present
        assert cand["level"] in (1, 2, 3, 4, 5)
        assert cand["template_id"]

    assert not bad, f"label conflicts: {bad[:3]}"
    assert not both_restricted, f"dom+dow both restricted: {both_restricted[:3]}"


def test_reserved_templates_never_land_in_train():
    rng = random.Random(99)
    for _ in range(5000):
        cand = grammar.sample_pair(rng)
        tids = cand["template_id"].split("+")
        split = grammar.assign_split(cand["phrase"], tids, cand["hour"], cand["minute"])
        reserved_exact = (cand["hour"] is not None
                          and grammar.is_reserved_combo(cand["hour"], cand["minute"]))
        if any(t in grammar.RESERVED_TEMPLATES for t in tids):
            # combo holdout takes priority over paraphrase for exact times
            assert split == ("combo" if reserved_exact else "paraphrase")
        if reserved_exact:
            assert split == "combo"
        if split == "train":
            assert all(t not in grammar.RESERVED_TEMPLATES for t in tids)


def test_online_training_examples_are_never_val_or_test():
    rng = random.Random(4242)
    for _ in range(2000):
        cand = grammar.sample_training_example(rng)
        tids = cand["template_id"].split("+")
        assert grammar.assign_split(cand["phrase"], tids,
                                    cand["hour"], cand["minute"]) == "train"


def test_reserved_templates_exist_in_the_tables():
    known = grammar.all_template_ids()
    missing = grammar.RESERVED_TEMPLATES - known
    assert not missing, f"reserved template ids not present in tables: {sorted(missing)}"


def test_level_distribution_roughly_respected():
    rng = random.Random(7)
    from collections import Counter
    counts = Counter(grammar.sample_pair(rng)["level"] for _ in range(20_000))
    total = sum(counts.values())
    expected = {1: 0.15, 2: 0.30, 3: 0.25, 4: 0.20, 5: 0.10}
    for level, share in expected.items():
        frac = counts[level] / total
        assert abs(frac - share) < 0.05, f"L{level}: expected ~{share}, got {frac:.3f}"


def test_split_buckets_are_stable():
    from cronlm.grammar import split_bucket, VAL_BUCKETS, TEST_BUCKETS
    # same input -> same bucket, always
    assert split_bucket("every weekday at 9am") == split_bucket("every weekday at 9am")
    a, b = split_bucket("a"), split_bucket("b")
    assert 0 <= a < 1000 and 0 <= b < 1000
