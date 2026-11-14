import torch

from cronlm.data import Collator, FileCronDataset
from cronlm.tokenizer import Tokenizer


def test_file_dataset_and_collator_masking():
    tok = Tokenizer()
    ids = tok.encode("at 9am", "0 9 * * *")
    phrase_len = len(tok.encode_prompt("at 9am")) - 1  # without bos
    sep_idx = ids.index(tok.sep_id)
    assert sep_idx == len("at 9am") + 1  # bos + phrase chars

    collate = Collator(tok)
    x, y = collate([ids])
    assert x.shape == y.shape == (1, len(ids) - 1)
    # prompt targets (positions up to and including <sep>) must be -100
    assert torch.all(y[0, :sep_idx] == -100)
    # targets after <sep> (cron tokens + <eos>) must be real ids
    assert torch.all(y[0, sep_idx:] != -100)
    assert y[0, sep_idx:].tolist() == ids[sep_idx + 1:]


def test_collator_dynamic_padding():
    tok = Tokenizer()
    a = tok.encode("every hour", "0 * * * *")
    b = tok.encode("every weekday at 9:30am", "30 9 * * 1-5")
    x, y = Collator(tok)([a, b])
    assert x.shape[1] == max(len(a), len(b)) - 1
    # padding collapses to -100 in labels and pad_id in input_ids
    assert torch.all(y[0, len(a) - 1:] == -100)
    assert torch.all(x[0, len(a) - 1:] == tok.pad_id)


def test_online_dataset_items_are_train_bucket(tmp_path):
    import itertools
    from cronlm.data import OnlineCronDataset
    from cronlm.grammar import split_bucket, TEST_BUCKETS

    tok = Tokenizer()
    ds = OnlineCronDataset(seed=123)
    checked = 0
    for i, ids in enumerate(itertools.islice(iter(ds), 300)):
        sep = ids.index(tok.sep_id)
        phrase = tok.decode(ids[1:sep])
        assert split_bucket(phrase) >= TEST_BUCKETS
        checked += 1
    assert checked == 300
