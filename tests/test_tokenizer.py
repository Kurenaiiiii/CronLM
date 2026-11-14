import json

import pytest

from cronlm.tokenizer import Tokenizer, BLOCK_SIZE


def test_vocab_shape():
    tok = Tokenizer()
    assert len(tok) == 48
    assert tok.pad_id == 0 and tok.bos_id == 1 and tok.sep_id == 2 and tok.eos_id == 3
    # specials first, then sorted allowed characters
    assert tok.tokens[:4] == ["<pad>", "<bos>", "<sep>", "<eos>"]
    assert tok.tokens[4:] == sorted(tok.tokens[4:])


def test_roundtrip():
    tok = Tokenizer()
    phrase, cron = "every weekday at 9:30am", "30 9 * * 1-5"
    ids = tok.encode(phrase, cron)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert ids[ids.index(tok.sep_id)] == tok.sep_id
    # prompt region round-trips to the phrase
    sep = ids.index(tok.sep_id)
    assert tok.decode(ids[:sep]) == phrase
    assert tok.decode(ids[sep + 1:]) == cron


def test_encode_prompt_ends_at_sep():
    tok = Tokenizer()
    ids = tok.encode_prompt("every hour")
    assert ids[0] == tok.bos_id and ids[-1] == tok.sep_id
    assert tok.decode(ids) == "every hour"


def test_unknown_character_error_lists_characters():
    tok = Tokenizer()
    with pytest.raises(ValueError) as exc:
        tok.encode("every WEEKDAY at 9am — really", "0 9 * * 1-5")
    msg = str(exc.value)
    assert "W" in msg or "E" in msg  # offending characters are listed
    assert "every WEEKDAY" in msg


def test_block_size_guard():
    tok = Tokenizer()
    with pytest.raises(ValueError):
        tok.encode("a" * (BLOCK_SIZE + 1), "0 9 * * *")


def test_vocab_save_load(tmp_path):
    path = tmp_path / "vocab.json"
    Tokenizer().save(str(path))
    tok = Tokenizer.load(str(path))
    assert tok.tokens == Tokenizer().tokens
    # vocab.json must be deterministic JSON of the tokens list
    data = json.loads(path.read_text())
    assert data["tokens"][:4] == ["<pad>", "<bos>", "<sep>", "<eos>"]


def test_vocab_load_mismatch_rejected(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"tokens": ["<pad>", "<bos>"]}))
    with pytest.raises(ValueError):
        Tokenizer.load(str(path))
