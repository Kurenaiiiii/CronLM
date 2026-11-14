import json
import os
import subprocess
import sys

import pytest
import torch

from cronlm import cli
from cronlm.tokenizer import Tokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY_CONFIG = os.path.join(REPO_ROOT, "configs", "tiny.yaml")
GOLDEN = os.path.join(REPO_ROOT, "tests", "golden.jsonl")


def _gen(tmp_path, train=256, val=64, test=64):
    out = tmp_path / "data"
    cli.main(["gen-data", "--out", str(out),
              "--train", str(train), "--val", str(val), "--test", str(test),
              "--combo", "24", "--paraphrase", "12",
              "--seed", "1337", "--golden-src", GOLDEN])
    return out


@pytest.fixture()
def tiny_data(tmp_path):
    return _gen(tmp_path)


def test_gen_data_outputs(tmp_path, tiny_data):
    out = tiny_data
    expected = {
        "train.jsonl": 256,
        "val.jsonl": 64,
        "test.jsonl": 64,
        "test_combo_holdout.jsonl": 24,
        "test_paraphrase_holdout.jsonl": 12,
    }
    for name, want in expected.items():
        path = out / name
        assert path.exists(), name
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        assert len(rows) == want, f"{name}: {len(rows)} != {want}"
        assert all(set(r) == {"phrase", "cron", "level", "template_id"} for r in rows)
    # golden was copied next to the generated splits
    golden = out / "golden.jsonl"
    assert golden.exists()
    assert len(golden.read_text().splitlines()) == 200
    # no duplicates in train
    train_phrases = [json.loads(l)["phrase"] for l in (out / "train.jsonl").read_text().splitlines()]
    assert len(train_phrases) == len(set(train_phrases))


def test_gen_data_is_deterministic(tmp_path):
    a = _gen(tmp_path / "a", train=64, val=16, test=24)
    b = _gen(tmp_path / "b", train=64, val=16, test=24)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl",
                 "test_combo_holdout.jsonl", "test_paraphrase_holdout.jsonl"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_smoke_train_resume_and_predict(tmp_path, tiny_data, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(12)
    rc = cli.main(["train", "--config", TINY_CONFIG, "--smoke", "--device", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "parameters" in out and "device: cpu" in out

    last = tmp_path / "runs" / "tiny" / "last.pt"
    best = tmp_path / "runs" / "tiny" / "best.pt"
    vocab = tmp_path / "runs" / "tiny" / "vocab.json"
    assert last.exists() and best.exists() and vocab.exists()
    assert (tmp_path / "runs" / "tiny" / "log.csv").exists()

    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 200
    for key in ("model", "optimizer", "config", "rng", "best"):
        assert key in ckpt

    # resume path prints the resume banner (it starts at step 200 = done)
    rc = cli.main(["train", "--config", TINY_CONFIG, "--smoke", "--device", "cpu",
                   "--resume", str(last)])
    out = capsys.readouterr().out
    assert "resumed" in out

    # checkpoint loads and predicts
    tok = Tokenizer.load(str(vocab))
    assert tok.encode("x", "y")  # sanity: vocab is consistent

    rc = cli.main(["predict", "--ckpt", str(best), "every weekday at 9am"])
    assert rc == 0


def test_online_smoke_train(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(12)
    # gen only val.jsonl for the eval; online never touches train.jsonl
    cli.main(["gen-data", "--out", str(tmp_path / "data"),
              "--train", "0", "--val", "32", "--test", "16",
              "--combo", "0", "--paraphrase", "0",
              "--seed", "1337", "--golden-src", GOLDEN])
    rc = cli.main(["train", "--config", TINY_CONFIG, "--online", "--smoke",
                   "--device", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--online" in out or "online" in out
    assert (tmp_path / "runs" / "tiny" / "last.pt").exists()
