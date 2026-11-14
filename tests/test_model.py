import math

import torch

from cronlm.model import GPT
from cronlm.tokenizer import Tokenizer


def _tiny():
    return GPT(d_model=128, n_layer=4, n_head=4, ff_dim=512)


def test_output_shapes():
    tok = Tokenizer()
    model = _tiny()
    a = tok.encode("every hour", "0 * * * *")
    b = tok.encode("at 9am", "0 9 * * *")
    width = max(len(a), len(b))
    ids = torch.tensor([a + [tok.pad_id] * (width - len(a)),
                        b + [tok.pad_id] * (width - len(b))])
    logits, loss = model(ids, targets=None)
    assert logits.shape == (2, width, len(tok))


def test_loss_is_scalar_and_finite():
    tok = Tokenizer()
    model = _tiny()
    ids = torch.tensor([tok.encode("every hour", "0 * * * *")])
    labels = ids.clone()
    logits, loss = model(ids, targets=labels)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_param_counts_match_configs():
    assert _tiny().num_params() == 819_968            # ~0.8M, tiny
    assert GPT(d_model=256, n_layer=6, n_head=8, ff_dim=1024).num_params() == 4_792_320   # ~4.8M, small
    assert GPT(d_model=384, n_layer=8, n_head=6, ff_dim=1536).num_params() == 14_276_352  # ~14M, base


def test_causality_change_future_token_no_effect_on_past():
    torch.manual_seed(0)
    tok = Tokenizer()
    model = _tiny().eval()
    ids = torch.tensor([tok.encode("every weekday at 9am", "0 9 * * 1-5")])
    with torch.no_grad():
        base, _ = model(ids)
        for t in range(5, ids.shape[1]):
            mutated = ids.clone()
            mutated[0, t] = (mutated[0, t] + 7) % len(tok)
            changed, _ = model(mutated)
            # logits at positions strictly before t must be identical
            assert torch.allclose(base[0, :t], changed[0, :t], atol=1e-6), f"pos {t}"


def test_right_padding_does_not_leak_into_real_logits():
    torch.manual_seed(0)
    tok = Tokenizer()
    model = _tiny().eval()
    single = tok.encode("at 9am", "0 9 * * *")
    other = tok.encode("every hour", "0 * * * *")
    width = max(len(single), len(other))
    ids = torch.tensor([single + [tok.pad_id] * (width - len(single)),
                        other + [tok.pad_id] * (width - len(other))],
                       dtype=torch.long)
    with torch.no_grad():
        padded_logits, _ = model(ids)
        single_logits, _ = model(torch.tensor([single], dtype=torch.long))
    assert torch.allclose(padded_logits[0, :len(single)],
                          single_logits[0, :len(single)], atol=1e-6)


def test_left_padded_generate_runs():
    tok = Tokenizer()
    model = _tiny().eval()
    p1 = tok.encode_prompt("every hour")
    p2 = tok.encode_prompt("at 9am on mondays")
    out = model.generate([p1, p2], max_new_tokens=8, eos_id=tok.eos_id)
    assert len(out) == 2
    assert all(isinstance(x, list) for x in out)


def test_tiny_overfit_32_examples():
    """Tiny overfit test (PRD Section 11.5): loss goes under 0.05 on 32
    examples within 300 steps (batch 32)."""
    torch.manual_seed(1337)
    tok = Tokenizer()
    from cronlm.data import Collator
    model = _tiny()
    pairs = [
        ("every weekday at 9am", "0 9 * * 1-5"),
        ("at 9:30am", "30 9 * * *"),
        ("every 15 minutes", "*/15 * * * *"),
        ("first of the month at midnight", "0 0 1 * *"),
        ("in march at 9am", "0 9 * 3 *"),
        ("every friday at 6pm", "0 18 * * 5"),
        ("hourly", "0 * * * *"),
        ("on the 15th at 9am", "0 9 15 * *"),
    ]
    batch_ids = [tok.encode(p, c) for p, c in pairs * 4]
    collate = Collator(tok)
    inputs, labels = collate(batch_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    model.train()
    loss = None
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets=labels)
        loss.backward()
        optimizer.step()
    assert loss is not None
    assert loss.item() < 0.05, f"expected loss < 0.05 after 300 steps, got {loss.item():.4f}"
