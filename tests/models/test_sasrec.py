import torch

from seqrec.models.sasrec import SASRec, SASRecConfig


def tiny_config() -> SASRecConfig:
    return SASRecConfig(num_items=20, max_length=8, hidden_size=16, num_layers=2, num_heads=1, dropout=0.0, ffn_size=16)


def test_causality():
    torch.manual_seed(0)
    model = SASRec(tiny_config()).eval()
    seqs = torch.randint(1, 21, (4, 8))
    seqs[0, :3] = 0  # left padding in one row
    changed = seqs.clone()
    j = 5
    changed[:, j] = changed[:, j] % 20 + 1  # always a different valid item
    with torch.no_grad():
        h1, h2 = model.encode(seqs), model.encode(changed)
    assert torch.allclose(h1[:, :j], h2[:, :j], atol=1e-6)
    assert (h1[:, j] - h2[:, j]).abs().max() > 1e-4


def test_padding_does_not_leak_and_no_nan():
    torch.manual_seed(0)
    model = SASRec(tiny_config()).eval()
    real = torch.randint(1, 21, (1, 4))
    short = real
    padded = torch.cat([torch.zeros(1, 4, dtype=torch.long), real], dim=1)
    all_pad = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        a = model.last_logits(short)
        b = model.last_logits(padded)
        c = model.last_logits(all_pad)
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.isnan(c).any()
    assert not torch.isnan(model.encode(all_pad)).any()


def test_config_round_trip():
    cfg = tiny_config()
    assert SASRecConfig.from_dict(cfg.to_dict()) == cfg
