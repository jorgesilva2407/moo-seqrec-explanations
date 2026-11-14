import subprocess
import sys
import textwrap

import torch

from seqrec.models.loader import load
from seqrec.models.sasrec import SASRec, SASRecConfig
from seqrec.training.checkpoint import build_checkpoint, save_checkpoint


def make_model() -> tuple[SASRec, SASRecConfig]:
    cfg = SASRecConfig(num_items=20, max_length=8, hidden_size=16, num_layers=2, num_heads=1, dropout=0.1, ffn_size=16)
    torch.manual_seed(0)
    return SASRec(cfg).eval(), cfg


def test_round_trip_in_fresh_process(tmp_path):
    model, cfg = make_model()
    path = tmp_path / "best.pt"
    save_checkpoint(path, build_checkpoint(model, cfg, {"seed": 1}, seed=1, data_hash="abc"))
    seqs = torch.randint(0, 21, (5, 8))
    seqs[:, -1] = torch.randint(1, 21, (5,))
    torch.save(seqs, tmp_path / "seqs.pt")
    script = textwrap.dedent(
        f"""
        import torch
        from seqrec.models.loader import load
        seqs = torch.load({str(tmp_path / 'seqs.pt')!r})
        scorer = load({str(path)!r}, device="cpu")
        torch.save(scorer.logits(seqs).clone(), {str(tmp_path / 'out.pt')!r})
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    fresh = torch.load(tmp_path / "out.pt")
    in_process = load(path, device="cpu").logits(seqs)
    torch.testing.assert_close(fresh, in_process, rtol=0, atol=1e-6)
    assert fresh.shape == (5, 20)


def test_scorer_semantics(tmp_path):
    model, cfg = make_model()
    path = tmp_path / "best.pt"
    save_checkpoint(path, build_checkpoint(model, cfg, {}, seed=0, data_hash="abc"))
    scorer = load(path, device="cpu", max_batch=2)
    assert (scorer.num_items, scorer.max_length, scorer.padding_id) == (20, 8, 0)
    assert len(scorer.checkpoint_id) == 12

    seqs = torch.randint(1, 21, (5, 6))
    full = scorer.logits(seqs, max_batch=100)
    torch.testing.assert_close(scorer.logits(seqs, max_batch=1), full, rtol=0, atol=1e-5)

    # a shorter input equals the same items left-padded; padding in the middle is compacted away
    with torch.no_grad():
        direct = model.last_logits(torch.cat([torch.zeros(5, 2, dtype=torch.long), seqs], dim=1))
    torch.testing.assert_close(full, direct, rtol=0, atol=1e-5)
    a = scorer.logits(torch.tensor([[3, 0, 4]]))
    b = scorer.logits(torch.tensor([[0, 3, 4]]))
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
