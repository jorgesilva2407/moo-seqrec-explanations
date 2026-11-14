import torch

from seqrec.training.metrics import metrics_from_ranks, target_ranks


def test_ranks_with_and_without_exclusion():
    logits = torch.tensor([[0.1, 0.9, 0.5]])
    target = torch.tensor([2])
    assert target_ranks(logits, target).tolist() == [1]
    seen = torch.tensor([[False, True, False]])
    assert target_ranks(logits, target, seen).tolist() == [0]
    target_seen = torch.tensor([[False, False, True]])
    assert target_ranks(logits, target, target_seen).tolist() == [3]  # never retrievable


def test_hr_and_ndcg():
    ranks = torch.tensor([0, 2, 30])
    m = metrics_from_ranks(ranks, (1, 5, 20))
    assert m["hr@1"].tolist() == [1.0, 0.0, 0.0]
    assert m["hr@5"].tolist() == [1.0, 1.0, 0.0]
    assert torch.isclose(m["ndcg@5"][1], torch.tensor(0.5))  # 1 / log2(2 + 2)
    assert m["ndcg@20"][2] == 0.0
