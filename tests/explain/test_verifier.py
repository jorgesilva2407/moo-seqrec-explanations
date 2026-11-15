import numpy as np
import torch

from seqrec.explain.instances import Instance
from seqrec.explain.verifier import Evaluation, Verifier

NUM_ITEMS, LENGTH = 12, 6


class LinearScorer:
    """Stand-in scorer: the logits are the sum of one fixed row per item present in the sequence."""

    num_items, max_length, padding_id, device = NUM_ITEMS, LENGTH, 0, torch.device("cpu")

    def __init__(self) -> None:
        self.table = torch.randn(NUM_ITEMS + 1, NUM_ITEMS, generator=torch.Generator().manual_seed(0))
        self.table[0] = 0.0
        self.seen: list[torch.Tensor] = []

    def logits(self, seqs: torch.Tensor, max_batch: int | None = None) -> torch.Tensor:
        self.seen.append(seqs.clone())
        return self.table[seqs].sum(dim=1)


def make_instance(history: list[int], target: int) -> Instance:
    items = np.array(history, dtype=np.int64)
    return Instance(0, 1, items, items, target, 0, 1, 0)


def margin(scorer: LinearScorer, seq: list[int], target: int, seen: list[int], k: int) -> float:
    logits = scorer.table[torch.tensor(seq)].sum(dim=0)
    others = [float(logits[i - 1]) for i in range(1, NUM_ITEMS + 1) if i != target and i not in seen]
    return float(logits[target - 1]) - sorted(others, reverse=True)[k - 1]


def test_worked_example_of_the_proposal():
    # n = 5, X = {2, 4}, M = 3
    nec = torch.tensor([[-1.2, -0.4, 0.3]])
    suf = torch.tensor([[0.8, 0.5, -0.2]])
    ev = Evaluation(nec=nec, suf=suf, size=torch.tensor([2]))
    torch.testing.assert_close(ev.objectives, torch.tensor([[-1.3 / 3, -1.1 / 3, 2.0]]))
    torch.testing.assert_close(ev.pn, torch.tensor([2 / 3]))
    torch.testing.assert_close(ev.ps, torch.tensor([2 / 3]))


def test_interventions_on_a_left_padded_history():
    scorer = LinearScorer()
    history, target, k = [1, 2, 3, 4], 9, 2
    fills = np.array([[5, 6, 7, 8], [8, 7, 6, 5]], dtype=np.int64)
    verifier = Verifier(scorer, [make_instance(history, target)], [fills], k=k)
    # X = second and fourth item; the bit on a padded column must be ignored
    mask = torch.tensor([[True, False, False, True, False, True]])
    ev = verifier.evaluate(torch.tensor([0]), mask)

    assert scorer.seen[-1].tolist() == [
        [0, 0, 1, 6, 3, 8],  # necessity: X replaced by each background
        [0, 0, 1, 7, 3, 5],
        [0, 0, 5, 2, 7, 4],  # sufficiency: everything but X replaced
        [0, 0, 8, 2, 6, 4],
    ]
    expected = [margin(scorer, seq, target, history, k) for seq in scorer.seen[-1].tolist()]
    torch.testing.assert_close(ev.nec, torch.tensor([expected[:2]]))
    torch.testing.assert_close(ev.suf, torch.tensor([expected[2:]]))
    assert ev.size.tolist() == [2]
    assert verifier.passes == 4


def test_batches_of_several_instances_are_deterministic_and_chunk_independent():
    scorer = LinearScorer()
    rng = np.random.default_rng(0)
    instances = [make_instance([1, 2, 3, 4, 5, 6], 10), make_instance([7, 8, 9], 11)]
    fills = [rng.integers(1, NUM_ITEMS + 1, size=(3, inst.n)) for inst in instances]
    index = torch.tensor([0, 1, 1, 0, 1])
    masks = torch.rand(5, LENGTH, generator=torch.Generator().manual_seed(1)) < 0.5
    big = Verifier(scorer, instances, fills, k=3, chunk=1000).evaluate(index, masks)
    small = Verifier(scorer, instances, fills, k=3, chunk=1).evaluate(index, masks)
    torch.testing.assert_close(big.objectives, small.objectives)

    # each row equals the same mask evaluated alone
    alone = Verifier(scorer, instances, fills, k=3).evaluate(index[2:3], masks[2:3])
    torch.testing.assert_close(alone.objectives, big.objectives[2:3])

    # full mask: necessity sees only backgrounds, sufficiency sees the untouched history
    verifier = Verifier(scorer, instances, fills, k=3)
    full = verifier.evaluate(torch.arange(2), verifier.full_masks())
    assert full.size.tolist() == [6, 3]
    assert torch.all(full.suf == full.suf[:, :1])
