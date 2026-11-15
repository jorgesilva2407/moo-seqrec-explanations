import numpy as np
import torch

from seqrec.data import Histories
from seqrec.explain.backgrounds import EVAL_STREAM, draw_backgrounds, nearest_neighbours
from seqrec.explain.instances import build_instances, eligible_users
from seqrec.explain.verifier import Verifier
from seqrec.models.loader import load
from seqrec.models.sasrec import SASRec, SASRecConfig
from seqrec.training.checkpoint import build_checkpoint, save_checkpoint

NUM_ITEMS, LENGTH = 60, 8
STRATA = ((4, 6), (7, 8))


def make_scorer(tmp_path):
    cfg = SASRecConfig(num_items=NUM_ITEMS, max_length=LENGTH, hidden_size=16, num_layers=2, num_heads=1, ffn_size=16)
    torch.manual_seed(0)
    path = tmp_path / "best.pt"
    save_checkpoint(path, build_checkpoint(SASRec(cfg), cfg, {}, seed=0, data_hash="abc"))
    return load(path, device="cpu")


def make_histories(num_users: int = 80) -> Histories:
    rng = np.random.default_rng(0)
    lengths = rng.integers(4, 15, size=num_users)  # includes validation and test items
    items = np.concatenate([rng.choice(NUM_ITEMS, size=n, replace=False) + 1 for n in lengths]).astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    split = np.zeros(len(items), dtype=np.uint8)
    split[offsets[1:] - 2], split[offsets[1:] - 1] = 1, 2
    return Histories(np.arange(num_users, dtype=np.int64) + 100, offsets, items, split)


def test_eligible_users_use_the_truncated_length():
    histories = make_histories()
    before_test = np.diff(histories.offsets) - 1
    low, high = eligible_users(histories, LENGTH, STRATA)
    assert set(low) == set(np.flatnonzero((before_test >= 4) & (before_test <= 6)))
    assert set(high) == set(np.flatnonzero(before_test >= 7))  # longer histories are cut to 8


def test_built_instances_are_valid_and_reproducible(tmp_path):
    scorer, histories = make_scorer(tmp_path), make_histories()
    popularity = np.bincount(histories.items, minlength=NUM_ITEMS + 1).astype(np.int64)
    kwargs = dict(per_stratum=5, seed=3, m=4, k=5, strata=STRATA, neighbours=3)
    instances, stats = build_instances(histories, scorer, popularity, **kwargs)
    again, _ = build_instances(histories, scorer, popularity, **kwargs)
    assert [(i.user_index, i.target) for i in instances] == [(i.user_index, i.target) for i in again]
    assert instances and all(s["kept"] <= 5 and s["proposed"] - s["discarded"] >= s["kept"] for s in stats["strata"])

    for inst in instances:
        low, high = STRATA[inst.stratum]
        assert low <= inst.n <= high
        assert inst.target not in inst.seen and 0 <= inst.target_rank < 5
        assert inst.history.tolist() == histories.full(inst.user_index)[:-1][-LENGTH:].tolist()
        assert inst.user_id == inst.user_index + 100

    embeddings = scorer.item_embeddings()
    fills = draw_backgrounds(instances, popularity, embeddings, 4, seed=3, neighbours=3)
    verifier = Verifier(scorer, instances, fills, k=5)
    full = verifier.evaluate(torch.arange(len(instances)), verifier.full_masks())
    assert torch.all(full.necessity < 0)  # the validity filter, recomputed from redrawn backgrounds
    assert torch.all(full.suf >= 0)  # the untouched history keeps the target in the top-k

    near = nearest_neighbours(embeddings, torch.tensor([i.target for i in instances]), 3).numpy()
    other = draw_backgrounds(instances, popularity, embeddings, 4, seed=3, stream=EVAL_STREAM, neighbours=3)
    for inst, a, b, banned in zip(instances, fills, other, near):
        assert a.shape == (4, inst.n) and not np.array_equal(a, b)
        drawn = set(a.ravel()) | set(b.ravel())
        assert not drawn & ({0, inst.target} | set(inst.seen) | set(banned))

    held_out, _ = build_instances(
        histories, scorer, popularity, exclude_users=[i.user_index for i in instances], **kwargs
    )
    assert not {i.user_index for i in held_out} & {i.user_index for i in instances}


def test_fill_variants(tmp_path):
    scorer, histories = make_scorer(tmp_path), make_histories()
    popularity = np.bincount(histories.items, minlength=NUM_ITEMS + 1).astype(np.int64)
    instances, _ = build_instances(histories, scorer, popularity, per_stratum=5, seed=3, m=4, k=5, strata=STRATA, neighbours=3)
    embeddings = scorer.item_embeddings()
    draw = lambda **kw: draw_backgrounds(instances, popularity, embeddings, 6, seed=3, neighbours=3, histories=histories, **kw)

    padding = draw(fill="padding")
    assert all(f.shape == (6, i.n) and not f.any() for f, i in zip(padding, instances))
    verifier = Verifier(scorer, instances, padding, k=5)
    result = verifier.evaluate(torch.arange(len(instances)), verifier.full_masks())
    assert torch.equal(result.size, torch.tensor([i.n for i in instances]))
    assert torch.allclose(result.nec, result.nec[:, :1].expand_as(result.nec))  # deletion has no randomness

    users = draw(fill="users")
    again = draw(fill="users")
    near = nearest_neighbours(embeddings, torch.tensor([i.target for i in instances]), 3).numpy()
    sequences = [histories.full(u)[:-1].tolist() for u in range(len(histories))]
    copied = 0
    for fills, same, inst, close in zip(users, again, instances, near):
        assert fills.shape == (6, inst.n) and np.array_equal(fills, same)
        assert fills.min() >= 1 and not np.isin(fills, np.concatenate([inst.seen, close, [inst.target]])).any()
        for row in fills.tolist():  # rows left untouched are a contiguous piece of some other user
            copied += any(row == seq[j : j + inst.n] for u, seq in enumerate(sequences) if u != inst.user_index
                          for j in range(len(seq) - inst.n + 1))
    assert copied > 0
