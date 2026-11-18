import numpy as np
import pytest
import torch
from test_verifier import NUM_ITEMS, LinearScorer, make_instance

from seqrec.explain.methods import DETERMINISTIC, METHODS, Context, nested
from seqrec.explain.scheduler import job_specs, load_job, method_seed, run_jobs, save_job
from seqrec.explain.verifier import Verifier

BUDGET = 40


def setup(budget: int = BUDGET):
    scorer = LinearScorer()
    rng = np.random.default_rng(0)
    instances = [make_instance([1, 2, 3, 4, 5, 6], 10), make_instance([7, 8, 9, 1, 2], 11)]
    fills = [rng.integers(1, NUM_ITEMS + 1, size=(3, inst.n)) for inst in instances]
    contexts = [
        Context(inst.n, np.array([1.0, 1.0, inst.n]), np.linspace(0, 1, inst.n), budget, pop_size=8, scalar_pop_size=4)
        for inst in instances
    ]
    return scorer, instances, fills, contexts


def run(methods, seeds=(0, 1), concurrency=64, budget=BUDGET, with_eval=False):
    scorer, instances, fills, contexts = setup(budget)
    verifier = Verifier(scorer, instances, fills, k=3)
    eval_verifier = Verifier(scorer, instances, [f[::-1].copy() for f in fills], k=3) if with_eval else None
    jobs = []
    run_jobs(
        verifier, job_specs(range(2), methods, seeds), contexts,
        lambda spec: method_seed(spec, spec.instance), jobs.append, concurrency, eval_verifier,
    )
    return jobs, verifier


@pytest.mark.parametrize("name", sorted(METHODS))
def test_every_method_respects_the_budget_and_stores_true_objectives(name):
    jobs, verifier = run([name])
    assert len(jobs) == 2 * (1 if name in DETERMINISTIC else 2)
    total = 0
    for job in jobs:
        masks = np.asarray(job.masks)
        total += len(masks)
        assert len(masks) and len({m.tobytes() for m in masks}) == len(masks)  # no mask is paid for twice
        if name == "enumeration":
            assert len(masks) == job.n + job.n * (job.n - 1) // 2
        else:
            assert len(masks) <= BUDGET
        padded = np.zeros((len(masks), verifier.length), dtype=bool)
        padded[:, verifier.length - job.n :] = masks
        direct = verifier.evaluate(torch.full((len(masks),), job.spec.instance), torch.from_numpy(padded))
        np.testing.assert_allclose(job.objectives, direct.objectives.numpy(), rtol=1e-5, atol=1e-6)
        front = job.objectives[job.front()]
        for row in job.objectives:  # nothing evaluated dominates a front member
            assert not np.any(np.all(row <= front, axis=1) & np.any(row < front, axis=1))
    assert total * 2 * verifier.m == verifier.passes - total * 2 * verifier.m  # search + the direct re-check


def test_results_do_not_depend_on_concurrency():
    together, _ = run(["nsga2", "scalar_ga", "random", "greedy"], concurrency=64)
    alone, _ = run(["nsga2", "scalar_ga", "random", "greedy"], concurrency=1)
    key = lambda job: (job.spec.instance, job.spec.method, job.spec.seed)
    for a, b in zip(sorted(together, key=key), sorted(alone, key=key)):
        assert key(a) == key(b)
        np.testing.assert_array_equal(np.asarray(a.masks), np.asarray(b.masks))
        np.testing.assert_allclose(a.objectives, b.objectives)


def test_seeds_differ_and_evolution_uses_the_whole_budget():
    jobs, _ = run(["nsga2"], budget=30)
    first, second = [job for job in jobs if job.spec.instance == 0]
    assert len(first.masks) == len(second.masks) == 30
    assert not np.array_equal(np.asarray(first.masks), np.asarray(second.masks))
    assert all(mask.any() for job in jobs for mask in job.masks)  # empty masks are repaired


def test_scripted_baselines_ask_the_expected_masks():
    assert nested(np.array([2, 0, 1])).tolist() == [[False, False, True], [True, False, True], [True, True, True]]
    jobs, _ = run(["recency", "similarity", "ablation"], budget=100)
    by_method = {job.spec.method: job for job in jobs if job.spec.instance == 0}
    recency = np.asarray(by_method["recency"].masks)
    assert recency[0].tolist() == [False] * 5 + [True] and recency[-1].all() and len(recency) == 6
    assert np.asarray(by_method["similarity"].masks)[0].tolist() == [False] * 5 + [True]  # most similar is last
    ablation = by_method["ablation"]
    assert np.array_equal(np.asarray(ablation.masks)[:6], np.eye(6, dtype=bool))
    best_single = int(np.argmin(ablation.objectives[:6, 0]))
    assert np.asarray(ablation.masks)[6][best_single]  # the first pair extends the most necessary position


def test_front_is_rescored_and_files_round_trip(tmp_path):
    jobs, _ = run(["nsga2"], seeds=(0,), with_eval=True)
    job = jobs[0]
    assert job.front_scores.shape == (len(job.front()), 5)
    np.testing.assert_array_equal(job.front_scores[:, 2], job.objectives[job.front(), 2])  # same sizes
    path = tmp_path / "nsga2" / "job.npz"
    save_job(path, job, {"method": "nsga2", "seed": 0})
    stored = load_job(path)
    assert stored["meta"]["evaluations"] == len(job.masks) and stored["meta"]["n"] == job.n
    np.testing.assert_array_equal(stored["masks"], np.asarray(job.masks))
    np.testing.assert_allclose(stored["objectives"], job.objectives, rtol=1e-6)
    np.testing.assert_allclose(stored["front_objectives"], job.front_scores[:, :3], rtol=1e-6)
    np.testing.assert_array_equal(stored["front"], job.front())


def test_operator_switches_change_the_search():
    from dataclasses import replace

    from seqrec.explain.methods import NSGA2Search, weight_lattice

    assert [len(weight_lattice(d)) for d in (1, 2, 3, 4)] == [3, 6, 10, 15]
    ctx = replace(setup()[3][0], n=40, pop_size=200)
    first = lambda **kw: NSGA2Search(replace(ctx, **kw), np.random.default_rng(0)).ask(10**6)
    assert first().sum(axis=1).mean() < 8 < 15 < first(init="uniform").sum(axis=1).mean()

    def offspring(**kw):
        search = NSGA2Search(replace(ctx, **kw), np.random.default_rng(0))
        parents = search.ask(10**6)
        search.tell(np.random.default_rng(1).random((len(parents), 3)))
        return parents, search.ask(10**6)

    parents, children = offspring(crossover=False, mutation_scale=1e-9)
    assert {c.tobytes() for c in children} <= {p.tobytes() for p in parents}  # copies only (duplicates dropped)
    _, mutated = offspring(crossover=False, mutation_scale=10.0)
    assert not {c.tobytes() for c in mutated} <= {p.tobytes() for p in parents}


def test_gece_edits_are_mapped_to_position_masks():
    from seqrec.explain.gece import run_gece

    scorer, instances, _, _ = setup()
    inst = instances[0]
    found = [run_gece(scorer, inst, seed, k=3, budget=2000, population=20, max_edits=3) for seed in range(6)]
    again = run_gece(scorer, inst, 0, k=3, budget=2000, population=20, max_edits=3)
    np.testing.assert_array_equal(found[0].masks, again.masks)  # reproducible from the seed
    assert any(r.chosen is not None for r in found)
    for r in found:
        assert r.masks.shape[1] == inst.n and r.masks.any(axis=1).all() and r.queries <= 2000
        assert len({m.tobytes() for m in r.masks}) == len(r.masks)
        if r.chosen is None:
            continue
        mask, cf = r.masks[r.chosen], np.asarray(r.counterfactual)
        assert 1 <= mask.sum() <= 3
        original = np.concatenate([np.zeros(scorer.max_length - inst.n, dtype=np.int64), inst.history])
        np.testing.assert_array_equal(mask, (cf != original)[-inst.n :])
        logits = scorer.table[torch.from_numpy(cf)].sum(dim=0)  # the target really left the top-3 candidates
        others = [float(logits[i - 1]) for i in range(1, NUM_ITEMS + 1)
                  if i != inst.target and i not in inst.seen and i not in cf]
        assert sum(o > float(logits[inst.target - 1]) for o in others) >= 3


def test_seeded_nsga2_starts_from_the_ablation_ranking():
    jobs, _ = run(["nsga2_seeded", "ablation"], seeds=(0,), budget=60)
    by_method = {job.spec.method: job for job in jobs if job.spec.instance == 0}
    seeded, ablation = by_method["nsga2_seeded"], by_method["ablation"]
    masks = np.asarray(seeded.masks)
    assert np.array_equal(masks[:6], np.eye(6, dtype=bool))  # single positions first
    scored = {m.tobytes() for m in masks}
    assert {m.tobytes() for m in np.asarray(ablation.masks)} <= scored  # everything the ranking scores
    assert len(masks) > len(ablation.masks)  # and then the evolution goes on


def test_shapley_is_exact_for_additive_objectives():
    from seqrec.explain.methods import ShapleyRanking

    n, weights = 6, np.array([[-3.0, 1.0], [2.0, -1.0], [0.5, -4.0], [-1.0, 0.0], [0.0, 2.0], [1.5, 0.5]])
    ctx = Context(n, np.ones(3), np.zeros(n), budget=200)
    method = ShapleyRanking(ctx, np.random.default_rng(0))
    while not method.done:
        masks = method.ask(10**6)
        method.tell(np.column_stack([masks @ weights, masks.sum(axis=1)]))
    np.testing.assert_allclose(method.shapley, weights)  # every permutation gives the exact value


def test_decomposition_and_indicator_searches_ask_whole_generations():
    from dataclasses import replace

    ctx = replace(setup()[3][0], n=40, pop_size=25, partitions=5)
    for name, size in (("moead", 21), ("nsga3", 21), ("smsemoa", 25), ("nsga2_nc", 25)):
        method = METHODS[name](ctx, np.random.default_rng(0))
        first = method.ask(10**6)
        method.tell(np.random.default_rng(1).random((len(first), 3)))
        assert len(first) == size and 0 < len(method.ask(10**6)) <= size, name
