import numpy as np
from test_search import run

from seqrec.commands.explain_rescore import rescore, top_per_size, write_variant
from seqrec.explain.analysis import load_jobs
from seqrec.explain.scheduler import load_job, save_job


def test_top_per_size_takes_the_lowest_necessity_of_each_size():
    objectives = np.array([[0.5, np.nan, 1], [0.1, np.nan, 1], [0.3, np.nan, 2], [0.2, np.nan, 1], [0.9, np.nan, 3]])
    assert top_per_size(objectives, 1, None).tolist() == [1, 2, 4]
    assert top_per_size(objectives, 2, None).tolist() == [1, 3, 2, 4]
    assert top_per_size(objectives, 2, 2).tolist() == [1, 3, 2]  # sizes above max_size are left out


def test_rescoring_reproduces_the_stored_front_and_writes_a_readable_variant(tmp_path):
    jobs, _ = run(["nsga2_nc"], seeds=(0,), budget=10, with_eval=True)
    job = jobs[0]
    source = tmp_path / "main" / "ckpt" / "nsga2_nc" / "u1_t10_s0.npz"
    save_job(source, job, {"method": "nsga2_nc", "seed": 0, "user_index": 1, "target": 10, "m": 3, "passes": job.passes})
    stored = load_job(source)
    # the evaluation verifier of run(): the reversed search backgrounds
    from test_search import setup
    from seqrec.explain.verifier import Verifier
    scorer, instances, fills, _ = setup()
    eval_verifier = Verifier(scorer, instances, [f[::-1].copy() for f in fills], k=3)
    again = rescore(eval_verifier, job.spec.instance, stored["masks"][stored["front"]])
    np.testing.assert_allclose(again[:, :3], stored["front_objectives"], rtol=1e-6)

    rows = top_per_size(stored["objectives"], 2, None)
    scores = rescore(eval_verifier, job.spec.instance, stored["masks"][rows])
    target = tmp_path / "rescore" / "ckpt" / "nsga2_nc_all_top2" / source.name
    write_variant(target, stored, rows, scores, {**stored["meta"], "method": "nsga2_nc_all_top2", "top_k": 2})
    [record] = load_jobs(tmp_path / "rescore" / "ckpt")
    assert record["method"] == "nsga2_nc_all_top2" and record["passes"] == job.passes
    assert not record["searched_s"]  # left out of the convergence curves
    np.testing.assert_allclose(record["front"], scores[:, :3], rtol=1e-6)
    np.testing.assert_array_equal(record["front_masks"], stored["masks"][rows])
