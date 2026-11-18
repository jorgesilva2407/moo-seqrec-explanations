"""Compare result files: hypervolume of each job's front, on a scale shared inside an instance.

Fronts are compared on the objectives re-scored with the independent evaluation backgrounds
(``front_objectives``), so searches that used a different M or budget are on equal terms. For
every instance (user, target) the three objectives are min-max scaled over all fronts being
compared and the reference point is 1.1 in every scaled objective, i.e. slightly worse than the
worst value any compared job reached for that instance.

Records are small dicts (one per result file); analyses that need the whole archive of a job
(convergence, correlation between objectives) read the file again through ``record["path"]``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from seqrec.explain.scheduler import load_job

REFERENCE = 1.1
_HV = HV(ref_point=np.full(3, REFERENCE))
_HV2 = HV(ref_point=np.full(2, REFERENCE))


def load_jobs(stage_dir: Path, source: str = "front_objectives") -> list[dict]:
    """One record per result file under ``stage_dir``.

    ``source`` is ``front_objectives`` (evaluation backgrounds) or ``objectives`` (search
    backgrounds, front rows only). ``front`` holds the points, ``front_masks`` their masks and
    ``front_pn`` / ``front_ps`` the matching sign fractions.
    """
    popularity = {}
    for shard in Path(stage_dir).glob("shard-*.json"):
        with open(shard, encoding="utf-8") as f:
            for inst in json.load(f)["instances"]:
                popularity[(inst["user_index"], inst["target"])] = inst["target_popularity"]
    records = []
    for path in sorted(Path(stage_dir).glob("*/*.npz")):
        job = load_job(path)
        meta = job["meta"]
        rows = job["front"]
        evaluated = source == "front_objectives"
        instance = (meta["user_index"], meta["target"])
        records.append(
            {
                "path": path,
                "method": meta["method"],
                "seed": meta["seed"],
                "instance": instance,
                "n": meta["n"],
                "m": meta["m"],
                "evaluations": meta["evaluations"],
                # model queries the method itself paid for (GECE queries sequences, not masks)
                "passes": meta.get("queries", meta["evaluations"] * 2 * meta["m"]),
                "target_popularity": popularity.get(instance),
                "front": (job["front_objectives"] if evaluated else job["objectives"][rows]).astype(np.float64),
                "front_pn": job["front_pn"] if evaluated else job["pn"][rows],
                "front_ps": job["front_ps"] if evaluated else job["ps"][rows],
                "front_masks": job["masks"][rows],
            }
        )
    return records


def non_dominated(points: np.ndarray) -> np.ndarray:
    """The non-dominated rows of ``points`` (re-scored fronts can contain dominated rows)."""
    if len(points) == 0:
        return points
    return points[NonDominatedSorting().do(points, only_non_dominated_front=True)]


def _restrict(points: np.ndarray, max_size: int | None) -> np.ndarray:
    return points if max_size is None else points[points[:, 2] <= max_size]


def _scaled_hv(points: np.ndarray, low: np.ndarray, span: np.ndarray) -> float:
    points = non_dominated(points)
    if not len(points):
        return 0.0
    return float((_HV if points.shape[1] == 3 else _HV2)((points - low) / span))


def _bounds(fronts: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    union = np.concatenate(fronts)
    low, high = union.min(axis=0), union.max(axis=0)
    return low, np.where(high > low, high - low, 1.0)


def _by_instance(records: Iterable[dict]) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[record["instance"]].append(record)
    return groups


def add_hypervolume(records: Iterable[dict], max_size: int | None = None, field: str = "hv",
                    columns: Sequence[int] = (0, 1, 2)) -> None:
    """Set ``record[field]``: hypervolume after scaling the objectives inside each instance.

    With ``max_size`` only the masks with at most that many positions count (small explanations),
    and the scaling bounds come from those masks only. ``columns`` picks the objectives (e.g.
    ``(0, 2)`` for the (N, C) projection; the reference point is 1.1 in each, so the largest
    possible value is 1.1 ** len(columns)).
    """
    columns = list(columns)
    for group in _by_instance(records).values():
        fronts = [_restrict(record["front"], max_size)[:, columns] for record in group]
        if not any(len(front) for front in fronts):
            for record in group:
                record[field] = 0.0
            continue
        low, span = _bounds([front for front in fronts if len(front)])
        for record, front in zip(group, fronts):
            record[field] = _scaled_hv(front, low, span)


def add_igd_plus(records: Iterable[dict], max_size: int | None = None, field: str = "igd_plus") -> None:
    """Set ``record[field]``: IGD+ of the job's front against the reference front of its instance.

    The reference is the non-dominated union of all compared fronts of the instance (restricted to
    ``max_size`` positions if given), scaled with the same per-instance bounds as the hypervolume.
    Lower is better; a job without masks in the region gets ``nan``.
    """
    for group in _by_instance(records).values():
        fronts = [_restrict(record["front"], max_size) for record in group]
        present = [front for front in fronts if len(front)]
        if not present:
            for record in group:
                record[field] = float("nan")
            continue
        low, span = _bounds(present)
        reference = non_dominated(np.concatenate(present))
        indicator = IGDPlus((reference - low) / span)
        for record, front in zip(group, fronts):
            record[field] = float(indicator((non_dominated(front) - low) / span)) if len(front) else float("nan")


def paired_statistics(x: Sequence[float], y: Sequence[float], resamples: int = 10_000, seed: int = 0) -> dict:
    """Effect size of paired values (one pair per instance): mean of x - y with an instance
    bootstrap 95% interval, and the matched-pairs rank-biserial correlation (+1: x always larger)."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(len(diff), size=(resamples, len(diff)))].mean(axis=1)
    nonzero = diff[diff != 0]
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1.0
    if len(nonzero):  # average ranks for ties in |diff|
        _, inverse, counts = np.unique(np.abs(nonzero), return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=ranks)
        ranks = (sums / counts)[inverse]
    total = ranks.sum()
    biserial = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / total) if total else 0.0
    return {"mean_diff": float(diff.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "rank_biserial": biserial}


def per_instance_means(records: Iterable[dict], key: str = "label", field: str = "hv") -> dict:
    """``{key value: {instance: mean of field over seeds}}`` (the unit of the paired tests)."""
    groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        groups[record[key]][record["instance"]].append(record[field])
    return {name: {inst: float(np.mean(v)) for inst, v in per.items()} for name, per in groups.items()}


def summarise(records: Sequence[dict], key: str = "label", fields: Sequence[str] = ("hv",)) -> dict:
    """Per value of ``key``: counts, mean of each field (seeds averaged inside each instance first),
    mean front size, mean evaluations and mean SASRec passes used."""
    out = defaultdict(dict)
    for field in fields:
        for name, per_instance in per_instance_means(records, key, field).items():
            values = np.array(list(per_instance.values()))
            out[name].update({"instances": len(values), f"{field}_mean": float(values.mean()),
                              f"{field}_median": float(np.median(values))})
    for name in out:
        mine = [record for record in records if record[key] == name]
        out[name].update(
            jobs=len(mine),
            front_size=float(np.mean([len(record["front"]) for record in mine])),
            evaluations=float(np.mean([record["evaluations"] for record in mine])),
            passes=float(np.mean([record["passes"] for record in mine])),
        )
    return dict(out)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def explanation_quality(records: Iterable[dict], max_size: int, key: str = "label") -> dict:
    """Per value of ``key``, over the front masks with at most ``max_size`` positions:

    * ``valid``: share of jobs with at least one mask where PN > 1/2 and PS > 1/2;
    * ``best_pn`` / ``best_ps``: mean over jobs of the highest PN and PS reached;
    * ``recency``: mean relative position of the selected items (0 = oldest, 1 = newest; 0.5 would
      be no preference).
    """
    groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        small = record["front"][:, 2] <= max_size
        if not small.any():
            continue
        pn, ps, masks = record["front_pn"][small], record["front_ps"][small], record["front_masks"][small]
        stats = groups[record[key]]
        stats["valid"].append(bool(np.any((pn > 0.5) & (ps > 0.5))))
        stats["best_pn"].append(float(pn.max()))
        stats["best_ps"].append(float(ps.max()))
        position = np.linspace(0.0, 1.0, record["n"])
        stats["recency"].append(float((masks * position).sum() / max(masks.sum(), 1)))
    return {name: {stat: float(np.mean(values)) for stat, values in stats.items()} for name, stats in groups.items()}


def convergence(records: Sequence[dict], checkpoints: Sequence[int], max_size: int | None = None, key: str = "label") -> dict:
    """``{key value: [mean hypervolume after the first c evaluations for c in checkpoints]}``.

    Uses the search objectives of every evaluated mask, in evaluation order, scaled inside each
    instance by the bounds of the final fronts. A job that stopped before a checkpoint keeps its
    final value.
    """
    curves = defaultdict(lambda: defaultdict(list))
    for group in _by_instance(records).values():
        archives = [_restrict(load_job(record["path"])["objectives"].astype(np.float64), None) for record in group]
        finals = [_restrict(non_dominated(archive), max_size) for archive in archives]
        if not any(len(front) for front in finals):
            continue
        low, span = _bounds([front for front in finals if len(front)])
        for record, archive in zip(group, archives):
            values = [_scaled_hv(_restrict(archive[:c], max_size), low, span) for c in checkpoints]
            curves[record[key]][record["instance"]].append(values)
    return {
        name: np.mean([np.mean(runs, axis=0) for runs in per_instance.values()], axis=0).tolist()
        for name, per_instance in curves.items()
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def objective_correlation(records: Sequence[dict], min_masks: int = 20) -> dict:
    """Spearman correlation between N-hat and S-hat among masks of the same size.

    Pools every evaluated mask of the given records inside each instance (use the ``random`` and
    ``enumeration`` jobs: their masks are not selected by fitness), computes the correlation for
    each exact size with at least ``min_masks`` masks and averages over instances. Returns
    ``{"overall": r ignoring size, "by_size": {relative size bin: r}, "size_1": r, "size_2": r}``;
    a strongly positive value means the two objectives rank masks alike (no conflict).
    """
    bins = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    overall, exact = [], defaultdict(list)
    binned = defaultdict(list)
    for group in _by_instance(records).values():
        archive = np.concatenate([load_job(record["path"])["objectives"] for record in group]).astype(np.float64)
        archive = np.unique(archive, axis=0)
        n = group[0]["n"]
        overall.append(_spearman(archive[:, 0], archive[:, 1]))
        per_bin = defaultdict(list)
        for size in np.unique(archive[:, 2]):
            rows = archive[archive[:, 2] == size]
            if len(rows) < min_masks:
                continue
            r = _spearman(rows[:, 0], rows[:, 1])
            if size in (1, 2):
                exact[int(size)].append(r)
            per_bin[int(np.searchsorted(bins, size / n, side="left")) - 1].append(r)
        for index, values in per_bin.items():
            binned[index].append(np.nanmean(values))
    label = lambda i: f"{int(bins[max(i, 0)] * 100)}-{int(bins[max(i, 0) + 1] * 100)}% of n"
    return {
        "overall": float(np.nanmean(overall)),
        "size_1": float(np.nanmean(exact[1])) if exact[1] else None,
        "size_2": float(np.nanmean(exact[2])) if exact[2] else None,
        "by_size": {label(i): float(np.nanmean(values)) for i, values in sorted(binned.items())},
    }
