"""Compare result files: hypervolume of each job's front, on a scale shared inside an instance.

Fronts are compared on the objectives re-scored with the independent evaluation backgrounds
(``front_objectives``), so searches that used a different M or budget are on equal terms. For
every instance (user, target) the three objectives are min-max scaled over all fronts being
compared and the reference point is 1.1 in every scaled objective, i.e. slightly worse than the
worst value any compared job reached for that instance.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from seqrec.explain.scheduler import load_job

REFERENCE = 1.1


def load_jobs(stage_dir: Path, source: str = "front_objectives") -> list[dict]:
    """One record per result file under ``stage_dir``: label, method, seed, instance key, front.

    ``source`` is ``front_objectives`` (evaluation backgrounds) or ``objectives`` (search
    backgrounds, front rows only).
    """
    records = []
    for path in sorted(Path(stage_dir).glob("*/*.npz")):
        job = load_job(path)
        meta = job["meta"]
        points = job[source] if source == "front_objectives" else job["objectives"][job["front"]]
        records.append(
            {
                "method": meta["method"],
                "seed": meta["seed"],
                "instance": (meta["user_index"], meta["target"]),
                "n": meta["n"],
                "evaluations": meta["evaluations"],
                "front": non_dominated(points.astype(np.float64)),
            }
        )
    return records


def non_dominated(points: np.ndarray) -> np.ndarray:
    """The non-dominated rows of ``points`` (re-scored fronts can contain dominated rows)."""
    if len(points) == 0:
        return points
    return points[NonDominatedSorting().do(points, only_non_dominated_front=True)]


def add_hypervolume(records: Iterable[dict]) -> None:
    """Set ``record["hv"]``: hypervolume after scaling the objectives inside each instance."""
    by_instance = defaultdict(list)
    for record in records:
        by_instance[record["instance"]].append(record)
    indicator = HV(ref_point=np.full(3, REFERENCE))
    for group in by_instance.values():
        union = np.concatenate([record["front"] for record in group])
        low, high = union.min(axis=0), union.max(axis=0)
        span = np.where(high > low, high - low, 1.0)
        for record in group:
            record["hv"] = float(indicator((record["front"] - low) / span)) if len(record["front"]) else 0.0


def summarise(records: Iterable[dict], key: str = "label") -> dict:
    """Per value of ``key``: jobs, mean hypervolume (seeds averaged inside each instance first),
    mean front size and mean evaluations."""
    groups = defaultdict(lambda: defaultdict(list))
    extra = defaultdict(lambda: {"front": [], "evaluations": []})
    for record in records:
        groups[record[key]][record["instance"]].append(record["hv"])
        extra[record[key]]["front"].append(len(record["front"]))
        extra[record[key]]["evaluations"].append(record["evaluations"])
    out = {}
    for name, per_instance in groups.items():
        means = np.array([np.mean(values) for values in per_instance.values()])
        out[name] = {
            "instances": len(means),
            "jobs": len(extra[name]["front"]),
            "hv_mean": float(means.mean()),
            "hv_median": float(np.median(means)),
            "front_size": float(np.mean(extra[name]["front"])),
            "evaluations": float(np.mean(extra[name]["evaluations"])),
        }
    return out


def per_instance_means(records: Iterable[dict], key: str = "label") -> dict:
    """``{key value: {instance: mean hypervolume over seeds}}`` for paired tests."""
    groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        groups[record[key]][record["instance"]].append(record["hv"])
    return {name: {inst: float(np.mean(v)) for inst, v in per.items()} for name, per in groups.items()}
