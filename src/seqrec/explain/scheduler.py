"""Run many (instance, method, seed) jobs in lockstep on one verifier.

Every round asks each active job for masks, scores all of them in one batch and hands the
objectives back. The budget is enforced here, in mask evaluations (one evaluation = 2 * M SASRec
passes): a mask already scored by the same job is answered from the job's cache and costs
nothing. Jobs are independent of each other, so results do not depend on ``concurrency``.

Result file of a job (``save_job`` / ``load_job``), E = evaluated masks, n = history length:

* ``masks``     uint8 [E, ceil(n / 8)]  ``np.packbits`` of the masks, in evaluation order
* ``objectives`` float32 [E, 3]         (N-hat, S-hat, |X|) on the search backgrounds
* ``pn``, ``ps`` float32 [E]            sign fractions on the search backgrounds
* ``front``     int64 [F]               rows of the non-dominated masks
* ``front_objectives`` / ``front_pn`` / ``front_ps``  the front re-scored on the independent
  evaluation backgrounds (absent when no evaluation verifier was given)
* ``meta``      JSON string
"""

from __future__ import annotations

import json
import os
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from seqrec.explain.methods import DETERMINISTIC, METHODS, UNBUDGETED, Context, Method
from seqrec.explain.verifier import Verifier

STALL_LIMIT = 20  # rounds without a new mask before a job is closed


@dataclass(frozen=True)
class JobSpec:
    instance: int  # row of the instance in the verifier
    method: str
    seed: int


@dataclass
class Job:
    spec: JobSpec
    method: Method
    n: int
    budget: int
    cache: dict = field(default_factory=dict)
    masks: list = field(default_factory=list)
    rows: list = field(default_factory=list)  # (N, S, C, PN, PS) per evaluated mask
    stalls: int = 0
    front_scores: np.ndarray | None = None

    @property
    def remaining(self) -> int:
        return self.budget - len(self.masks)

    @property
    def objectives(self) -> np.ndarray:
        return np.asarray(self.rows, dtype=np.float64).reshape(-1, 5)[:, :3]

    def front(self) -> np.ndarray:
        """Rows of the non-dominated evaluated masks. The empty mask (scored by methods that need a
        baseline value) is not an explanation and never enters the front."""
        candidates = np.flatnonzero(self.objectives[:, 2] > 0) if self.rows else np.zeros(0, dtype=np.int64)
        if not len(candidates):
            return np.zeros(0, dtype=np.int64)
        kept = NonDominatedSorting().do(self.objectives[candidates], only_non_dominated_front=True)
        return np.sort(candidates[kept])


def job_specs(instances: Sequence[int], methods: Sequence[str], seeds: Sequence[int]) -> list[JobSpec]:
    """All jobs; deterministic methods get the first seed only."""
    specs = []
    for instance in instances:
        for method in methods:
            for seed in seeds[:1] if method in DETERMINISTIC else seeds:
                specs.append(JobSpec(instance, method, seed))
    return specs


def _scores(evaluation) -> np.ndarray:
    """float64 [R, 5]: N, S, C, PN, PS."""
    columns = [evaluation.necessity, evaluation.sufficiency, evaluation.size.float(), evaluation.pn, evaluation.ps]
    return torch.stack(columns, dim=1).cpu().numpy().astype(np.float64)


def _padded(masks: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros((masks.shape[0], length), dtype=bool)
    if masks.shape[1]:
        out[:, length - masks.shape[1] :] = masks
    return out


def run_jobs(
    verifier: Verifier,
    specs: Sequence[JobSpec],
    contexts: Sequence[Context],
    job_rng: Callable[[JobSpec], np.random.Generator],
    on_finish: Callable[[Job], None],
    concurrency: int = 64,
    eval_verifier: Verifier | None = None,
    on_round: Callable[[int, int], None] | None = None,
) -> None:
    """Run ``specs`` to completion, ``concurrency`` jobs at a time.

    ``contexts[i]`` belongs to instance ``i`` of ``verifier``. Finished jobs are re-scored on
    ``eval_verifier`` (their front only, not charged to the budget) and passed to ``on_finish``.
    """
    pending = list(reversed(specs))
    active: list[Job] = []
    finished = 0
    while pending or active:
        while pending and len(active) < concurrency:
            spec = pending.pop()
            ctx = contexts[spec.instance]
            budget = sys.maxsize if spec.method in UNBUDGETED else ctx.budget
            active.append(Job(spec, METHODS[spec.method](ctx, job_rng(spec)), ctx.n, budget))

        asked, index, batch = [], [], []
        for job in active:
            masks = np.zeros((0, job.n), dtype=bool) if job.method.done else np.asarray(job.method.ask(job.remaining))
            keys = [row.tobytes() for row in masks]
            fresh: dict = {}
            cut = False
            for row, key in enumerate(keys):
                if key in job.cache or key in fresh:
                    continue
                if len(fresh) >= job.remaining:
                    cut = True
                    break
                fresh[key] = row
            asked.append((masks, keys, fresh, cut))
            index += [job.spec.instance] * len(fresh)
            batch.append(_padded(masks[list(fresh.values())].reshape(-1, job.n), verifier.length))

        scores = np.zeros((0, 5))
        if index:
            scores = _scores(verifier.evaluate(torch.tensor(index), torch.from_numpy(np.concatenate(batch))))

        closing, offset = [], 0
        for job, (masks, keys, fresh, cut) in zip(active, asked):
            for key, row in fresh.items():
                job.cache[key] = len(job.masks)
                job.masks.append(masks[row])
                job.rows.append(scores[offset])
                offset += 1
            job.stalls = 0 if fresh else job.stalls + 1
            if cut or job.remaining <= 0 or job.method.done or job.stalls >= STALL_LIMIT:
                closing.append(job)
                continue
            job.method.tell(np.asarray([job.rows[job.cache[key]][:3] for key in keys]).reshape(-1, 3))
            if job.method.done:
                closing.append(job)

        if closing:
            _rescore(closing, eval_verifier)
            for job in closing:
                on_finish(job)
            done = {id(job) for job in closing}
            active = [job for job in active if id(job) not in done]
            finished += len(closing)
        if on_round:
            on_round(finished, len(specs))


def _rescore(jobs: Sequence[Job], eval_verifier: Verifier | None) -> None:
    """Score the front of each job on the independent evaluation backgrounds."""
    if eval_verifier is None:
        return
    fronts = [job.front() for job in jobs]
    index = [job.spec.instance for job, front in zip(jobs, fronts) for _ in front]
    if not index:
        return
    masks = np.concatenate([_padded(np.asarray(job.masks)[front], eval_verifier.length) for job, front in zip(jobs, fronts)])
    scores = _scores(eval_verifier.evaluate(torch.tensor(index), torch.from_numpy(masks)))
    offset = 0
    for job, front in zip(jobs, fronts):
        job.front_scores = scores[offset : offset + len(front)]
        offset += len(front)


def method_seed(spec: JobSpec, user_index: int) -> np.random.Generator:
    """Random generator of a job: a function of the seed, the user and the method name only."""
    return np.random.default_rng([spec.seed, user_index, zlib.crc32(spec.method.encode())])


def save_job(path: Path, job: Job, meta: dict) -> None:
    """Write the job's archive atomically (see the module docstring for the layout)."""
    scores = np.asarray(job.rows, dtype=np.float64).reshape(-1, 5)
    masks = np.asarray(job.masks, dtype=bool).reshape(-1, job.n)
    arrays = {
        "masks": np.packbits(masks, axis=1),
        "objectives": scores[:, :3].astype(np.float32),
        "pn": scores[:, 3].astype(np.float32),
        "ps": scores[:, 4].astype(np.float32),
        "front": job.front(),
        "meta": np.array(json.dumps({**meta, "n": job.n, "evaluations": len(job.masks)})),
    }
    if job.front_scores is not None:
        arrays["front_objectives"] = job.front_scores[:, :3].astype(np.float32)
        arrays["front_pn"] = job.front_scores[:, 3].astype(np.float32)
        arrays["front_ps"] = job.front_scores[:, 4].astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_job(path: Path) -> dict:
    """Read a result file; ``masks`` is unpacked to bool [E, n] and ``meta`` parsed."""
    with np.load(path) as z:
        out = {name: z[name] for name in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    out["masks"] = np.unpackbits(out["masks"], axis=1, count=out["meta"]["n"]).astype(bool)
    return out
