"""Search methods over masks, all behind the same ask/tell protocol.

A method never calls the model. ``ask(remaining)`` returns the masks it wants scored (bool [R, n],
one bit per history position) and ``tell(F)`` receives their objectives (float [R, 3] =
(N-hat, S-hat, |X|), all minimised) in the same order. The scheduler owns the budget: it may stop
a method at any round, so a method keeps no result of its own; everything it had scored is the
job's archive.

Scalarised methods use ``Context.scale`` to put the three objectives on comparable units:
``|N|`` and ``|S|`` of the full mask and the history length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterator

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.problems.static import StaticProblem

def weight_lattice(divisions: int) -> np.ndarray:
    """float [W, 3]: weight vectors over (N, S, C) on the simplex lattice (3 divisions -> 10 vectors)."""
    grid = np.ndindex(divisions + 1, divisions + 1, divisions + 1)
    return np.array([w for w in grid if sum(w) == divisions], dtype=np.float64) / divisions


DEFAULT_WEIGHTS = weight_lattice(3)
SPARSE_BITS = 5  # expected bits on in an initial individual


@dataclass(frozen=True)
class Context:
    """What a method may know about its instance.

    Attributes:
        n: number of positions (mask length).
        scale: float [3], divides (N, S, C) before a weighted sum.
        similarity: float [n], cosine similarity of each history item to the target.
        budget: mask evaluations available to the job.
        pop_size: population of NSGA-II (and batch size of the random baseline).
        scalar_pop_size: population of each scalarised GA run.
        weights: float [W, 3], weight vectors of the scalarised GA.
        init: ``sparse`` (about ``SPARSE_BITS`` bits on) or ``uniform`` (each bit on with 1/2).
        crossover: apply uniform crossover (False: offspring are mutated copies of the parents).
        mutation_scale: the per-bit flip rate is ``mutation_scale / n``.
    """

    n: int
    scale: np.ndarray
    similarity: np.ndarray
    budget: int
    pop_size: int = 100
    scalar_pop_size: int = 20
    weights: np.ndarray = field(default_factory=lambda: DEFAULT_WEIGHTS)
    init: str = "sparse"
    crossover: bool = True
    mutation_scale: float = 1.0


def scalarise(objectives: np.ndarray, weights: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """float [R]: weighted sum of the scaled objectives (lower is better)."""
    return (objectives / scale) @ weights


class Method:
    """Base class: ``done`` turns True when the method has nothing more to ask."""

    done = False

    def __init__(self, ctx: Context, rng: np.random.Generator) -> None:
        self.ctx, self.rng = ctx, rng

    def ask(self, remaining: int) -> np.ndarray:
        raise NotImplementedError

    def tell(self, objectives: np.ndarray) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------------- evolutionary


class SparseSampling(Sampling):
    """Each bit on with probability ``SPARSE_BITS / n`` (small explanations first)."""

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        return random_state.random((n_samples, problem.n_var)) < min(0.5, SPARSE_BITS / problem.n_var)


class NonEmptyRepair(Repair):
    """Turn one random bit on in every empty mask."""

    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__()
        self.rng = rng

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X).astype(bool)
        empty = np.flatnonzero(~X.any(axis=1))
        X[empty, self.rng.integers(problem.n_var, size=len(empty))] = True
        return X


class _Evolution:
    """A pymoo genetic algorithm driven by ask/tell on a binary mask."""

    def __init__(self, algorithm_class, ctx: Context, n_obj: int, pop_size: int, rng: np.random.Generator) -> None:
        if ctx.init not in ("sparse", "uniform"):
            raise ValueError(f"init must be sparse or uniform, got {ctx.init!r}")
        self.problem = Problem(n_var=ctx.n, n_obj=n_obj, xl=0, xu=1, vtype=bool)
        self.algorithm = algorithm_class(
            pop_size=pop_size,
            sampling=SparseSampling() if ctx.init == "sparse" else BinaryRandomSampling(),
            crossover=UX(prob=1.0 if ctx.crossover else 0.0),
            mutation=BitflipMutation(prob_var=min(0.5, ctx.mutation_scale / ctx.n)),
            repair=NonEmptyRepair(rng),
            eliminate_duplicates=True,
        )
        seed = int(rng.integers(2**31))
        self.algorithm.setup(self.problem, termination=NoTermination(), seed=seed, verbose=False)
        self.population = None

    def ask(self) -> np.ndarray:
        # pymoo returns nothing when every offspring duplicates an existing individual
        self.population = self.algorithm.ask()
        if self.population is None or len(self.population) == 0:
            self.population = None
            return np.zeros((0, self.problem.n_var), dtype=bool)
        return self.population.get("X").astype(bool)

    def tell(self, fitness: np.ndarray) -> None:
        if self.population is None:
            return
        Evaluator().eval(StaticProblem(self.problem, F=fitness), self.population)
        self.algorithm.tell(infills=self.population)


class NSGA2Search(Method):
    """NSGA-II on (N, S, C): uniform crossover, bit-flip mutation 1/n, sparse initialisation."""

    def __init__(self, ctx: Context, rng: np.random.Generator) -> None:
        super().__init__(ctx, rng)
        self.evolution = _Evolution(NSGA2, ctx, 3, ctx.pop_size, rng)

    def ask(self, remaining: int) -> np.ndarray:
        return self.evolution.ask()

    def tell(self, objectives: np.ndarray) -> None:
        self.evolution.tell(objectives)


class ScalarGASearch(Method):
    """Single-objective GA with the same operators, run once per weight vector.

    The budget is split evenly over the weight vectors; the runs are independent restarts.
    """

    def __init__(self, ctx: Context, rng: np.random.Generator) -> None:
        super().__init__(ctx, rng)
        self.share = max(ctx.budget // len(ctx.weights), 1)
        self.run = -1
        self._next_run()

    def _next_run(self) -> None:
        self.run += 1
        self.asked = 0
        if self.run >= len(self.ctx.weights):
            self.done = True
            return
        self.evolution = _Evolution(GA, self.ctx, 1, self.ctx.scalar_pop_size, self.rng)

    def ask(self, remaining: int) -> np.ndarray:
        masks = self.evolution.ask()
        self.asked += len(masks)
        return masks

    def tell(self, objectives: np.ndarray) -> None:
        fitness = scalarise(objectives, self.ctx.weights[self.run], self.ctx.scale)
        self.evolution.tell(fitness[:, None])
        if self.asked >= self.share:
            self._next_run()


# -------------------------------------------------------------------------------------- scripted


class Scripted(Method):
    """A method written as a generator: ``objectives = yield masks``."""

    def __init__(self, ctx: Context, rng: np.random.Generator) -> None:
        super().__init__(ctx, rng)
        self._script = self.script()
        self._masks = next(self._script, None)
        self.done = self._masks is None

    def script(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def ask(self, remaining: int) -> np.ndarray:
        return self._masks

    def tell(self, objectives: np.ndarray) -> None:
        try:
            self._masks = self._script.send(objectives)
        except StopIteration:
            self.done = True


def nested(order: np.ndarray) -> np.ndarray:
    """bool [n, n]: row k selects the first k + 1 positions of ``order``."""
    n = len(order)
    masks = np.zeros((n, n), dtype=bool)
    for k in range(n):
        masks[k, order[: k + 1]] = True
    return masks


class RandomSubsets(Scripted):
    """Subsets with a size uniform in 1..n and uniform positions, until the budget ends."""

    def script(self):
        n = self.ctx.n
        while True:
            masks = np.zeros((self.ctx.pop_size, n), dtype=bool)
            for row, size in zip(masks, self.rng.integers(1, n + 1, size=len(masks))):
                row[self.rng.choice(n, size=size, replace=False)] = True
            yield masks


class RecencySubsets(Scripted):
    """The k most recent positions, for every k."""

    def script(self):
        yield nested(np.arange(self.ctx.n)[::-1])


class SimilaritySubsets(Scripted):
    """The k items most similar to the target, for every k."""

    def script(self):
        yield nested(np.argsort(-self.ctx.similarity, kind="stable"))


class AblationRanking(Scripted):
    """Score every single position, then take the top-k by necessity and by sufficiency."""

    def script(self):
        single = yield np.eye(self.ctx.n, dtype=bool)
        yield nested(np.argsort(single[:, 0], kind="stable"))
        yield nested(np.argsort(single[:, 1], kind="stable"))


class Enumeration(Scripted):
    """Every mask with one or two positions (exact reference for |X| <= 2)."""

    batch = 2048

    def script(self):
        n = self.ctx.n
        subsets = [(i,) for i in range(n)] + list(combinations(range(n), 2))
        for start in range(0, len(subsets), self.batch):
            block = subsets[start : start + self.batch]
            masks = np.zeros((len(block), n), dtype=bool)
            for row, positions in zip(masks, block):
                row[list(positions)] = True
            yield masks


class Greedy(Scripted):
    """Scalarised greedy on N + S (scaled): forward selection from the empty mask, then backward
    elimination from the full mask with what is left of the budget."""

    weights = np.array([1.0, 1.0, 0.0])

    def script(self):
        n = self.ctx.n
        eye = np.eye(n, dtype=bool)
        current = np.zeros(n, dtype=bool)
        while not current.all():
            free = np.flatnonzero(~current)
            objectives = yield current | eye[free]
            current[free[np.argmin(scalarise(objectives, self.weights, self.ctx.scale))]] = True
        while current.sum() > 1:
            used = np.flatnonzero(current)
            objectives = yield current & ~eye[used]
            current[used[np.argmin(scalarise(objectives, self.weights, self.ctx.scale))]] = False


METHODS: dict[str, Callable[[Context, np.random.Generator], Method]] = {
    "nsga2": NSGA2Search,
    "scalar_ga": ScalarGASearch,
    "random": RandomSubsets,
    "recency": RecencySubsets,
    "similarity": SimilaritySubsets,
    "ablation": AblationRanking,
    "greedy": Greedy,
    "enumeration": Enumeration,
}
# one seed is enough for these
DETERMINISTIC = frozenset({"recency", "similarity", "ablation", "greedy", "enumeration"})
# reference methods that run to completion whatever the budget
UNBUDGETED = frozenset({"enumeration"})
