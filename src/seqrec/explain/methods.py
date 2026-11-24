"""Search methods over masks, all behind the same ask/tell protocol.

A method never calls the model. ``ask(remaining)`` returns the masks it wants scored (bool [R, n],
one bit per history position) and ``tell(F)`` receives their objectives (float [R, 3] =
(N-hat, S-hat, |X|), all minimised) in the same order. A method with ``needs_sufficiency = False``
is never charged for S-hat, which it receives as NaN. The scheduler owns the budget: it may stop
a method at any round, so a method keeps no result of its own; everything it had scored is the
job's archive.

Scalarised methods use ``Context.scale`` to put the three objectives on comparable units:
``|N|`` and ``|S|`` of the full mask and the history length.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Callable, Iterator

import numpy as np
from pymoo.algorithms.moo.moead import ParallelMOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.termination import NoTermination
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.problems.static import StaticProblem
from pymoo.util.ref_dirs import get_reference_directions

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
        budget: mask evaluations available to the job when each mask is scored on N and S (2M
            passes); a method that does not need S pays M passes per mask and gets twice as many.
        pop_size: population of NSGA-II (and batch size of the random baseline).
        scalar_pop_size: population of each scalarised GA run.
        weights: float [W, 3], weight vectors of the scalarised GA.
        init: ``sparse`` (about ``SPARSE_BITS`` bits on) or ``uniform`` (each bit on with 1/2).
        crossover: apply uniform crossover (False: offspring are mutated copies of the parents).
        mutation_scale: the per-bit flip rate is ``mutation_scale / n``.
        objectives: which of (N, S, C) the multi-objective searches see (indices); without S (1),
            S-hat is not estimated during the search.
        partitions: Das-Dennis partitions of the reference directions of MOEA/D and NSGA-III
            (3 objectives: 5 -> 21 directions, 6 -> 28); also their population size.
        decomposition: ``tchebicheff`` or ``pbi`` (MOEA/D).
        neighbours: MOEA/D neighbourhood size (clamped to the number of directions).
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
    objectives: tuple = (0, 1, 2)
    partitions: int = 5
    decomposition: str = "tchebicheff"
    neighbours: int = 20


def scalarise(objectives: np.ndarray, weights: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """float [R]: weighted sum of the scaled objectives (lower is better)."""
    return (objectives / scale) @ weights


class Method:
    """Base class: ``done`` turns True when the method has nothing more to ask; ``needs_sufficiency``
    says whether its masks must be scored on S-hat (otherwise S-hat is not estimated nor charged)."""

    done = False
    needs_sufficiency = True

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

    def __init__(self, algorithm_class, ctx: Context, n_obj: int, pop_size: int | None, rng: np.random.Generator,
                 initial: np.ndarray | None = None, fitness_scale: np.ndarray | None = None, **options) -> None:
        """``initial`` (bool [P, n]) replaces the sampled initial population. ``fitness_scale``
        divides what ``tell`` receives (for algorithms that do not normalise objectives).
        ``pop_size=None`` leaves the population size to the algorithm (MOEA/D: one per direction);
        ``options`` go to the algorithm's constructor."""
        if ctx.init not in ("sparse", "uniform"):
            raise ValueError(f"init must be sparse or uniform, got {ctx.init!r}")
        self.problem = Problem(n_var=ctx.n, n_obj=n_obj, xl=0, xu=1, vtype=bool)
        self.fitness_scale = fitness_scale
        settings = dict(
            sampling=initial if initial is not None else SparseSampling() if ctx.init == "sparse" else BinaryRandomSampling(),
            crossover=UX(prob=1.0 if ctx.crossover else 0.0),
            mutation=BitflipMutation(prob_var=min(0.5, ctx.mutation_scale / ctx.n)),
            repair=NonEmptyRepair(rng),
        )
        if pop_size is not None:
            settings["pop_size"] = pop_size
        if algorithm_class is not ParallelMOEAD:  # MOEA/D sets its own (no) duplicate elimination
            settings["eliminate_duplicates"] = True
        self.algorithm = algorithm_class(**settings, **options)
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
        if self.fitness_scale is not None:
            fitness = fitness / self.fitness_scale
        Evaluator().eval(StaticProblem(self.problem, F=fitness), self.population)
        self.algorithm.tell(infills=self.population)


class _MultiObjective(Method):
    """A pymoo multi-objective algorithm on the objectives in ``ctx.objectives``."""

    def __init__(self, ctx: Context, rng: np.random.Generator) -> None:
        super().__init__(ctx, rng)
        self.columns = list(ctx.objectives)
        self.needs_sufficiency = 1 in self.columns
        self.evolution = self.build(ctx, rng, len(self.columns))

    def build(self, ctx: Context, rng: np.random.Generator, n_obj: int) -> _Evolution:
        raise NotImplementedError

    def ask(self, remaining: int) -> np.ndarray:
        return self.evolution.ask()

    def tell(self, objectives: np.ndarray) -> None:
        self.evolution.tell(objectives[:, self.columns])


class NSGA2Search(_MultiObjective):
    """NSGA-II on (N, S, C): uniform crossover, bit-flip mutation 1/n, sparse initialisation."""

    def build(self, ctx, rng, n_obj):
        return _Evolution(NSGA2, ctx, n_obj, ctx.pop_size, rng)


def _directions(ctx: Context, n_obj: int) -> np.ndarray:
    return get_reference_directions("das-dennis", n_obj, n_partitions=ctx.partitions)


class MOEADSearch(_MultiObjective):
    """MOEA/D (generation-wise variant): one subproblem per Das-Dennis direction, neighbourhood
    mating, Tchebicheff or PBI decomposition. The objectives are divided by ``ctx.scale`` first,
    because decompositions compare raw objective values and C is two orders larger than N, S."""

    def build(self, ctx, rng, n_obj):
        directions = _directions(ctx, n_obj)
        decomposition = {"tchebicheff": Tchebicheff, "pbi": PBI}[ctx.decomposition]()
        return _Evolution(ParallelMOEAD, ctx, n_obj, None, rng, fitness_scale=ctx.scale[self.columns],
                          ref_dirs=directions, n_neighbors=min(ctx.neighbours, len(directions)),
                          decomposition=decomposition)


class NSGA3Search(_MultiObjective):
    """NSGA-III: non-dominated sorting with reference-direction niching (normalises internally)."""

    def build(self, ctx, rng, n_obj):
        directions = _directions(ctx, n_obj)
        return _Evolution(NSGA3, ctx, n_obj, len(directions), rng, ref_dirs=directions)


class SMSEMOASearch(_MultiObjective):
    """SMS-EMOA: survival by least hypervolume contribution (normalises internally). Generates a
    whole population of offspring per step so each ask is one GPU batch."""

    def build(self, ctx, rng, n_obj):
        return _Evolution(SMSEMOA, ctx, n_obj, ctx.pop_size, rng, n_offsprings=ctx.pop_size)


class SeededNSGA2Search(Method):
    """NSGA-II whose initial population comes from the ablation ranking.

    First it scores what ``AblationRanking`` scores (every single position, then the top-k
    positions by necessity and by sufficiency for every k: at most 3n masks). The initial
    population is the ``pop_size`` of those masks that NSGA-II's own survival (non-dominated rank,
    then crowding) keeps; the scheduler answers them from the job's cache, so they are not paid
    for twice. The rest of the budget is the usual NSGA-II.
    """

    def __init__(self, ctx: Context, rng: np.random.Generator) -> None:
        super().__init__(ctx, rng)
        self.phase = 0
        self.seeds = np.eye(ctx.n, dtype=bool)
        self.scores = np.zeros((0, 3))
        self.evolution = None

    def ask(self, remaining: int) -> np.ndarray:
        if self.phase == 0:
            return self.seeds
        if self.phase == 1:
            return self.ranked
        return self.evolution.ask()

    def tell(self, objectives: np.ndarray) -> None:
        if self.phase == 0:
            self.scores = objectives
            self.ranked = np.concatenate([nested(np.argsort(objectives[:, column], kind="stable")) for column in (0, 1)])
            self.phase = 1
        elif self.phase == 1:
            masks, index = np.unique(np.concatenate([self.seeds, self.ranked]), axis=0, return_index=True)
            scores = np.concatenate([self.scores, objectives])[index]
            problem = Problem(n_var=self.ctx.n, n_obj=3, xl=0, xu=1, vtype=bool)
            kept = RankAndCrowding().do(problem, Population.new(X=masks, F=scores), n_survive=min(self.ctx.pop_size, len(masks)))
            self.evolution = _Evolution(NSGA2, self.ctx, 3, self.ctx.pop_size, self.rng, initial=kept.get("X").astype(bool))
            self.phase = 2
        else:
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


class ShapleyRanking(Scripted):
    """Monte Carlo Shapley values of each position for N-hat and S-hat, then nested top-k.

    The value of a set X is (N-hat(X), S-hat(X)); the empty mask is scored once. Each random
    permutation scores its n prefix masks, so one permutation gives a marginal contribution of every
    position to both objectives. Permutations are drawn until only 2n evaluations are left, which go
    to the nested top-k masks by the Shapley value for N and for S (most negative first, i.e. the
    positions whose presence in X lowers the objective most).
    """

    def script(self):
        n = self.ctx.n
        base = (yield np.zeros((1, n), dtype=bool))[0, :2]
        left = self.ctx.budget - 1 - 2 * n
        total, count = np.zeros((n, 2)), 0
        while left >= n or count == 0:
            order = self.rng.permutation(n)
            values = np.vstack([base, (yield nested(order))[:, :2]])
            total[order] += np.diff(values, axis=0)
            count += 1
            left -= n
        self.shapley = total / count
        yield nested(np.argsort(self.shapley[:, 0], kind="stable"))
        yield nested(np.argsort(self.shapley[:, 1], kind="stable"))


METHODS: dict[str, Callable[[Context, np.random.Generator], Method]] = {
    "nsga2": NSGA2Search,
    "nsga2_nc": lambda ctx, rng: NSGA2Search(replace(ctx, objectives=(0, 2)), rng),
    "moead": MOEADSearch,
    "nsga3": NSGA3Search,
    "smsemoa": SMSEMOASearch,
    "shapley": ShapleyRanking,
    "nsga2_seeded": SeededNSGA2Search,
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
