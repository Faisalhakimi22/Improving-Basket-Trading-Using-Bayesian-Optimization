"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
"""

import numpy as np
import cma
from typing import Callable, Optional, Tuple
from .base_optimizer import BaseOptimizer


class CMAESOptimizer(BaseOptimizer):
    """
    CMA-ES optimizer for basket trading weights (Hansen, 2016), using the
    `cma` package with box constraints on the free weights.
    """

    def __init__(
        self,
        objective_func: Callable,
        n_assets: int,
        n_trials: int = 50,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        normalize_weights: bool = True,
        seed: int = 42,
        popsize: Optional[int] = None,
        sigma0: Optional[float] = None
    ):
        """
        Parameters (in addition to BaseOptimizer):
        -----------
        popsize : int, optional
            Population size per generation (default: CMA-ES default,
            4 + 3 * ln(dim))
        sigma0 : float, optional
            Initial step size (default: a quarter of the bounds width)
        """
        super().__init__(objective_func, n_assets, n_trials, bounds, normalize_weights, seed)
        self.popsize = popsize
        self.sigma0 = sigma0 if sigma0 is not None else 0.25 * (self.upper - self.lower)

    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Run CMA-ES until the evaluation budget is used or CMA-ES converges.
        """
        x0 = np.full(self.dim, (self.lower + self.upper) / 2)
        options = {
            'bounds': [self.lower, self.upper],
            'seed': int(self.rng.integers(1, 2 ** 31)),  # cma treats seed 0 as random
            'verbose': -9,
        }
        if self.popsize is not None:
            options['popsize'] = self.popsize

        es = cma.CMAEvolutionStrategy(x0, self.sigma0, options)

        while not es.stop() and self.budget_left > 0:
            solutions = es.ask()

            if len(solutions) > self.budget_left:
                # Not enough budget for a full generation: spend the rest and stop
                for x in solutions[:self.budget_left]:
                    self.evaluate(x)
                break

            # CMA-ES minimizes, so negate the score
            es.tell(solutions, [-self.evaluate(x) for x in solutions])

            if verbose:
                print(f"  Generation {es.countiter}: best score = {self.best_score:.4f}")

        return self.best_weights, self.best_score
