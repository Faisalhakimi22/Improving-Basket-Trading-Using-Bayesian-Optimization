"""
TuRBO (Trust Region Bayesian Optimization) optimizer.
"""

import warnings
import numpy as np
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from typing import Callable, Optional, Tuple
from .base_optimizer import BaseOptimizer


class TuRBOOptimizer(BaseOptimizer):
    """
    TuRBO-1 optimizer for basket trading weights (Eriksson et al., 2019).

    A local Gaussian Process model is fitted to the points in the current run,
    and Thompson sampling picks the next point inside a trust region around
    the best point. The trust region grows after repeated successes, shrinks
    after repeated failures, and the run restarts when it becomes too small.
    All runs share the n_trials evaluation budget.
    """

    def __init__(
        self,
        objective_func: Callable,
        n_assets: int,
        n_trials: int = 50,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        normalize_weights: bool = True,
        seed: int = 42,
        n_init: Optional[int] = None,
        n_candidates: Optional[int] = None,
        length_init: float = 0.8,
        length_min: float = 0.5 ** 7,
        length_max: float = 1.6
    ):
        """
        Parameters (in addition to BaseOptimizer):
        -----------
        n_init : int, optional
            Latin hypercube points at the start of each run (default: 2 * dim,
            at least 5)
        n_candidates : int, optional
            Thompson sampling candidates per step, rounded up to a power of 2
            (default: min(100 * dim, 1000))
        length_init, length_min, length_max : float
            Trust region side length, relative to the unit cube
        """
        super().__init__(objective_func, n_assets, n_trials, bounds, normalize_weights, seed)
        self.n_init = n_init if n_init is not None else max(5, 2 * self.dim)
        self.n_candidates = n_candidates if n_candidates is not None else min(100 * self.dim, 1000)
        self.sobol_m = int(np.ceil(np.log2(self.n_candidates)))  # Sobol needs a power of 2
        self.length_init = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.success_tol = 3
        self.failure_tol = max(4, self.dim)
        self.n_restarts = 0

    def _from_unit(self, z: np.ndarray) -> np.ndarray:
        return self.lower + z * (self.upper - self.lower)

    def _fit_gp(self, Z: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=np.full(self.dim, 0.5), length_scale_bounds=(5e-3, 2.0), nu=2.5)
            + WhiteKernel(1e-3, (1e-6, 1e-1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=int(self.rng.integers(2 ** 31))
        )

        # Failed evaluations score -1e6; cap them at the worst valid score so
        # they don't dominate the GP fit
        valid = y > -1e5
        if valid.any():
            y = np.where(valid, y, y[valid].min())

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            gp.fit(Z, y)
        return gp

    def _trust_region(self, gp: GaussianProcessRegressor, center: np.ndarray,
                      length: float) -> Tuple[np.ndarray, np.ndarray]:
        """Box around center, stretched along dimensions with long lengthscales."""
        lengthscales = np.atleast_1d(gp.kernel_.k1.k2.length_scale)
        scale = lengthscales / np.prod(lengthscales) ** (1.0 / self.dim)
        lb = np.clip(center - scale * length / 2, 0.0, 1.0)
        ub = np.clip(center + scale * length / 2, 0.0, 1.0)
        return lb, ub

    def _run(self, verbose: bool) -> None:
        """One TuRBO run from a fresh Latin hypercube design."""
        n_init = min(self.n_init, self.budget_left)
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        Z = sampler.random(n_init)
        y = np.array([self.evaluate(self._from_unit(z)) for z in Z])

        length = self.length_init
        n_success = n_failure = 0

        while self.budget_left > 0 and length >= self.length_min:
            gp = self._fit_gp(Z, y)
            center = Z[np.argmax(y)]
            lb, ub = self._trust_region(gp, center, length)

            sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
            candidates = lb + (ub - lb) * sobol.random_base2(self.sobol_m)

            # Thompson sampling: one posterior draw, take its maximizer
            draw = gp.sample_y(candidates, random_state=int(self.rng.integers(2 ** 31))).ravel()
            z_next = candidates[np.argmax(draw)]
            y_next = self.evaluate(self._from_unit(z_next))

            if y_next > y.max() + 1e-3 * abs(y.max()):
                n_success, n_failure = n_success + 1, 0
            else:
                n_success, n_failure = 0, n_failure + 1

            if n_success == self.success_tol:
                length = min(2.0 * length, self.length_max)
                n_success = 0
            elif n_failure == self.failure_tol:
                length /= 2.0
                n_failure = 0

            Z = np.vstack([Z, z_next])
            y = np.append(y, y_next)

            if verbose:
                print(f"  Eval {self.n_evals}: best score = {self.best_score:.4f}, length = {length:.3f}")

    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Run TuRBO, restarting when the trust region collapses, until the
        evaluation budget is used.
        """
        self._run(verbose)
        while self.budget_left > 0:
            self.n_restarts += 1
            if verbose:
                print(f"  Trust region collapsed, restart {self.n_restarts}")
            self._run(verbose)

        return self.best_weights, self.best_score
