"""
Common interface for basket weight optimizers.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple


class BaseOptimizer(ABC):
    """
    Base class for optimizers that search for basket weights.

    Every optimizer searches the same space as BayesianOptimizer: n_assets - 1
    free weights within `bounds`, with the last weight set by normalization
    (weights sum to zero, then scaled so the last weight is -1). The objective
    is maximized and the evaluation budget is n_trials calls, so results are
    directly comparable with Bayesian Optimization.
    """

    def __init__(
        self,
        objective_func: Callable,
        n_assets: int,
        n_trials: int = 50,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        normalize_weights: bool = True,
        seed: int = 42
    ):
        """
        Initialize optimizer.

        Parameters:
        -----------
        objective_func : Callable
            Function that takes full weights (length n_assets) and returns a
            performance score to maximize, e.g. from create_objective_function
        n_assets : int
            Number of assets in the basket
        n_trials : int
            Maximum number of objective evaluations
        bounds : Tuple[float, float]
            Lower and upper bound for each free weight
        normalize_weights : bool
            Whether to normalize weights so the last weight is -1
        seed : int
            Random seed
        """
        if n_assets < 2:
            raise ValueError("A basket needs at least 2 assets")

        self.objective_func = objective_func
        self.n_assets = n_assets
        self.n_trials = n_trials
        self.lower, self.upper = bounds
        self.normalize_weights = normalize_weights
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Free search dimensions (last weight is determined by normalization)
        self.dim = n_assets - 1

        # Store optimization results
        self.n_evals = 0
        self.best_weights = None
        self.best_score = -np.inf
        self.history: List[float] = []  # Best score after each evaluation

    @property
    def budget_left(self) -> int:
        return self.n_trials - self.n_evals

    def to_weights(self, x: np.ndarray) -> np.ndarray:
        """
        Map free weights to full normalized weights.

        Uses the same convention as BayesianOptimizer.normalize_weights_array.
        """
        x = np.asarray(x, dtype=float)
        weights = np.append(x, -np.sum(x))

        if self.normalize_weights and weights[-1] != 0:
            weights = weights / abs(weights[-1])
            if weights[-1] > 0:
                weights = -weights

        return weights

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate free weights x, counting the call against the budget.

        Returns:
        --------
        float
            Score to maximize (-1e6 for invalid scores)
        """
        if self.budget_left <= 0:
            raise RuntimeError("Evaluation budget exhausted")

        x = np.clip(np.asarray(x, dtype=float), self.lower, self.upper)
        weights = self.to_weights(x)
        score = float(self.objective_func(weights))
        if np.isnan(score) or np.isinf(score):
            score = -1e6

        self.n_evals += 1
        if score > self.best_score:
            self.best_score = score
            self.best_weights = weights.copy()
        self.history.append(self.best_score)

        return score

    @abstractmethod
    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Run optimization.

        Returns:
        --------
        Tuple[np.ndarray, float]
            (best_weights, best_score) tuple, as BayesianOptimizer.optimize
        """
