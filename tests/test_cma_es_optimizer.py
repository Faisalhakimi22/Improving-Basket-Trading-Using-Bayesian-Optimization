import numpy as np

from src.optimizers import CMAESOptimizer

# Representable optimum: free weights (0.5, 0.5) give full weights (0.5, 0.5, -1)
TARGET = np.array([0.5, 0.5, -1.0])


def objective(weights):
    return -np.sum((weights - TARGET) ** 2)


def test_cma_es_finds_optimum_within_budget():
    optimizer = CMAESOptimizer(objective, n_assets=3, n_trials=120, seed=0)
    weights, score = optimizer.optimize()

    assert optimizer.n_evals <= 120
    assert len(weights) == 3
    assert weights[-1] == -1.0
    assert score > -1e-2
    np.testing.assert_allclose(weights, TARGET, atol=0.1)


def test_cma_es_history_and_seed():
    first = CMAESOptimizer(objective, n_assets=3, n_trials=30, seed=1)
    first.optimize()
    second = CMAESOptimizer(objective, n_assets=3, n_trials=30, seed=1)
    second.optimize()

    assert len(first.history) == first.n_evals
    assert np.all(np.diff(first.history) >= 0)
    assert first.history == second.history
