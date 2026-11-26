import numpy as np

from src.optimizers import TuRBOOptimizer

# Representable optimum: free weights (0.5, 0.5) give full weights (0.5, 0.5, -1)
TARGET = np.array([0.5, 0.5, -1.0])


def objective(weights):
    return -np.sum((weights - TARGET) ** 2)


def test_turbo_finds_optimum_within_budget():
    optimizer = TuRBOOptimizer(objective, n_assets=3, n_trials=40, seed=0)
    weights, score = optimizer.optimize()

    assert optimizer.n_evals == 40
    assert len(weights) == 3
    assert weights[-1] == -1.0
    assert score > -1e-2
    np.testing.assert_allclose(weights, TARGET, atol=0.1)


def test_turbo_restarts_share_budget():
    # A large minimum length makes the trust region collapse early
    optimizer = TuRBOOptimizer(objective, n_assets=3, n_trials=30, seed=0, length_min=0.4)
    optimizer.optimize()

    assert optimizer.n_restarts > 0
    assert optimizer.n_evals == 30
    assert len(optimizer.history) == 30
    assert np.all(np.diff(optimizer.history) >= 0)
