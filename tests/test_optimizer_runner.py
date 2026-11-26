import numpy as np
import pandas as pd

from src.optimizer_runner import OptimizationRunner, train_test_split


def cointegrated_log_prices(n_days=500, seed=0):
    """Two random walks plus a third asset tied to them by a mean-reverting spread."""
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.normal(0, 0.01, n_days)) + 4.0
    b = np.cumsum(rng.normal(0, 0.01, n_days)) + 4.5
    spread = np.zeros(n_days)
    for t in range(1, n_days):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.01)
    c = 0.5 * a + 0.5 * b + spread
    index = pd.bdate_range('2020-01-01', periods=n_days)
    return pd.DataFrame({'A': a, 'B': b, 'C': c}, index=index)


def test_runner_compares_optimizers_on_held_out_data():
    train, test = train_test_split(cointegrated_log_prices(), test_size=0.3)
    assert train.index[-1] < test.index[0]

    runner = OptimizationRunner(train, test, n_trials=15)
    summary = runner.run(verbose=False)

    assert list(summary.index) == ['bo', 'cmaes', 'turbo']
    assert (summary['n_evals'] <= 15).all()
    assert np.isfinite(summary['test_sharpe_ratio']).all()
    for weights in summary['weights']:
        assert len(weights) == 3
        assert weights[-1] == -1.0
