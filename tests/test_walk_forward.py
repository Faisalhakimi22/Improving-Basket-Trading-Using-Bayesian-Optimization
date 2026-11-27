import numpy as np

from src.walk_forward import walk_forward, walk_forward_splits
from tests.test_optimizer_runner import cointegrated_log_prices


def test_splits_are_consecutive_and_out_of_sample():
    splits = walk_forward_splits(n=1000, train_size=400, test_size=150)

    assert splits == [(0, 400, 400, 550), (150, 550, 550, 700),
                      (300, 700, 700, 850), (450, 850, 850, 1000)]
    for (_, _, _, prev_end), (_, _, next_start, _) in zip(splits, splits[1:]):
        assert prev_end == next_start


def test_anchored_splits_start_at_zero():
    splits = walk_forward_splits(n=1000, train_size=400, test_size=150, anchored=True)

    assert [s[0] for s in splits] == [0, 0, 0, 0]
    assert [s[1] for s in splits] == [400, 550, 700, 850]


def test_walk_forward_summarizes_each_optimizer():
    prices = cointegrated_log_prices(n_days=500)
    per_window, summary = walk_forward(
        prices, optimizers=['cmaes', 'turbo'], train_size=250, test_size=100,
        n_trials=8, verbose=False
    )

    assert len(per_window) == 2 * 2  # 2 windows x 2 optimizers
    assert (per_window['n_evals'] <= 8).all()
    assert list(summary.index) == ['cmaes', 'turbo']
    assert (summary[('windows', 'count')] == 2).all()
    assert np.isfinite(summary[('test_sharpe_ratio', 'mean')]).all()
